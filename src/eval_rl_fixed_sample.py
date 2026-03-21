import json
import argparse
import random
import sqlite3
import time
import re
import os
from pathlib import Path

# 🔥 fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from prompting import encode_prompt
from src.sql_validator import validate_sql_schema

# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize_sql(sql):
    sql = sql.replace('"', "'")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip().lower().rstrip(";")

def normalize_result(res):
    try:
        return sorted([str(r) for r in res])
    except:
        return []

# -------------------------------
# EXECUTION CHECK
# -------------------------------
def check_execution(pred_sql, gold_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')

        start = time.monotonic()

        def timeout():
            return 1 if (time.monotonic() - start) > 2 else 0

        conn.set_progress_handler(timeout, 10000)

        cur = conn.cursor()
        cur.execute(pred_sql)
        pred = cur.fetchall()

        cur.execute(gold_sql)
        gold = cur.fetchall()

        conn.close()

        return normalize_result(pred) == normalize_result(gold)

    except:
        return False

# -------------------------------
# CLEAN SQL
# -------------------------------
def clean_sql(sql):
    sql = sql.strip()
    sql = sql.replace("\n", " ")
    sql = re.sub(r"\s+", " ", sql)

    low = sql.lower()

    if low.startswith("select"):
        return sql

    if " from " in low:
        return "select " + sql

    if any(x in low for x in ["count", "avg", "sum", "min", "max"]):
        return "select " + sql

    return "select 1"

# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--shuffle_dev", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    db_root = root / "data/database"
    dev_path = root / "data/dev.json"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

    base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)

    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model = model.merge_and_unload()
    model.eval()

    with open(dev_path) as f:
        data = json.load(f)

    if args.shuffle_dev:
        random.Random(args.shuffle_seed).shuffle(data)

    data = data[: args.num_samples]
    total = len(data)

    em = 0
    ex = 0
    constraint_ok = 0

    print(f"\n🚀 Evaluating {total} samples...\n")

    for i, item in enumerate(data, 1):

        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]

        db_path = db_root / db_id / f"{db_id}.sqlite"

        # -------------------------------
        # GENERATE
        # -------------------------------
        inputs = encode_prompt(
            tokenizer,
            question,
            db_id,
            device=device,
            max_input_tokens=512
        ).unsqueeze(0)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=120,
            num_beams=8,
            repetition_penalty=1.2   # 🔥 improves quality
        )

        pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        pred_sql = clean_sql(pred_sql)

        # -------------------------------
        # CONSTRAINT CHECK
        # -------------------------------
        try:
            valid, _ = validate_sql_schema(pred_sql, str(db_path))
        except:
            valid = False

        # 🔥 SMART RETRY (KEY IMPROVEMENT)
        if not valid:

            retry_prompt = f"""
Fix this SQL query.

SQL:
{pred_sql}

Schema:
{db_id}

Return ONLY corrected SQL.
"""

            retry_inputs = encode_prompt(
                tokenizer,
                retry_prompt,
                db_id,
                device=device,
                max_input_tokens=512
            ).unsqueeze(0)

            retry_outputs = model.generate(
                input_ids=retry_inputs,
                max_new_tokens=120,
                num_beams=8
            )

            retry_sql = tokenizer.decode(retry_outputs[0], skip_special_tokens=True).strip()
            retry_sql = clean_sql(retry_sql)

            try:
                valid_retry, _ = validate_sql_schema(retry_sql, str(db_path))
            except:
                valid_retry = False

            if valid_retry:
                pred_sql = retry_sql
                valid = True

        if valid:
            constraint_ok += 1

        # -------------------------------
        # METRICS
        # -------------------------------
        if normalize_sql(pred_sql) == normalize_sql(gold_sql):
            em += 1

        if check_execution(pred_sql, gold_sql, db_path):
            ex += 1

        if i % 20 == 0 or i == total:
            print(
                f"Progress: {i}/{total} | "
                f"EM: {em/i:.3f} | "
                f"EX: {ex/i:.3f} | "
                f"Constraint: {constraint_ok/i:.3f}"
            )

    print("\n==========================================")
    print("🎯 FINAL RESULTS (CONSTRAINED)")
    print("==========================================")

    print(f"Exact Match Accuracy  : {em/total:.4f}")
    print(f"Execution Accuracy    : {ex/total:.4f}")
    print(f"Constraint Rate       : {constraint_ok/total:.4f}")

    print("==========================================\n")


if __name__ == "__main__":
    main()