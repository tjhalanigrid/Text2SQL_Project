import json
import subprocess
import sys
import argparse
import random
import sqlite3
import time
import re
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from prompting import encode_prompt

# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize_sql(sql):
    sql = sql.replace('"', "'")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip().lower().rstrip(";")


# -------------------------------
# 🔥 SAFE RESULT NORMALIZATION (FIX)
# -------------------------------
def normalize_result(res):
    try:
        return sorted([str(r) for r in res])
    except:
        return []


# -------------------------------
# EXECUTION CHECK (FIXED)
# -------------------------------
def check_execution(pred_sql, gold_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')

        start_time = time.monotonic()

        def timeout_handler():
            return 1 if (time.monotonic() - start_time) > 2.0 else 0

        conn.set_progress_handler(timeout_handler, 10000)

        cursor = conn.cursor()

        cursor.execute(pred_sql)
        pred_res = cursor.fetchall()

        cursor.execute(gold_sql)
        gold_res = cursor.fetchall()

        conn.close()

        # 🔥 FIXED COMPARISON
        return normalize_result(pred_res) == normalize_result(gold_res)

    except Exception:
        return False


# -------------------------------
# SPIDER PARSER
# -------------------------------
def _parse_spider_accuracy(stdout: str, metric_type: str):
    for line in stdout.splitlines():
        if metric_type == "exec" and line.strip().startswith("execution"):
            try:
                return float(line.split()[-1])
            except:
                pass
        elif metric_type == "match" and line.strip().startswith("exact"):
            try:
                return float(line.split()[-1])
            except:
                pass
    return None


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default= 500)
    parser.add_argument("--shuffle_dev", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter_dir = project_root / args.adapter

    db_root = project_root / "data" / "database"
    table_json = project_root / "data" / "tables.json"
    dev_json = project_root / "data" / "dev.json"

    pred_path = project_root / "temp_predictions.txt"
    temp_gold_path = project_root / "temp_gold.sql"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    BASE_MODEL = "Salesforce/codet5-base"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n📦 Loading Model: {args.adapter}")

    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    adapter_for_peft = os.path.relpath(adapter_dir, project_root)

    model = PeftModel.from_pretrained(
        base,
        adapter_for_peft,
        local_files_only=True
    ).to(device)

    model = model.merge_and_unload()
    model.eval()

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    with dev_json.open() as f:
        dev = json.load(f)

    if args.shuffle_dev:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(dev)

    dev = dev[: args.num_samples]
    total = len(dev)

    gen_kwargs = dict(
        max_new_tokens=160,
        num_beams=8,
        length_penalty=0.8,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"\n🚀 Evaluating {total} samples...\n")

    em_correct = 0
    ex_correct = 0

    with pred_path.open("w") as out_pred, temp_gold_path.open("w") as out_gold, torch.no_grad():
        for i, ex in enumerate(dev, start=1):

            db_id = ex["db_id"]
            question = ex["question"]
            gold_query = ex["query"]
            db_path = db_root / db_id / f"{db_id}.sqlite"

            # -------------------------------
            # GENERATE SQL
            # -------------------------------
            input_ids = encode_prompt(
                tokenizer,
                question,
                db_id,
                device=device,
                max_input_tokens=512
            )

            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # -------------------------------
            # SAVE FOR SPIDER EVAL
            # -------------------------------
            out_pred.write(f"{pred_sql}\n")
            out_gold.write(f"{gold_query}\t{db_id}\n")

            # -------------------------------
            # LIVE METRICS
            # -------------------------------
            if normalize_sql(pred_sql) == normalize_sql(gold_query):
                em_correct += 1

            if check_execution(pred_sql, gold_query, db_path):
                ex_correct += 1

            if i % 20 == 0 or i == total:
                print(
                    f"Progress: {i}/{total} | "
                    f"EM: {(em_correct/i)*100:.2f}% | "
                    f"EX: {(ex_correct/i)*100:.2f}%"
                )

    print("\n🚀 Running Official Spider Evaluation...\n")

    eval_script = project_root / "spider_eval" / "evaluation.py"

    # EXACT MATCH
    cmd_match = [
        sys.executable, str(eval_script),
        "--gold", str(temp_gold_path),
        "--pred", str(pred_path),
        "--etype", "match",
        "--db", str(db_root),
        "--table", str(table_json),
    ]

    proc_match = subprocess.run(cmd_match, capture_output=True, text=True)
    exact_acc = _parse_spider_accuracy(proc_match.stdout, "match")

    # EXECUTION
    cmd_exec = [
        sys.executable, str(eval_script),
        "--gold", str(temp_gold_path),
        "--pred", str(pred_path),
        "--etype", "exec",
        "--db", str(db_root),
        "--table", str(table_json),
    ]

    proc_exec = subprocess.run(cmd_exec, capture_output=True, text=True)
    exec_acc = _parse_spider_accuracy(proc_exec.stdout, "exec")

    print("==========================================")
    print(f"🎯 OFFICIAL SPIDER RESULTS FOR: {args.adapter}")
    print("==========================================")

    print(f"Exact Match Accuracy  : {exact_acc*100:.2f}%" if exact_acc else "EM parsing failed")
    print(f"Execution Accuracy    : {exec_acc*100:.2f}%" if exec_acc else "EX parsing failed")

    print("==========================================\n")


if __name__ == "__main__":
    main()


