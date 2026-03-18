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

from src.prompting import encode_prompt  # ✅ FIXED IMPORT

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

        return normalize_result(pred_res) == normalize_result(gold_res)

    except Exception:
        return False


# -------------------------------
# 🔥 SCHEMA CONSTRAINT
# -------------------------------
def extract_schema_tokens(schema_text):
    schema_text = schema_text.lower()

    tables = set(re.findall(r'(\w+)\s*\(', schema_text))
    columns = set()

    for block in re.findall(r'\((.*?)\)', schema_text):
        for col in block.split(","):
            col_name = col.strip().split()[0]
            if col_name:
                columns.add(col_name)

    return tables, columns


def is_schema_valid(sql, tables, columns):
    sql = sql.lower()

    used_tables = re.findall(r'(?:from|join)\s+(\w+)', sql)

    if used_tables and not any(t in tables for t in used_tables):
        return False

    col_blocks = re.findall(r'select\s+(.*?)\s+from', sql)

    for block in col_blocks:
        for col in block.split(","):
            col = col.strip()

            if col == "*" or "(" in col:
                continue

            if col not in columns:
                return False

    return True


# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=700)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter_dir = project_root / args.adapter

    db_root = project_root / "data" / "database"
    dev_json = project_root / "data" / "dev.json"
    table_json = project_root / "data" / "tables.json"

    pred_base = project_root / "pred_base.txt"
    pred_const = project_root / "pred_const.txt"
    gold_file = project_root / "gold.sql"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\n📦 Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)

    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model = model.merge_and_unload()
    model.eval()

    with dev_json.open() as f:
        data = json.load(f)[:args.num_samples]

    gen_kwargs = dict(max_new_tokens=160, num_beams=5)

    em_base = ex_base = 0
    em_const = ex_const = 0
    
    # 📊 ADDED: Trackers for constraint satisfaction
    constraint_valid_base = 0
    constraint_valid_const = 0

    with open(pred_base, "w") as pb, open(pred_const, "w") as pc, open(gold_file, "w") as gf:

        for i, ex in enumerate(data, 1):

            db_id = ex["db_id"]
            question = ex["question"]
            gold_sql = ex["query"]

            db_path = db_root / db_id / f"{db_id}.sqlite"

            # ---------------- BASE MODEL ----------------
            input_ids = encode_prompt(tokenizer, question, db_id, device=device)
            input_ids = input_ids.unsqueeze(0).to(device)

            outputs = model.generate(input_ids=input_ids, **gen_kwargs)
            pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            pb.write(pred_sql + "\n")
            gf.write(f"{gold_sql}\t{db_id}\n")

            # metrics
            if normalize_sql(pred_sql) == normalize_sql(gold_sql):
                em_base += 1

            if check_execution(pred_sql, gold_sql, db_path):
                ex_base += 1

            # ---------------- CONSTRAINED ----------------
            schema = ""  # optional: load schema if needed
            tables, columns = extract_schema_tokens(schema)

            # 📊 ADDED: Check base constraint valid
            constraint_valid_base += 1 if is_schema_valid(pred_sql, tables, columns) else 0

            pred_sql_const = pred_sql

            if tables and not is_schema_valid(pred_sql, tables, columns):
                # retry with constraint prompt
                constraint_prompt = f"""
Schema tables: {', '.join(tables)}
Columns: {', '.join(columns)}
Question: {question}
SQL:
"""
                input_ids = tokenizer(constraint_prompt, return_tensors="pt").to(device)
                outputs = model.generate(**input_ids, **gen_kwargs)
                pred_sql_const = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            pc.write(pred_sql_const + "\n")

            if normalize_sql(pred_sql_const) == normalize_sql(gold_sql):
                em_const += 1

            if check_execution(pred_sql_const, gold_sql, db_path):
                ex_const += 1
                
            # 📊 ADDED: Check constrained constraint valid
            constraint_valid_const += 1 if is_schema_valid(pred_sql_const, tables, columns) else 0

            if i % 20 == 0:
                print(f"{i}/{len(data)} done...")

    total = len(data)
    
    # 📊 ADDED: Calculate the rates
    constraint_rate_base = constraint_valid_base / total
    constraint_rate_const = constraint_valid_const / total

    print("\n==========================================")
    print("🎯 FINAL RESULTS (FAIR COMPARISON)")
    print("==========================================")
    # 📊 ADDED: Appended Constraint Satisfaction Rate to prints
    print(f"Unconstrained → EM: {em_base/total:.2%} | EX: {ex_base/total:.2%} | Constraint Satisfaction Rate: {constraint_rate_base:.2%}")
    print(f"Constrained   → EM: {em_const/total:.2%} | EX: {ex_const/total:.2%} | Constraint Satisfaction Rate: {constraint_rate_const:.2%}")
    print("==========================================\n")


if __name__ == "__main__":
    main()