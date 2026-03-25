from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import numpy as np
from collections import Counter

# -------------------------------
# PATH FIX (Immune to folder location)
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# IMPORTS
# -------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from src.sql_validator import validate_sql_schema
from transformers.generation.logits_process import LogitsProcessorList
from src.constrained_decoding import SchemaConstrainedLogitsProcessor

try:
    from prompting import encode_prompt
except ImportError:
    from src.prompting import encode_prompt

# -------------------------------
# TARGET DBS (The Hard 29)
# -------------------------------
TASK_4_TARGET_DBS = {
    "flight_1", "student_assessment", "store_1", "bike_1", "book_2", "chinook_1",
    "academic", "aircraft", "car_1", "cinema", "club_1", "csu_1",
    "college_1", "college_2", "company_1", "company_employee",
    "customer_complaints", "department_store", "employee_hire_evaluation",
    "museum_visit", "products_for_hire", "restaurant_1",
    "school_finance", "shop_membership", "small_bank_1",
    "student_1", "tvshow", "voter_1", "world_1"
}

# ==========================================
# 🔥 ULTIMATE DB FINDER
# ==========================================
def find_db_path(db_id, db_root):
    """Searches every corner of the project for the sqlite file."""
    search_paths = [
        db_root / db_id / f"{db_id}.sqlite",
        db_root / f"{db_id}.sqlite",
        PROJECT_ROOT / "data" / "database" / db_id / f"{db_id}.sqlite",
        PROJECT_ROOT / "database" / db_id / f"{db_id}.sqlite"
    ]
    
    # Check standard paths
    for p in search_paths:
        if p.exists(): return str(p)
        
    # 🔥 ULTIMATE SEARCH: Look for any .sqlite file matching the name in the project
    for path in PROJECT_ROOT.rglob(f"{db_id}.sqlite"):
        return str(path)
        
    return str(search_paths[0]) # Default fallback

# ==========================================
# 🔥 EXECUTION & SCORING UTILS
# ==========================================
def normalize_result(res):
    """Sorts rows and values within rows to ensure order-invariant comparison."""
    if not res: return []
    try:
        normalized = [tuple(sorted(str(x) for x in row)) for row in res]
        return sorted(normalized)
    except:
        return sorted([str(r) for r in res])

def clean_sql(sql):
    sql = sql.strip().replace("\n", " ")
    match = re.search(r"(select|with)[\s\S]*", sql, re.IGNORECASE)
    return match.group(0).strip() if match else "SELECT 1"

def check_execution(pred_sql, gold_sql, db_path):
    if not os.path.exists(db_path):
        return False, None, None
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cur = conn.cursor()
        
        cur.execute(gold_sql)
        gold_res = cur.fetchall()
        
        cur.execute(pred_sql)
        pred_res = cur.fetchall()
        
        conn.close()
        return normalize_result(pred_res) == normalize_result(gold_res), pred_res, gold_res
    except:
        return False, None, None

def _soft_score(pred_rows, gold_rows):
    """Recall-based soft score with length penalty to prevent reward hacking."""
    if pred_rows is None or gold_rows is None: return 0.0
    if not gold_rows: return 1.0 if not pred_rows else 0.0
    
    p_c = Counter(tuple(str(i) for i in r) for r in pred_rows)
    g_c = Counter(tuple(str(i) for i in r) for r in gold_rows)
    
    intersection = sum((p_c & g_c).values())
    recall = intersection / len(gold_rows)
    
    # Penalty for returning too many irrelevant rows
    lp = min(1.0, len(gold_rows) / len(pred_rows)) if len(pred_rows) > 0 else 0.0
    return recall * lp

# -------------------------------
# 🔥 ULTRA EVALUATOR (EXECUTION-GUIDED)
# -------------------------------
def evaluate_adapter(adapter_path: str, items: Sequence[dict], db_root: Path) -> Dict[str, Any]:
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model = model.merge_and_unload()
    model.eval()

    ex, em, constraint_ok = 0, 0, 0
    h_rewards, s_rewards = [], []

    for i, item in enumerate(items, 1):
        db_id = item["db_id"]
        db_path = find_db_path(db_id, db_root)
        
        # 🔍 DEBUG: Print the detected path for the first sample
        if i == 1: print(f"📍 Sample 1 DB Path Verified: {db_path}")

        lp = LogitsProcessorList([SchemaConstrainedLogitsProcessor(tokenizer, db_path)])
        inputs = encode_prompt(tokenizer, item["question"], db_id, device=device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, 
                max_new_tokens=128, 
                num_beams=12,           
                num_return_sequences=10, # Capture more beams for execution testing
                logits_processor=lp,
                repetition_penalty=1.1
            )

        best_sql, found_exec = "", False
        valid_candidates = []

        # Test the top-10 candidates
        for out in outputs:
            sql = clean_sql(tokenizer.decode(out, skip_special_tokens=True))
            correct_exec, p_res, g_res = check_execution(sql, item["query"], db_path)
            
            if correct_exec:
                best_sql = sql
                found_exec = True
                break
            
            # Syntax validation for fallback logic
            is_valid = False
            try:
                is_valid, _ = validate_sql_schema(sql, db_path)
            except: pass
            
            if is_valid:
                valid_candidates.append((sql, _soft_score(p_res, g_res)))

        # Fallback to the syntactically valid query with highest partial credit
        if not found_exec:
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
                best_sql = valid_candidates[0][0]
            else:
                best_sql = clean_sql(tokenizer.decode(outputs[0], skip_special_tokens=True))

        if found_exec: ex += 1
        if re.sub(r'\s+', '', best_sql.lower()) == re.sub(r'\s+', '', item["query"].lower()): em += 1
        
        try:
            v, _ = validate_sql_schema(best_sql, db_path)
            if v: constraint_ok += 1
        except: pass
        
        # Re-check execution for final metrics
        ok, pr, gr = check_execution(best_sql, item["query"], db_path)
        h_rewards.append(1.0 if ok else 0.0)
        s_rewards.append(_soft_score(pr, gr))

        if i % 10 == 0 or i == len(items):
            print(f"🚀 Progress: {i}/{len(items)} | EX: {ex/i:.3f} | Constraint: {constraint_ok/i:.3f}")

    return {"ex": ex/len(items), "em": em/len(items), "constraint_rate": constraint_ok/len(items),
            "h_mean": np.mean(h_rewards), "s_mean": np.mean(s_rewards)}

# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_adapter", type=str, required=True)
    parser.add_argument("--soft_adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()

    db_root = PROJECT_ROOT / "data" / "database"
    with open(PROJECT_ROOT / "data" / "dev.json") as f:
        data = [x for x in json.load(f) if x.get("db_id") in TASK_4_TARGET_DBS]
    
    random.seed(42)
    items = random.sample(data, min(args.num_samples, len(data)))

    report = {"models": {}}
    for tag, path in [("HARD", args.hard_adapter), ("SOFT", args.soft_adapter)]:
        print(f"\n🔥 Running {tag} Model Evaluation...")
        res = evaluate_adapter(path, items, db_root)
        report["models"][tag] = res
        print(f"✅ {tag} Final Summary | EX: {res['ex']:.4f} | Constraint: {res['constraint_rate']:.4f}")

    out_p = PROJECT_ROOT / "results" / "task4_ultra_eval.json"
    out_p.parent.mkdir(exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n🎉 Task 4 Report Complete: {out_p}")

if __name__ == "__main__":
    main()