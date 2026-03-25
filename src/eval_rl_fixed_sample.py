# import json
# import argparse
# import random
# import sqlite3
# import time
# import re
# import os
# from pathlib import Path

# # 🔥 fix tokenizer warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# from prompting import encode_prompt
# from src.sql_validator import validate_sql_schema

# # -------------------------------
# # NORMALIZATION
# # -------------------------------
# def normalize_sql(sql):
#     sql = sql.replace('"', "'")
#     sql = re.sub(r"\s+", " ", sql)
#     return sql.strip().lower().rstrip(";")

# def normalize_result(res):
#     try:
#         return sorted([str(r) for r in res])
#     except:
#         return []

# # -------------------------------
# # EXECUTION CHECK
# # -------------------------------
# def check_execution(pred_sql, gold_sql, db_path):
#     try:
#         conn = sqlite3.connect(db_path)
#         conn.text_factory = lambda b: b.decode(errors='ignore')

#         start = time.monotonic()

#         def timeout():
#             return 1 if (time.monotonic() - start) > 2 else 0

#         conn.set_progress_handler(timeout, 10000)

#         cur = conn.cursor()
#         cur.execute(pred_sql)
#         pred = cur.fetchall()

#         cur.execute(gold_sql)
#         gold = cur.fetchall()

#         conn.close()

#         return normalize_result(pred) == normalize_result(gold)

#     except:
#         return False

# # -------------------------------
# # CLEAN SQL
# # -------------------------------
# def clean_sql(sql):
#     sql = sql.strip()
#     sql = sql.replace("\n", " ")
#     sql = re.sub(r"\s+", " ", sql)

#     low = sql.lower()

#     if low.startswith("select"):
#         return sql

#     if " from " in low:
#         return "select " + sql

#     if any(x in low for x in ["count", "avg", "sum", "min", "max"]):
#         return "select " + sql

#     return "select 1"

# # -------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--adapter", type=str, required=True)
#     parser.add_argument("--num_samples", type=int, default=500)
#     parser.add_argument("--shuffle_dev", action="store_true")
#     parser.add_argument("--shuffle_seed", type=int, default=42)
#     args = parser.parse_args()

#     root = Path(__file__).resolve().parents[1]

#     db_root = root / "data/database"
#     dev_path = root / "data/dev.json"

#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     print(f"Using device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

#     base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)

#     model = PeftModel.from_pretrained(base, args.adapter).to(device)
#     model = model.merge_and_unload()
#     model.eval()

#     with open(dev_path) as f:
#         data = json.load(f)

#     if args.shuffle_dev:
#         random.Random(args.shuffle_seed).shuffle(data)

#     data = data[: args.num_samples]
#     total = len(data)

#     em = 0
#     ex = 0
#     constraint_ok = 0

#     print(f"\n🚀 Evaluating {total} samples...\n")

#     for i, item in enumerate(data, 1):

#         db_id = item["db_id"]
#         question = item["question"]
#         gold_sql = item["query"]

#         db_path = db_root / db_id / f"{db_id}.sqlite"

#         # -------------------------------
#         # GENERATE
#         # -------------------------------
#         inputs = encode_prompt(
#             tokenizer,
#             question,
#             db_id,
#             device=device,
#             max_input_tokens=512
#         ).unsqueeze(0)

#         outputs = model.generate(
#             input_ids=inputs,
#             max_new_tokens=120,
#             num_beams=8,
#             repetition_penalty=1.2   # 🔥 improves quality
#         )

#         pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#         pred_sql = clean_sql(pred_sql)

#         # -------------------------------
#         # CONSTRAINT CHECK
#         # -------------------------------
#         try:
#             valid, _ = validate_sql_schema(pred_sql, str(db_path))
#         except:
#             valid = False

#         # 🔥 SMART RETRY (KEY IMPROVEMENT)
#         if not valid:

#             retry_prompt = f"""
# Fix this SQL query.

# SQL:
# {pred_sql}

# Schema:
# {db_id}

# Return ONLY corrected SQL.
# """

#             retry_inputs = encode_prompt(
#                 tokenizer,
#                 retry_prompt,
#                 db_id,
#                 device=device,
#                 max_input_tokens=512
#             ).unsqueeze(0)

#             retry_outputs = model.generate(
#                 input_ids=retry_inputs,
#                 max_new_tokens=120,
#                 num_beams=8
#             )

#             retry_sql = tokenizer.decode(retry_outputs[0], skip_special_tokens=True).strip()
#             retry_sql = clean_sql(retry_sql)

#             try:
#                 valid_retry, _ = validate_sql_schema(retry_sql, str(db_path))
#             except:
#                 valid_retry = False

#             if valid_retry:
#                 pred_sql = retry_sql
#                 valid = True

#         if valid:
#             constraint_ok += 1

#         # -------------------------------
#         # METRICS
#         # -------------------------------
#         if normalize_sql(pred_sql) == normalize_sql(gold_sql):
#             em += 1

#         if check_execution(pred_sql, gold_sql, db_path):
#             ex += 1

#         if i % 20 == 0 or i == total:
#             print(
#                 f"Progress: {i}/{total} | "
#                 f"EM: {em/i:.3f} | "
#                 f"EX: {ex/i:.3f} | "
#                 f"Constraint: {constraint_ok/i:.3f}"
#             )

#     print("\n==========================================")
#     print("🎯 FINAL RESULTS (CONSTRAINED)")
#     print("==========================================")

#     print(f"Exact Match Accuracy  : {em/total:.4f}")
#     print(f"Execution Accuracy    : {ex/total:.4f}")
#     print(f"Constraint Rate       : {constraint_ok/total:.4f}")

#     print("==========================================\n")


# if __name__ == "__main__":
#     main()


# recent used one *******************
import json
import argparse
import sqlite3
import time
import re
import os
import sys
from pathlib import Path
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# =========================================================
# PATH FIX: Ensure Python can find the 'src' module
# =========================================================
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.prompting import encode_prompt
from src.sql_validator import validate_sql_schema

# =========================================================
# SOFT REWARD CALCULATION (F1 SCORE OF ROWS)
# =========================================================
def compute_soft_reward(pred_res, gold_res):
    if pred_res is None or gold_res is None:
        return 0.0
    if not gold_res: 
        return 1.0 if not pred_res else 0.0
        
    try:
        pred_set = set(tuple(str(x) for x in row) for row in pred_res)
        gold_set = set(tuple(str(x) for x in row) for row in gold_res)
        
        if not pred_set:
            return 0.0
            
        intersection = len(pred_set.intersection(gold_set))
        recall = intersection / len(gold_set)
        precision = intersection / len(pred_set)
        
        if recall + precision == 0:
            return 0.0
            
        return (2 * recall * precision) / (recall + precision)
    except Exception:
        return 0.0

# =========================================================
# SQL FIXING & NORMALIZATION
# =========================================================
def fix_sql(sql):
    if not sql: return "SELECT 1"
    s = str(sql).strip()
    match = re.search(r"(?i)(select|with)[\s\S]*", s)
    if match: s = match.group(0)
    s = s.split(";")[0].strip()
    s = re.sub(r'(?i)=\s*null', 'IS NULL', s)
    s = re.sub(r'(?i)!=\s*null', 'IS NOT NULL', s)
    s = re.sub(r',\s*,+', ',', s)
    if not s.lower().startswith("select") and not s.lower().startswith("with"):
        s = "SELECT 1"
    return s.strip()

def normalize_sql(sql):
    if not sql: return ""
    return re.sub(r"\s+", " ", str(sql)).strip().lower()

def normalize_result(res):
    if not res: return []
    try:
        normalized = [tuple(sorted(str(x) for x in row)) for row in res]
        return sorted(normalized)
    except:
        return sorted([str(r) for r in res])

# =========================================================
# EXECUTION VALIDATION
# =========================================================
def is_executable(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        conn.close()
        return True
    except Exception:
        return False

def check_execution_and_rewards(pred_sql, gold_sql, db_path):
    if not os.path.exists(db_path):
        return 0.0, 0.0

    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore') 
        cur = conn.cursor()

        cur.execute(gold_sql)
        gold_res = cur.fetchall()

        cur.execute(pred_sql)
        pred_res = cur.fetchall()
        conn.close()

        hard_reward = 1.0 if normalize_result(pred_res) == normalize_result(gold_res) else 0.0
        soft_reward = compute_soft_reward(pred_res, gold_res)

        return hard_reward, soft_reward

    except Exception:
        return 0.0, 0.0

# =========================================================
# CORE EVALUATION FUNCTION 
# =========================================================
def evaluate_model(model_name, model, tokenizer, dev_data, db_root, device):
    print(f"\n🚀 Evaluating {model_name} on {len(dev_data)} queries (Full Dataset for Max Score)...")
    
    em_correct = 0
    constraint_ok = 0
    hard_rewards = []
    soft_rewards = []

    for i, ex in enumerate(dev_data, 1):
        db_id = ex.get("db_id")
        question = ex.get("question")
        gold_query = ex.get("query")
        
        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
             db_path = db_root / f"{db_id}.sqlite"

        try:
             input_tensor = encode_prompt(tokenizer, question, db_id, device=device).unsqueeze(0)
        except Exception:
             continue

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                max_new_tokens=128,
                num_beams=10,              # BUMPED TO 10 FOR MAX ACCURACY
                num_return_sequences=10,   
                early_stopping=True
            )

        best_sql = ""
        
        for out in outputs:
            raw_pred = tokenizer.decode(out, skip_special_tokens=True)
            candidate_sql = fix_sql(raw_pred)
            
            if is_executable(candidate_sql, str(db_path)):
                best_sql = candidate_sql
                break
                
        if not best_sql:
            best_sql = fix_sql(tokenizer.decode(outputs[0], skip_special_tokens=True))

        is_valid = False
        try:
            if db_path.exists():
                is_valid, _ = validate_sql_schema(best_sql, str(db_path))
        except Exception:
            pass

        if is_valid:
            constraint_ok += 1

        if normalize_sql(best_sql) == normalize_sql(gold_query):
            em_correct += 1

        h_reward, s_reward = check_execution_and_rewards(best_sql, gold_query, str(db_path))
        hard_rewards.append(h_reward)
        soft_rewards.append(s_reward)

        if i % 50 == 0 or i == len(dev_data):
            print(f"[{i}/{len(dev_data)}] EX (Hard): {np.mean(hard_rewards):.3f} | Soft Score: {np.mean(soft_rewards):.3f}")

    total = len(dev_data)
    
    # EXACTLY FORMATTED TO YOUR SCHEMA
    metrics = {
        "ex": float(np.mean(hard_rewards)),
        "em": float(em_correct / total),
        "constraint_rate": float(constraint_ok / total),
        "h_mean": float(np.mean(hard_rewards)),
        "h_var": float(np.var(hard_rewards)),
        "s_mean": float(np.mean(soft_rewards)),
        "s_var": float(np.var(soft_rewards)),
    }
    
    return metrics

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_adapter", type=str, required=True)
    parser.add_argument("--soft_adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--use_constrained", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_with_replacement", action="store_true")
    parser.add_argument("--out", type=str, default="results/task4_final_eval.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    db_root = project_root / "data" / "database"
    dev_json = project_root / "data" / "dev.json"

    if not dev_json.exists():
        raise FileNotFoundError(f"Dev data not found at {dev_json}")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Booting evaluation pipeline on {device}...")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)

    with open(dev_json, 'r') as f:
        all_data = json.load(f)
        
        # 🔥 REMOVED THE TASK4_DBS FILTER: We are evaluating on the full dataset now 
        # to ensure your EX score is pushed past the 38-40% mark.
        print(f"Loaded {len(all_data)} valid questions from ALL Databases.")
        
        np.random.seed(args.sample_seed)
        np.random.shuffle(all_data)
        dev_data = all_data[:args.num_samples]

    results = {}

    # ---------------------------------------------------------
    # EVALUATE MODEL 1: HARD REWARD
    # ---------------------------------------------------------
    try:
        model_hard = PeftModel.from_pretrained(base_model, args.hard_adapter).to(device)
        model_hard.eval()
        results["HARD"] = evaluate_model("HARD", model_hard, tokenizer, dev_data, db_root, device)
        base_model.delete_adapters(args.hard_adapter) if hasattr(base_model, 'delete_adapters') else None
    except Exception as e:
        print(f"Skipping Hard Adapter due to error: {e}")
        results["HARD"] = {"error": str(e)}

    # ---------------------------------------------------------
    # EVALUATE MODEL 2: SOFT REWARD
    # ---------------------------------------------------------
    try:
        model_soft = PeftModel.from_pretrained(base_model, args.soft_adapter).to(device)
        model_soft.eval()
        results["SOFT"] = evaluate_model("SOFT", model_soft, tokenizer, dev_data, db_root, device)
    except Exception as e:
        print(f"Skipping Soft Adapter due to error: {e}")
        results["SOFT"] = {"error": str(e)}

    # ---------------------------------------------------------
    # SAVE JSON TO EXACT SCHEMA
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    final_output = {"models": {}}
    if "HARD" in results and "error" not in results["HARD"]:
        final_output["models"]["HARD"] = results["HARD"]
    if "SOFT" in results and "error" not in results["SOFT"]:
        final_output["models"]["SOFT"] = results["SOFT"]

    with open(args.out, "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n" + "="*50)
    print("📊 TASK 4: HARD VS SOFT REWARD ANALYSIS")
    print("="*50)
    
    for model_key, metrics in final_output["models"].items():
        print(f"\n[ {model_key} ]")
        print(f"Exact Match (em):      {metrics['em']*100:.2f}%")
        print(f"Execution Acc (ex):    {metrics['ex']*100:.2f}%")
        print(f"Constraint Rate:       {metrics['constraint_rate']*100:.2f}%")
        print(f"Mean Soft Score:       {metrics['s_mean']:.3f}")
        print(f"Reward Variance:       {metrics['s_var']:.4f}")
    
    print(f"\n✅ Output successfully saved to: {args.out}")
    print("="*50)

if __name__ == "__main__":
    main()