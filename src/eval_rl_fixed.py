# import json
# import subprocess
# import sys
# import argparse
# import random
# import sqlite3
# import time
# import re
# import os
# from pathlib import Path

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# from prompting import encode_prompt

# # -------------------------------
# # NORMALIZATION
# # -------------------------------
# def normalize_sql(sql):
#     sql = sql.replace('"', "'")
#     sql = re.sub(r"\s+", " ", sql)
#     return sql.strip().lower().rstrip(";")


# # -------------------------------
# # 🔥 SAFE RESULT NORMALIZATION (FIX)
# # -------------------------------
# def normalize_result(res):
#     try:
#         return sorted([str(r) for r in res])
#     except:
#         return []


# # -------------------------------
# # EXECUTION CHECK (FIXED)
# # -------------------------------
# def check_execution(pred_sql, gold_sql, db_path):
#     try:
#         conn = sqlite3.connect(db_path)
#         conn.text_factory = lambda b: b.decode(errors='ignore')

#         start_time = time.monotonic()

#         def timeout_handler():
#             return 1 if (time.monotonic() - start_time) > 2.0 else 0

#         conn.set_progress_handler(timeout_handler, 10000)

#         cursor = conn.cursor()

#         cursor.execute(pred_sql)
#         pred_res = cursor.fetchall()

#         cursor.execute(gold_sql)
#         gold_res = cursor.fetchall()

#         conn.close()

#         # 🔥 FIXED COMPARISON
#         return normalize_result(pred_res) == normalize_result(gold_res)

#     except Exception:
#         return False


# # -------------------------------
# # SPIDER PARSER
# # -------------------------------
# def _parse_spider_accuracy(stdout: str, metric_type: str):
#     for line in stdout.splitlines():
#         if metric_type == "exec" and line.strip().startswith("execution"):
#             try:
#                 return float(line.split()[-1])
#             except:
#                 pass
#         elif metric_type == "match" and line.strip().startswith("exact"):
#             try:
#                 return float(line.split()[-1])
#             except:
#                 pass
#     return None


# # -------------------------------
# # MAIN
# # -------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--adapter", type=str, required=True)
#     parser.add_argument("--num_samples", type=int, default=700)
#     parser.add_argument("--shuffle_dev", action="store_true")
#     parser.add_argument("--shuffle_seed", type=int, default=42)
#     args = parser.parse_args()

#     project_root = Path(__file__).resolve().parents[1]
#     adapter_dir = project_root / args.adapter

#     db_root = project_root / "data" / "database"
#     table_json = project_root / "data" / "tables.json"
#     dev_json = project_root / "data" / "dev.json"

#     pred_path = project_root / "temp_predictions.txt"
#     temp_gold_path = project_root / "temp_gold.sql"

#     if not adapter_dir.exists():
#         raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

#     device = "mps" if torch.backends.mps.is_available() else (
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#     print(f"Using device: {device}")

#     BASE_MODEL = "Salesforce/codet5-base"
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     print(f"\n📦 Loading Model: {args.adapter}")

#     base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

#     adapter_for_peft = os.path.relpath(adapter_dir, project_root)

#     model = PeftModel.from_pretrained(
#         base,
#         adapter_for_peft,
#         local_files_only=True
#     ).to(device)

#     model = model.merge_and_unload()
#     model.eval()

#     # -------------------------------
#     # LOAD DATA
#     # -------------------------------
#     with dev_json.open() as f:
#         dev = json.load(f)

#     if args.shuffle_dev:
#         rng = random.Random(args.shuffle_seed)
#         rng.shuffle(dev)

#     dev = dev[: args.num_samples]
#     total = len(dev)

#     gen_kwargs = dict(
#         max_new_tokens=160,
#         num_beams=8,
#         length_penalty=0.8,
#         do_sample=False,
#         early_stopping=True,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     print(f"\n🚀 Evaluating {total} samples...\n")

#     em_correct = 0
#     ex_correct = 0

#     with pred_path.open("w") as out_pred, temp_gold_path.open("w") as out_gold, torch.no_grad():
#         for i, ex in enumerate(dev, start=1):

#             db_id = ex["db_id"]
#             question = ex["question"]
#             gold_query = ex["query"]
#             db_path = db_root / db_id / f"{db_id}.sqlite"

#             # -------------------------------
#             # GENERATE SQL
#             # -------------------------------
#             input_ids = encode_prompt(
#                 tokenizer,
#                 question,
#                 db_id,
#                 device=device,
#                 max_input_tokens=512
#             )

#             input_ids = input_ids.unsqueeze(0).to(device)
#             attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

#             outputs = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 **gen_kwargs
#             )

#             pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#             # -------------------------------
#             # SAVE FOR SPIDER EVAL
#             # -------------------------------
#             out_pred.write(f"{pred_sql}\n")
#             out_gold.write(f"{gold_query}\t{db_id}\n")

#             # -------------------------------
#             # LIVE METRICS
#             # -------------------------------
#             if normalize_sql(pred_sql) == normalize_sql(gold_query):
#                 em_correct += 1

#             if check_execution(pred_sql, gold_query, db_path):
#                 ex_correct += 1

#             if i % 20 == 0 or i == total:
#                 print(
#                     f"Progress: {i}/{total} | "
#                     f"EM: {(em_correct/i)*100:.2f}% | "
#                     f"EX: {(ex_correct/i)*100:.2f}%"
#                 )

#     print("\n🚀 Running Official Spider Evaluation...\n")

#     eval_script = project_root / "spider_eval" / "evaluation.py"

#     # EXACT MATCH
#     cmd_match = [
#         sys.executable, str(eval_script),
#         "--gold", str(temp_gold_path),
#         "--pred", str(pred_path),
#         "--etype", "match",
#         "--db", str(db_root),
#         "--table", str(table_json),
#     ]

#     proc_match = subprocess.run(cmd_match, capture_output=True, text=True)
#     exact_acc = _parse_spider_accuracy(proc_match.stdout, "match")

#     # EXECUTION
#     cmd_exec = [
#         sys.executable, str(eval_script),
#         "--gold", str(temp_gold_path),
#         "--pred", str(pred_path),
#         "--etype", "exec",
#         "--db", str(db_root),
#         "--table", str(table_json),
#     ]

#     proc_exec = subprocess.run(cmd_exec, capture_output=True, text=True)
#     exec_acc = _parse_spider_accuracy(proc_exec.stdout, "exec")

#     print("==========================================")
#     print(f"🎯 OFFICIAL SPIDER RESULTS FOR: {args.adapter}")
#     print("==========================================")

#     print(f"Exact Match Accuracy  : {exact_acc*100:.2f}%" if exact_acc else "EM parsing failed")
#     print(f"Execution Accuracy    : {exec_acc*100:.2f}%" if exec_acc else "EX parsing failed")

#     print("==========================================\n")


# if __name__ == "__main__":
#     main()






# import json
# import sqlite3
# import re
# import time
# import sys
# import argparse
# from pathlib import Path

# # ==========================================
# # PATH SETUP
# # ==========================================
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from src.text2sql_engine import get_engine
# from src.sql_validator import validate_sql_schema

# # ==========================================
# # CONFIG
# # ==========================================
# DATA_PATH = PROJECT_ROOT / "data" / "dev.json"
# DB_ROOT = PROJECT_ROOT / "data" / "database"

# # ==========================================
# # NORMALIZATION
# # ==========================================
# def normalize_sql(sql):
#     if not isinstance(sql, str):
#         return ""
#     sql = sql.replace('"', "'")
#     sql = re.sub(r"\s+", " ", sql)
#     return sql.strip().lower().rstrip(";")

# def normalize_result(res):
#     try:
#         return sorted([tuple(map(str, r)) for r in res])
#     except:
#         return []

# # ==========================================
# # EXECUTION
# # ==========================================
# def execute_sql(db_path, sql):
#     try:
#         conn = sqlite3.connect(db_path)

#         start = time.time()
#         def timeout():
#             return 1 if (time.time() - start) > 2 else 0

#         conn.set_progress_handler(timeout, 10000)

#         cur = conn.cursor()
#         cur.execute(sql)
#         res = cur.fetchall()

#         conn.close()
#         return res

#     except Exception:
#         return None

# # ==========================================
# # EVALUATION
# # ==========================================
# def evaluate(engine, data, is_constrained=False, debug=False):

#     attempted = 0
#     total = 0
#     exact_match = 0
#     execution_match = 0
#     constraint_ok = 0

#     skipped_missing_db = 0
#     skipped_exception = 0
#     skipped_no_sql = 0

#     total_time = 0

#     for i, item in enumerate(data, 1):

#         question = item.get("question", "")
#         gold_sql = item.get("query", "")
#         db_id = item.get("db_id", "")

#         db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

#         if not db_path.exists():
#             skipped_missing_db += 1
#             continue

#         try:
#             start = time.time()
#             result = engine.ask(question, db_id)
#             total_time += (time.time() - start)
#         except Exception:
#             skipped_exception += 1
#             continue

#         if not isinstance(result, dict):
#             continue

#         pred_sql = result.get("sql", "")

#         # DEBUG
#         if debug:
#             print(f"\nQ: {question}")
#             print(f"PRED: {pred_sql}")
#             print(f"GOLD: {gold_sql}")

#         if not pred_sql:
#             skipped_no_sql += 1
#             continue

#         attempted += 1
#         total += 1

#         # CONSTRAINT CHECK
#         if is_constrained:
#             try:
#                 is_valid, _ = validate_sql_schema(pred_sql, str(db_path))
#                 if is_valid:
#                     constraint_ok += 1
#             except:
#                 pass

#         # EXACT MATCH
#         if normalize_sql(pred_sql) == normalize_sql(gold_sql):
#             exact_match += 1

#         # EXECUTION MATCH
#         pred_res = execute_sql(str(db_path), pred_sql)
#         gold_res = execute_sql(str(db_path), gold_sql)

#         if pred_res is not None and gold_res is not None:
#             if normalize_result(pred_res) == normalize_result(gold_res):
#                 execution_match += 1

#         # PROGRESS
#         if i % 10 == 0:
#             print(
#                 f"[{i}/{len(data)}] "
#                 f"EM: {exact_match/max(total,1):.3f} | "
#                 f"EX: {execution_match/max(total,1):.3f} | "
#                 f"Constraint: {(constraint_ok/max(total,1)) if is_constrained else 0:.3f}"
#             )

#     avg_latency = total_time / max(attempted, 1)

#     return {
#         "exact_match": exact_match / total if total > 0 else 0,
#         "execution_accuracy": execution_match / total if total > 0 else 0,
#         "constraint_rate": (constraint_ok / total if (is_constrained and total > 0) else 0),
#         "avg_latency": avg_latency,
#         "total": total,
#         "attempted": attempted,
#         "skipped_missing_db": skipped_missing_db,
#         "skipped_exception": skipped_exception,
#         "skipped_no_sql": skipped_no_sql,
#     }

# # ==========================================
# # MAIN
# # ==========================================
# if __name__ == "__main__":

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--num-samples", type=int, default=100)
#     ap.add_argument("--adapter", type=str, default="checkpoints/best_rlhf_model")
#     ap.add_argument("--debug", action="store_true")
#     args = ap.parse_args()

#     print(f"\n📥 Loading dataset from {DATA_PATH}...")

#     with open(str(DATA_PATH)) as f:
#         data = json.load(f)[: args.num_samples]

#     # ==========================================
#     # 🔴 BASE MODEL
#     # ==========================================
#     print("\n🚀 Running BASE MODEL...\n")

#     engine_base = get_engine(
#         adapter_path="checkpoints/sft_adapter_codet5"  ,  # 🔥 change this 
#         use_lora=True,
#         use_constrained=False
#     )

#     res_base = evaluate(engine_base, data, is_constrained=False, debug=args.debug)

#     # ==========================================
#     # 🟡 RLHF (NO CONSTRAINT)
#     # ==========================================
#     print("\n🚀 Running RLHF (NO CONSTRAINT)...\n")

#     engine_rlhf = get_engine(
#        adapter_path="checkpoints/best_rlhf_model",
#         use_lora=True,
#         use_constrained=False
#     )

#     res_rlhf = evaluate(engine_rlhf, data, is_constrained=False, debug=args.debug)

#     # ==========================================
#     # 🟢 RLHF + CONSTRAINT
#     # ==========================================
#     print("\n🚀 Running RLHF + CONSTRAINED...\n")

#     engine_const = get_engine(
#         adapter_path="checkpoints/best_rlhf_model_2",
#         use_lora=True,
#         use_constrained=True
#     )

#     res_const = evaluate(engine_const, data, is_constrained=True, debug=args.debug)

#     # ==========================================
#     # FINAL RESULTS
#     # ==========================================
#     print("\n==========================================")
#     print("🎯 FINAL RESULTS (3-WAY COMPARISON)")
#     print("==========================================")

#     print(f"Base Model       → EM: {res_base['exact_match']*100:.2f}% | "
#           f"EX: {res_base['execution_accuracy']*100:.2f}%")

#     print(f"RLHF             → EM: {res_rlhf['exact_match']*100:.2f}% | "
#           f"EX: {res_rlhf['execution_accuracy']*100:.2f}%")

#     print(f"RLHF + Constrain → EM: {res_const['exact_match']*100:.2f}% | "
#           f"EX: {res_const['execution_accuracy']*100:.2f}% | "
#           f"Constraint: {res_const['constraint_rate']*100:.2f}%")

#     print("==========================================\n")


import json
import argparse
import sqlite3
import time
import re
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Import handling
try:
    from prompting import encode_prompt
    from src.sql_validator import validate_sql_schema
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.prompting import encode_prompt
    from src.sql_validator import validate_sql_schema

# =========================================================
# ERROR LOGGING
# =========================================================
ERROR_LOG_FILE = "results/error_logs.json"

def classify_error(sql, error_msg=""):
    sql = sql.lower()
    error_msg = str(error_msg).lower()

    if "no such column" in error_msg:
        return "wrong_column"
    if "no such table" in error_msg:
        return "wrong_table"
    if "syntax error" in error_msg:
        return "syntax_error"
    if "ambiguous column" in error_msg:
        return "ambiguous_column"
    if "join" in sql and " on " not in sql:
        return "missing_join"

    return "other"

def log_error(question, sql, error, error_type):
    os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)

    entry = {
        "question": question,
        "sql": sql,
        "error": str(error),
        "error_type": error_type,
        "timestamp": time.time()
    }

    logs = []
    if os.path.exists(ERROR_LOG_FILE):
        try:
            with open(ERROR_LOG_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
        except:
            logs = []

    logs.append(entry)

    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# =========================================================
# 🔥 FINAL FIX_SQL (BALANCED VERSION)
# =========================================================
def fix_sql(sql):
    if not sql:
        return "SELECT 1"

    s = str(sql).strip()

    # Extract SQL only
    match = re.search(r"(?i)(select|with)[\s\S]*", s)
    if match:
        s = match.group(0)

    s = s.split(";")[0].strip()

    # NULL fixes
    s = re.sub(r'(?i)=\s*null', 'IS NULL', s)
    s = re.sub(r'(?i)!=\s*null', 'IS NOT NULL', s)

    # Fix commas
    s = re.sub(r',\s*,+', ',', s)
    s = re.sub(r'(?i),\s*from', ' FROM', s)

    # 🔥 LIGHT COLUMN SAFETY (main improvement)
    if "select" in s.lower():
        if len(re.findall(r'\w+\.\w+', s)) > 3:
            s = re.sub(r'(?i)select\s+.*?\s+from', 'SELECT * FROM', s)

    # 🔥 JOIN fix
    if "join" in s.lower() and " on " not in s.lower():
        s = re.sub(r'join\s+(\w+)', r'JOIN \1 ON 1=1', s, flags=re.I)

    # Ensure valid SQL
    if not s.lower().startswith(("select", "with")):
        return "SELECT 1"

    return s.strip()

# =========================================================
# NORMALIZATION
# =========================================================
def normalize_sql(sql):
    if not sql:
        return ""
    return re.sub(r"\s+", " ", str(sql)).strip().lower()

def normalize_result(res):
    if not res:
        return []
    try:
        normalized = [tuple(sorted(str(x) for x in row)) for row in res]
        return sorted(normalized)
    except:
        return sorted([str(r) for r in res])

# =========================================================
# EXECUTION HELPERS
# =========================================================
def is_executable(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        conn.close()
        return True
    except:
        return False

def check_execution(pred_sql, gold_sql, db_path, question):
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cur = conn.cursor()

        cur.execute(gold_sql)
        gold_res = cur.fetchall()

        cur.execute(pred_sql)
        pred_res = cur.fetchall()

        conn.close()

        return normalize_result(pred_res) == normalize_result(gold_res)

    except Exception as e:
        error_type = classify_error(pred_sql, str(e))
        log_error(question, pred_sql, str(e), error_type)
        return False

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=700)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    if project_root.name in ["scripts", "src"]:
        project_root = project_root.parent

    db_root = project_root / "data" / "database"
    dev_json = project_root / "data" / "dev.json"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)

    model = PeftModel.from_pretrained(base_model, args.adapter).to(device)
    model = model.merge_and_unload()
    model.eval()

    with open(dev_json, "r") as f:
        dev_data = json.load(f)[:args.num_samples]

    em_correct = 0
    ex_correct = 0
    constraint_ok = 0

    print(f"\n🚀 Evaluating {len(dev_data)} samples...\n")

    for i, ex in enumerate(dev_data, 1):
        db_id = ex["db_id"]
        question = ex["question"]
        gold_query = ex["query"]

        db_path = db_root / db_id / f"{db_id}.sqlite"

        input_tensor = encode_prompt(tokenizer, question, db_id, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                max_new_tokens=128,
                num_beams=8,
                num_return_sequences=8
            )

        best_sql = ""

        # 🔥 EXECUTION-GUIDED SELECTION
        for out in outputs:
            raw_pred = tokenizer.decode(out, skip_special_tokens=True)
            candidate_sql = fix_sql(raw_pred)

            if is_executable(candidate_sql, str(db_path)):
                best_sql = candidate_sql
                break

        if not best_sql:
            best_sql = fix_sql(tokenizer.decode(outputs[0], skip_special_tokens=True))

        try:
            is_valid, _ = validate_sql_schema(best_sql, str(db_path))
        except:
            is_valid = False

        if is_valid:
            constraint_ok += 1

        if normalize_sql(best_sql) == normalize_sql(gold_query):
            em_correct += 1

        if check_execution(best_sql, gold_query, str(db_path), question):
            ex_correct += 1

        if i % 50 == 0:
            print(f"{i}/{len(dev_data)} done")

    print("\n========================================")
    print("🎯 FINAL EVALUATION RESULTS")
    print("========================================")
    print(f"Exact Match (EM):      {(em_correct/len(dev_data))*100:.2f}%")
    print(f"Execution Acc (EX):    {(ex_correct/len(dev_data))*100:.2f}%")
    print(f"Constraint Rate:       {(constraint_ok/len(dev_data))*100:.2f}%")
    print("========================================")
    print(f"Errors logged to: {ERROR_LOG_FILE}")

if __name__ == "__main__":
    main()
