# import json
# import argparse
# import sqlite3
# import time
# import re
# import os
# import gc
# import sys
# from pathlib import Path

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# print("Script is alive!")

# # Import handling
# try:
#     from prompting import encode_prompt
#     from src.sql_validator import validate_sql_schema
# except ImportError:
#     sys.path.append(str(Path(__file__).resolve().parents[1]))
#     from src.prompting import encode_prompt
#     from src.sql_validator import validate_sql_schema

# # =========================================================
# # ERROR LOGGING
# # =========================================================
# ERROR_LOG_FILE = "results/error_logs.json"

# def classify_error(sql, error_msg=""):
#     sql = sql.lower()
#     error_msg = str(error_msg).lower()

#     if "no such column" in error_msg:
#         return "wrong_column"
#     if "no such table" in error_msg:
#         return "wrong_table"
#     if "syntax error" in error_msg:
#         return "syntax_error"
#     if "ambiguous column" in error_msg:
#         return "ambiguous_column"
#     if "join" in sql and " on " not in sql:
#         return "missing_join"

#     return "other"

# def log_error(question, sql, error, error_type, eval_type="unknown"):
#     os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)

#     entry = {
#         "question": question,
#         "sql": sql,
#         "error": str(error),
#         "error_type": error_type,
#         "eval_type": eval_type,
#         "timestamp": time.time()
#     }

#     logs = []
#     if os.path.exists(ERROR_LOG_FILE):
#         try:
#             with open(ERROR_LOG_FILE, "r") as f:
#                 content = f.read().strip()
#                 if content:
#                     logs = json.loads(content)
#         except:
#             logs = []

#     logs.append(entry)

#     with open(ERROR_LOG_FILE, "w") as f:
#         json.dump(logs, f, indent=2)

# # =========================================================
# # 🔥 FINAL FIX_SQL (BALANCED VERSION)
# # =========================================================
# def fix_sql(sql):
#     if not sql:
#         return "SELECT 1"

#     s = str(sql).strip()

#     match = re.search(r"(?i)(select|with)[\s\S]*", s)
#     if match:
#         s = match.group(0)

#     s = s.split(";")[0].strip()

#     s = re.sub(r'(?i)=\s*null', 'IS NULL', s)
#     s = re.sub(r'(?i)!=\s*null', 'IS NOT NULL', s)
#     s = re.sub(r',\s*,+', ',', s)
#     s = re.sub(r'(?i),\s*from', ' FROM', s)

#     if "select" in s.lower():
#         if len(re.findall(r'\w+\.\w+', s)) > 3:
#             s = re.sub(r'(?i)select\s+.*?\s+from', 'SELECT * FROM', s)

#     if "join" in s.lower() and " on " not in s.lower():
#         s = re.sub(r'join\s+(\w+)', r'JOIN \1 ON 1=1', s, flags=re.I)

#     if not s.lower().startswith(("select", "with")):
#         return "SELECT 1"

#     return s.strip()

# # =========================================================
# # NORMALIZATION & EXECUTION HELPERS
# # =========================================================
# def normalize_sql(sql):
#     if not sql:
#         return ""
#     return re.sub(r"\s+", " ", str(sql)).strip().lower()

# def normalize_result(res):
#     if not res:
#         return []
#     try:
#         normalized = [tuple(sorted(str(x) for x in row)) for row in res]
#         return sorted(normalized)
#     except:
#         return sorted([str(r) for r in res])

# def is_executable(sql, db_path):
#     try:
#         conn = sqlite3.connect(db_path)
#         cur = conn.cursor()
#         cur.execute(sql)
#         conn.close()
#         return True
#     except:
#         return False

# def check_execution(pred_sql, gold_sql, db_path, question, eval_type="unknown"):
#     try:
#         conn = sqlite3.connect(db_path)
#         conn.text_factory = lambda b: b.decode(errors='ignore')
#         cur = conn.cursor()

#         cur.execute(gold_sql)
#         gold_res = cur.fetchall()

#         cur.execute(pred_sql)
#         pred_res = cur.fetchall()

#         conn.close()

#         return normalize_result(pred_res) == normalize_result(gold_res)

#     except Exception as e:
#         error_type = classify_error(pred_sql, str(e))
#         log_error(question, pred_sql, str(e), error_type, eval_type)
#         return False


# # =========================================================
# # EVALUATION RUNNER WITH SANITY CHECK
# # =========================================================
# def evaluate_model(adapter_path, mode, dev_data, db_root, tokenizer, device):
#     print(f"\n⚙️ Loading [{mode.upper()}] model from: {adapter_path}")
    
#     # 🔥 THE FIX: Strict Path & File Checking
#     adapter_dir = Path(adapter_path).resolve()
#     config_file = adapter_dir / "adapter_config.json"
    
#     if not adapter_dir.exists() or not config_file.exists():
#         print("\n" + "!"*60)
#         print(f"❌ ERROR: Could not find the LoRA adapter at:\n   {adapter_dir}")
#         print("!"*60)
#         print("WHY THIS HAPPENED:")
#         print("1. You haven't downloaded the 'checkpoints' folder from Google Drive.")
#         print("2. The folder is in the wrong location (it should be in the project root).")
#         print("\nPLEASE FIX:")
#         print("Go to the README, download the checkpoints via the Drive link,")
#         print("and make sure 'adapter_config.json' is inside that folder.")
#         print("!"*60 + "\n")
#         sys.exit(1) # Stop the script so it doesn't evaluate the wrong model!

#     base_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)
#     # Using str(adapter_dir) prevents relative path issues
#     model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(device)
#     model = model.merge_and_unload()
#     model.eval()

#     em_correct = 0
#     ex_correct = 0
#     constraint_ok = 0

#     print(f"🚀 Running {mode} evaluation...")

#     for i, ex in enumerate(dev_data, 1):
#         db_id = ex["db_id"]
#         question = ex["question"]
#         gold_query = ex["query"]
#         db_path = db_root / db_id / f"{db_id}.sqlite"

#         input_tensor = encode_prompt(tokenizer, question, db_id, device=device).unsqueeze(0)

#         with torch.no_grad():
#             if mode == "unconstrained":
#                 # Only need top 1 beam for unconstrained
#                 outputs = model.generate(input_ids=input_tensor, max_new_tokens=128, num_beams=4, num_return_sequences=1)
                
#                 raw_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 pred_sql = fix_sql(raw_pred)

#                 if normalize_sql(pred_sql) == normalize_sql(gold_query):
#                     em_correct += 1
#                 if check_execution(pred_sql, gold_query, str(db_path), question, eval_type="unconstrained"):
#                     ex_correct += 1

#             elif mode == "constrained":
#                 # Need multiple beams to filter through execution
#                 outputs = model.generate(input_ids=input_tensor, max_new_tokens=128, num_beams=8, num_return_sequences=8)
                
#                 pred_sql = ""
#                 for out in outputs:
#                     candidate_sql = fix_sql(tokenizer.decode(out, skip_special_tokens=True))
#                     if is_executable(candidate_sql, str(db_path)):
#                         pred_sql = candidate_sql
#                         break 
                
#                 if not pred_sql:
#                     pred_sql = fix_sql(tokenizer.decode(outputs[0], skip_special_tokens=True))

#                 try:
#                     is_valid, _ = validate_sql_schema(pred_sql, str(db_path))
#                 except:
#                     is_valid = False

#                 if is_valid:
#                     constraint_ok += 1

#                 if normalize_sql(pred_sql) == normalize_sql(gold_query):
#                     em_correct += 1
#                 if check_execution(pred_sql, gold_query, str(db_path), question, eval_type="constrained"):
#                     ex_correct += 1

#         if i % 50 == 0:
#             print(f"  {i}/{len(dev_data)} samples processed...")

#     # Free up memory before loading the next model
#     del model
#     del base_model
#     gc.collect()
#     if device == "mps":
#         torch.mps.empty_cache()
#     elif device == "cuda":
#         torch.cuda.empty_cache()

#     return em_correct, ex_correct, constraint_ok

# # =========================================================
# # MAIN
# # =========================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--adapter_unconstrained", type=str, required=True, help="Path to unconstrained adapter")
#     parser.add_argument("--adapter_constrained", type=str, required=True, help="Path to constrained adapter")
#     parser.add_argument("--num_samples", type=int, default=200)
#     args = parser.parse_args()

#     # 🔥 Robust Path Finding Logic
#     script_dir = Path(__file__).resolve().parent
#     project_root = script_dir.parent if script_dir.name in ["scripts", "src", "project3"] else script_dir

#     # Search multiple possible locations for dev.json
#     candidate_paths = [
#         project_root / "data" / "dev.json",
#         project_root / "spider" / "dev.json",
#         script_dir / "data" / "dev.json",
#         script_dir / "spider" / "dev.json",
#         project_root / "dev.json",
#         script_dir / "dev.json",
#     ]

#     dev_json = None
#     for path in candidate_paths:
#         if path.exists():
#             dev_json = path
#             break

#     if not dev_json:
#         raise FileNotFoundError(
#             f"Could not find 'dev.json'. Searched in standard project folders near {project_root}. "
#             "Please verify where your spider dataset is located."
#         )

#     data_folder = dev_json.parent
#     db_root = data_folder / "database"

#     device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

#     tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

#     with open(dev_json, "r") as f:
#         dev_data = json.load(f)[:args.num_samples]

#     print(f"\n========================================")
#     print(f"Starting Evaluation Pipeline on {device.upper()}")
#     print(f"Total Samples: {len(dev_data)}")
#     print(f"Dataset found at: {dev_json}")
#     print(f"========================================")

#     # Run Model 1
#     em_unc, ex_unc, _ = evaluate_model(
#         args.adapter_unconstrained, "unconstrained", dev_data, db_root, tokenizer, device
#     )

#     # Run Model 2
#     em_con, ex_con, ok_con = evaluate_model(
#         args.adapter_constrained, "constrained", dev_data, db_root, tokenizer, device
#     )

#     # Print Final Comparison
#     print("\n========================================")
#     print("🎯 FINAL COMPARISON RESULTS")
#     print("========================================")
#     print("📊 UNCONSTRAINED MODEL:")
#     print(f"  Exact Match (EM):      {(em_unc/len(dev_data))*100:.2f}%")
#     print(f"  Execution Acc (EX):    {(ex_unc/len(dev_data))*100:.2f}%")
#     print("\n🛡️ CONSTRAINED MODEL:")
#     print(f"  Exact Match (EM):      {(em_con/len(dev_data))*100:.2f}%")
#     print(f"  Execution Acc (EX):    {(ex_con/len(dev_data))*100:.2f}%")
#     print(f"  Constraint Valid Rate: {(ok_con/len(dev_data))*100:.2f}%")
#     print("========================================")
#     print(f"Errors logged to: {ERROR_LOG_FILE}")

# if __name__ == "__main__":
#     main()



import json
import argparse
import sqlite3
import time
import re
import os
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

print("Script is alive!")

# Import handling
try:
    from prompting import encode_prompt
    from src.sql_validator import validate_sql_schema
except ImportError:
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

def log_error(question, sql, error, error_type, eval_type="unknown"):
    os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)

    entry = {
        "question": question,
        "sql": sql,
        "error": str(error),
        "error_type": error_type,
        "eval_type": eval_type,
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

    match = re.search(r"(?i)(select|with)[\s\S]*", s)
    if match:
        s = match.group(0)

    s = s.split(";")[0].strip()

    s = re.sub(r'(?i)=\s*null', 'IS NULL', s)
    s = re.sub(r'(?i)!=\s*null', 'IS NOT NULL', s)
    s = re.sub(r',\s*,+', ',', s)
    s = re.sub(r'(?i),\s*from', ' FROM', s)

    if "select" in s.lower():
        if len(re.findall(r'\w+\.\w+', s)) > 3:
            s = re.sub(r'(?i)select\s+.*?\s+from', 'SELECT * FROM', s)

    if "join" in s.lower() and " on " not in s.lower():
        s = re.sub(r'join\s+(\w+)', r'JOIN \1 ON 1=1', s, flags=re.I)

    if not s.lower().startswith(("select", "with")):
        return "SELECT 1"

    return s.strip()

# =========================================================
# NORMALIZATION & EXECUTION HELPERS
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

def is_executable(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        conn.close()
        return True
    except:
        return False

def check_execution(pred_sql, gold_sql, db_path, question, eval_type="unknown"):
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
        log_error(question, pred_sql, str(e), error_type, eval_type)
        return False

# =========================================================
# MODEL LOADING (GITHUB CLONE OPTIMIZED)
# =========================================================
def load_model(adapter_path, mode, device):
    print(f"\n⚙️ Loading [{mode.upper()}] model from: {adapter_path}")

    # Smart pathing to handle different computers
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name in ["scripts", "src", "project3"] else script_dir
    
    # Resolve the absolute path based on the project root
    adapter_dir = project_root / adapter_path
    config_file = adapter_dir / "adapter_config.json"
    
    if not adapter_dir.exists() or not config_file.exists():
        print("\n" + "!"*60)
        print(f"❌ ERROR: Could not find the LoRA adapter at:\n   {adapter_dir}")
        print("!"*60)
        print("WHY THIS HAPPENED:")
        print("The files are missing. Make sure you successfully pushed the .bin")
        print("and .json files inside your checkpoints folder to GitHub.")
        print("!"*60 + "\n")
        sys.exit(1)

    # Load adapter config and base model
    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        peft_config.base_model_name_or_path
    ).to(device)

    # Load adapter weights and merge
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    
    return model

# =========================================================
# EVALUATION RUNNER
# =========================================================
def evaluate_model(adapter_path, mode, dev_data, db_root, tokenizer, device):
    print(f"\n==============================")
    print(f"🚀 Running {mode.upper()} evaluation...")
    print(f"==============================")

    model = load_model(adapter_path, mode, device)

    em_correct = 0
    ex_correct = 0
    constraint_ok = 0

    for i, ex in enumerate(dev_data, 1):
        db_id = ex["db_id"]
        question = ex["question"]
        gold_query = ex["query"]
        db_path = db_root / db_id / f"{db_id}.sqlite"

        input_tensor = encode_prompt(tokenizer, question, db_id, device=device).unsqueeze(0)

        with torch.no_grad():
            if mode == "unconstrained":
                # Only need top 1 beam for unconstrained
                outputs = model.generate(input_ids=input_tensor, max_new_tokens=128, num_beams=4, num_return_sequences=1)
                
                raw_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_sql = fix_sql(raw_pred)

                if normalize_sql(pred_sql) == normalize_sql(gold_query):
                    em_correct += 1
                if check_execution(pred_sql, gold_query, str(db_path), question, eval_type="unconstrained"):
                    ex_correct += 1

            elif mode == "constrained":
                # Need multiple beams to filter through execution
                outputs = model.generate(input_ids=input_tensor, max_new_tokens=128, num_beams=8, num_return_sequences=8)
                
                pred_sql = ""
                for out in outputs:
                    candidate_sql = fix_sql(tokenizer.decode(out, skip_special_tokens=True))
                    if is_executable(candidate_sql, str(db_path)):
                        pred_sql = candidate_sql
                        break 
                
                if not pred_sql:
                    pred_sql = fix_sql(tokenizer.decode(outputs[0], skip_special_tokens=True))

                try:
                    is_valid, _ = validate_sql_schema(pred_sql, str(db_path))
                except:
                    is_valid = False

                if is_valid:
                    constraint_ok += 1

                if normalize_sql(pred_sql) == normalize_sql(gold_query):
                    em_correct += 1
                if check_execution(pred_sql, gold_query, str(db_path), question, eval_type="constrained"):
                    ex_correct += 1

        if i % 50 == 0:
            print(f"  {i}/{len(dev_data)} samples processed...")

    # Free up memory before loading the next model
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return em_correct, ex_correct, constraint_ok

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_unconstrained", type=str, required=True, help="Path to unconstrained adapter")
    parser.add_argument("--adapter_constrained", type=str, required=True, help="Path to constrained adapter")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()

    # 🔥 Robust Path Finding Logic
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name in ["scripts", "src", "project3"] else script_dir

    # Search multiple possible locations for dev.json
    candidate_paths = [
        project_root / "data" / "dev.json",
        project_root / "spider" / "dev.json",
        script_dir / "data" / "dev.json",
        script_dir / "spider" / "dev.json",
        project_root / "dev.json",
        script_dir / "dev.json",
    ]

    dev_json = None
    for path in candidate_paths:
        if path.exists():
            dev_json = path
            break

    if not dev_json:
        raise FileNotFoundError(
            f"Could not find 'dev.json'. Searched in standard project folders near {project_root}. "
            "Please verify where your spider dataset is located."
        )

    data_folder = dev_json.parent
    db_root = data_folder / "database"

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

    with open(dev_json, "r") as f:
        dev_data = json.load(f)[:args.num_samples]

    print(f"\n========================================")
    print(f"Starting Evaluation Pipeline on {device.upper()}")
    print(f"Total Samples: {len(dev_data)}")
    print(f"Dataset found at: {dev_json}")
    print(f"========================================")

    # Run Model 1
    em_unc, ex_unc, _ = evaluate_model(
        args.adapter_unconstrained, "unconstrained", dev_data, db_root, tokenizer, device
    )

    # Run Model 2
    em_con, ex_con, ok_con = evaluate_model(
        args.adapter_constrained, "constrained", dev_data, db_root, tokenizer, device
    )

    # Print Final Comparison
    print("\n========================================")
    print("🎯 FINAL COMPARISON RESULTS")
    print("========================================")
    print("📊 UNCONSTRAINED MODEL:")
    print(f"  Exact Match (EM):      {(em_unc/len(dev_data))*100:.2f}%")
    print(f"  Execution Acc (EX):    {(ex_unc/len(dev_data))*100:.2f}%")
    print("\n🛡️ CONSTRAINED MODEL:")
    print(f"  Exact Match (EM):      {(em_con/len(dev_data))*100:.2f}%")
    print(f"  Execution Acc (EX):    {(ex_con/len(dev_data))*100:.2f}%")
    print(f"  Constraint Valid Rate: {(ok_con/len(dev_data))*100:.2f}%")
    print("========================================")
    print(f"Errors logged to: {ERROR_LOG_FILE}")

if __name__ == "__main__":
    main()