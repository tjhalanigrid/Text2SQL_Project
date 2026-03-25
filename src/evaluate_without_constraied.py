
# *********** code till task 3 ************ 

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
#     parser.add_argument("--num_samples", type=int, default= 500)
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




# *********** for task 2 ****************************************
import json
import argparse
import random
import sqlite3
import re
import os
from pathlib import Path
from collections import defaultdict

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

def normalize_result(res):
    try:
        return sorted([str(r) for r in res])
    except:
        return []

# -------------------------------
# STEP 1: EXECUTION
# -------------------------------
def execute_with_error(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        res = cur.fetchall()
        conn.close()
        return res, None
    except Exception as e:
        return None, str(e)

# -------------------------------
# STEP 2: ERROR CLASSIFICATION
# -------------------------------
def classify_error(sql, error_msg):
    if error_msg is None:
        return "correct"

    err = error_msg.lower()
    sql_l = sql.lower()

    if "syntax" in err:
        return "syntax_error"
    if "no such table" in err:
        return "wrong_table"
    if "no such column" in err:
        return "wrong_column"
    if "ambiguous" in err:
        return "missing_join"
    if "datatype mismatch" in err:
        return "type_error"
    if "where" not in sql_l and any(x in sql_l for x in ["=", ">", "<"]):
        return "missing_where"

    return "other"

# -------------------------------
# STEP 4: HINTS
# -------------------------------
def generate_hint(error_type):
    hints = {
        "missing_join": "Try using JOIN between related tables.",
        "wrong_column": "Check column names in schema.",
        "missing_where": "Add WHERE condition.",
        "syntax_error": "Fix SQL syntax.",
        "wrong_table": "Verify table names.",
        "type_error": "Check data types.",
        "other": "Review SQL logic."
    }
    return hints.get(error_type, "")

# -------------------------------
# STEP 2 EXTRA: LIGHT ATTRIBUTION
# -------------------------------
def extract_keywords(question):
    return [w for w in re.findall(r"\w+", question.lower()) if len(w) > 3]

# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    db_root = project_root / "data" / "database"
    dev_json = project_root / "data" / "dev.json"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)

    model = PeftModel.from_pretrained(
        base,
        os.path.relpath(project_root / args.adapter, project_root),
        local_files_only=True
    ).to(device)

    model = model.merge_and_unload()
    model.eval()

    with open(dev_json) as f:
        dev = json.load(f)

    dev = dev[:args.num_samples]

    # STORAGE
    error_counter = defaultdict(int)
    error_examples = defaultdict(list)
    success_examples = []
    hint_examples = defaultdict(list)
    operation_counter = defaultdict(int)
    attribution_map = defaultdict(list)

    em, ex = 0, 0

    print(f"\n🚀 Evaluating {len(dev)} samples...\n")

    for i, sample in enumerate(dev, 1):

        db_id = sample["db_id"]
        q = sample["question"]
        gold = sample["query"]
        db_path = db_root / db_id / f"{db_id}.sqlite"

        input_ids = encode_prompt(tokenizer, q, db_id, device=device).unsqueeze(0)

        out = model.generate(input_ids=input_ids, max_new_tokens=120, num_beams=8)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # operation analysis
        s = pred.lower()
        if "select" in s: operation_counter["SELECT"] += 1
        if "where" in s: operation_counter["WHERE"] += 1
        if "join" in s: operation_counter["JOIN"] += 1
        if "group by" in s: operation_counter["GROUP_BY"] += 1
        if "order by" in s: operation_counter["ORDER_BY"] += 1

        pred_res, err = execute_with_error(pred, db_path)
        gold_res, _ = execute_with_error(gold, db_path)

        error_type = classify_error(pred, err)
        error_counter[error_type] += 1

        # attribution
        if err:
            attribution_map[error_type].append(extract_keywords(q))

        # examples
        if len(error_examples[error_type]) < 3:
            error_examples[error_type].append(pred)

        # hints
        if error_type != "correct":
            hint = generate_hint(error_type)
            if len(hint_examples[error_type]) < 3:
                hint_examples[error_type].append((pred, hint))

        # metrics
        if normalize_sql(pred) == normalize_sql(gold):
            em += 1

        if pred_res and gold_res and normalize_result(pred_res) == normalize_result(gold_res):
            ex += 1
            if len(success_examples) < 5:
                success_examples.append(pred)

        if i % 20 == 0:
            print(f"[{i}] EM: {em/i:.2f} | EX: {ex/i:.2f}")

    # -------------------------------
    # OUTPUT
    # -------------------------------
    print("\n🎯 FINAL RESULTS")
    print(f"EM: {em/len(dev)*100:.2f}%")
    print(f"EX: {ex/len(dev)*100:.2f}%")

    print("\n🔥 ERROR SUMMARY")
    for k, v in error_counter.items():
        print(k, ":", v)

    print("\n🔥 ERROR EXAMPLES")
    for k in error_examples:
        print("\n", k)
        for e in error_examples[k]:
            print("  ", e)

    print("\n🔥 HINTS")
    for k in hint_examples:
        print("\n", k)
        for sql, h in hint_examples[k]:
            print("  ", sql)
            print("  →", h)

    print("\n🔥 ATTRIBUTION (KEYWORDS)")
    for k in attribution_map:
        print(k, ":", attribution_map[k][:3])

    print("\n🔥 SQL OPERATIONS")
    for k, v in operation_counter.items():
        print(k, ":", v)

    # -------------------------------
    # ADVERSARIAL
    # -------------------------------
    print("\n🔥 ADVERSARIAL TESTS")

    adv = [
        "Find most expensive product",
        "Top 3 students by marks",
        "Average salary per department"
    ]

    for q in adv:
        inp = encode_prompt(tokenizer, q, dev[0]["db_id"], device=device).unsqueeze(0)
        out = model.generate(input_ids=inp, max_new_tokens=120)
        print("\nQ:", q)
        print("SQL:", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()