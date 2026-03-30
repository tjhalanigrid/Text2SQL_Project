# import json
# import argparse
# import sqlite3
# import re
# import gc
# import os
# import sys
# from pathlib import Path

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel, PeftConfig

# # Import handling
# try:
#     from prompting import encode_prompt
# except ImportError:
#     sys.path.append(str(Path(__file__).resolve().parents[1]))
#     from src.prompting import encode_prompt

# # =========================================================
# # HELPER FUNCTIONS
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
#     if not s.lower().startswith(("select", "with")):
#         return "SELECT 1"
#     return s.strip()

# def normalize_sql(sql):
#     return re.sub(r"\s+", " ", str(sql)).strip().lower() if sql else ""

# def normalize_result(res):
#     if not res:
#         return []
#     try:
#         return sorted([tuple(sorted(str(x) for x in row)) for row in res])
#     except:
#         return sorted([str(r) for r in res])

# def check_execution(pred_sql, gold_sql, db_path):
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
#     except:
#         return False

# # =========================================================
# # MODEL LOADING WITH SANITY CHECK
# # =========================================================
# def load_model(adapter_path, device):
#     print(f"\n⚙️ Loading model from: {adapter_path}")

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
#         sys.exit(1) # Stop the script so it doesn't crash terribly!

#     # Load adapter config
#     peft_config = PeftConfig.from_pretrained(str(adapter_dir))

#     base_model = AutoModelForSeq2SeqLM.from_pretrained(
#         peft_config.base_model_name_or_path
#     ).to(device)

#     # Load adapter weights and merge
#     model = PeftModel.from_pretrained(base_model, str(adapter_dir))
#     model = model.merge_and_unload()
#     model = model.to(device)
#     model.eval()

#     print("\n🔍 DEBUG INFO:")
#     print("Model type:", type(model))

#     return model

# # =========================================================
# # EVALUATION
# # =========================================================
# def evaluate_model(adapter_path, model_name, dev_data, db_root, tokenizer, device):
#     print(f"\n==============================")
#     print(f"🚀 Evaluating: {model_name}")
#     print(f"==============================")

#     model = load_model(adapter_path, device)

#     em_correct = 0
#     ex_correct = 0

#     for i, ex in enumerate(dev_data, 1):
#         db_id = ex["db_id"]
#         question = ex["question"]
#         gold_query = ex["query"]

#         db_path = db_root / db_id / f"{db_id}.sqlite"

#         input_tensor = encode_prompt(
#             tokenizer,
#             question,
#             db_id,
#             device=device
#         ).unsqueeze(0)

#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_tensor,
#                 max_new_tokens=128,
#                 num_beams=4
#             )

#         pred_sql = fix_sql(
#             tokenizer.decode(outputs[0], skip_special_tokens=True)
#         )

#         # Exact Match
#         if normalize_sql(pred_sql) == normalize_sql(gold_query):
#             em_correct += 1

#         # Execution Accuracy
#         if check_execution(pred_sql, gold_query, str(db_path)):
#             ex_correct += 1

#         if i % 50 == 0:
#             print(f"  {i}/{len(dev_data)} samples processed...")

#     # Cleanup
#     del model
#     gc.collect()

#     if device == "mps":
#         torch.mps.empty_cache()
#     elif device == "cuda":
#         torch.cuda.empty_cache()

#     return em_correct, ex_correct

# # =========================================================
# # MAIN
# # =========================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--adapter_hard", type=str, required=True)
#     parser.add_argument("--adapter_soft", type=str, required=True)
#     parser.add_argument("--num_samples", type=int, default=200)

#     args = parser.parse_args()

#     # Path setup
#     script_dir = Path(__file__).resolve().parent
#     project_root = (
#         script_dir.parent
#         if script_dir.name in ["scripts", "src", "project3"]
#         else script_dir
#     )

#     dev_json = project_root / "data" / "dev.json"
#     if not dev_json.exists():
#         dev_json = project_root / "spider" / "dev.json"

#     db_root = dev_json.parent / "database"

#     device = (
#         "mps"
#         if torch.backends.mps.is_available()
#         else ("cuda" if torch.cuda.is_available() else "cpu")
#     )

#     print(f"\n💻 Using device: {device}")

#     tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

#     with open(dev_json, "r") as f:
#         dev_data = json.load(f)[:args.num_samples]

#     print(f"\n📊 Total samples: {len(dev_data)}")

#     # Run evaluations
#     em_hard, ex_hard = evaluate_model(
#         args.adapter_hard,
#         "Hard Reward",
#         dev_data,
#         db_root,
#         tokenizer,
#         device
#     )

#     em_soft, ex_soft = evaluate_model(
#         args.adapter_soft,
#         "Soft Reward",
#         dev_data,
#         db_root,
#         tokenizer,
#         device
#     )

#     # Results
#     print("\n========================================")
#     print("🎯 FINAL TASK 4 COMPARISON RESULTS")
#     print("========================================")

#     print("\n🧱 HARD REWARD MODEL:")
#     print(f"  Exact Match (EM):      {(em_hard/len(dev_data))*100:.2f}%")
#     print(f"  Execution Acc (EX):    {(ex_hard/len(dev_data))*100:.2f}%")

#     print("\n☁️ SOFT REWARD MODEL:")
#     print(f"  Exact Match (EM):      {(em_soft/len(dev_data))*100:.2f}%")
#     print(f"  Execution Acc (EX):    {(ex_soft/len(dev_data))*100:.2f}%")

#     print("========================================")

# if __name__ == "__main__":
#     main()





import json
import argparse
import sqlite3
import re
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

# Import handling
try:
    from prompting import encode_prompt
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.prompting import encode_prompt

# =========================================================
# HELPER FUNCTIONS
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
    if not s.lower().startswith(("select", "with")):
        return "SELECT 1"
    return s.strip()

def normalize_sql(sql):
    return re.sub(r"\s+", " ", str(sql)).strip().lower() if sql else ""

def normalize_result(res):
    if not res:
        return []
    try:
        return sorted([tuple(sorted(str(x) for x in row)) for row in res])
    except:
        return sorted([str(r) for r in res])

def check_execution(pred_sql, gold_sql, db_path):
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
    except:
        return False

# =========================================================
# MODEL LOADING (GITHUB CLONE OPTIMIZED)
# =========================================================
def load_model(adapter_path, device):
    print(f"\n⚙️ Loading model from: {adapter_path}")

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

    print("✅ LoRA adapter loaded and merged successfully.")
    return model

# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(adapter_path, model_name, dev_data, db_root, tokenizer, device):
    print(f"\n==============================")
    print(f"🚀 Evaluating: {model_name}")
    print(f"==============================")

    model = load_model(adapter_path, device)

    em_correct = 0
    ex_correct = 0

    for i, ex in enumerate(dev_data, 1):
        db_id = ex["db_id"]
        question = ex["question"]
        gold_query = ex["query"]
        db_path = db_root / db_id / f"{db_id}.sqlite"

        input_tensor = encode_prompt(tokenizer, question, db_id, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_tensor, max_new_tokens=128, num_beams=4)

        pred_sql = fix_sql(tokenizer.decode(outputs[0], skip_special_tokens=True))

        if normalize_sql(pred_sql) == normalize_sql(gold_query):
            em_correct += 1
        if check_execution(pred_sql, gold_query, str(db_path)):
            ex_correct += 1

        if i % 50 == 0:
            print(f"  {i}/{len(dev_data)} samples processed...")

    del model
    gc.collect()
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()

    return em_correct, ex_correct

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_hard", type=str, required=True, help="Relative path to Hard Reward model")
    parser.add_argument("--adapter_soft", type=str, required=True, help="Relative path to Soft Reward model")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name in ["scripts", "src", "project3"] else script_dir

    dev_json = project_root / "data" / "dev.json"
    if not dev_json.exists():
        dev_json = project_root / "spider" / "dev.json"
    db_root = dev_json.parent / "database"

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    with open(dev_json, "r") as f:
        dev_data = json.load(f)[:args.num_samples]

    em_hard, ex_hard = evaluate_model(args.adapter_hard, "Hard Reward", dev_data, db_root, tokenizer, device)
    em_soft, ex_soft = evaluate_model(args.adapter_soft, "Soft Reward", dev_data, db_root, tokenizer, device)

    print("\n========================================")
    print("🎯 FINAL TASK 4 COMPARISON RESULTS")
    print("========================================")
    print("\n🧱 HARD REWARD MODEL:")
    print(f"  Exact Match (EM):      {(em_hard/len(dev_data))*100:.2f}%")
    print(f"  Execution Acc (EX):    {(ex_hard/len(dev_data))*100:.2f}%")
    print("\n☁️ SOFT REWARD MODEL:")
    print(f"  Exact Match (EM):      {(em_soft/len(dev_data))*100:.2f}%")
    print(f"  Execution Acc (EX):    {(ex_soft/len(dev_data))*100:.2f}%")
    print("========================================")

if __name__ == "__main__":
    main()