
# ********** main *******************************



import sqlite3
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from src.sql_validator import SQLValidator, validate_sql_schema
from src.schema_encoder import SchemaEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"

# ==========================================
def is_valid_question(q: str):
    return len(re.findall(r"[a-zA-Z0-9]+", q)) >= 1

def normalize_question(q: str):
    return re.sub(r"\s+", " ", q.lower().strip())

def semantic_fix(question, sql):
    match = re.search(r'\b(?:show|list|top|get)\s+(\d+)\b', question.lower())
    if match and "limit" not in sql.lower():
        sql += f" LIMIT {match.group(1)}"
    return sql

# ==========================================
class Text2SQLEngine:

    def __init__(self,
                 adapter_path="checkpoints/best_rlhf_model",
                 base_model_name="Salesforce/codet5-base",
                 use_lora=True,
                 use_constrained_decoding=False):

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.validator = SQLValidator(DB_ROOT)
        self.schema_encoder = SchemaEncoder(DB_ROOT)

        self.use_constrained_decoding = use_constrained_decoding
        self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate|create)\b'

        print(f"\n📦 Loading model on {self.device}...")

        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if adapter_path is not None:
            adapter_path = Path(adapter_path)

        if use_lora and adapter_path and adapter_path.exists():
            try:
                self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
                print(f"✅ LoRA loaded from {adapter_path}")
            except:
                self.model = base.to(self.device)
        else:
            self.model = base.to(self.device)

        self.model.eval()

    # ==========================================
    def build_prompt(self, question, schema):
        return f"""
You are an expert SQL generator.

IMPORTANT:
- Use correct tables and columns
- Use JOINs only when necessary

Schema:
{schema}

Question:
{question}

SQL:
"""

    def get_schema(self, db_id):
        return self.schema_encoder.structured_schema(db_id)

    def extract_sql(self, text):
        match = re.search(r"(select|with)[\s\S]*", text, re.IGNORECASE)
        return match.group(0).split(";")[0].strip() if match else ""

    def clean_sql(self, sql):
        return re.sub(r"\s+", " ", sql.replace('"', "'")).strip()

    # ==========================================
    def generate_sql(self, prompt):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            do_sample=False
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = self.clean_sql(self.extract_sql(decoded))

        return sql

    # ==========================================
    def ask(self, question, db_id):

        question = normalize_question(question)
        question_context = f"Database question: {question}"

        if re.search(self.dml_keywords, question_context, re.IGNORECASE):
            return {"sql": "", "error": "Blocked dangerous query"}

        if not is_valid_question(question_context):
            return {"sql": "", "error": "Invalid input"}

        schema = self.get_schema(db_id)

        # 🔥 STEP 1: GENERATE
        sql = self.generate_sql(
            self.build_prompt(question_context, schema)
        )

        # 🔥 STEP 2: JOIN PENALTY (safe retry)
        if self.use_constrained_decoding:

            if " join " in sql.lower() and any(x in question.lower() for x in ["count", "avg", "average", "total"]):

                retry_prompt = f"""
Generate SIMPLE SQL.

Rules:
- Avoid unnecessary JOIN
- Prefer single table queries

{question_context}
Schema:
{schema}
"""

                sql_retry = self.generate_sql(retry_prompt)

                if sql_retry:
                    sql = sql_retry

        # 🔥 STEP 3: SAFE VALIDATION (REAL CONSTRAINT)
        if self.use_constrained_decoding:

            db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

            try:
                is_valid, _ = validate_sql_schema(sql, str(db_path))
            except:
                is_valid = False

            if not is_valid or not sql:

                retry_prompt = f"""
Fix this SQL query.

SQL:
{sql}

Schema:
{schema}

Return corrected SQL only.
"""

                sql_retry = self.generate_sql(retry_prompt)

                if sql_retry:
                    sql = sql_retry

        # 🔥 STEP 4: EXECUTE
        final_sql, cols, rows, error = self.execute_sql(question_context, sql, db_id)

        return {
            "question": question_context,
            "sql": final_sql,
            "columns": cols,
            "rows": rows,
            "error": error
        }

    # ==========================================
    def execute_sql(self, question, sql, db_id):

        if re.search(self.dml_keywords, sql, re.IGNORECASE):
            return "", [], [], "Blocked dangerous SQL"

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
        sql = semantic_fix(question, sql)

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(sql)

            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []

            conn.close()
            return sql, cols, rows, None

        except Exception as e:
            return sql, [], [], str(e)

# ==========================================
def get_engine(
    adapter_path="checkpoints/best_rlhf_model",
    base_model_name="Salesforce/codet5-base",
    use_lora=True,
    use_constrained=True
):
    return Text2SQLEngine(
        adapter_path=adapter_path,
        base_model_name=base_model_name,
        use_lora=use_lora,
        use_constrained_decoding=use_constrained
    )




#  ************ 2 ********************************* 



# import sqlite3
# import torch
# import re
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# from src.sql_validator import SQLValidator, validate_sql_schema
# from src.schema_encoder import SchemaEncoder

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DB_ROOT = PROJECT_ROOT / "data" / "database"

# # ==========================================
# def is_valid_question(q: str):
#     return len(re.findall(r"[a-zA-Z0-9]+", q)) >= 1

# def normalize_question(q: str):
#     return re.sub(r"\s+", " ", q.lower().strip())

# def semantic_fix(question, sql):
#     match = re.search(r'\b(?:show|list|top|get)\s+(\d+)\b', question.lower())
#     if match and "limit" not in sql.lower():
#         sql += f" LIMIT {match.group(1)}"
#     return sql

# # ==========================================
# class Text2SQLEngine:

#     def __init__(self,
#                  adapter_path="checkpoints/best_rlhf_model",
#                  base_model_name="Salesforce/codet5-base",
#                  use_lora=True,
#                  use_constrained_decoding=False):

#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"

#         self.validator = SQLValidator(DB_ROOT)
#         self.schema_encoder = SchemaEncoder(DB_ROOT)

#         self.use_constrained_decoding = use_constrained_decoding
#         self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate|create)\b'

#         print(f"\n📦 Loading model on {self.device}...")

#         base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#         if adapter_path is not None:
#             adapter_path = Path(adapter_path)

#         if use_lora and adapter_path and adapter_path.exists():
#             try:
#                 self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
#                 print(f"✅ LoRA loaded from {adapter_path}")
#             except:
#                 self.model = base.to(self.device)
#         else:
#             self.model = base.to(self.device)

#         self.model.eval()

#     # ==========================================
#     def build_prompt(self, question, schema):
#         return f"""
# You are an expert SQL generator.

# IMPORTANT:
# - Use correct tables and columns
# - Use JOINs only when necessary

# Schema:
# {schema}

# Question:
# {question}

# SQL:
# """

#     def get_schema(self, db_id):
#         return self.schema_encoder.structured_schema(db_id)

#     def extract_sql(self, text):
#         match = re.search(r"(select|with)[\s\S]*", text, re.IGNORECASE)
#         return match.group(0).split(";")[0].strip() if match else ""

#     def clean_sql(self, sql):
#         return re.sub(r"\s+", " ", sql.replace('"', "'")).strip()

#     # ==========================================
#     # 🔥 MULTI-GENERATION (BOOST ACCURACY)
#     # ==========================================
#     def generate_sql(self, prompt):

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

#         candidates = []

#         for _ in range(2):   # 🔥 generate 2 candidates
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=80,
#                 num_beams=4,
#                 do_sample=False
#             )

#             decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             sql = self.clean_sql(self.extract_sql(decoded))

#             if sql:
#                 candidates.append(sql)

#         # pick simplest SQL (shortest)
#         return min(candidates, key=len) if candidates else ""

#     # ==========================================
#     def ask(self, question, db_id):

#         question = normalize_question(question)
#         question_context = f"Database question: {question}"

#         if re.search(self.dml_keywords, question_context, re.IGNORECASE):
#             return {"sql": "", "error": "Blocked dangerous query"}

#         if not is_valid_question(question_context):
#             return {"sql": "", "error": "Invalid input"}

#         schema = self.get_schema(db_id)

#         # STEP 1: generate
#         sql = self.generate_sql(
#             self.build_prompt(question_context, schema)
#         )

#         # STEP 2: JOIN penalty retry
#         if self.use_constrained_decoding:
#             if " join " in sql.lower() and any(x in question.lower() for x in ["count", "avg", "average", "total"]):

#                 retry_prompt = f"""
# Generate SIMPLE SQL.

# Rules:
# - Avoid unnecessary JOIN
# - Prefer single table queries

# {question_context}
# Schema:
# {schema}
# """

#                 sql_retry = self.generate_sql(retry_prompt)
#                 if sql_retry:
#                     sql = sql_retry

#         # STEP 3: SCHEMA VALIDATION (REAL CONSTRAINT)
#         if self.use_constrained_decoding:
#             db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

#             try:
#                 is_valid, _ = validate_sql_schema(sql, str(db_path))
#             except:
#                 is_valid = False

#             if not is_valid or not sql:

#                 retry_prompt = f"""
# Fix this SQL query.

# SQL:
# {sql}

# Schema:
# {schema}

# Return corrected SQL only.
# """

#                 sql_retry = self.generate_sql(retry_prompt)
#                 if sql_retry:
#                     sql = sql_retry

#         # STEP 4: EXECUTION
#         final_sql, cols, rows, error = self.execute_sql(question_context, sql, db_id)

#         return {
#             "question": question_context,
#             "sql": final_sql,
#             "columns": cols,
#             "rows": rows,
#             "error": error
#         }

#     # ==========================================
#     def execute_sql(self, question, sql, db_id):

#         if re.search(self.dml_keywords, sql, re.IGNORECASE):
#             return "", [], [], "Blocked dangerous SQL"

#         db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
#         sql = semantic_fix(question, sql)

#         try:
#             conn = sqlite3.connect(db_path)
#             cur = conn.cursor()
#             cur.execute(sql)

#             rows = cur.fetchall()
#             cols = [d[0] for d in cur.description] if cur.description else []

#             conn.close()
#             return sql, cols, rows, None

#         except Exception as e:
#             return sql, [], [], str(e)

# # ==========================================
# def get_engine(
#     adapter_path="checkpoints/best_rlhf_model",
#     base_model_name="Salesforce/codet5-base",
#     use_lora=True,
#     use_constrained=True
# ):
#     return Text2SQLEngine(
#         adapter_path=adapter_path,
#         base_model_name=base_model_name,
#         use_lora=use_lora,
#         use_constrained_decoding=use_constrained
#     )