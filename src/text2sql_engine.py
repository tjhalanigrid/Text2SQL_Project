# import sqlite3
# import torch
# import re
# import time
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel
# from src.sql_validator import SQLValidator
# from src.schema_encoder import SchemaEncoder

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# # ================================
# # DATABASE PATH AUTO DETECTION
# # ================================
# if (PROJECT_ROOT / "data/database").exists():
#     DB_ROOT = PROJECT_ROOT / "data/database"
# else:
#     DB_ROOT = PROJECT_ROOT / "final_databases"


# def normalize_question(q: str):
#     q = q.lower().strip()
#     q = re.sub(r"distinct\s+(\d+)", r"\1 distinct", q)
#     q = re.sub(r"\s+", " ", q)
#     return q


# def semantic_fix(question, sql):
#     q = question.lower().strip()
#     s = sql.lower()

#     num_match = re.search(r'\b(?:show|list|top|limit|get|first|last)\s+(\d+)\b', q)

#     if num_match and "limit" not in s and "count(" not in s:
#         limit_val = num_match.group(1)
#         sql = sql.rstrip(";")
#         sql = f"{sql.strip()} LIMIT {limit_val}"

#     return sql


# class Text2SQLEngine:
#     def __init__(self,
#                  adapter_path=None,
#                  base_model_name="Salesforce/codet5-base",
#                  use_lora=True):

#         self.device = "mps" if torch.backends.mps.is_available() else (
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         self.validator = SQLValidator(DB_ROOT)
#         self.schema_encoder = SchemaEncoder(DB_ROOT)

#         self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate)\b'

#         print("Loading base model...")
#         base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

#         if not use_lora:
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#             self.model = base.to(self.device)
#             self.model.eval()
#             return

#         if (PROJECT_ROOT / "checkpoints/best_rlhf_model").exists():
#             adapter_path = PROJECT_ROOT / "checkpoints/best_rlhf_model"
#         else:
#             adapter_path = PROJECT_ROOT / "best_rlhf_model"

#         adapter_path = adapter_path.resolve()

#         print("Loading tokenizer and LoRA adapter...")

#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 str(adapter_path),
#                 local_files_only=True
#             )
#         except Exception:
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#         self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
#         self.model.eval()

#         print("✅ RLHF model ready\n")

#     def build_prompt(self, question, schema):
#         return f"""You are an expert SQL generator.
# Database schema:
# {schema}
# Generate a valid SQLite query for the question.
# Question:
# {question}
# SQL:
# """

#     def get_schema(self, db_id):
#         return self.schema_encoder.structured_schema(db_id)

#     def extract_sql(self, text: str):

#         text = text.strip()

#         if "SQL:" in text:
#             text = text.split("SQL:")[-1]

#         match = re.search(r"select[\s\S]*", text, re.IGNORECASE)

#         if match:
#             text = match.group(0)

#         return text.split(";")[0].strip()

#     def clean_sql(self, sql: str):

#         sql = sql.replace('"', "'")
#         sql = re.sub(r"\s+", " ", sql)

#         return sql.strip()

#     def generate_sql(self, prompt):

#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512
#         ).to(self.device)

#         with torch.no_grad():

#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 num_beams=5,
#                 early_stopping=True
#             )

#         decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return self.clean_sql(self.extract_sql(decoded))

#     def execute_sql(self, question, sql, db_id):

#         if re.search(self.dml_keywords, sql, re.IGNORECASE):
#             return sql, [], [], "❌ Security Alert"

#         # FIXED DATABASE PATH
#         db_path = DB_ROOT / f"{db_id}.sqlite"

#         sql = self.clean_sql(sql)
#         sql = semantic_fix(question, sql)

#         try:

#             conn = sqlite3.connect(db_path)

#             cursor = conn.cursor()

#             cursor.execute(sql)

#             rows = cursor.fetchall()

#             columns = [d[0] for d in cursor.description] if cursor.description else []

#             conn.close()

#             return sql, columns, rows, None

#         except Exception as e:

#             return sql, [], [], str(e)

#     def ask(self, question, db_id):

#         question = normalize_question(question)

#         if re.search(self.dml_keywords, question, re.IGNORECASE):

#             return {
#                 "question": question,
#                 "sql": "-- BLOCKED",
#                 "columns": [],
#                 "rows": [],
#                 "error": "Malicious prompt"
#             }

#         schema = self.get_schema(db_id)

#         prompt = self.build_prompt(question, schema)

#         raw_sql = self.generate_sql(prompt)

#         final_sql, cols, rows, error = self.execute_sql(question, raw_sql, db_id)

#         return {
#             "question": question,
#             "sql": final_sql,
#             "columns": cols,
#             "rows": rows,
#             "error": error
#         }


# _engine = None


# def get_engine():

#     global _engine

#     if _engine is None:
#         _engine = Text2SQLEngine()

#     return _engine



import sqlite3
import torch
import re
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from src.sql_validator import SQLValidator
from src.schema_encoder import SchemaEncoder, build_schema_graph # Added build_schema_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ================================
# DATABASE PATH AUTO DETECTION
# ================================
if (PROJECT_ROOT / "data/database").exists():
    DB_ROOT = PROJECT_ROOT / "data/database"
else:
    DB_ROOT = PROJECT_ROOT / "final_databases"


# ==========================================
# INPUT VALIDATION & RELEVANCE (From Code 1)
# ==========================================
def is_valid_question(q: str):
    """Extremely relaxed valid question checker. As long as there is 1 word, it passes."""
    words = re.findall(r"[a-zA-Z0-9]+", q)
    return len(words) >= 1

def is_relevant_to_db(question: str, schema_graph: dict):
    """
    Lexical heuristic to block completely out-of-domain questions
    while allowing valid plurals.
    """
    q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
    stop_words = {"show", "list", "all", "and", "the", "get", "find", "how", "many", "what", "where", "which", "who", "give", "display", "count", "from", "for", "with", "that", "have", "has", "are", "there"}
    q_words = q_words - stop_words
    
    if not q_words:
        return True
        
    schema_words = set()
    for table, cols in schema_graph.items():
        schema_words.update(re.findall(r'\b[a-z]{3,}\b', table.lower()))
        for col in cols:
            schema_words.update(re.findall(r'\b[a-z]{3,}\b', col.lower()))
            
    synonyms = {
        "customer": ["client", "buyer", "shopper", "person", "people", "user"],
        "employee": ["staff", "worker", "boss", "manager", "person", "people"],
        "track": ["song", "music", "audio", "tune"],
        "album": ["record", "cd", "music"],
        "artist": ["singer", "band", "musician", "creator"],
        "invoice": ["bill", "receipt", "purchase", "sale", "order", "buy", "bought", "cost"],
        "city": ["town", "location", "place"],
        "country": ["nation", "location", "place"],
        "flight": ["plane", "airline", "trip", "fly", "airport"],
        "student": ["pupil", "learner", "kid", "child"],
        "club": ["group", "organization", "team"],
        "course": ["class", "subject"],
        "cinema": ["movie", "film", "theater", "screen"]
    }
    
    extended_schema_words = set(schema_words)
    for db_word in schema_words:
        if db_word in synonyms:
            extended_schema_words.update(synonyms[db_word])
            
    extended_schema_words.update({"id", "name", "total", "sum", "average", "avg", "min", "max", "number", "amount", "record", "data", "info", "information", "detail", "first", "last", "most", "least", "cheapest", "expensive", "best"})
    
    # Check if the word OR its singular form is in the schema
    for qw in q_words:
        qw_singular = qw[:-1] if qw.endswith('s') else qw
        if qw in extended_schema_words or qw_singular in extended_schema_words:
            return True
            
    return False

# ==========================================
# SCHEMA CONSTRAINTS (From Code 1)
# ==========================================
def apply_schema_constraints(sql, schema_graph):
    sql = sql.lower()

    used_tables = [t[1] for t in re.findall(r'(from|join)\s+(\w+)', sql)]
    for t in used_tables:
        if t not in schema_graph:
            return None

    valid_columns = set()
    for cols in schema_graph.values():
        valid_columns.update(cols)

    col_blocks = re.findall(r'select\s+(.*?)\s+from', sql)
    for block in col_blocks:
        for c in block.split(","):
            c = c.strip().split()[-1]
            if "." in c:
                c = c.split(".")[-1]
            
            if c != "*" and "(" not in c and c != "":
                if c not in valid_columns:
                    return None

    return sql


def normalize_question(q: str):
    q = q.lower().strip()
    q = re.sub(r"distinct\s+(\d+)", r"\1 distinct", q)
    q = re.sub(r"\s+", " ", q)
    return q


def semantic_fix(question, sql):
    q = question.lower().strip()
    s = sql.lower()

    num_match = re.search(r'\b(?:show|list|top|limit|get|first|last)\s+(\d+)\b', q)

    if num_match and "limit" not in s and "count(" not in s:
        limit_val = num_match.group(1)
        sql = sql.rstrip(";")
        sql = f"{sql.strip()} LIMIT {limit_val}"

    return sql


class Text2SQLEngine:
    def __init__(self,
                 adapter_path=None,
                 base_model_name="Salesforce/codet5-base",
                 use_lora=True,
                 use_constrained_decoding=True): # Added constrained decoding flag

        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.validator = SQLValidator(DB_ROOT)
        self.schema_encoder = SchemaEncoder(DB_ROOT)
        self.use_constrained_decoding = use_constrained_decoding

        self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate)\b'

        print("Loading base model...")
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        if not use_lora:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = base.to(self.device)
            self.model.eval()
            return

        if (PROJECT_ROOT / "checkpoints/best_rlhf_model").exists():
            adapter_path = PROJECT_ROOT / "checkpoints/best_rlhf_model"
        else:
            adapter_path = PROJECT_ROOT / "best_rlhf_model"

        adapter_path = adapter_path.resolve()

        print("Loading tokenizer and LoRA adapter...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_path),
                local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
        self.model.eval()

        print("✅ RLHF model ready\n")

    def build_prompt(self, question, schema):
        return f"""You are an expert SQL generator.
Database schema:
{schema}
Generate a valid SQLite query for the question.
Question:
{question}
SQL:
"""

    def get_schema(self, db_id):
        return self.schema_encoder.structured_schema(db_id)

    def extract_sql(self, text: str):
        text = text.strip()
        if "SQL:" in text:
            text = text.split("SQL:")[-1]
        match = re.search(r"select[\s\S]*", text, re.IGNORECASE)
        if match:
            text = match.group(0)
        return text.split(";")[0].strip()

    def clean_sql(self, sql: str):
        sql = sql.replace('"', "'")
        sql = re.sub(r"\s+", " ", sql)
        return sql.strip()

    def generate_sql(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                early_stopping=True
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_sql(self.extract_sql(decoded))

    def execute_sql(self, question, sql, db_id):
        if re.search(self.dml_keywords, sql, re.IGNORECASE):
            return sql, [], [], "❌ Security Alert"

        # FIXED DATABASE PATH (From Code 2)
        db_path = DB_ROOT / f"{db_id}.sqlite"

        sql = self.clean_sql(sql)
        sql = semantic_fix(question, sql)

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []
            conn.close()
            return sql, columns, rows, None
        except Exception as e:
            return sql, [], [], str(e)

    def ask(self, question, db_id):
        # 1. Normalize
        question_norm = normalize_question(question)
        question_context = f"Database question: {question_norm}"

        # 2. Block dangerous inputs
        if re.search(self.dml_keywords, question_context, re.IGNORECASE):
            return {
                "question": question_norm,
                "sql": "-- BLOCKED",
                "columns": [],
                "rows": [],
                "error": "❌ Malicious prompt"
            }

        # 3. Check basic validity of question
        if not is_valid_question(question_context):
            return {"sql": "", "error": "❌ Invalid input. Please type words."}

        schema = self.get_schema(db_id)
        schema_graph = build_schema_graph(schema)

        # 4. LEXICAL RELEVANCE GUARDRAIL 
        if not is_relevant_to_db(question_norm, schema_graph):
            return {"sql": "", "error": "❌ Question is completely out of domain for the selected database."}

        # 5. INITIAL GENERATION
        prompt = self.build_prompt(question_context, schema)
        raw_sql = self.generate_sql(prompt)

        # 6. STRONGER CONSTRAINT LOGIC
        if self.use_constrained_decoding:
            filtered_sql = apply_schema_constraints(raw_sql, schema_graph)

            if filtered_sql is None:
                constraint_prompt = f"""Use ONLY valid schema.
Database schema:
{schema}
Generate a valid SQLite query for the question.
Question:
{question_context}
SQL:
"""
                sql_retry = self.generate_sql(constraint_prompt)
                filtered_sql = apply_schema_constraints(sql_retry, schema_graph)

                if filtered_sql:
                    raw_sql = filtered_sql
                else:
                    raw_sql = sql_retry

        # 7. EXECUTION
        final_sql, cols, rows, error = self.execute_sql(question_norm, raw_sql, db_id)

        return {
            "question": question_norm,
            "sql": final_sql,
            "columns": cols,
            "rows": rows,
            "error": error
        }


_engine = None

def get_engine(use_constrained=True): # Added parameter to control constraints
    global _engine

    if _engine is None:
        _engine = Text2SQLEngine(use_constrained_decoding=use_constrained)

    return _engine

# import sqlite3
# import torch
# import re
# import os
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel
# from src.sql_validator import SQLValidator
# from src.schema_encoder import SchemaEncoder  # Removed build_schema_graph import

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# # ================================
# # DATABASE PATH AUTO DETECTION
# # ================================
# if (PROJECT_ROOT / "data/database").exists():
#     DB_ROOT = PROJECT_ROOT / "data/database"
# else:
#     DB_ROOT = PROJECT_ROOT / "final_databases"


# # ==========================================
# # SCHEMA PARSING
# # ==========================================
# def build_schema_graph(schema_text):
#     """
#     Parses a structured schema text string into a dictionary graph.
#     Matches formats like: table_name(col1, col2, col3)
#     """
#     tables = {}
#     for match in re.findall(r'(\w+)\s*\((.*?)\)', schema_text):
#         table = match[0]
#         cols = [c.strip().split()[0] for c in match[1].split(",")]
#         tables[table] = cols
#     return tables


# # ==========================================
# # INPUT VALIDATION & RELEVANCE
# # ==========================================
# def is_valid_question(q: str):
#     q = q.strip().lower()

#     if len(q) < 3:
#         return False

#     words = re.findall(r"[a-zA-Z]+", q)
#     if len(words) < 1:
#         return False

#     return True


# def is_relevant_to_db(question: str, schema_graph: dict):
#     q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
#     stop_words = {"show", "list", "all", "and", "the", "get", "find", "how", "many", "what", "where", "which", "who", "give", "display", "count", "from", "for", "with", "that", "have", "has", "are", "there"}
#     q_words = q_words - stop_words
    
#     if not q_words:
#         return True
        
#     schema_words = set()
#     for table, cols in schema_graph.items():
#         schema_words.update(re.findall(r'\b[a-z]{3,}\b', table.lower()))
#         for col in cols:
#             schema_words.update(re.findall(r'\b[a-z]{3,}\b', col.lower()))
            
#     synonyms = {
#         "customer": ["client", "buyer", "shopper", "person", "people", "user"],
#         "employee": ["staff", "worker", "boss", "manager", "person", "people"],
#         "track": ["song", "music", "audio", "tune"],
#         "album": ["record", "cd", "music"],
#         "artist": ["singer", "band", "musician", "creator"],
#         "invoice": ["bill", "receipt", "purchase", "sale", "order", "buy", "bought", "cost"],
#         "city": ["town", "location", "place"],
#         "country": ["nation", "location", "place"],
#         "flight": ["plane", "airline", "trip", "fly", "airport"],
#         "student": ["pupil", "learner", "kid", "child"],
#         "club": ["group", "organization", "team"],
#         "course": ["class", "subject"],
#         "cinema": ["movie", "film", "theater", "screen"]
#     }
    
#     extended_schema_words = set(schema_words)
#     for db_word in schema_words:
#         if db_word in synonyms:
#             extended_schema_words.update(synonyms[db_word])
            
#     extended_schema_words.update({"id", "name", "total", "sum", "average", "avg", "min", "max", "number", "amount", "record", "data", "info", "information", "detail", "first", "last", "most", "least", "cheapest", "expensive", "best"})
    
#     for qw in q_words:
#         qw_singular = qw[:-1] if qw.endswith('s') else qw
#         if qw in extended_schema_words or qw_singular in extended_schema_words:
#             return True
            
#     return False

# def normalize_question(q: str):
#     return re.sub(r"\s+", " ", q.lower().strip())

# def semantic_fix(question, sql):
#     q = question.lower()
#     num_match = re.search(r'\b(?:show|list|top|get)\s+(\d+)\b', q)

#     if num_match and "limit" not in sql.lower():
#         sql = f"{sql} LIMIT {num_match.group(1)}"

#     return sql

# # ==========================================
# # SCHEMA CONSTRAINTS (SIMULATED LOGIT BLOCKING)
# # ==========================================
# def apply_schema_constraints(sql, schema_graph):
#     sql = sql.lower()

#     used_tables = [t[1] for t in re.findall(r'(from|join)\s+(\w+)', sql)]
#     for t in used_tables:
#         if t not in schema_graph:
#             return None

#     valid_columns = set()
#     for cols in schema_graph.values():
#         valid_columns.update(cols)

#     col_blocks = re.findall(r'select\s+(.*?)\s+from', sql)
#     for block in col_blocks:
#         for c in block.split(","):
#             c = c.strip().split()[-1]
#             if "." in c:
#                 c = c.split(".")[-1]
            
#             if c != "*" and "(" not in c and c != "":
#                 if c not in valid_columns:
#                     return None

#     return sql

# # ==========================================
# # ENGINE
# # ==========================================
# class Text2SQLEngine:

#     def __init__(self,
#                  adapter_path="checkpoints/best_rlhf_model_2",
#                  base_model_name="Salesforce/codet5-base",
#                  use_lora=True,
#                  use_constrained_decoding=False):

#         self.device = "mps" if torch.backends.mps.is_available() else (
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         self.validator = SQLValidator(DB_ROOT)
#         self.schema_encoder = SchemaEncoder(DB_ROOT)

#         self.use_constrained_decoding = use_constrained_decoding
#         self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate|create)\b'

#         print(f"\n📦 Loading model on {self.device}...")

#         base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
#         # Override the redundant special tokens to prevent the tokenizer crash
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             base_model_name,
#             use_fast=False,
#             additional_special_tokens=[]
#         )

#         # 🔥 FIXED LOADA ADAPTER PATH LOGIC
#         if use_lora:
#             if adapter_path and (PROJECT_ROOT / adapter_path).exists():
#                 adapter_path = PROJECT_ROOT / adapter_path
#             elif (PROJECT_ROOT / "checkpoints/best_rlhf_model_2").exists():
#                 adapter_path = PROJECT_ROOT / "checkpoints/best_rlhf_model_2"
#             else:
#                 adapter_path = PROJECT_ROOT / "best_rlhf_model_2"
                
#             adapter_path = adapter_path.resolve()

#             if adapter_path.exists():
#                 try:
#                     self.model = PeftModel.from_pretrained(
#                         base,
#                         str(adapter_path),
#                         local_files_only=True
#                     ).to(self.device)
#                     print(f"✅ LoRA loaded from {adapter_path}")
#                 except Exception as e:
#                     print(f"⚠️ LoRA load failed: {e}")
#                     self.model = base.to(self.device)
#             else:
#                 print(f"⚠️ Adapter not found at {adapter_path}, using base model")
#                 self.model = base.to(self.device)
#         else:
#             self.model = base.to(self.device)

#         self.model.eval()

#     def build_prompt(self, question, schema):
#         return f"""
# You are an expert SQL generator.

# IMPORTANT:
# - Use correct tables and columns
# - Use JOINs when needed

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

#     def generate_sql(self, prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 num_beams=8,
#                 length_penalty=0.8,
#                 early_stopping=True
#             )

#         decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return self.clean_sql(self.extract_sql(decoded))

#     def execute_sql(self, question, sql, db_id):

#         if re.search(self.dml_keywords, sql, re.IGNORECASE):
#             return "", [], [], "❌ Blocked malicious SQL (Contains INSERT/UPDATE/DELETE/DROP)"

#         # 🔥 FIXED DATABASE PATH
#         db_path = DB_ROOT / f"{db_id}.sqlite"
#         sql = semantic_fix(question, sql)

#         try:
#             conn = sqlite3.connect(db_path)
#             cursor = conn.cursor()
#             cursor.execute(sql)

#             rows = cursor.fetchall()
#             columns = [d[0] for d in cursor.description] if cursor.description else []

#             conn.close()
#             return sql, columns, rows, None

#         except Exception as e:
#             return sql, [], [], str(e)

#     def ask(self, question, db_id):

#         question = normalize_question(question)
#         question_context = f"Database question: {question}"

#         if re.search(self.dml_keywords, question_context, re.IGNORECASE):
#             return {"sql": "", "error": "❌ Blocked dangerous query from input text."}

#         if not is_valid_question(question_context):
#             return {"sql": "", "error": "❌ Invalid input. Please type words."}

#         schema = self.get_schema(db_id)
#         schema_graph = build_schema_graph(schema)

#         if not is_relevant_to_db(question, schema_graph):
#             return {"sql": "", "error": "❌ Question is completely out of domain for the selected database."}

#         sql = self.generate_sql(self.build_prompt(question_context, schema))

#         if self.use_constrained_decoding:
#             filtered_sql = apply_schema_constraints(sql, schema_graph)

#             if filtered_sql is None:
#                 constraint_prompt = f"""
# Use ONLY valid schema.
# Schema:
# {schema}

# Question:
# {question_context}

# SQL:
# """
#                 sql_retry = self.generate_sql(constraint_prompt)
#                 filtered_sql = apply_schema_constraints(sql_retry, schema_graph)

#                 if filtered_sql:
#                     sql = filtered_sql
#                 else:
#                     sql = sql_retry

#         final_sql, cols, rows, error = self.execute_sql(question_context, sql, db_id)

#         return {
#             "question": question_context,
#             "sql": final_sql,
#             "columns": cols,
#             "rows": rows,
#             "error": error
#         }

# def get_engine(
#     adapter_path="checkpoints/best_rlhf_model_2",
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