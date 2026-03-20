
import sqlite3
import torch
import re
import time
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.generation.logits_process import LogitsProcessorList
from peft import PeftModel

# Fix imports to ensure src modules can be found
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sql_validator import SQLValidator
from src.schema_encoder import SchemaEncoder  
# 🔥 IMPORT THE CONSTRAINED DECODING LOGIC
from src.constrained_decoding import SchemaConstraintGraph, SchemaConstrainedLogitsProcessor

DB_ROOT = PROJECT_ROOT / "data" / "database"

# ==========================================
# ✅ ADVANCED INPUT VALIDATION
# ==========================================
def is_valid_question(q: str):
    q = q.strip().lower()
    if len(q) < 5: return False
    words = re.findall(r"[a-zA-Z]+", q)
    if len(words) < 2: return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 12: return False
    vowel_ratio = sum(c in "aeiou" for c in q) / len(q)
    if vowel_ratio < 0.2: return False
    if not re.search(r"\b(is|are|was|were|find|list|show|get|count|give|display|which|who|what|where|when)\b", q):
        return False
    if len(set(words)) < len(words) * 0.5: return False
    return True

# ==========================================
# NORMALIZATION & FIXES
# ==========================================
def normalize_question(q: str):
    q = q.lower().strip()
    q = re.sub(r"distinct\s+(\d+)", r"\1 distinct", q)
    q = re.sub(r"\s+", " ", q)
    return q

def semantic_fix(question, sql):
    q = question.lower().strip()
    s = sql.lower()
    num_match = re.search(r'\b(?:show|list|top|limit|get|first|last)\s+(?:me\s+)?(\d+)\b', q)
    if num_match and "limit" not in s and "count(" not in s:
        sql = sql.rstrip(";")
        sql = f"{sql.strip()} LIMIT {num_match.group(1)}"
    return sql

def extract_sql(text):
    if "SQL:" in text: text = text.split("SQL:")[-1]
    match = re.search(r"(select|with)[\s\S]*", text, re.IGNORECASE)
    return match.group(0).split(";")[0].strip() if match else ""

def clean_sql(sql: str):
    sql = sql.replace('"', "'")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()

# ==========================================
# ENGINE
# ==========================================
class Text2SQLEngine:

    def __init__(self,
                 adapter_path="checkpoints/best_rlhf_model_2",
                 base_model_name="Salesforce/codet5-base",
                 use_lora=True,
                 use_constrained_decoding=False):

        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.validator = SQLValidator(DB_ROOT)
        self.schema_encoder = SchemaEncoder(DB_ROOT)
        self.use_constrained_decoding = use_constrained_decoding
        self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate|create|replace|rename|grant|revoke)\b'

        print(f"\n📦 Loading model on {self.device}...")

        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        adapter_path = Path(PROJECT_ROOT) / adapter_path

        if use_lora and adapter_path.exists():
            self.model = PeftModel.from_pretrained(
                base,
                str(adapter_path),
                local_files_only=True
            ).to(self.device)
            # ✅ merge LoRA → stable + faster UI generation
            self.model = self.model.merge_and_unload()
            print(f"✅ LoRA loaded and merged from {adapter_path}")
        else:
            print(f"⚠️ Adapter not found at {adapter_path}, using base model")
            self.model = base.to(self.device)

        self.model.eval()

    # 🔥 FIX: THIS IS THE FUNCTION THE UI WAS MISSING
    def get_schema(self, db_id):
        return self.schema_encoder.structured_schema(db_id)

    def build_prompt(self, question, schema):
        return f"""You are an expert SQL generator.
Schema:
{schema}
Question:
{question}
SQL:
"""

    def build_repair_prompt(self, question, schema, bad_sql, error_msg):
        return f"""Fix SQL.
Schema:
{schema}
Bad SQL:
{bad_sql}
Error:
{error_msg}
SQL:
"""

    def repair_logic(self, question, sql):
        q = question.lower()
        s = sql.lower()
        if re.search(r'\b(never|no|without)\b', q):
            m = re.search(r"from\s+(\w+).*join\s+(\w+)", s)
            if m:
                left, right = m.group(1), m.group(2)
                key = re.search(r"on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", s)
                if key:
                    sql = f"SELECT {left}.* FROM {left} LEFT JOIN {right} ON {key.group(1)} = {key.group(2)} WHERE {key.group(2)} IS NULL"
        if any(w in q for w in ["contain", "with", "include"]):
            if any(col in s for col in ["name", "title", "description"]):
                sql = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql)
        return sql

    def generate_sql(self, prompt, db_id=None, is_repair=False):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 6, 
            "length_penalty": 0.9, 
            "early_stopping": True
        }

        if is_repair:
            gen_kwargs.update({"do_sample": True, "temperature": 0.5, "top_p": 0.9, "num_beams": 1})

        # 🔥 TASK 3: INJECT CONSTRAINTS AT INFERENCE TIME!
        if self.use_constrained_decoding and db_id:
            db_path = str(DB_ROOT / db_id / f"{db_id}.sqlite")
            schema_graph = SchemaConstraintGraph(db_path)
            logits_processor = LogitsProcessorList([
                SchemaConstrainedLogitsProcessor(self.tokenizer, schema_graph)
            ])
            gen_kwargs["logits_processor"] = logits_processor

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_sql(extract_sql(decoded))

    # 🔥 FIX: WE NEED THIS TO ACTUALLY RUN THE DB QUERY FOR THE UI TABLE
    def execute_sql(self, question, sql, db_id):
        if re.search(self.dml_keywords, sql, re.IGNORECASE):
            return sql, [], [], "❌ Blocked malicious SQL"

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return sql, [], [], f"Database not found: {db_path}"

        sql = self.repair_logic(question, sql)
        sql = clean_sql(sql)
        sql = semantic_fix(question, sql)

        schema = self.get_schema(db_id).lower()
        tables = re.findall(r"from\s+(\w+)", sql.lower())
        for t in tables:
            if t not in schema:
                return sql, [], [], f"❌ Invalid table detected: {t}"

        is_valid, reason = self.validator.validate(sql, db_id)
        if not is_valid:
            return sql, [], [], f"Blocked: {reason}"

        try:
            conn = sqlite3.connect(db_path)
            start_time = time.monotonic()
            def timeout_handler():
                if (time.monotonic() - start_time) > 5.0: raise TimeoutError("Query timeout")
                return 0
            conn.set_progress_handler(timeout_handler, 10000)

            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []
            conn.close()
            return sql, columns, rows, None
        except Exception as e:
            return sql, [], [], str(e)

    def ask(self, question, db_id):
        question = normalize_question(question)

        # 🛑 1. SECURITY FIRST
        if re.search(self.dml_keywords, question, re.IGNORECASE):
            return {"question": question, "sql": "-- BLOCKED", "columns": [], "rows": [], "error": "❌ Blocked: Destructive operations not allowed."}

        # 🧠 2. VALIDATION
        if not is_valid_question(question):
            return {"question": question, "sql": "-- INVALID INPUT", "columns": [], "rows": [], "error": "❌ Please enter a meaningful database question."}

        schema = self.get_schema(db_id)
        prompt = self.build_prompt(question, schema)
        
        # GENERATE (Passing db_id so constraints can activate!)
        raw_sql = self.generate_sql(prompt, db_id=db_id)

        if not raw_sql:
            return {"question": question, "sql": "-- NO_SQL", "columns": [], "rows": [], "error": "Model failed"}

        # EXECUTE
        final_sql, cols, rows, error = self.execute_sql(question, raw_sql, db_id)

        # 🔁 REPAIR LOOP
        if (error or not rows):
            repair_prompt = self.build_repair_prompt(question, schema, final_sql, error or "Empty result")
            repaired_sql = self.generate_sql(repair_prompt, db_id=db_id, is_repair=True)
            final_sql, cols, rows, error = self.execute_sql(question, repaired_sql, db_id)

        return {
            "question": question,
            "sql": final_sql,
            "columns": cols,
            "rows": rows,
            "error": error
        }

# ==========================================
# FACTORY
# ==========================================
_engine = None

def get_engine(adapter_path="checkpoints/best_rlhf_model", base_model_name="Salesforce/codet5-base", use_lora=True, use_constrained=False):
    global _engine

    if _engine is None:
        _engine = Text2SQLEngine(
            adapter_path=adapter_path, 
            base_model_name=base_model_name, 
            use_lora=use_lora,
            use_constrained_decoding=use_constrained
        )
    else:
        # Prevent reloading massive models on every click
        _engine.use_constrained_decoding = use_constrained

    return _engine