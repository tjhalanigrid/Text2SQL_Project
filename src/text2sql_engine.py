import sqlite3
import torch
import re
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from src.sql_validator import SQLValidator
from src.schema_encoder import SchemaEncoder, build_schema_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"

# ==========================================
# INPUT VALIDATION & RELEVANCE
# ==========================================
def is_valid_question(q: str):
    """Extremely relaxed valid question checker. As long as there is 1 word, it passes."""
    words = re.findall(r"[a-zA-Z0-9]+", q)
    return len(words) >= 1

def is_relevant_to_db(question: str, schema_graph: dict):
    """
    Lexical heuristic to block completely out-of-domain questions (cats, elephants)
    while allowing valid plurals (employees -> employee).
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
    
    # 🔥 THE PLURAL FIX: Check if the word OR its singular form is in the schema
    for qw in q_words:
        qw_singular = qw[:-1] if qw.endswith('s') else qw
        if qw in extended_schema_words or qw_singular in extended_schema_words:
            return True
            
    return False


def normalize_question(q: str):
    return re.sub(r"\s+", " ", q.lower().strip())

def semantic_fix(question, sql):
    q = question.lower()
    num_match = re.search(r'\b(?:show|list|top|get)\s+(\d+)\b', q)

    if num_match and "limit" not in sql.lower():
        sql = f"{sql} LIMIT {num_match.group(1)}"

    return sql


# ==========================================
# SCHEMA CONSTRAINTS (SIMULATED LOGIT BLOCKING)
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


# ==========================================
# ENGINE
# ==========================================
class Text2SQLEngine:

    def __init__(self,
                 adapter_path="checkpoints/best_rlhf_model",
                 base_model_name="Salesforce/codet5-base",
                 use_lora=True,
                 use_constrained_decoding=False):

        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.validator = SQLValidator(DB_ROOT)
        self.schema_encoder = SchemaEncoder(DB_ROOT)

        self.use_constrained_decoding = use_constrained_decoding
        self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate|create)\b'

        print(f"\n📦 Loading model on {self.device}...")

        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        adapter_path = Path(adapter_path)

        if use_lora and adapter_path.exists():
            try:
                self.model = PeftModel.from_pretrained(
                    base,
                    str(adapter_path),
                    local_files_only=True
                ).to(self.device)
                print(f"✅ LoRA loaded from {adapter_path}")
            except Exception as e:
                print(f"⚠️ LoRA load failed: {e}")
                self.model = base.to(self.device)
        else:
            print("⚠️ Adapter not found, using base model")
            self.model = base.to(self.device)

        self.model.eval()

    def build_prompt(self, question, schema):
        return f"""
You are an expert SQL generator.

IMPORTANT:
- Use correct tables and columns
- Use JOINs when needed

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

    def generate_sql(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=8,
                length_penalty=0.8,
                early_stopping=True
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_sql(self.extract_sql(decoded))

    def execute_sql(self, question, sql, db_id):

        if re.search(self.dml_keywords, sql, re.IGNORECASE):
            return "", [], [], "❌ Blocked malicious SQL (Contains INSERT/UPDATE/DELETE/DROP)"

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
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

        question = normalize_question(question)
        question_context = f"Database question: {question}"

        # 1. Block dangerous inputs from the prompt itself
        if re.search(self.dml_keywords, question_context, re.IGNORECASE):
            return {"sql": "", "error": "❌ Blocked dangerous query from input text."}

        # 2. Check basic validity of question (Now practically un-failable)
        if not is_valid_question(question_context):
            return {"sql": "", "error": "❌ Invalid input. Please type words."}

        schema = self.get_schema(db_id)
        schema_graph = build_schema_graph(schema)

        # 3. LEXICAL RELEVANCE GUARDRAIL (With Plural Fix!)
        if not is_relevant_to_db(question, schema_graph):
            return {"sql": "", "error": "❌ Question is completely out of domain for the selected database."}

        # 4. INITIAL GENERATION
        sql = self.generate_sql(self.build_prompt(question_context, schema))

        # ==========================================
        # STRONGER CONSTRAINT (Task 3 Logic)
        # ==========================================
        if self.use_constrained_decoding:
            filtered_sql = apply_schema_constraints(sql, schema_graph)

            if filtered_sql is None:
                constraint_prompt = f"""
Use ONLY valid schema.
Schema:
{schema}

Question:
{question_context}

SQL:
"""
                sql_retry = self.generate_sql(constraint_prompt)
                filtered_sql = apply_schema_constraints(sql_retry, schema_graph)

                if filtered_sql:
                    sql = filtered_sql
                else:
                    sql = sql_retry

        # 5. EXECUTION
        final_sql, cols, rows, error = self.execute_sql(question_context, sql, db_id)

        return {
            "question": question_context,
            "sql": final_sql,
            "columns": cols,
            "rows": rows,
            "error": error
        }


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