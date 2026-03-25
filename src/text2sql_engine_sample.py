

import sqlite3
import torch
import re
import hashlib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections import defaultdict

from src.schema_encoder import SchemaEncoder, build_schema_graph
from src.schema_utils import get_foreign_keys
from src.sql_validator import validate_sql_schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"

# ===============================
# 🔥 TASK 1 GLOBALS
# ===============================
EXECUTOR = ThreadPoolExecutor(max_workers=10)
CONNECTION_POOL = {}
QUERY_CACHE = {}

# ===============================
# 🔥 TASK 2 GLOBALS
# ===============================
ERROR_LOG = defaultdict(int)
ERROR_EXAMPLES = defaultdict(list)
SQL_OP_STATS = defaultdict(int)

# ===============================
# SECURITY GUARDRAILS (THE DML BLOCKER)
# ===============================
def is_safe_query(text):
    """Blocks destructive commands (DELETE, DROP, UPDATE, etc.)"""
    dml_keywords = r'\b(delete|update|insert|drop|alter|truncate)\b'
    if re.search(dml_keywords, text, re.IGNORECASE):
        return False
    return True

# ===============================
# INPUT FILTERS
# ===============================
def normalize_question(q):
    return re.sub(r"\s+", " ", q.strip())

def is_relevant_to_db(question, schema_graph):
    words = set(re.findall(r"\b[a-z]{3,}\b", question.lower()))
    
    # 🔥 FIX: Subtract command words FIRST so "list cats" actually blocks "cats"
    stop_words = {"show", "list", "all", "and", "the", "get", "find", "how", "many", "what", "where", "which", "who", "give", "display", "count", "from", "for", "with", "that", "have", "has", "are", "there", "distinct", "top", "limit", "sample"}
    words = words - stop_words

    if not words:
        return True

    schema_words = set()
    for t, cols in schema_graph.items():
        schema_words.update(t.lower().split())
        for c in cols:
            schema_words.update(c.lower().split())

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

    for w in words:
        w_singular = w[:-1] if w.endswith('s') else w
        if w in extended_schema_words or w_singular in extended_schema_words:
            return True

    return False

# ===============================
# LIMIT & HALLUCINATION FIX
# ===============================
def semantic_fix(question, sql):
    q = question.lower().strip()
    s = sql.lower()
    
    num_match = re.search(r'\b(?:show|list|top|limit|get|first|last|sample)\s+(\d+)\b', q)
    if num_match:
        limit_val = num_match.group(1)
        bad_count_pattern = rf"(?i)\s*(?:where|having|and)?\s*count\s*\(\s*\*\s*\)\s*=\s*{limit_val}"
        sql = re.sub(bad_count_pattern, "", sql)
        
        sql = re.sub(r"(?i)\s+where\s*$", "", sql)
        sql = re.sub(r"(?i)\s+having\s*$", "", sql)
        sql = re.sub(r"(?i)\s+group by\s*$", "", sql)
        
        if "limit" not in sql.lower():
            sql = sql.rstrip(";")
            sql = f"{sql.strip()} LIMIT {limit_val}"
            
    return sql

# ===============================
# FK FIX
# ===============================
def fix_joins_with_fk(sql, db_path):
    fks = get_foreign_keys(db_path)
    sql_lower = sql.lower()

    for table, col, ref_table, ref_col in fks:
        if table in sql_lower and ref_table in sql_lower:
            correct = f"{table}.{col} = {ref_table}.{ref_col}"
            if " on " in sql_lower and "=" not in sql_lower:
                sql = re.sub(r"on\s+[a-zA-Z0-9_.\s]+", f"ON {correct}", sql, flags=re.IGNORECASE)

    return sql

# ===============================
# TASK 1: CONNECTION POOL
# ===============================
def get_connection(db_path):
    if db_path not in CONNECTION_POOL:
        CONNECTION_POOL[db_path] = sqlite3.connect(db_path, check_same_thread=False)
    return CONNECTION_POOL[db_path]

# ===============================
# TASK 1: CACHE
# ===============================
def get_query_hash(sql, db_path):
    return hashlib.md5((sql + db_path).encode()).hexdigest()

# ===============================
# TASK 1: EXECUTION
# ===============================
def execute_sql(sql, db_path, timeout=2):

    # 🔥 Secondary Security Check right before execution
    if not is_safe_query(sql):
        raise Exception("Security Alert: Blocked Data Modification Query")

    key = get_query_hash(sql, db_path)

    if key in QUERY_CACHE:
        return QUERY_CACHE[key]

    def run():
        conn = get_connection(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return rows, cols

    future = EXECUTOR.submit(run)

    try:
        result = future.result(timeout=timeout)
        QUERY_CACHE[key] = result
        return result
    except TimeoutError:
        return [], []
    except Exception as e:
        raise e

# ===============================
# TASK 2: ERROR CLASSIFIER
# ===============================
def classify_error(sql, err):
    err = str(err).lower()
    sql = sql.lower()

    if "no such column" in err:
        return "wrong_column"
    if "no such table" in err:
        return "wrong_table"
    if "syntax error" in err:
        return "syntax_error"
    if "join" in sql and " on " not in sql:
        return "missing_join"
    if "where" in sql and "=" not in sql:
        return "wrong_where"

    return "other"

def get_hint(err_type):
    return {
        "wrong_column": "Check column names in schema.",
        "wrong_table": "Verify table names.",
        "syntax_error": "Fix SQL syntax.",
        "missing_join": "Add JOIN condition.",
        "wrong_where": "Fix WHERE clause.",
        "other": "Review SQL."
    }.get(err_type, "Review SQL.")

# ===============================
# TASK 2: SQL OPS
# ===============================
def track_sql_ops(sql):
    s = sql.lower()

    if "select" in s: SQL_OP_STATS["SELECT"] += 1
    if "where" in s: SQL_OP_STATS["WHERE"] += 1
    if "join" in s: SQL_OP_STATS["JOIN"] += 1
    if "group by" in s: SQL_OP_STATS["GROUP_BY"] += 1
    if "order by" in s: SQL_OP_STATS["ORDER_BY"] += 1

# ===============================
# ENGINE
# ===============================
class Text2SQLEngine:

    def __init__(self, adapter_path, base_model_name="Salesforce/codet5-base", use_lora=True):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        self.model = PeftModel.from_pretrained(base, adapter_path).to(self.device)
        self.model = self.model.merge_and_unload()
        self.model.eval()

        self.schema_encoder = SchemaEncoder(DB_ROOT)

    def get_schema(self, db_id):
        return self.schema_encoder.structured_schema(db_id)

    def build_prompt(self, question, schema):
        return f"""
Generate SQL query.
Use only given schema.
IMPORTANT: If the question asks to show a specific number of items (e.g. "Show 5"), use the LIMIT clause (e.g. LIMIT 5). Do NOT use HAVING count(*) or WHERE count(*).

Schema:
{schema}
Question:
{question}
SQL:
"""

    def generate_sql(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=8,
            repetition_penalty=1.2
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_sql(decoded)

    def clean_sql(self, text):
        match = re.search(r"(select|with)[\s\S]*", text, re.IGNORECASE)
        return match.group(0).strip() if match else ""

    def ask(self, question, db_id):

        # 🔥 Primary Security Check on the User Prompt
        if not is_safe_query(question):
            return {"sql": "-- BLOCKED", "columns": [], "rows": [], "error": "❌ Security Alert: Malicious prompt detected."}

        question = normalize_question(question)
        schema = self.get_schema(db_id)

        schema_graph = build_schema_graph(schema)

        if not is_relevant_to_db(question, schema_graph):
            return {"sql": "", "columns": [], "rows": [], "error": "Out of domain"}

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            db_path = DB_ROOT / f"{db_id}.sqlite"
        if not db_path.exists():
            db_path = PROJECT_ROOT / "final_databases" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            db_path = PROJECT_ROOT / "final_databases" / f"{db_id}.sqlite"

        sql = self.generate_sql(self.build_prompt(question, schema))
        sql = fix_joins_with_fk(sql, str(db_path))
        sql = semantic_fix(question, sql)

        track_sql_ops(sql)

        try:
            valid, _ = validate_sql_schema(sql, str(db_path))
        except:
            valid = False

        if not valid:
            return {"sql": sql, "columns": [], "rows": [], "error": "Invalid schema"}

        try:
            rows, cols = execute_sql(sql, str(db_path))

            return {
                "sql": sql,
                "columns": cols,
                "rows": rows,
                "error": None
            }

        except Exception as e:
            # Catch the specific security alert so it displays properly
            if "Security Alert" in str(e):
                return {"sql": "-- BLOCKED", "columns": [], "rows": [], "error": str(e)}

            err_type = classify_error(sql, e)
            ERROR_LOG[err_type] += 1

            if len(ERROR_EXAMPLES[err_type]) < 3:
                ERROR_EXAMPLES[err_type].append(sql)

            hint = get_hint(err_type)

            return {
                "sql": sql,
                "columns": [],
                "rows": [],
                "error": f"{err_type} → {hint}"
            }

# ===============================
def get_engine(adapter_path, base_model_name="Salesforce/codet5-base", use_lora=True, use_constrained=True):
    return Text2SQLEngine(adapter_path, base_model_name, use_lora)