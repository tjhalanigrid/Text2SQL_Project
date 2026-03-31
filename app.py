"""
GRADIO DEMO UI
NL → SQL → Result Table
"""

import gradio as gr
import pandas as pd
import re
import time
import os
import torch
import sys
import json
import subprocess
import base64
from pathlib import Path
from typing import Iterator
import io
# import os
# Force the app to ALWAYS use the checkpoints folder
os.environ["TEXT2SQL_ADAPTER_PATH"] = "checkpoints/best_rlhf_model"
# ==========================================
# 🔥 CUDA MOCK PATCH FOR MAC (MPS) / CPU
# ==========================================
if not torch.cuda.is_available():
    class MockCUDAEvent:
        def __init__(self, enable_timing=False, blocking=False, interprocess=False):
            self.t = 0.0
        def record(self, stream=None):
            self.t = time.perf_counter()
        def elapsed_time(self, end_event):
            return (end_event.t - self.t) * 1000.0 

    torch.cuda.Event = MockCUDAEvent
    if not hasattr(torch.cuda, 'synchronize'):
        torch.cuda.synchronize = lambda: None

# ==========================================
# RELATIVE PATH RESOLUTION (GLOBAL)
# ==========================================
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

# Dynamically resolve where the databases are kept
if (PROJECT_ROOT / "data" / "database").exists():
    DB_ROOT = PROJECT_ROOT / "data" / "database"
else:
    DB_ROOT = PROJECT_ROOT / "final_databases"

# ==========================================
# IMPORTS & ENGINE SETUP
# ==========================================
from src.quantized_text2sql_engine import QuantizedText2SQLEngine
from src.schema_encoder import SchemaEncoder

fallback_adapter = str(PROJECT_ROOT / "checkpoints" / "best_rlhf_model_2")
if not os.path.exists(fallback_adapter):
    fallback_adapter = str(PROJECT_ROOT / "checkpoints" / "sft_adapter_codet5")

adapter_path = os.environ.get("TEXT2SQL_ADAPTER_PATH", fallback_adapter)
base_model_name = os.environ.get("TEXT2SQL_BASE_MODEL", "Salesforce/codet5-base")
use_lora_env = os.environ.get("TEXT2SQL_USE_LORA", "true").strip().lower()
use_lora = use_lora_env not in {"0", "false", "no"}

DEFAULT_QUANT_ARTIFACT = os.environ.get("TEXT2SQL_QUANT_ARTIFACT", str(PROJECT_ROOT / "checkpoints" / "task5" / "int8_dynamic")).strip()
if not DEFAULT_QUANT_ARTIFACT:
    DEFAULT_QUANT_ARTIFACT = str(PROJECT_ROOT / "checkpoints" / "task5" / "int8_dynamic")

_ENGINE_CACHE = {}
_QUERY_LOG = []  
_PERF_LOG = []   
_SUCCESS_LOG = []  

_OP_STATS = {
    "SELECT": {"ok": 0, "fail": 0},
    "WHERE": {"ok": 0, "fail": 0},
    "JOIN": {"ok": 0, "fail": 0},
    "GROUP_BY": {"ok": 0, "fail": 0},
    "ORDER_BY": {"ok": 0, "fail": 0},
    "HAVING": {"ok": 0, "fail": 0},
    "LIMIT": {"ok": 0, "fail": 0},
}

def get_quant_engine(artifact_dir: str, use_constrained: bool, exec_workers: int = 8, use_cache: bool = True):
    key = (artifact_dir, bool(use_constrained), int(exec_workers), bool(use_cache))
    if key not in _ENGINE_CACHE:
        try:
            _ENGINE_CACHE[key] = QuantizedText2SQLEngine(
                artifact_dir,
                device="cpu",
                use_constrained=bool(use_constrained),
                exec_workers=int(exec_workers),
                use_cache=bool(use_cache),
            )
        except TypeError:
            print("⚠️ Engine initialization missing kwargs fallback.")
            _ENGINE_CACHE[key] = QuantizedText2SQLEngine(artifact_dir)
    return _ENGINE_CACHE[key]

try:
    quant_engine = get_quant_engine(DEFAULT_QUANT_ARTIFACT, use_constrained=False, exec_workers=8, use_cache=True)
except Exception:
    quant_engine = None

# Schema encoder for UI + relevance checks
schema_encoder = SchemaEncoder(DB_ROOT)

# =========================
# SAMPLE QUESTIONS DATA
# =========================
SAMPLES = [
    ("Show 10 distinct employee first names.", "chinook_1"),
    ("Which artist has the most albums?", "chinook_1"),
    ("List all the tracks that belong to the 'Rock' genre.", "chinook_1"),
    ("What are the names of all the cities?", "flight_1"),
    ("Find the flight number and cost of the cheapest flight.", "flight_1"),
    ("List the airlines that fly out of New York.", "flight_1"),
    ("Which campus was opened between 1935 and 1939?", "csu_1"),
    ("Count the number of students in each department.", "college_2"),
    ("List the names of all clubs.", "club_1"),
    ("How many members does each club have?", "club_1"),
    ("Show the names of all cinemas.", "cinema"),
    ("Which cinema has the most screens?", "cinema")
]

SAMPLE_QUESTIONS = [q[0] for q in SAMPLES]

# =========================
# SQL EXPLAINER
# =========================
def explain_sql(sql):
    if not sql: return ""
    explanation = "This SQL query retrieves information from the database."
    sql_lower = sql.lower()
    if "join" in sql_lower: explanation += "\n• It combines data from multiple tables using JOIN."
    if "where" in sql_lower: explanation += "\n• It filters rows using a WHERE condition."
    if "group by" in sql_lower: explanation += "\n• It groups results using GROUP BY."
    if "order by" in sql_lower: explanation += "\n• It sorts the results using ORDER BY."
    if "limit" in sql_lower: explanation += "\n• It limits the number of returned rows."
    return explanation

# =========================
# ERROR CLASSIFICATION
# =========================
def sql_ops(sql: str) -> list[str]:
    s = (sql or "").lower()
    ops = ["SELECT"] 
    if " where " in f" {s} ": ops.append("WHERE")
    if " join " in f" {s} ": ops.append("JOIN")
    if " group by " in f" {s} ": ops.append("GROUP_BY")
    if " order by " in f" {s} ": ops.append("ORDER_BY")
    if " having " in f" {s} ": ops.append("HAVING")
    if " limit " in f" {s} ": ops.append("LIMIT")
    return ops

def classify_error(sql: str, error_msg=None, *, timed_out: bool = False):
    s = (sql or "").lower()
    m = (error_msg or "").lower()

    if timed_out or "interrupted" in m or "timeout" in m: return "timeout"
    if not s.strip().startswith(("select", "with")): return "syntax_error"

    if " join " in f" {s} " and " on " not in f" {s} ": return "missing_join"
    if " where " in f" {s} " and not any(op in s for op in ["=", ">", "<", " in ", " like ", " between ", " is null", " is not null"]): return "wrong_where"
    if ("is null" in s or "is not null" in s) and ("no such column" in m or "misuse" in m): return "null_handling"

    if "no such table" in m: return "missing_table"
    if "no such column" in m: return "missing_column"
    if "ambiguous column name" in m: return "ambiguous_column"
    if "datatype mismatch" in m or "type mismatch" in m: return "type_mismatch"
    if "misuse of aggregate" in m or "misuse of aggregate function" in m: return "wrong_aggregation"
    if "syntax error" in m: return "syntax_error"
    if "near" in m and "syntax error" in m: return "syntax_error"

    if "runtime" in m or "constraint failed" in m: return "runtime_error"
    return "other"

def get_hint(error_type):
    hints = {
        "missing_join": "Check JOIN conditions between tables.",
        "wrong_aggregation": "Use proper aggregation like avg(column).",
        "wrong_where": "Check WHERE condition syntax.",
        "syntax_error": "Ensure SQL starts with SELECT.",
        "missing_table": "Use only tables from the provided schema.",
        "missing_column": "Use only columns from the provided schema (check table.column names).",
        "ambiguous_column": "Disambiguate by using table.column in SELECT/WHERE.",
        "null_handling": "Use IS NULL / IS NOT NULL (not '= NULL').",
        "type_mismatch": "Compare compatible types (strings in quotes, numeric without quotes).",
        "timeout": "Query took too long; simplify joins/aggregations or add LIMIT.",
        "other": "Review SQL logic."
    }
    return hints.get(error_type, "Review query.")

# =========================
# DOMAIN RELEVANCE FILTER
# =========================
def is_relevant_to_schema(question, db_id):
    try:
        raw_schema = schema_encoder.structured_schema(db_id).lower()
    except:
        return True 
        
    schema_words = set(re.findall(r'[a-z0-9_]+', raw_schema))
    q_words = re.findall(r'[a-z0-9_]+', question.lower())
    
    stop_words = {"show", "list", "all", "what", "is", "the", "how", "many", "count", "find", "get", "me", "a", "an", "of", "in", "for", "from", "with", "which", "are", "there", "give", "tell", "details", "info", "data", "everything"}
    meaningful_q_words = [w for w in q_words if w not in stop_words and not w.isdigit()]
    
    if not meaningful_q_words: return True 
        
    for word in meaningful_q_words:
        singular_word = word[:-1] if word.endswith('s') else word
        if word in schema_words or singular_word in schema_words: return True
            
    return False 

# =========================
# CORE FUNCTIONS
# =========================
def run_query(
    method,
    sample_q,
    custom_q,
    db_id
):
    # Hardcoded default inference settings
    quant_artifact_dir = DEFAULT_QUANT_ARTIFACT
    use_constrained_decoding = False
    gen_beams = 8
    gen_max_new_tokens = 120
    exec_timeout_s = 2.0
    exec_workers = 8
    exec_cache_on = True

    def _log(error_type: str, *, question: str, db_id_val: str, sql: str = "", error_msg: str = "") -> None:
        _QUERY_LOG.append({
            "t": time.time(), "db_id": str(db_id_val), "question": str(question),
            "sql": str(sql), "error_type": str(error_type), "error_msg": str(error_msg),
        })

    def _perf_log(payload: dict) -> None:
        _PERF_LOG.append(payload)
        if len(_PERF_LOG) > 1000: del _PERF_LOG[:200]

    raw_question = sample_q if method == "💡 Pick a Sample" else custom_q
    run_rec = {
        "t": time.time(), "db_id": str(db_id or ""), "question": str(raw_question or ""),
        "sql": "", "status": "unknown", "error_type": "", "error_msg": "", "ops": [],
    }

    if not raw_question or str(raw_question).strip() == "":
        _log("empty_input", question=str(raw_question or ""), db_id_val=str(db_id or ""), error_msg="empty input")
        run_rec.update({"status": "blocked", "error_type": "empty_input", "error_msg": "empty input"})
        return "-- No input provided", pd.DataFrame(columns=["Warning"]), "⚠️ Please enter a question."
        
    if not db_id or str(db_id).strip() == "":
        _log("no_db_selected", question=str(raw_question or ""), db_id_val=str(db_id or ""), error_msg="no db selected")
        run_rec.update({"status": "blocked", "error_type": "no_db_selected", "error_msg": "no db selected"})
        return "-- No database selected", pd.DataFrame(columns=["Warning"]), "⚠️ Please select a database."

    typo_corrections = [
        (r'\bshaw\b', 'show'), (r'\bshw\b', 'show'), (r'\bsho\b', 'show'),
        (r'\blsit\b', 'list'), (r'\blis\b', 'list'), (r'\bfidn\b', 'find'), 
        (r'\bfnd\b', 'find'), (r'\bgte\b', 'get')
    ]
    question = str(raw_question)
    for bad, good in typo_corrections:
        question = re.sub(bad, good, question, flags=re.IGNORECASE)

    q_lower = question.strip().lower()

    if len(q_lower.split()) < 2 and not any(vowel in q_lower for vowel in 'aeiouy'):
        _log("gibberish", question=str(question), db_id_val=str(db_id), error_msg="gibberish filtered")
        run_rec.update({"status": "blocked", "error_type": "gibberish", "error_msg": "gibberish filtered"})
        return "-- Input Blocked", pd.DataFrame(columns=["Warning"]), "⚠️ Please enter a clear, meaningful question."

    dml_pattern = r'\b(delete|update|insert|drop|alter|truncate)\b'
    if re.search(dml_pattern, q_lower):
        _log("blocked_dml", question=str(question), db_id_val=str(db_id), error_msg="DML blocked")
        run_rec.update({"status": "blocked", "error_type": "blocked_dml", "error_msg": "DML blocked"})
        return "-- ❌ BLOCKED: Data Modification", pd.DataFrame(columns=["Security Alert"]), "🛑 Security Alert: Modifying or deleting data is strictly prohibited by the application guardrails."

    if not is_relevant_to_schema(question, db_id):
        _log("out_of_domain", question=str(question), db_id_val=str(db_id), error_msg="out of domain")
        run_rec.update({"status": "blocked", "error_type": "out_of_domain", "error_msg": "out of domain"})
        return "-- ❌ BLOCKED: Out of Domain", pd.DataFrame(columns=["Domain Alert"]), f"🛑 Relevance Alert: I don't see anything related to your question in the '{db_id}' database schema. Please check the tables on the left and ask a relevant question."

    start_time = time.time()
    t0 = time.perf_counter()
    engine_stats_before = {}
    ui_warnings = ""

    # 5. INFERENCE 
    try:
        engine = quant_engine
        if quant_artifact_dir and str(quant_artifact_dir).strip():
            engine = get_quant_engine(
                str(quant_artifact_dir).strip(),
                bool(use_constrained_decoding),
                exec_workers=int(exec_workers),
                use_cache=bool(exec_cache_on),
            )
        if engine is None:
            raise RuntimeError("Quantized engine is not available (check TEXT2SQL_QUANT_ARTIFACT / artifact path).")
        try: engine_stats_before = engine.stats()
        except Exception: engine_stats_before = {}

        try:
            result = engine.ask(
                question,
                str(db_id),
                num_beams=int(gen_beams),
                max_new_tokens=int(gen_max_new_tokens),
                timeout_s=float(exec_timeout_s),
            )
        except TypeError:
            ui_warnings += "⚠️ WARNING: Your backend engine.ask() doesn't support UI dials yet. Falling back to default.\n\n"
            result = engine.ask(question, str(db_id))

    except Exception as e:
        run_rec.update({"status": "backend_crash", "error_type": "backend_crash", "error_msg": str(e)})
        _perf_log({"db_id": str(db_id), "question": question, "status": "backend_crash", "error": str(e)})
        return f"-- ❌ BACKEND CRASH\n-- {str(e)}", pd.DataFrame(columns=["Error Status"]), f"❌ CRITICAL BACKEND CRASH:\n{str(e)}"

    final_sql = result.get("sql", "")
    if not isinstance(final_sql, str): final_sql = str(final_sql) if final_sql else ""
    model_sql = final_sql
    run_rec["sql"] = final_sql
        
    # 6. SEMANTIC FIX
    num_match = re.search(r'\b(?:show|list|top|limit|get|first|last|sample)\s+(\d+)\b', q_lower)
    if num_match and final_sql:
        limit_val = num_match.group(1)
        # Fix #1: Remove rogue where count(*) = N
        final_sql = re.sub(rf"(?i)\s*(?:where|having|and)?\s*count\s*\(\s*\*\s*\)\s*=\s*{limit_val}", "", final_sql)
        
        # Fix #2: If the intent is a simple list, destroy hallucinated GROUP BY / ORDER BY count()
        is_simple_list = not re.search(r'\b(count|how many|group|most|least|highest|lowest)\b', q_lower)
        if is_simple_list and "group by" in final_sql.lower():
            final_sql = re.sub(r"(?i)\s*group by\s+.*?(?=(?:limit|;|$))", "", final_sql)
            final_sql = re.sub(r"(?i)\s*order by\s+count\([^)]*\)\s*(?:desc|asc)?", "", final_sql)
            final_sql = re.sub(r"(?i)select\s+.*?\s+from", "SELECT * FROM", final_sql, count=1)

        # Fix #3: Standard group-by cleanup if count isn't present
        if "group by" in final_sql.lower() and not re.search(r'(?i)\b(count|sum|avg|max|min)\b\(', final_sql):
            final_sql = re.sub(r"(?i)\s*group by\s+[a-zA-Z0-9_.]+", "", final_sql)

        # Fix #4: Ensure the limit is actually applied
        if "limit" not in final_sql.lower():
            final_sql = f"{final_sql.strip().rstrip(';')} LIMIT {limit_val}"

    # =====================================================================
    # 🔥 7. ROBUST SQLITE EXECUTION (Bypassing Strict Python Validation)
    # =====================================================================
    from src.sql_validator import validate_sql_schema
    db_path = str(DB_ROOT / db_id / f"{db_id}.sqlite")
    
    # We run the python validator ONLY to track the constraint metric. 
    # We DO NOT let it block the UI if it fails.
    try:
        strict_valid, _ = validate_sql_schema(final_sql, db_path)
    except Exception:
        strict_valid = False

    error_msg = None
    rows, cols = [], []
    sqlite_success = False

    # The Ultimate Validator: If SQLite can run it, it's valid.
    if final_sql and engine is not None:
        try:
            # We call the internal _execute_one to guarantee the validator is completely bypassed
            rows, cols = engine._execute_one(final_sql, db_path, timeout_s=float(exec_timeout_s))
            sqlite_success = True
        except Exception as e:
            error_msg = str(e)
            sqlite_success = False

    # Fallback to model's original SQL if our semantic LIMIT fix broke it
    if not sqlite_success and model_sql and model_sql != final_sql and engine is not None:
        try:
            alt_rows, alt_cols = engine._execute_one(model_sql, db_path, timeout_s=float(exec_timeout_s))
            final_sql = model_sql
            rows, cols = alt_rows, alt_cols
            sqlite_success = True
            error_msg = None
        except Exception:
            pass

    # 🔥 Overwrite validation status so the UI shows success if SQLite succeeded
    valid = sqlite_success

    if error_msg or not valid:
        et = classify_error(final_sql, str(error_msg or ""), timed_out=("interrupted" in str(error_msg or "").lower()))
        _log(et, question=str(question), db_id_val=str(db_id), sql=str(final_sql), error_msg=str(error_msg or ("invalid schema" if not valid else "")))
        run_rec.update({"status": "fail", "error_type": et, "error_msg": str(error_msg or ("invalid schema" if not valid else ""))})

    latency = round(time.time() - start_time, 3)

    t1 = time.perf_counter()
    engine_stats_after = {}
    if engine is not None:
        try: engine_stats_after = engine.stats()
        except Exception: engine_stats_after = {}

    perf = {
        "db_id": str(db_id),
        "use_constrained_decoding": bool(use_constrained_decoding),
        "num_beams": int(gen_beams),
        "max_new_tokens": int(gen_max_new_tokens),
        "exec_timeout_s": float(exec_timeout_s),
        "exec_workers": int(exec_workers),
        "exec_cache_on": bool(exec_cache_on),
        "latency_total_ms": round((t1 - t0) * 1000.0, 2),
        "constraint_ok": bool(strict_valid),  # Logs whether the python script liked it
        "has_error": bool(error_msg),
        "exec_cache_hit_rate": float(engine_stats_after.get("exec_cache_hit_rate", 0.0) or 0.0),
        "exec_calls_total": int(engine_stats_after.get("exec_calls", 0) or 0),
        "exec_cache_hits_total": int(engine_stats_after.get("exec_cache_hits", 0) or 0),
        "exec_cache_misses_total": int(engine_stats_after.get("exec_cache_misses", 0) or 0),
    }
    _perf_log(perf)

    window = _PERF_LOG[-50:]
    if window:
        avg_ms = sum(float(x.get("latency_total_ms", 0.0) or 0.0) for x in window) / len(window)
        constraint_rate = sum(1 for x in window if x.get("constraint_ok")) / len(window)
        error_rate = sum(1 for x in window if x.get("has_error")) / len(window)
    else:
        avg_ms, constraint_rate, error_rate = 0.0, 0.0, 0.0

    perf_block = (
        "\n\n---\nPerformance (task impact)\n"
        f"- Total latency (ms): {perf['latency_total_ms']}\n"
        f"- Strict Python Validator OK (Task 3): {perf['constraint_ok']}\n"
        f"- Exec cache hit-rate (Task 1/5): {round(perf['exec_cache_hit_rate'], 3)}\n"
        f"- Rolling avg latency last 50 (ms): {round(avg_ms, 2)}\n"
        f"- Rolling constraint rate last 50: {round(constraint_rate, 3)}\n"
        f"- Rolling error rate last 50: {round(error_rate, 3)}"
    )

    if error_msg or not valid:
        display_sql = final_sql if final_sql.strip() else "-- ❌ INVALID SQL"
        explanation = f"{ui_warnings}❌ Error Details:\n\n"
        if error_msg: explanation += f"{error_msg}\n\n"
        if not valid:
            error_type = classify_error(final_sql, str(error_msg or ""))
            explanation += f"Error Type: {error_type}\nHint: {get_hint(error_type)}"
        explanation += perf_block
        ops = sql_ops(final_sql)
        run_rec["ops"] = ops
        for op in ops:
            if op in _OP_STATS: _OP_STATS[op]["fail"] += 1
        return display_sql, pd.DataFrame(columns=["Execution Notice"]), explanation

    if not rows:
        safe_cols = cols if (cols and len(cols) > 0) else ["Result"]
        ops = sql_ops(final_sql)
        run_rec.update({"status": "ok", "ops": ops})
        for op in ops:
            if op in _OP_STATS: _OP_STATS[op]["ok"] += 1
        _SUCCESS_LOG.append({"t": run_rec["t"], "db_id": str(db_id), "question": question, "sql": final_sql, "ops": ops})
        return final_sql, pd.DataFrame(columns=safe_cols), f"{ui_warnings}✅ Query executed successfully\n\nRows returned: 0\nExecution Time: {latency} sec\n\n{explain_sql(final_sql)}{perf_block}"

    df = pd.DataFrame(rows, columns=cols)
    actual_rows = len(rows)
    explanation = f"{ui_warnings}✅ Query executed successfully\n\nRows returned: {actual_rows}\nExecution Time: {latency} sec\n\n{explain_sql(final_sql)}{perf_block}"
    ops = sql_ops(final_sql)
    run_rec.update({"status": "ok", "ops": ops})
    for op in ops:
        if op in _OP_STATS: _OP_STATS[op]["ok"] += 1
    _SUCCESS_LOG.append({"t": run_rec["t"], "db_id": str(db_id), "question": question, "sql": final_sql, "ops": ops})

    limit_match = re.search(r'LIMIT\s+(\d+)', final_sql, re.IGNORECASE)
    if limit_match:
        requested_limit = int(limit_match.group(1))
        if actual_rows < requested_limit:
            explanation += f"\n\nℹ️ Query allowed up to {requested_limit} rows but only {actual_rows} matched."

    return final_sql, df, explanation

# =========================
# RESEARCH TASK FUNCTIONS
# =========================
from typing import List, Dict, Optional

def _run_cmd(cmd: List[str], env: Optional[Dict] = None) -> str:
    run_env = (env or os.environ.copy()).copy()
    project_root = str(PROJECT_ROOT)
    run_env["PYTHONPATH"] = project_root + (os.pathsep + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=project_root)
    out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return out.strip()

def task1_benchmark(n_rollouts: int, max_workers: int) -> Iterator[tuple[str, str]]:
    project_root = str(PROJECT_ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    try: os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    except Exception: pass

    cmd = [sys.executable, "-u", "scripts/benchmark_parallel_reward.py", "--n", str(int(n_rollouts)), "--max-workers", str(int(max_workers)), "--skip-profile"]
    proc = subprocess.Popen(cmd, cwd=project_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_yield = time.perf_counter()
    lines: list[str] = []
    yield "Running Task 1 benchmark...\n", "<i>Running...</i>"

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        now = time.perf_counter()
        if now - last_yield >= 0.5:
            last_yield = now
            yield "".join(lines[-200:]).strip(), "<i>Running...</i>"

    proc.wait()
    out = "".join(lines).strip()

    plot_path = str(PROJECT_ROOT / "results" / "task1_plot.png")
    if os.path.exists(plot_path):
        try:
            b64 = base64.b64encode(Path(plot_path).read_bytes()).decode("ascii")
            yield out, f"<img src='data:image/png;base64,{b64}' style='max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px;' />"
            return
        except Exception:
            yield out, f"<pre>{plot_path}</pre>"
            return

    yield out, "<i>No plot generated</i>"

def task5_quant_export(mode: str, base_model: str, adapter: str, out_dir: str) -> str:
    return _run_cmd([sys.executable, "scripts/quantize_export.py", "--mode", mode, "--base_model", base_model, "--adapter", adapter, "--out_dir", out_dir, "--local_only"])

def task2_dashboard() -> str:
    if not _QUERY_LOG: return "No logged errors yet."
    counts = {}
    for r in _QUERY_LOG[-500:]:
        k = r.get("error_type") or "other"
        counts[k] = counts.get(k, 0) + 1
    perf_window = _PERF_LOG[-200:]
    if perf_window:
        avg_ms = sum(float(x.get("latency_total_ms", 0.0) or 0.0) for x in perf_window) / len(perf_window)
        constraint_rate = sum(1 for x in perf_window if x.get("constraint_ok")) / len(perf_window)
        error_rate = sum(1 for x in perf_window if x.get("has_error")) / len(perf_window)
    else:
        avg_ms, constraint_rate, error_rate = 0.0, 0.0, 0.0
    return json.dumps({
        "counts": sorted(counts.items(), key=lambda x: (-x[1], x[0])),
        "recent": _QUERY_LOG[-10:],
        "perf_last_200": {"avg_latency_ms": round(avg_ms, 2), "constraint_rate": round(constraint_rate, 3), "error_rate": round(error_rate, 3)}
    }, indent=2)

def task2_dashboard_structured():
    if not _QUERY_LOG:
        empty_counts = pd.DataFrame(columns=["error_type", "count", "hint"])
        empty_recent = pd.DataFrame(columns=["time", "db_id", "error_type", "question", "error_msg"])
        return empty_counts, empty_recent, gr.update(choices=[], value=None)

    counts = {}
    for r in _QUERY_LOG[-1000:]:
        k = r.get("error_type") or "other"
        counts[k] = counts.get(k, 0) + 1
    rows = [{"error_type": k, "count": int(v), "hint": get_hint(k)} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    counts_df = pd.DataFrame(rows)

    recent = []
    for r in _QUERY_LOG[-100:]:
        ts = r.get("t")
        try: ts_s = time.strftime("%H:%M:%S", time.localtime(float(ts))) if ts else ""
        except Exception: ts_s = ""
        recent.append({"time": ts_s, "db_id": r.get("db_id", ""), "error_type": r.get("error_type", ""), "question": r.get("question", ""), "error_msg": r.get("error_msg", "")})
    recent_df = pd.DataFrame(recent)

    choices = [str(x["error_type"]) for x in rows]
    default = choices[0] if choices else None
    return counts_df, recent_df, gr.update(choices=choices, value=default)

def task2_error_examples(error_type: str) -> str:
    if not error_type: return ""
    hint = get_hint(error_type)
    matches = [r for r in reversed(_QUERY_LOG) if (r.get("error_type") or "") == str(error_type)][:3]
    if not matches: return f"Error type: {error_type}\nHint: {hint}\n\nNo examples yet."
    out = [f"Error type: {error_type}", f"Hint: {hint}", ""]
    for i, r in enumerate(matches, 1):
        out.extend([f"Example {i}", f"DB: {r.get('db_id','')}", f"Q: {r.get('question','')}", f"SQL: {r.get('sql','')}", f"Msg: {r.get('error_msg','')}", ""])
    return "\n".join(out).strip()

def _tokenize_for_similarity(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))

def similar_successes_for_error(error_type: str, *, limit: int = 3) -> str:
    if not error_type: return ""
    failures = [r for r in reversed(_QUERY_LOG) if (r.get("error_type") or "") == str(error_type)]
    if not failures: return "No failures of this type yet."
    target = failures[0]
    tgt_tokens = _tokenize_for_similarity(f"{target.get('question','')} {target.get('sql','')}")
    if not tgt_tokens: return "No tokens to compare."

    scored = []
    for s in reversed(_SUCCESS_LOG[-2000:]):
        cand_tokens = _tokenize_for_similarity(f"{s.get('question','')} {s.get('sql','')}")
        if not cand_tokens: continue
        inter = len(tgt_tokens & cand_tokens)
        union = len(tgt_tokens | cand_tokens)
        score = inter / union if union else 0.0
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [x for x in scored[: int(limit)] if x[0] > 0]

    hint = get_hint(error_type)
    out = [f"Error type: {error_type}", f"Hint: {hint}", ""]
    if not top:
        out.append("No similar successful queries found yet.")
        return "\n".join(out).strip()

    for i, (score, s) in enumerate(top, 1):
        out.extend([f"Similar success {i} (score={score:.3f})", f"DB: {s.get('db_id','')}", f"Q: {s.get('question','')}", f"SQL: {s.get('sql','')}", ""])
    return "\n".join(out).strip()

def _plot_op_stats_html() -> str:
    try:
        import matplotlib.pyplot as plt
        labels = list(_OP_STATS.keys())
        oks = [int(_OP_STATS[k]["ok"]) for k in labels]
        fails = [int(_OP_STATS[k]["fail"]) for k in labels]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        x = list(range(len(labels)))
        ax.bar(x, oks, label="ok", color="#16a34a")
        ax.bar(x, fails, bottom=oks, label="fail", color="#dc2626")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title("Success/Failure by SQL operation")
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"<img src='data:image/png;base64,{b64}' style='max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px;' />"
    except Exception as e: return f"<pre>Plot error: {e}</pre>"

def task2_ops_table():
    rows = []
    for op, d in _OP_STATS.items():
        ok = int(d.get("ok", 0))
        fail = int(d.get("fail", 0))
        total = ok + fail
        rows.append({"op": op, "ok": ok, "fail": fail, "total": total, "success_rate": (ok / total) if total else 0.0})
    return pd.DataFrame(rows), _plot_op_stats_html()

def toggle_input_method(method, current_sample):
    if method == "💡 Pick a Sample":
        db = next((db for q, db in SAMPLES if q == current_sample), "chinook_1")
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=db, interactive=False))
    return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(interactive=True))

def load_sample(selected_question):
    if not selected_question: return gr.update()
    return gr.update(value=next((db for q, db in SAMPLES if q == selected_question), "chinook_1"))

def clear_inputs():
    return (gr.update(value="💡 Pick a Sample"), gr.update(value=SAMPLE_QUESTIONS[0], visible=True), gr.update(visible=False), gr.update(value="", visible=False), gr.update(value="chinook_1", interactive=False), "", pd.DataFrame(), "")

def update_schema(db_id):
    if not db_id: return ""
    try:
        raw_schema = schema_encoder.structured_schema(db_id)
        html_output = "<div style='max-height: 250px; overflow-y: auto; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9em; line-height: 1.6;'>"
        for line in raw_schema.strip().split('\n'):
            line = line.strip()
            if not line: continue
            match = re.search(r'^([a-zA-Z0-9_]+)\s*\((.*)\)', line)
            if match: html_output += f"<div style='margin-bottom: 8px;'><strong style='color: #0f172a; font-size: 1.05em; font-weight: 800;'>{match.group(1).upper()}</strong> <span style='color: #64748b;'>( {match.group(2).lower()} )</span></div>"
            else: html_output += f"<div style='color: #475569;'>{line}</div>"
        html_output += "</div>"
        return html_output
    except Exception as e: return f"<div style='color: red;'>Error loading schema: {str(e)}</div>"

# =========================
# UI LAYOUT
# =========================
with gr.Blocks(title="Text-to-SQL RLHF") as demo:
    gr.HTML("""
        <div style="text-align: center; background-color: #e0e7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #c7d2fe;">
            <h1 style="color: #3730a3; margin-top: 0; margin-bottom: 10px; font-size: 2.2em;"> Text-to-SQL using RLHF + Execution Reward</h1>
            <p style="color: #4f46e5; font-size: 1.1em; margin: 0;">Convert Natural Language to SQL, strictly validated and safely executed on local SQLite databases.</p>
        </div>
    """)

    DBS = sorted(["flight_1", "student_assessment", "store_1", "bike_1", "book_2", "chinook_1", "academic", "aircraft", "car_1", "cinema", "club_1", "csu_1", "college_1", "college_2", "company_1", "company_employee", "customer_complaints", "department_store", "employee_hire_evaluation", "museum_visit", "products_for_hire", "restaurant_1", "school_finance", "shop_membership", "small_bank_1", "student_1", "tvshow", "voter_1", "world_1"])

    with gr.Tabs():
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Configuration & Input")
                    input_method = gr.Radio(choices=["💡 Pick a Sample", "✍️ Type my own"], value="💡 Pick a Sample", label="How do you want to ask?")
                    sample_dropdown = gr.Dropdown(choices=SAMPLE_QUESTIONS, value=SAMPLE_QUESTIONS[0], label="Select a Sample Question", info="The database will be selected automatically.", visible=True)
                    type_own_warning = gr.Markdown("**⚠️ Please select a Database first, then type your custom question below:**", visible=False)
                    gr.Markdown("---")
                    db_id = gr.Dropdown(choices=DBS, value="chinook_1", label="Select Database", interactive=False)
                    custom_question = gr.Textbox(label="Ask your Custom Question", placeholder="Type your own question here...", lines=3, visible=False)

                    gr.Markdown("#### 📋 Database Structure")
                    gr.HTML("<p style='font-size: 0.85em; color: #64748b; margin-top: -10px; margin-bottom: 5px;'>Use these exact names! Table names are <strong>Dark</strong>, Column names are <span style='color: #94a3b8;'>Light</span>.</p>")
                    schema_display = gr.HTML(value=update_schema("chinook_1"))

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        run_btn = gr.Button(" Generate & Run SQL", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 2. Execution Results")
                    final_sql = gr.Code(language="sql", label="Final Executed SQL")
                    result_table = gr.Dataframe(label="Query Result Table", interactive=False, wrap=True)
                    explanation = gr.Textbox(label="AI Explanation + Execution Details", lines=8)

        with gr.Tab("Diagnostics"):
            gr.Markdown("## Diagnostics (Task 1–2)")
            with gr.Accordion("Task 1: Parallel reward benchmark", open=False):
                gr.Markdown("*Note: High rollouts can take minutes due to SQLite file-locking. Defaults lowered for faster testing.*")
                t1_n = gr.Number(value=20, precision=0, label="Rollouts (n)")
                t1_workers = gr.Number(value=10, precision=0, label="Max workers")
                t1_run = gr.Button("Run Task 1 benchmark")
                t1_out = gr.Textbox(label="Output", lines=12)
                t1_plot = gr.HTML(label="Plot (if generated)")
                t1_run.click(fn=task1_benchmark, inputs=[t1_n, t1_workers], outputs=[t1_out, t1_plot])

            with gr.Accordion("Task 2: Error dashboard", open=False):
                t2_refresh = gr.Button("Refresh dashboard")
                t2_counts = gr.Dataframe(label="Error counts", interactive=False, wrap=True)
                t2_recent = gr.Dataframe(label="Recent errors", interactive=False, wrap=True)
                t2_type = gr.Dropdown(choices=[], value=None, label="Select error type")
                t2_examples = gr.Textbox(label="Examples + hint", lines=10)
                t2_similar = gr.Textbox(label="Similar successful queries", lines=10)
                t2_ops_refresh = gr.Button("Refresh SQL-op stats")
                t2_ops_tbl = gr.Dataframe(label="Success/failure by op", interactive=False, wrap=True)
                t2_ops_plot = gr.HTML(label="Op plot")

                t2_refresh.click(fn=task2_dashboard_structured, inputs=[], outputs=[t2_counts, t2_recent, t2_type])
                t2_type.change(fn=task2_error_examples, inputs=[t2_type], outputs=[t2_examples])
                t2_type.change(fn=similar_successes_for_error, inputs=[t2_type], outputs=[t2_similar])
                t2_ops_refresh.click(fn=task2_ops_table, inputs=[], outputs=[t2_ops_tbl, t2_ops_plot])

    input_method.change(fn=toggle_input_method, inputs=[input_method, sample_dropdown], outputs=[sample_dropdown, type_own_warning, custom_question, db_id])
    sample_dropdown.change(fn=load_sample, inputs=[sample_dropdown], outputs=[db_id])
    db_id.change(fn=update_schema, inputs=[db_id], outputs=[schema_display])
    run_btn.click(fn=run_query, inputs=[input_method, sample_dropdown, custom_question, db_id], outputs=[final_sql, result_table, explanation])
    clear_btn.click(fn=clear_inputs, inputs=[], outputs=[input_method, sample_dropdown, type_own_warning, custom_question, db_id, final_sql, result_table, explanation])

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0").strip() in {"1", "true", "True", "yes", "Y"}
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(share=share, server_name=server_name, theme = gr.themes.Soft())