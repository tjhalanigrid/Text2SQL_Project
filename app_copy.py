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
# IMPORTS & ENGINE SETUP
# ==========================================
from src.quantized_text2sql_engine import QuantizedText2SQLEngine
from src.schema_encoder import SchemaEncoder

fallback_adapter = "checkpoints/best_rlhf_model_2"
if not os.path.exists(fallback_adapter):
    fallback_adapter = "checkpoints/sft_adapter_codet5"

adapter_path = os.environ.get("TEXT2SQL_ADAPTER_PATH", fallback_adapter)
base_model_name = os.environ.get("TEXT2SQL_BASE_MODEL", "Salesforce/codet5-base")
use_lora_env = os.environ.get("TEXT2SQL_USE_LORA", "true").strip().lower()
use_lora = use_lora_env not in {"0", "false", "no"}

DEFAULT_QUANT_ARTIFACT = os.environ.get("TEXT2SQL_QUANT_ARTIFACT", "checkpoints/task5/int8_dynamic").strip()
if not DEFAULT_QUANT_ARTIFACT:
    DEFAULT_QUANT_ARTIFACT = "checkpoints/task5/int8_dynamic"

_ENGINE_CACHE = {}
_QUERY_LOG = []  # Task2 dashboard: recent queries/errors
_PERF_LOG = []   # Per-query perf metrics (Task impact)
_SUCCESS_LOG = []  # Successful queries for "similar successful queries"

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

# Schema encoder for UI + relevance checks (no HF model downloads)
PROJECT_ROOT = Path(__file__).resolve().parent
DB_ROOT = PROJECT_ROOT / "data" / "database"
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
    ops = ["SELECT"]  # everything in this app is expected to be SELECT-ish
    if " where " in f" {s} ":
        ops.append("WHERE")
    if " join " in f" {s} ":
        ops.append("JOIN")
    if " group by " in f" {s} ":
        ops.append("GROUP_BY")
    if " order by " in f" {s} ":
        ops.append("ORDER_BY")
    if " having " in f" {s} ":
        ops.append("HAVING")
    if " limit " in f" {s} ":
        ops.append("LIMIT")
    return ops


def classify_error(sql: str, error_msg: str | None = None, *, timed_out: bool = False):
    """
    Task-2 style categorization: uses BOTH the SQL text and SQLite runtime message.
    """
    s = (sql or "").lower()
    m = (error_msg or "").lower()

    if timed_out or "interrupted" in m or "timeout" in m:
        return "timeout"

    if not s.strip().startswith(("select", "with")):
        return "syntax_error"

    # Heuristics from SQL text
    if " join " in f" {s} " and " on " not in f" {s} ":
        return "missing_join"
    if " where " in f" {s} " and not any(op in s for op in ["=", ">", "<", " in ", " like ", " between ", " is null", " is not null"]):
        return "wrong_where"
    if ("is null" in s or "is not null" in s) and ("no such column" in m or "misuse" in m):
        return "null_handling"

    # SQLite runtime message parsing
    if "no such table" in m:
        return "missing_table"
    if "no such column" in m:
        return "missing_column"
    if "ambiguous column name" in m:
        return "ambiguous_column"
    if "datatype mismatch" in m or "type mismatch" in m:
        return "type_mismatch"
    if "misuse of aggregate" in m or "misuse of aggregate function" in m:
        return "wrong_aggregation"
    if "syntax error" in m:
        return "syntax_error"
    if "near" in m and "syntax error" in m:
        return "syntax_error"

    # Fallbacks
    if "runtime" in m or "constraint failed" in m:
        return "runtime_error"
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
    
    if not meaningful_q_words:
        return True 
        
    for word in meaningful_q_words:
        singular_word = word[:-1] if word.endswith('s') else word
        if word in schema_words or singular_word in schema_words:
            return True
            
    return False 

# =========================
# CORE FUNCTIONS
# =========================
def run_query(
    method,
    sample_q,
    custom_q,
    db_id,
    quant_artifact_dir,
    use_constrained_decoding,
    gen_beams,
    gen_max_new_tokens,
    exec_timeout_s,
    exec_workers,
    exec_cache_on,
):
    def _log(error_type: str, *, question: str, db_id_val: str, sql: str = "", error_msg: str = "") -> None:
        _QUERY_LOG.append(
            {
                "t": time.time(),
                "db_id": str(db_id_val),
                "question": str(question),
                "sql": str(sql),
                "error_type": str(error_type),
                "error_msg": str(error_msg),
            }
        )

    def _perf_log(payload: dict) -> None:
        _PERF_LOG.append(payload)
        if len(_PERF_LOG) > 1000:
            del _PERF_LOG[:200]

    raw_question = sample_q if method == "💡 Pick a Sample" else custom_q
    run_rec = {
        "t": time.time(),
        "db_id": str(db_id or ""),
        "question": str(raw_question or ""),
        "sql": "",
        "status": "unknown",
        "error_type": "",
        "error_msg": "",
        "ops": [],
    }

    # 1. EMPTY CHECK
    if not raw_question or str(raw_question).strip() == "":
        _log("empty_input", question=str(raw_question or ""), db_id_val=str(db_id or ""), error_msg="empty input")
        run_rec.update({"status": "blocked", "error_type": "empty_input", "error_msg": "empty input"})
        return "-- No input provided", pd.DataFrame(columns=["Warning"]), "⚠️ Please enter a question."
    if not db_id or str(db_id).strip() == "":
        _log("no_db_selected", question=str(raw_question or ""), db_id_val=str(db_id or ""), error_msg="no db selected")
        run_rec.update({"status": "blocked", "error_type": "no_db_selected", "error_msg": "no db selected"})
        return "-- No database selected", pd.DataFrame(columns=["Warning"]), "⚠️ Please select a database."

    # 🔥 AUTO-CORRECT PRE-PROCESSOR (The Typo Fixer)
    typo_corrections = [
        (r'\bshaw\b', 'show'), (r'\bshw\b', 'show'), (r'\bsho\b', 'show'),
        (r'\blsit\b', 'list'), (r'\blis\b', 'list'),
        (r'\bfidn\b', 'find'), (r'\bfnd\b', 'find'),
        (r'\bgte\b', 'get')
    ]
    question = str(raw_question)
    for bad, good in typo_corrections:
        question = re.sub(bad, good, question, flags=re.IGNORECASE)

    q_lower = question.strip().lower()

    # 2. GIBBERISH FILTER
    if len(q_lower.split()) < 2 and not any(vowel in q_lower for vowel in 'aeiouy'):
        _log("gibberish", question=str(question), db_id_val=str(db_id), error_msg="gibberish filtered")
        run_rec.update({"status": "blocked", "error_type": "gibberish", "error_msg": "gibberish filtered"})
        return "-- Input Blocked", pd.DataFrame(columns=["Warning"]), "⚠️ Please enter a clear, meaningful question."

    # 3. DML (DELETE/UPDATE) BLOCKER
    dml_pattern = r'\b(delete|update|insert|drop|alter|truncate)\b'
    if re.search(dml_pattern, q_lower):
        _log("blocked_dml", question=str(question), db_id_val=str(db_id), error_msg="DML blocked")
        run_rec.update({"status": "blocked", "error_type": "blocked_dml", "error_msg": "DML blocked"})
        return "-- ❌ BLOCKED: Data Modification", pd.DataFrame(columns=["Security Alert"]), "🛑 Security Alert: Modifying or deleting data is strictly prohibited by the application guardrails."

    # 4. OUT-OF-DOMAIN FILTER
    if not is_relevant_to_schema(question, db_id):
        _log("out_of_domain", question=str(question), db_id_val=str(db_id), error_msg="out of domain")
        run_rec.update({"status": "blocked", "error_type": "out_of_domain", "error_msg": "out of domain"})
        return "-- ❌ BLOCKED: Out of Domain", pd.DataFrame(columns=["Domain Alert"]), f"🛑 Relevance Alert: I don't see anything related to your question in the '{db_id}' database schema. Please check the tables on the left and ask a relevant question."

    start_time = time.time()
    t0 = time.perf_counter()
    engine_stats_before = {}

    # 5. INFERENCE (Quantized deployment engine with Fallbacks)
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
        try:
            engine_stats_before = engine.stats()
        except Exception:
            engine_stats_before = {}

        # Try with the new UI dials, fallback to default ask() if backend doesn't support kwargs
        try:
            result = engine.ask(
                question,
                str(db_id),
                num_beams=int(gen_beams),
                max_new_tokens=int(gen_max_new_tokens),
                timeout_s=float(exec_timeout_s),
            )
        except TypeError:
            print("⚠️ Engine ask() doesn't support UI dials yet. Falling back to default.")
            result = engine.ask(question, str(db_id))

    except Exception as e:
        run_rec.update({"status": "backend_crash", "error_type": "backend_crash", "error_msg": str(e)})
        _perf_log({"db_id": str(db_id), "question": question, "status": "backend_crash", "error": str(e)})
        return f"-- ❌ BACKEND CRASH\n-- {str(e)}", pd.DataFrame(columns=["Error Status"]), f"❌ CRITICAL BACKEND CRASH:\n{str(e)}"

    final_sql = result.get("sql", "")
    if not isinstance(final_sql, str): final_sql = str(final_sql) if final_sql else ""
    model_sql = final_sql  # keep original model output for fallback
    run_rec["sql"] = final_sql
        
    # 6. SEMANTIC FIX (The "Show 5" Bug + Invalid Group By fix)
    num_match = re.search(r'\b(?:show|list|top|limit|get|first|last|sample)\s+(\d+)\b', q_lower)
    if num_match and final_sql:
        limit_val = num_match.group(1)
        final_sql = re.sub(rf"(?i)\s*(?:where|having|and)?\s*count\s*\(\s*\*\s*\)\s*=\s*{limit_val}", "", final_sql)
        
        # 🔥 Erase unwarranted GROUP BY clauses
        if "group by" in final_sql.lower() and not re.search(r'(?i)\b(count|sum|avg|max|min)\b\(', final_sql):
            final_sql = re.sub(r"(?i)\s*group by\s+[a-zA-Z0-9_.]+", "", final_sql)
            final_sql = re.sub(r"(?i)select\s+[a-zA-Z0-9_.]+\s+from", "SELECT * FROM", final_sql)

        if "limit" not in final_sql.lower():
            final_sql = f"{final_sql.strip().rstrip(';')} LIMIT {limit_val}"

    error_msg = result.get("error", None)
    rows = result.get("rows", [])
    cols = result.get("columns", [])

    latency = round(time.time() - start_time, 3)

    # 7. SCHEMA VALIDATION
    from src.sql_validator import validate_sql_schema
    PROJECT_ROOT = Path(__file__).resolve().parent
    if (PROJECT_ROOT / "data" / "database").exists():
        db_path = str(PROJECT_ROOT / "data" / "database" / db_id / f"{db_id}.sqlite")
    else:
        db_path = str(PROJECT_ROOT / "final_databases" / db_id / f"{db_id}.sqlite")

    def _is_valid(sql_text: str) -> bool:
        try:
            ok, _ = validate_sql_schema(sql_text, db_path)
            return bool(ok)
        except Exception:
            return False

    valid = _is_valid(final_sql)
    model_valid = _is_valid(model_sql) if model_sql and model_sql != final_sql else valid

    # If we modified SQL (LIMIT fixes etc.), re-execute so the result table matches the final SQL.
    if not error_msg and valid and engine is not None:
        try:
            try:
                rows, cols = engine.execute_sql(final_sql, str(db_id), timeout_s=float(exec_timeout_s), validate_schema=False)
            except TypeError:
                print("⚠️ Engine execute_sql() doesn't support UI dials yet. Falling back to default.")
                rows, cols = engine.execute_sql(final_sql, str(db_id))
        except Exception as e:
            error_msg = str(e)
            rows, cols = [], []

    # Fallback: if our semantic fixes made things worse, prefer the model's original SQL.
    if (error_msg or not valid) and model_sql and model_sql != final_sql and engine is not None and model_valid:
        try:
            try:
                alt_rows, alt_cols = engine.execute_sql(model_sql, str(db_id), timeout_s=float(exec_timeout_s), validate_schema=False)
            except TypeError:
                alt_rows, alt_cols = engine.execute_sql(model_sql, str(db_id))
            final_sql = model_sql
            valid = True
            error_msg = None
            rows, cols = alt_rows, alt_cols
        except Exception:
            pass

    # Task 2 dashboard logging
    if error_msg or not valid:
        et = classify_error(final_sql, str(error_msg or ""), timed_out=("interrupted" in str(error_msg or "").lower()))
        _log(et, question=str(question), db_id_val=str(db_id), sql=str(final_sql), error_msg=str(error_msg or ("invalid schema" if not valid else "")))
        run_rec.update({"status": "fail", "error_type": et, "error_msg": str(error_msg or ("invalid schema" if not valid else ""))})

    # --------------------------------------------------
    # Task impact metrics (Tasks 1/3/5)
    # --------------------------------------------------
    t1 = time.perf_counter()
    engine_stats_after = {}
    if engine is not None:
        try:
            engine_stats_after = engine.stats()
        except Exception:
            engine_stats_after = {}

    perf = {
        "db_id": str(db_id),
        "use_constrained_decoding": bool(use_constrained_decoding),
        "num_beams": int(gen_beams),
        "max_new_tokens": int(gen_max_new_tokens),
        "exec_timeout_s": float(exec_timeout_s),
        "exec_workers": int(exec_workers),
        "exec_cache_on": bool(exec_cache_on),
        "latency_total_ms": round((t1 - t0) * 1000.0, 2),
        "constraint_ok": bool(valid),
        "has_error": bool(error_msg),
        "exec_cache_hit_rate": float(engine_stats_after.get("exec_cache_hit_rate", 0.0) or 0.0),
        "exec_calls_total": int(engine_stats_after.get("exec_calls", 0) or 0),
        "exec_cache_hits_total": int(engine_stats_after.get("exec_cache_hits", 0) or 0),
        "exec_cache_misses_total": int(engine_stats_after.get("exec_cache_misses", 0) or 0),
    }
    _perf_log(perf)

    # Rolling summary (last 50)
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
        f"- Constraint OK (Task 3): {perf['constraint_ok']}\n"
        f"- Exec cache hit-rate (Task 1/5): {round(perf['exec_cache_hit_rate'], 3)}\n"
        f"- Rolling avg latency last 50 (ms): {round(avg_ms, 2)}\n"
        f"- Rolling constraint rate last 50: {round(constraint_rate, 3)}\n"
        f"- Rolling error rate last 50: {round(error_rate, 3)}"
    )

    # 8. ERROR HANDLING
    if error_msg or not valid:
        display_sql = final_sql if final_sql.strip() else "-- ❌ INVALID SQL"
        explanation = "❌ Error Details:\n\n"
        if error_msg: explanation += f"{error_msg}\n\n"
        if not valid:
            error_type = classify_error(final_sql, str(error_msg or ""))
            explanation += f"Error Type: {error_type}\nHint: {get_hint(error_type)}"
        explanation += perf_block
        ops = sql_ops(final_sql)
        run_rec["ops"] = ops
        for op in ops:
            if op in _OP_STATS:
                _OP_STATS[op]["fail"] += 1
        return display_sql, pd.DataFrame(columns=["Execution Notice"]), explanation

    # 9. SUCCESS HANDLING
    if not rows:
        safe_cols = cols if (cols and len(cols) > 0) else ["Result"]
        ops = sql_ops(final_sql)
        run_rec.update({"status": "ok", "ops": ops})
        for op in ops:
            if op in _OP_STATS:
                _OP_STATS[op]["ok"] += 1
        _SUCCESS_LOG.append({"t": run_rec["t"], "db_id": str(db_id), "question": question, "sql": final_sql, "ops": ops})
        return (
            final_sql,
            pd.DataFrame(columns=safe_cols),
            f"✅ Query executed successfully\n\nRows returned: 0\nExecution Time: {latency} sec\n\n{explain_sql(final_sql)}{perf_block}",
        )

    df = pd.DataFrame(rows, columns=cols)
    actual_rows = len(rows)
    explanation = f"✅ Query executed successfully\n\nRows returned: {actual_rows}\nExecution Time: {latency} sec\n\n{explain_sql(final_sql)}{perf_block}"
    ops = sql_ops(final_sql)
    run_rec.update({"status": "ok", "ops": ops})
    for op in ops:
        if op in _OP_STATS:
            _OP_STATS[op]["ok"] += 1
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
def _run_cmd(cmd: list[str], env: dict | None = None) -> str:
    run_env = (env or os.environ.copy()).copy()
    project_root = str(Path(__file__).resolve().parent)
    run_env["PYTHONPATH"] = project_root + (os.pathsep + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
    res = subprocess.run(cmd, capture_output=True, text=True, env=run_env, cwd=project_root)
    out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return out.strip()

def task1_benchmark(n_rollouts: int, max_workers: int) -> Iterator[tuple[str, str]]:
    project_root = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    try:
        os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    except Exception:
        pass

    cmd = [
        sys.executable,
        "-u",
        "scripts/benchmark_parallel_reward.py",
        "--n",
        str(int(n_rollouts)),
        "--max-workers",
        str(int(max_workers)),
        "--skip-profile",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_yield = time.perf_counter()
    lines: list[str] = []
    yield "Running Task 1 benchmark...\n", "<i>Running...</i>"

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        now = time.perf_counter()
        if now - last_yield >= 0.5:
            last_yield = now
            tail = "".join(lines[-200:]).strip()
            yield tail, "<i>Running...</i>"

    proc.wait()
    out = "".join(lines).strip()

    plot_path = "results/task1_plot.png"
    if os.path.exists(plot_path):
        try:
            b64 = base64.b64encode(Path(plot_path).read_bytes()).decode("ascii")
            html = f"<img src='data:image/png;base64,{b64}' style='max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px;' />"
            yield out, html
            return
        except Exception:
            yield out, f"<pre>{plot_path}</pre>"
            return

    yield out, "<i>No plot generated</i>"

def task4_eval(hard_adapter: str, soft_adapter: str, candidates: int, repairs: int) -> str:
    env = os.environ.copy()
    env["EVAL_NUM_CANDIDATES"] = str(int(candidates))
    env["EVAL_MAX_REPAIRS"] = str(int(repairs))
    out = _run_cmd(
        [
            sys.executable,
            "src/eval_rl_fixed_sample.py",
            "--hard_adapter",
            hard_adapter,
            "--soft_adapter",
            soft_adapter,
            "--use_constrained",
            "--num_samples",
            "500",
            "--sample_seed",
            "42",
            "--sample_with_replacement",
            "--out",
            "results/task4_eval.json",
        ],
        env=env,
    )
    if os.path.exists("results/task4_eval.json"):
        return Path("results/task4_eval.json").read_text()
    return out


def task4_train(reward_type: str, use_constrained: bool) -> Iterator[str]:
    project_root = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CONSTRAINED_DECODING"] = "1" if use_constrained else "0"

    if reward_type == "soft":
        script = "src/train_rl_codet5_reward_soft.py"
    else:
        script = "src/train_rl_codet5.py"

    cmd = [sys.executable, "-u", script]
    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    last_yield = time.perf_counter()
    yield f"Starting {reward_type.upper()} RL training (constrained={use_constrained})...\nCommand: {' '.join(cmd)}\n"

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        now = time.perf_counter()
        if now - last_yield >= 0.5:
            last_yield = now
            yield "".join(lines[-300:]).strip()

    proc.wait()
    yield "".join(lines).strip()

def task5_quant_export(mode: str, base_model: str, adapter: str, out_dir: str) -> str:
    return _run_cmd(
        [
            sys.executable,
            "scripts/quantize_export.py",
            "--mode",
            mode,
            "--base_model",
            base_model,
            "--adapter",
            adapter,
            "--out_dir",
            out_dir,
            "--local_only",
        ]
    )

def task2_dashboard() -> str:
    if not _QUERY_LOG:
        return "No logged errors yet."
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
    return json.dumps(
        {
            "counts": sorted(counts.items(), key=lambda x: (-x[1], x[0])),
            "recent": _QUERY_LOG[-10:],
            "perf_last_200": {
                "avg_latency_ms": round(avg_ms, 2),
                "constraint_rate": round(constraint_rate, 3),
                "error_rate": round(error_rate, 3),
            },
        },
        indent=2,
    )


def task2_dashboard_structured():
    """
    UI-friendly Task-2 outputs: counts table + recent errors table + dropdown choices.
    """
    if not _QUERY_LOG:
        empty_counts = pd.DataFrame(columns=["error_type", "count", "hint"])
        empty_recent = pd.DataFrame(columns=["time", "db_id", "error_type", "question", "error_msg"])
        return empty_counts, empty_recent, gr.update(choices=[], value=None)

    counts = {}
    for r in _QUERY_LOG[-1000:]:
        k = r.get("error_type") or "other"
        counts[k] = counts.get(k, 0) + 1
    rows = []
    for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        rows.append({"error_type": k, "count": int(v), "hint": get_hint(k)})
    counts_df = pd.DataFrame(rows)

    recent = []
    for r in _QUERY_LOG[-100:]:
        ts = r.get("t")
        try:
            ts_s = time.strftime("%H:%M:%S", time.localtime(float(ts))) if ts else ""
        except Exception:
            ts_s = ""
        recent.append(
            {
                "time": ts_s,
                "db_id": r.get("db_id", ""),
                "error_type": r.get("error_type", ""),
                "question": r.get("question", ""),
                "error_msg": r.get("error_msg", ""),
            }
        )
    recent_df = pd.DataFrame(recent)

    choices = [str(x["error_type"]) for x in rows]
    default = choices[0] if choices else None
    return counts_df, recent_df, gr.update(choices=choices, value=default)


def task2_error_examples(error_type: str) -> str:
    if not error_type:
        return ""
    hint = get_hint(error_type)
    matches = [r for r in reversed(_QUERY_LOG) if (r.get("error_type") or "") == str(error_type)]
    matches = matches[:3]
    if not matches:
        return f"Error type: {error_type}\nHint: {hint}\n\nNo examples yet."
    out = [f"Error type: {error_type}", f"Hint: {hint}", ""]
    for i, r in enumerate(matches, 1):
        out.append(f"Example {i}")
        out.append(f"DB: {r.get('db_id','')}")
        out.append(f"Q: {r.get('question','')}")
        out.append(f"SQL: {r.get('sql','')}")
        out.append(f"Msg: {r.get('error_msg','')}")
        out.append("")
    return "\n".join(out).strip()


def _tokenize_for_similarity(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def similar_successes_for_error(error_type: str, *, limit: int = 3) -> str:
    """
    Finds similar successful queries to the most recent failure of this error type.
    Similarity is a simple Jaccard over question+sql tokens (no external deps).
    """
    if not error_type:
        return ""
    failures = [r for r in reversed(_QUERY_LOG) if (r.get("error_type") or "") == str(error_type)]
    if not failures:
        return "No failures of this type yet."
    target = failures[0]
    tgt_tokens = _tokenize_for_similarity(f"{target.get('question','')} {target.get('sql','')}")
    if not tgt_tokens:
        return "No tokens to compare."

    scored = []
    for s in reversed(_SUCCESS_LOG[-2000:]):
        cand_tokens = _tokenize_for_similarity(f"{s.get('question','')} {s.get('sql','')}")
        if not cand_tokens:
            continue
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
        out.append(f"Similar success {i} (score={score:.3f})")
        out.append(f"DB: {s.get('db_id','')}")
        out.append(f"Q: {s.get('question','')}")
        out.append(f"SQL: {s.get('sql','')}")
        out.append("")
    return "\n".join(out).strip()


ADVERSARIAL_CASES = [
    {"name": "Missing JOIN", "db_id": "chinook_1", "question": "List track names and their artist names.", "expected": "missing_join"},
    {"name": "Wrong WHERE", "db_id": "flight_1", "question": "Show flights that cost more than 500.", "expected": "wrong_where"},
    {"name": "NULL handling", "db_id": "student_1", "question": "List students with no advisor.", "expected": "null_handling"},
    {"name": "Type mismatch", "db_id": "store_1", "question": "List orders where order_id equals 'abc'.", "expected": "type_mismatch"},
]


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
    except Exception as e:
        return f"<pre>Plot error: {e}</pre>"


def task2_ops_table():
    rows = []
    for op, d in _OP_STATS.items():
        ok = int(d.get("ok", 0))
        fail = int(d.get("fail", 0))
        total = ok + fail
        rows.append({"op": op, "ok": ok, "fail": fail, "total": total, "success_rate": (ok / total) if total else 0.0})
    return pd.DataFrame(rows), _plot_op_stats_html()


def run_adversarial_suite(quant_artifact_dir: str, use_constrained: bool, gen_beams: int, gen_max_new: int, exec_timeout_s: float, exec_workers: int, exec_cache_on: bool):
    """
    Runs a small adversarial suite and reports error taxonomy outcomes.
    """
    engine = quant_engine
    if quant_artifact_dir and str(quant_artifact_dir).strip():
        engine = get_quant_engine(str(quant_artifact_dir).strip(), bool(use_constrained), exec_workers=int(exec_workers), use_cache=bool(exec_cache_on))
    if engine is None:
        return pd.DataFrame(columns=["name", "db_id", "expected", "got", "status"]), "Engine unavailable."

    out_rows = []
    for c in ADVERSARIAL_CASES:
        dbid = c["db_id"]
        q = c["question"]
        expected = c.get("expected", "")
        try:
            try:
                res = engine.ask(q, dbid, num_beams=int(gen_beams), max_new_tokens=int(gen_max_new), timeout_s=float(exec_timeout_s))
            except TypeError:
                res = engine.ask(q, dbid)
            sql = str(res.get("sql", "") or "")
            err = res.get("error", None)
            got = classify_error(sql, str(err or "")) if err else ""
            status = "ok" if not err else "fail"
        except Exception as e:
            sql = ""
            got = "backend_crash"
            status = "crash"
            err = str(e)
        out_rows.append({"name": c["name"], "db_id": dbid, "expected": expected, "got": got, "status": status})

    df = pd.DataFrame(out_rows)
    summary = df["status"].value_counts().to_dict()
    return df, json.dumps({"summary": summary}, indent=2)


_ATTRIBUTION_CACHE = {}


def _load_attribution_model(base_model: str, adapter: str, device: str):
    key = (base_model, adapter, device)
    if key in _ATTRIBUTION_CACHE:
        return _ATTRIBUTION_CACHE[key]
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    if adapter and str(adapter).strip():
        m = PeftModel.from_pretrained(base, adapter).to(device)
        try:
            m = m.merge_and_unload()
        except Exception:
            pass
    else:
        m = base
    m.eval()
    _ATTRIBUTION_CACHE[key] = (tok, m)
    return tok, m


def gradient_attribution_for_error(
    base_model: str,
    adapter: str,
    db_id: str,
    question: str,
    sql: str,
    top_k: int = 12,
) -> str:
    """
    Simple gradient-based saliency over prompt tokens:
    - loss = NLL of the generated SQL tokens given the prompt
    - score per prompt token = ||grad(embedding)||_2
    This is slow and should be used only for debugging.
    """
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer, model = _load_attribution_model(base_model, adapter, device)
    except Exception as e:
        return f"Model load failed: {e}"

    try:
        schema = schema_encoder.structured_schema(str(db_id))
    except Exception:
        schema = ""

    prompt = (
        "You are a SQLite expert.\n\n"
        f"Database: {db_id}\n\n"
        "Schema:\n"
        f"{schema}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "SQL:"
    )

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    labels = tokenizer(str(sql), return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
    # Standard seq2seq: use -100 for padding label positions
    labels = labels.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    model.zero_grad(set_to_none=True)

    # Hook into input embeddings to get gradients
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(enc.input_ids)
    inputs_embeds.retain_grad()

    out = model(inputs_embeds=inputs_embeds, attention_mask=enc.attention_mask, labels=labels)
    loss = out.loss
    if loss is None:
        return "No loss computed."
    loss.backward()

    grads = inputs_embeds.grad  # [1, seq, dim]
    if grads is None:
        return "No gradients."
    scores = torch.norm(grads[0], dim=-1)  # [seq]
    scores = scores.detach().float().cpu()

    toks = tokenizer.convert_ids_to_tokens(enc.input_ids[0].detach().cpu().tolist())
    pairs = list(zip(range(len(toks)), toks, scores.tolist()))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[: int(top_k)]

    lines = [f"loss={float(loss.detach().cpu()):.4f}", "Top prompt tokens by gradient norm:"]
    for idx, tok, sc in top:
        lines.append(f"- {idx}: {tok} -> {sc:.6f}")
    return "\n".join(lines).strip()

def toggle_input_method(method, current_sample):
    if method == "💡 Pick a Sample":
        db = next((db for q, db in SAMPLES if q == current_sample), "chinook_1")
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=db, interactive=False))
    else:
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(interactive=True))

def load_sample(selected_question):
    if not selected_question: return gr.update()
    db = next((db for q, db in SAMPLES if q == selected_question), "chinook_1")
    return gr.update(value=db)

def clear_inputs():
    return (
        gr.update(value="💡 Pick a Sample"), gr.update(value=SAMPLE_QUESTIONS[0], visible=True), 
        gr.update(visible=False), gr.update(value="", visible=False),                 
        gr.update(value="chinook_1", interactive=False), "", pd.DataFrame(), ""                              
    )

def update_schema(db_id):
    if not db_id: return ""
    try:
        raw_schema = schema_encoder.structured_schema(db_id)
        html_output = "<div style='max-height: 250px; overflow-y: auto; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9em; line-height: 1.6;'>"
        for line in raw_schema.strip().split('\n'):
            line = line.strip()
            if not line: continue
            match = re.search(r'^([a-zA-Z0-9_]+)\s*\((.*)\)', line)
            if match:
                table_name = match.group(1).upper()
                columns = match.group(2).lower()
                html_output += f"<div style='margin-bottom: 8px;'><strong style='color: #0f172a; font-size: 1.05em; font-weight: 800;'>{table_name}</strong> <span style='color: #64748b;'>( {columns} )</span></div>"
            else:
                html_output += f"<div style='color: #475569;'>{line}</div>"
        html_output += "</div>"
        return html_output
    except Exception as e:
        return f"<div style='color: red;'>Error loading schema: {str(e)}</div>"

# =========================
# UI LAYOUT
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Text-to-SQL RLHF") as demo:

    gr.HTML(
        """
        <div style="text-align: center; background-color: #e0e7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #c7d2fe;">
            <h1 style="color: #3730a3; margin-top: 0; margin-bottom: 10px; font-size: 2.2em;"> Text-to-SQL using RLHF + Execution Reward</h1>
            <p style="color: #4f46e5; font-size: 1.1em; margin: 0;">Convert Natural Language to SQL, strictly validated and safely executed on local SQLite databases.</p>
        </div>
        """
    )

    DBS = sorted([
        "flight_1", "student_assessment", "store_1", "bike_1", "book_2", "chinook_1",
        "academic", "aircraft", "car_1", "cinema", "club_1", "csu_1",
        "college_1", "college_2", "company_1", "company_employee",
        "customer_complaints", "department_store", "employee_hire_evaluation",
        "museum_visit", "products_for_hire", "restaurant_1",
        "school_finance", "shop_membership", "small_bank_1",
         "student_1", "tvshow", "voter_1", "world_1"
    ])

    with gr.Tabs():
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Configuration & Input")
                    input_method = gr.Radio(
                        choices=["💡 Pick a Sample", "✍️ Type my own"],
                        value="💡 Pick a Sample",
                        label="How do you want to ask?",
                    )
                    sample_dropdown = gr.Dropdown(
                        choices=SAMPLE_QUESTIONS,
                        value=SAMPLE_QUESTIONS[0],
                        label="Select a Sample Question",
                        info="The database will be selected automatically.",
                        visible=True,
                    )
                    type_own_warning = gr.Markdown("**⚠️ Please select a Database first, then type your custom question below:**", visible=False)
                    gr.Markdown("---")
                    db_id = gr.Dropdown(choices=DBS, value="chinook_1", label="Select Database", interactive=False)
                    custom_question = gr.Textbox(label="Ask your Custom Question", placeholder="Type your own question here...", lines=3, visible=False)

                    gr.Markdown("#### ⚙️ Inference Settings")
                    quant_artifact_dir = gr.Textbox(label="Quant Artifact Dir", value=DEFAULT_QUANT_ARTIFACT)
                    use_constrained_decoding = gr.Checkbox(label="Use schema-aware constrained decoding", value=False)

                    gen_beams = gr.Number(value=8, precision=0, label="Beam search (num_beams)")
                    gen_max_new = gr.Number(value=120, precision=0, label="Max new tokens")
                    exec_timeout_s = gr.Number(value=2.0, precision=2, label="SQL execution timeout (s)")
                    exec_workers = gr.Number(value=8, precision=0, label="SQL exec workers (pool)")
                    exec_cache_on = gr.Checkbox(label="Cache SQL results (per DB)", value=True)

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

            with gr.Accordion("Task 2: Adversarial suite", open=False):
                adv_run = gr.Button("Run adversarial suite")
                adv_out = gr.Dataframe(label="Adversarial results", interactive=False, wrap=True)
                adv_summary = gr.Textbox(label="Summary", lines=8)
                adv_run.click(
                    fn=run_adversarial_suite,
                    inputs=[quant_artifact_dir, use_constrained_decoding, gen_beams, gen_max_new, exec_timeout_s, exec_workers, exec_cache_on],
                    outputs=[adv_out, adv_summary],
                )

            with gr.Accordion("Task 2: Gradient attribution (slow)", open=False):
                gr.Markdown("Computes gradient saliency over prompt tokens for the latest failed query of a selected error type.")
                attr_base = gr.Textbox(label="Base model", value=base_model_name)
                attr_adapter = gr.Textbox(label="Adapter (optional)", value=adapter_path)
                attr_type = gr.Dropdown(choices=[], value=None, label="Pick error type (uses latest failure)")
                attr_topk = gr.Number(value=12, precision=0, label="Top-K tokens")
                attr_run = gr.Button("Compute attribution")
                attr_out = gr.Textbox(label="Attribution output", lines=14)

                # Reuse the same refresh as dashboard to populate error types.
                def _populate_types():
                    if not _QUERY_LOG:
                        return gr.update(choices=[], value=None)
                    counts = {}
                    for r in _QUERY_LOG[-1000:]:
                        k = r.get("error_type") or "other"
                        counts[k] = counts.get(k, 0) + 1
                    choices = [k for k, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
                    return gr.update(choices=choices, value=(choices[0] if choices else None))

                attr_refresh = gr.Button("Refresh error types")
                attr_refresh.click(fn=_populate_types, inputs=[], outputs=[attr_type])

                def _run_attr(base_model: str, adapter: str, error_type: str, top_k: int):
                    if not error_type:
                        return "Pick an error type."
                    failures = [r for r in reversed(_QUERY_LOG) if (r.get("error_type") or "") == str(error_type)]
                    if not failures:
                        return "No failures of this type yet."
                    r = failures[0]
                    return gradient_attribution_for_error(
                        base_model=str(base_model),
                        adapter=str(adapter),
                        db_id=str(r.get("db_id", "")),
                        question=str(r.get("question", "")),
                        sql=str(r.get("sql", "")),
                        top_k=int(top_k),
                    )

                attr_run.click(fn=_run_attr, inputs=[attr_base, attr_adapter, attr_type, attr_topk], outputs=[attr_out])

        with gr.Tab("Training / Eval"):
            gr.Markdown("## Training + Evaluation (Task 4)")
            with gr.Accordion("Train RL (hard vs soft)", open=False):
                t4_train_reward = gr.Dropdown(choices=["soft", "hard"], value="soft", label="Reward type")
                t4_train_constrained = gr.Checkbox(label="Use constrained decoding (CONSTRAINED_DECODING=1)", value=True)
                t4_train_run = gr.Button("Start training (streams logs)")
                t4_train_logs = gr.Textbox(label="Training logs", lines=16)
                t4_train_run.click(fn=task4_train, inputs=[t4_train_reward, t4_train_constrained], outputs=[t4_train_logs])
            with gr.Accordion("Task 4: Hard vs Soft eval", open=False):
                t4_hard = gr.Textbox(label="Hard adapter path", value="checkpoints/best_rlhf_model_2")
                t4_soft = gr.Textbox(label="Soft adapter path", value="checkpoints/best_rlhf_codet5_soft")
                t4_cand = gr.Number(value=8, precision=0, label="Candidates per prompt")
                t4_rep = gr.Number(value=2, precision=0, label="Repair attempts")
                t4_run = gr.Button("Run Task 4 eval (500 samples)")
                t4_out = gr.Textbox(label="Eval JSON", lines=14)
                t4_run.click(fn=task4_eval, inputs=[t4_hard, t4_soft, t4_cand, t4_rep], outputs=[t4_out])

        with gr.Tab("Deployment"):
            gr.Markdown("## Deployment (Task 5)")
            with gr.Accordion("Task 5: Quant export", open=False):
                t5_mode = gr.Dropdown(choices=["fp32", "int8_dynamic", "int8_decoder_dynamic"], value="int8_dynamic", label="Mode")
                t5_base = gr.Textbox(label="Base model", value=base_model_name)
                t5_adapter = gr.Textbox(label="Adapter", value=adapter_path)
                t5_outdir = gr.Textbox(label="Out dir", value="checkpoints/task5/int8_dynamic")
                t5_run = gr.Button("Export quant artifact")
                t5_out = gr.Textbox(label="Export log", lines=12)
                t5_run.click(fn=task5_quant_export, inputs=[t5_mode, t5_base, t5_adapter, t5_outdir], outputs=[t5_out])

    # EVENT LISTENERS
    input_method.change(fn=toggle_input_method, inputs=[input_method, sample_dropdown], outputs=[sample_dropdown, type_own_warning, custom_question, db_id])
    sample_dropdown.change(fn=load_sample, inputs=[sample_dropdown], outputs=[db_id])
    db_id.change(fn=update_schema, inputs=[db_id], outputs=[schema_display])
    run_btn.click(
        fn=run_query,
        inputs=[
            input_method,
            sample_dropdown,
            custom_question,
            db_id,
            quant_artifact_dir,
            use_constrained_decoding,
            gen_beams,
            gen_max_new,
            exec_timeout_s,
            exec_workers,
            exec_cache_on,
        ],
        outputs=[final_sql, result_table, explanation],
    )
    clear_btn.click(fn=clear_inputs, inputs=[], outputs=[input_method, sample_dropdown, type_own_warning, custom_question, db_id, final_sql, result_table, explanation])

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0").strip() in {"1", "true", "True", "yes", "Y"}
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(share=share, server_name=server_name, show_api=False)
