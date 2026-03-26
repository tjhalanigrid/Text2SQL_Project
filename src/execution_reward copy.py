

# from __future__ import annotations

# import hashlib
# import os
# import queue
# import re
# import sqlite3
# import threading
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

# # --- CACHE CONTROL ---
# USE_CACHE = True
# _REWARD_CACHE: Dict[str, float] = {}

# def set_use_cache(enabled: bool):
#     """Dynamically toggle the reward cache for benchmarks."""
#     global USE_CACHE
#     USE_CACHE = enabled

# def _normalize_sql(sql: str) -> str:
#     if not isinstance(sql, str):
#         return ""
#     s = sql.strip()
#     if s.startswith("```"):
#         s = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", s).strip()
#         s = re.sub(r"\n?```$", "", s).strip()
#     if s.lower().startswith("sql:"):
#         s = s[4:].strip()
#     if ";" in s:
#         s = s.split(";", 1)[0].strip()
#     return s

# def _connect_readonly(db_path: str) -> sqlite3.Connection:
#     uri = f"file:{os.path.abspath(db_path)}?mode=ro"
#     conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
#     conn.execute("PRAGMA query_only = ON;")
#     conn.execute("PRAGMA foreign_keys = ON;")
#     return conn

# DEFAULT_QUERY_TIMEOUT_S = 2.0

# def _with_timeout(conn: sqlite3.Connection, timeout_s: float = DEFAULT_QUERY_TIMEOUT_S) -> None:
#     start = time.monotonic()
#     def _handler() -> int:
#         return 1 if (time.monotonic() - start) > timeout_s else 0
#     conn.set_progress_handler(_handler, 10_000)

# def _list_tables(conn: sqlite3.Connection) -> List[str]:
#     try:
#         cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
#         return [r[0] for r in cur.fetchall() if r and isinstance(r[0], str)]
#     except sqlite3.Error:
#         return []

# def _contains_table_name(sql: str, table_names: Sequence[str]) -> bool:
#     s = sql.lower()
#     for t in table_names:
#         tl = t.lower()
#         if not tl:
#             continue
#         if re.search(rf"\b{re.escape(tl)}\b", s):
#             return True
#     return False

# def _explain_query_plan(conn: sqlite3.Connection, sql: str) -> bool:
#     try:
#         _with_timeout(conn, timeout_s=DEFAULT_QUERY_TIMEOUT_S)
#         conn.execute(f"EXPLAIN QUERY PLAN {sql}")
#         return True
#     except sqlite3.Error:
#         return False

# def _execute(conn: sqlite3.Connection, sql: str, max_rows: int = 1000) -> Tuple[bool, List[Tuple], Optional[str]]:
#     try:
#         _with_timeout(conn, timeout_s=DEFAULT_QUERY_TIMEOUT_S)
#         cur = conn.execute(sql)
#         rows = cur.fetchmany(max_rows)
#         norm_rows = [tuple(r) for r in rows]
#         return True, norm_rows, None
#     except sqlite3.Error as e:
#         return False, [], str(e)

# _SQL_KEYWORDS_TO_IGNORE = {
#     "select", "from", "where", "join", "inner", "left", "right", "full", "outer", 
#     "on", "group", "by", "order", "limit", "having", "distinct", "union", "intersect", 
#     "except", "as", "and", "or", "not", "in", "is", "null", "like", "between", "case", 
#     "when", "then", "else", "end", "asc", "desc"
# }

# _SQL_FUNCTIONS_TO_IGNORE = {
#     "count", "avg", "min", "max", "sum", "lower", "upper", "substr", "coalesce", 
#     "round", "date", "datetime", "strftime"
# }

# # --- LIGHTWEIGHT PARSING ---
# def is_valid_select(sql: str):
#     sql = sql.strip().lower()
#     return sql.startswith("select") or sql.startswith("with")

# def extract_tables(sql: str) -> List[str]:
#     sql = sql.lower()
#     if "join" not in sql:
#         tables = re.findall(r'from\s+(\w+)', sql)
#         return list(set(tables))

#     tables = re.findall(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql)
#     joins = re.findall(r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql)
#     return list(set(tables + joins))

# def extract_columns(sql: str) -> List[str]:
#     sql = sql.lower()
#     match = re.search(r'select\s+(.*?)\s+from', sql)
#     if not match:
#         return []
#     cols = match.group(1)
#     if cols.strip() == "*":
#         return ["*"]
#     return [c.strip() for c in cols.split(",")]

# def _get_db_tables_and_columns(conn: sqlite3.Connection) -> Tuple[Set[str], Set[str]]:
#     tables = set()
#     columns = set()
#     for t in _list_tables(conn):
#         tl = t.lower()
#         if not tl:
#             continue
#         tables.add(tl)
#         try:
#             cur = conn.execute(f'PRAGMA table_info("{t}")')
#             for row in cur.fetchall():
#                 if row and isinstance(row[1], str):
#                     columns.add(row[1].lower())
#         except sqlite3.Error:
#             continue
#     return tables, columns

# def _safe_results_equal(a: List[Tuple], b: List[Tuple]) -> bool:
#     return a == b

# @dataclass
# class RewardDebugStats:
#     total: int = 0
#     parsed_ok: int = 0
#     table_match: int = 0
#     column_match: int = 0
#     executed_ok: int = 0
#     exact_match: int = 0

# _DEBUG = RewardDebugStats()

# def reset_debug_metrics() -> None:
#     global _DEBUG
#     _DEBUG = RewardDebugStats()

# def get_debug_metrics() -> dict:
#     denom = max(_DEBUG.total, 1)
#     return {
#         "valid_sql_rate": _DEBUG.parsed_ok / denom,
#         "table_match_rate": _DEBUG.table_match / denom,
#         "column_match_rate": _DEBUG.column_match / denom,
#         "execution_accuracy": _DEBUG.exact_match / denom,
#     }

# EXECUTION_ERROR = "EXECUTION_ERROR"

# _RESULT_CACHE_LOCK = threading.Lock()
# _RESULT_CACHE: "Dict[str, Union[List[Tuple], str]]" = {}
# _RESULT_CACHE_MAX = 100_000

# def clear_result_cache() -> None:
#     """Clear both DB query cache and reward cache."""
#     with _RESULT_CACHE_LOCK:
#         _RESULT_CACHE.clear()
#     _REWARD_CACHE.clear()

# def _db_state_fingerprint(db_path: str) -> str:
#     try:
#         st = os.stat(db_path)
#         return f"{st.st_mtime_ns}:{st.st_size}"
#     except OSError:
#         return "missing"

# def _result_cache_key(db_path: str, sql: str) -> str:
#     fp = _db_state_fingerprint(db_path)
#     payload = f"{fp}\0{sql}".encode("utf-8", errors="ignore")
#     return hashlib.sha256(payload).hexdigest()

# class _ConnectionPool:
#     def __init__(self, db_path: str, maxsize: int = 1) -> None:
#         self.db_path = db_path
#         self.pool = queue.LifoQueue(maxsize=maxsize)
#         self.lock = threading.Lock()

#     def acquire(self) -> sqlite3.Connection:
#         try:
#             return self.pool.get_nowait()
#         except queue.Empty:
#             with self.lock:
#                 try:
#                     return self.pool.get_nowait()
#                 except queue.Empty:
#                     return _connect_readonly(self.db_path)

#     def release(self, conn: sqlite3.Connection) -> None:
#         try:
#             self.pool.put_nowait(conn)
#         except queue.Full:
#             try:
#                 conn.close()
#             except Exception:
#                 pass

# _POOL_LOCK = threading.Lock()
# _POOLS: Dict[str, _ConnectionPool] = {}

# def _get_pool(db_path: str) -> _ConnectionPool:
#     with _POOL_LOCK:
#         pool = _POOLS.get(db_path)
#         if pool is None:
#             pool = _ConnectionPool(db_path=db_path, maxsize=1)
#             _POOLS[db_path] = pool
#         return pool

# class _PooledConnection:
#     def __init__(self, db_path: str) -> None:
#         self.db_path = db_path
#         self.pool = _get_pool(db_path)
#         self.conn: Optional[sqlite3.Connection] = None

#     def __enter__(self) -> sqlite3.Connection:
#         self.conn = self.pool.acquire()
#         return self.conn

#     def __exit__(self, exc_type, exc, tb) -> None:
#         if self.conn is not None:
#             self.pool.release(self.conn)
#             self.conn = None

# def _cache_get(key: str) -> Optional[Union[List[Tuple], str]]:
#     with _RESULT_CACHE_LOCK:
#         return _RESULT_CACHE.get(key)

# def _cache_put(key: str, value: Union[List[Tuple], str]) -> None:
#     with _RESULT_CACHE_LOCK:
#         if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
#             _RESULT_CACHE.clear()
#         _RESULT_CACHE[key] = value

# def execute_sql(conn: sqlite3.Connection, sql: str, *, max_rows: int = 1000) -> Union[List[Tuple], str]:
#     try:
#         _with_timeout(conn, timeout_s=DEFAULT_QUERY_TIMEOUT_S)
#         cur = conn.execute(sql)
#         rows = cur.fetchmany(max_rows)
#         return [tuple(r) for r in rows]
#     except Exception:
#         return EXECUTION_ERROR

# def execute_sql_cached(db_path: str, sql: str, *, max_rows: int = 1000) -> Union[List[Tuple], str]:
#     if not USE_CACHE:
#         with _PooledConnection(db_path) as conn:
#             return execute_sql(conn, sql, max_rows=max_rows)
            
#     key = _result_cache_key(db_path, sql)
#     cached = _cache_get(key)
#     if cached is not None:
#         return cached
#     with _PooledConnection(db_path) as conn:
#         res = execute_sql(conn, sql, max_rows=max_rows)
#     _cache_put(key, res)
#     return res

# def execution_reward_timed(
#     pred_sql: str, db_path: str, gold_sql: str, *, measure_plan: bool = False,
# ) -> Tuple[float, Dict[str, float]]:
#     timings = {"parse_s": 0.0, "plan_s": 0.0, "exec_s": 0.0}
#     t0 = time.perf_counter()
#     sql = _normalize_sql(pred_sql)
#     gold = _normalize_sql(gold_sql)

#     if not is_valid_select(sql):
#         timings["parse_s"] = time.perf_counter() - t0
#         return 0.0, timings

#     t1 = time.perf_counter()
#     timings["parse_s"] = t1 - t0

#     if measure_plan:
#         with _PooledConnection(db_path) as conn:
#             p0 = time.perf_counter()
#             _explain_query_plan(conn, sql)
#             _explain_query_plan(conn, gold)
#             timings["plan_s"] = time.perf_counter() - p0

#     e0 = time.perf_counter()
#     pred_res = execute_sql_cached(db_path, sql)
#     if pred_res == EXECUTION_ERROR:
#         timings["exec_s"] = time.perf_counter() - e0
#         return 0.0, timings
#     gold_res = execute_sql_cached(db_path, gold)
#     timings["exec_s"] = time.perf_counter() - e0
#     if gold_res == EXECUTION_ERROR:
#         return 0.0, timings

#     reward = -0.2
#     reward += 0.2
#     if _safe_results_equal(pred_res, gold_res):
#         return 1.0, timings
#     return max(-1.0, min(1.0, reward)), timings

# def execution_reward(pred_sql: str, db_path: str, gold_sql: str) -> float:
#     try:
#         sql = _normalize_sql(pred_sql)
#         gold = _normalize_sql(gold_sql)

#         if not is_valid_select(sql):
#             return -1.0

#         reward = -0.2

#         pred_tables = set(extract_tables(sql))
#         gold_tables = set(extract_tables(gold))

#         if pred_tables == gold_tables and len(gold_tables) > 0:
#             reward += 0.3

#         pred_cols = set(extract_columns(sql))
#         gold_cols = set(extract_columns(gold))

#         if gold_cols:
#             overlap = len(pred_cols & gold_cols) / len(gold_cols)
#             reward += 0.3 * overlap

#         pred_res = execute_sql_cached(db_path, sql)
#         if pred_res == EXECUTION_ERROR:
#             return 0.0
#         reward += 0.2

#         gold_res = execute_sql_cached(db_path, gold)
#         if gold_res == EXECUTION_ERROR:
#             return 0.0
#         if _safe_results_equal(pred_res, gold_res):
#             return 1.0

#         return max(-1.0, min(1.0, reward))

#     except Exception:
#         return 0.0

# def cached_execution_reward(pred_sql: str, db_path: str, gold_sql: str) -> float:
#     if not USE_CACHE:
#         return execution_reward(pred_sql, db_path, gold_sql)
        
#     key = f"{db_path}|{pred_sql}|{gold_sql}"
#     if key not in _REWARD_CACHE:
#         _REWARD_CACHE[key] = execution_reward(pred_sql, db_path, gold_sql)
#     return _REWARD_CACHE[key]

# def execution_reward_batch_sequential(rollouts: Sequence[Tuple[str, str, str]]) -> List[float]:
#     return [cached_execution_reward(pred_sql, db_path, gold_sql) for pred_sql, db_path, gold_sql in rollouts]

# def execution_reward_batch_parallel(rollouts: Sequence[Tuple[str, str, str]], *, max_workers: int = 20) -> List[float]:
#     if not rollouts:
#         return []
        
#     unique_dbs = {db_path for _, db_path, _ in rollouts}
#     worker_count = max(1, min(max_workers, len(unique_dbs)))
#     results: List[Optional[float]] = [None] * len(rollouts)
    
#     with ThreadPoolExecutor(max_workers=worker_count) as executor:
#         futures = {
#             executor.submit(cached_execution_reward, pred_sql, db_path, gold_sql): i
#             for i, (pred_sql, db_path, gold_sql) in enumerate(rollouts)
#         }
#         for fut in as_completed(futures):
#             idx = futures[fut]
#             try:
#                 results[idx] = float(fut.result())
#             except Exception:
#                 results[idx] = 0.0
                
#     return [r if r is not None else 0.0 for r in results]

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

from src.sql_validator import validate_sql_schema

# =========================================================
# 🔥 CONFIG FLAGS
# =========================================================
USE_SCHEMA_VALIDATION = True
USE_CACHE = True
DEFAULT_QUERY_TIMEOUT_S = 2.0

EXECUTION_ERROR = "EXECUTION_ERROR"

_REWARD_CACHE: Dict[str, float] = {}

# =========================================================
# 🔥 TASK 2: ERROR ANALYSIS + LOGGING
# =========================================================
ERROR_LOG_FILE = "results/error_logs.json"


def classify_error(sql: str) -> str:
    sql = sql.lower()

    if "join" in sql and " on " not in sql:
        return "missing_join"

    if "where" in sql and "=" not in sql and ">" not in sql and "<" not in sql:
        return "wrong_where"

    if "null" in sql:
        return "null_handling"

    if "group by" in sql and "count" not in sql:
        return "wrong_groupby"

    return "other"


def get_hint(error_type: str) -> str:
    hints = {
        "missing_join": "Add proper JOIN condition using ON.",
        "wrong_where": "Check WHERE clause conditions.",
        "null_handling": "Handle NULL values using IS NULL.",
        "wrong_groupby": "Use aggregation functions with GROUP BY.",
        "other": "Check SQL syntax and logic."
    }
    return hints.get(error_type, "Check query.")


def log_error(question: str, sql: str, error: str, error_type: str):
    os.makedirs("results", exist_ok=True)

    entry = {
        "question": question,
        "sql": sql,
        "error": error,
        "error_type": error_type,
        "timestamp": time.time()
    }

    if os.path.exists(ERROR_LOG_FILE):
        with open(ERROR_LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(entry)

    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# =========================================================
# CACHE/VALIDATION TOGGLES (Task 1)
# =========================================================
def set_use_cache(enabled: bool) -> None:
    global USE_CACHE
    USE_CACHE = bool(enabled)


def set_use_schema_validation(enabled: bool) -> None:
    global USE_SCHEMA_VALIDATION
    USE_SCHEMA_VALIDATION = bool(enabled)


# =========================================================
# SQL CLEANING
# =========================================================
def _normalize_sql(sql: str) -> str:
    if not isinstance(sql, str):
        return ""
    s = sql.strip()

    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", s).strip()
        s = re.sub(r"\n?```$", "", s).strip()

    if s.lower().startswith("sql:"):
        s = s[4:].strip()

    if ";" in s:
        s = s.split(";", 1)[0].strip()

    return s


# =========================================================
# DB EXECUTION
# =========================================================
def _connect_readonly(db_path: str):
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _with_timeout(conn: sqlite3.Connection, timeout_s: float = DEFAULT_QUERY_TIMEOUT_S):
    start = time.monotonic()

    def handler():
        return 1 if (time.monotonic() - start) > timeout_s else 0

    conn.set_progress_handler(handler, 10_000)


def execute_sql(conn, sql):
    try:
        _with_timeout(conn, timeout_s=DEFAULT_QUERY_TIMEOUT_S)
        cur = conn.execute(sql)
        return cur.fetchall()
    except Exception:
        return EXECUTION_ERROR


_RESULT_CACHE = {}
_RESULT_LOCK = threading.Lock()


def execute_sql_cached(db_path, sql):
    key = f"{db_path}|{sql}"

    if USE_CACHE:
        with _RESULT_LOCK:
            if key in _RESULT_CACHE:
                return _RESULT_CACHE[key]

    conn = _connect_readonly(db_path)
    result = execute_sql(conn, sql)
    conn.close()

    if USE_CACHE:
        with _RESULT_LOCK:
            _RESULT_CACHE[key] = result

    return result


def execute_sql_cached_conn(conn: sqlite3.Connection, db_path: str, sql: str):
    """
    Like execute_sql_cached(), but reuses an existing connection.
    Intended for 1-thread-per-DB workloads (Task 1).
    """
    key = f"{db_path}|{sql}"
    if USE_CACHE:
        with _RESULT_LOCK:
            if key in _RESULT_CACHE:
                return _RESULT_CACHE[key]

    result = execute_sql(conn, sql)

    if USE_CACHE:
        with _RESULT_LOCK:
            _RESULT_CACHE[key] = result

    return result


def clear_result_cache() -> None:
    global _RESULT_CACHE, _REWARD_CACHE
    with _RESULT_LOCK:
        _RESULT_CACHE.clear()
    _REWARD_CACHE.clear()


# =========================================================
# SQL PARSING
# =========================================================
def is_valid_select(sql):
    return sql.lower().startswith("select") or sql.lower().startswith("with")


def extract_tables(sql):
    return re.findall(r'from\s+(\w+)', sql.lower())


def extract_columns(sql):
    match = re.search(r'select\s+(.*?)\s+from', sql.lower())
    if not match:
        return []
    cols = match.group(1)
    return ["*"] if cols.strip() == "*" else [c.strip() for c in cols.split(",")]


def get_sql_operations(sql: str):
    sql = sql.lower()
    ops = []

    if "select" in sql: ops.append("SELECT")
    if "where" in sql: ops.append("WHERE")
    if "join" in sql: ops.append("JOIN")
    if "group by" in sql: ops.append("GROUP_BY")
    if "order by" in sql: ops.append("ORDER_BY")

    return ops


def _explain_query_plan(conn: sqlite3.Connection, sql: str) -> bool:
    try:
        _with_timeout(conn, timeout_s=DEFAULT_QUERY_TIMEOUT_S)
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        return True
    except Exception:
        return False


def execution_reward_timed(pred_sql: str, db_path: str, gold_sql: str, measure_plan: bool = False):
    """
    Returns (reward, timings) where timings keys: parse_s, plan_s, exec_s.
    Used by Task-1 benchmark to profile bottlenecks.
    """
    timings = {"parse_s": 0.0, "plan_s": 0.0, "exec_s": 0.0}
    t0 = time.perf_counter()

    sql = _normalize_sql(pred_sql)
    gold = _normalize_sql(gold_sql)

    if not is_valid_select(sql):
        timings["parse_s"] = time.perf_counter() - t0
        return 0.0, timings

    t1 = time.perf_counter()
    timings["parse_s"] = t1 - t0

    conn = _connect_readonly(db_path)
    try:
        if measure_plan:
            p0 = time.perf_counter()
            _explain_query_plan(conn, sql)
            _explain_query_plan(conn, gold)
            timings["plan_s"] = time.perf_counter() - p0

        e0 = time.perf_counter()
        pred_res = execute_sql_cached_conn(conn, db_path, sql)
        if pred_res == EXECUTION_ERROR:
            timings["exec_s"] = time.perf_counter() - e0
            return 0.0, timings
        gold_res = execute_sql_cached_conn(conn, db_path, gold)
        timings["exec_s"] = time.perf_counter() - e0
        if gold_res == EXECUTION_ERROR:
            return 0.0, timings

        reward = -0.2 + 0.2
        if pred_res == gold_res:
            return 1.0, timings
        return max(-1.0, min(1.0, reward)), timings
    finally:
        try:
            conn.close()
        except Exception:
            pass


# =========================================================
# 🔥 FINAL REWARD FUNCTION (TASK 2 INTEGRATED)
# =========================================================
def execution_reward(pred_sql: str, db_path: str, gold_sql: str) -> float:
    try:
        sql = _normalize_sql(pred_sql)
        gold = _normalize_sql(gold_sql)

        if not is_valid_select(sql):
            return -1.0

        reward = -0.2

        # =========================
        # SCHEMA VALIDATION (Task 3)
        # =========================
        if USE_SCHEMA_VALIDATION:
            valid, _ = validate_sql_schema(sql, db_path)
            if not valid:
                error_type = classify_error(sql)
                log_error("UNKNOWN", sql, "schema_invalid", error_type)
                return 0.1

        # =========================
        # EXECUTION
        # =========================
        pred_res = execute_sql_cached(db_path, sql)

        if pred_res == "EXECUTION_ERROR":
            error_type = classify_error(sql)

            log_error(
                question="UNKNOWN",
                sql=sql,
                error="execution_error",
                error_type=error_type
            )

            print(f"[ERROR] {error_type}")
            print(f"[HINT] {get_hint(error_type)}")

            return 0.1

        reward += 0.2

        gold_res = execute_sql_cached(db_path, gold)

        if gold_res == "EXECUTION_ERROR":
            return 0.1

        if pred_res == gold_res:
            return 1.0

        return max(-1.0, min(1.0, reward))

    except Exception as e:
        log_error("UNKNOWN", pred_sql, str(e), "runtime_error")
        return 0.0


# =========================================================
# BATCH EXECUTION (Task 1)
# =========================================================
def cached_execution_reward(pred_sql: str, db_path: str, gold_sql: str) -> float:
    if not USE_CACHE:
        return float(execution_reward(pred_sql, db_path, gold_sql))
    key = f"{db_path}|{pred_sql}|{gold_sql}"
    if key in _REWARD_CACHE:
        return float(_REWARD_CACHE[key])
    r = float(execution_reward(pred_sql, db_path, gold_sql))
    _REWARD_CACHE[key] = r
    return r


def execution_reward_batch_sequential(rollouts):
    return [cached_execution_reward(p, d, g) for (p, d, g) in rollouts]


def execution_reward_batch_parallel(rollouts, max_workers=10):
    results = [0.0] * len(rollouts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(cached_execution_reward, p, d, g): i
            for i, (p, d, g) in enumerate(rollouts)
        }

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = 0.0

    return results


def execution_reward_batch_parallel_by_db(rollouts, max_workers: int = 20):
    """
    1 thread per DB path. Reuses a single readonly connection per DB worker.
    Preserves input order.
    """
    if not rollouts:
        return []

    by_db = {}
    for idx, (pred_sql, db_path, gold_sql) in enumerate(rollouts):
        by_db.setdefault(db_path, []).append((idx, pred_sql, gold_sql))

    results = [0.0 for _ in range(len(rollouts))]

    def _reward_with_conn(conn: sqlite3.Connection, pred_sql: str, db_path: str, gold_sql: str) -> float:
        try:
            sql = _normalize_sql(pred_sql)
            gold = _normalize_sql(gold_sql)

            if not is_valid_select(sql):
                return -1.0

            reward = -0.2

            if USE_SCHEMA_VALIDATION:
                valid, _ = validate_sql_schema(sql, db_path)
                if not valid:
                    error_type = classify_error(sql)
                    log_error("UNKNOWN", sql, "schema_invalid", error_type)
                    return 0.1

            pred_res = execute_sql_cached_conn(conn, db_path, sql)
            if pred_res == EXECUTION_ERROR:
                error_type = classify_error(sql)
                log_error("UNKNOWN", sql, "execution_error", error_type)
                return 0.1

            reward += 0.2
            gold_res = execute_sql_cached_conn(conn, db_path, gold)
            if gold_res == EXECUTION_ERROR:
                return 0.1
            if pred_res == gold_res:
                return 1.0
            return max(-1.0, min(1.0, reward))
        except Exception:
            return 0.0

    def _worker(db_path: str, items):
        conn = _connect_readonly(db_path)
        try:
            for idx, pred, gold in items:
                results[idx] = _reward_with_conn(conn, pred, db_path, gold)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futures = [ex.submit(_worker, db_path, items) for db_path, items in by_db.items()]
        for fut in as_completed(futures):
            fut.result()

    return results
