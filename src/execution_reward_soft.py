import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.execution_reward import (
    _normalize_sql,
    is_valid_select,
    execute_sql_cached,
    execute_sql_cached_conn,
    EXECUTION_ERROR,
    validate_sql_schema,
    USE_SCHEMA_VALIDATION,
    _connect_readonly,
)

# =========================================================
# 🔥 SOFT REWARD CORE
# =========================================================
def compute_soft_reward(pred_res, gold_res, sample_k=10):
    try:
        # =================================================
        # 1. EDGE CASES
        # =================================================
        if not gold_res:
            return 1.0 if not pred_res else 0.3

        if not pred_res:
            return -0.05

        # =================================================
        # 2. SAFE HASHING
        # =================================================
        def make_hashable(row):
            return tuple(str(item) for item in row)

        pred_counter = Counter(make_hashable(r) for r in pred_res)

        # =================================================
        # 3. SAMPLING
        # =================================================
        k = min(sample_k, len(gold_res))
        sample = random.sample(gold_res, k)

        # =================================================
        # 4. MATCH COUNT
        # =================================================
        match = 0
        for row in sample:
            key = make_hashable(row)
            if pred_counter.get(key, 0) > 0:
                pred_counter[key] -= 1
                match += 1

        score = match / max(len(sample), 1)

        # =================================================
        # 5. 🔥 ANTI-CHEAT LENGTH PENALTY
        # =================================================
        len_ratio = len(pred_res) / max(len(gold_res), 1)

        if len_ratio > 1.5:
            score = score / (len_ratio ** 0.5)   # 🔥 smoother penalty

        # =================================================
        # 6. CLAMP SCORE (IMPORTANT FOR STABILITY)
        # =================================================
        score = max(0.0, min(1.0, score))

        # =================================================
        # 7. FINAL REWARD
        # =================================================
        return 0.3 + 0.7 * score

    except Exception:
        return -0.05


# =========================================================
# 🔥 MAIN EXECUTION REWARD
# =========================================================
_TLS = threading.local()


def _get_thread_conn(db_path: str):
    conns = getattr(_TLS, "conns", None)
    if conns is None:
        conns = {}
        _TLS.conns = conns
    conn = conns.get(db_path)
    if conn is None:
        conn = _connect_readonly(db_path)
        conns[db_path] = conn
    return conn


def execution_reward_soft_pooled(pred_sql, db_path, gold_sql, *, sample_k: int = 10):
    """
    Soft execution reward, but reuses a per-thread read-only SQLite connection.
    This avoids connect/close overhead in RL loops.
    """
    try:
        sql = _normalize_sql(pred_sql)
        gold = _normalize_sql(gold_sql)

        if not is_valid_select(sql):
            return -0.05

        if USE_SCHEMA_VALIDATION:
            ok, _ = validate_sql_schema(sql, db_path)
            if not ok:
                return -0.05

        conn = _get_thread_conn(db_path)
        pred_res = execute_sql_cached_conn(conn, db_path, sql)
        if pred_res == EXECUTION_ERROR:
            return -0.05

        gold_res = execute_sql_cached_conn(conn, db_path, gold)
        if gold_res == EXECUTION_ERROR:
            return -0.05

        return compute_soft_reward(pred_res, gold_res, sample_k=int(sample_k))
    except Exception:
        return -0.05


def execution_reward_soft(pred_sql, db_path, gold_sql):
    try:
        sql = _normalize_sql(pred_sql)
        gold = _normalize_sql(gold_sql)

        # =================================================
        # BASIC VALIDATION
        # =================================================
        if not is_valid_select(sql):
            return -0.05

        if USE_SCHEMA_VALIDATION:
            ok, _ = validate_sql_schema(sql, db_path)
            if not ok:
                return -0.05

        # =================================================
        # EXECUTION
        # =================================================
        pred_res = execute_sql_cached(db_path, sql)
        if pred_res == EXECUTION_ERROR:
            return -0.05

        gold_res = execute_sql_cached(db_path, gold)
        if gold_res == EXECUTION_ERROR:
            return -0.05

        return compute_soft_reward(pred_res, gold_res)

    except Exception:
        return -0.05


def execution_reward_soft_batch_parallel_by_db(rollouts, *, max_workers: int = 20, sample_k: int = 10):
    """
    rollouts: Sequence[(pred_sql, db_path, gold_sql)]
    Executes with 1-thread-per-DB grouping for better connection reuse.
    Returns rewards in the same order as input.
    """
    if not rollouts:
        return []

    # Group by DB so each worker can hold a single connection and reuse it.
    by_db = {}
    for idx, (pred_sql, db_path, gold_sql) in enumerate(rollouts):
        by_db.setdefault(db_path, []).append((idx, pred_sql, gold_sql))

    out = [0.0 for _ in range(len(rollouts))]

    def _worker(db_path: str, items):
        conn = _connect_readonly(db_path)
        try:
            for idx, pred_sql, gold_sql in items:
                # Do NOT use the global thread-local here; this worker owns the connection.
                try:
                    sql = _normalize_sql(pred_sql)
                    gold = _normalize_sql(gold_sql)
                    if not is_valid_select(sql):
                        out[idx] = -0.05
                        continue
                    if USE_SCHEMA_VALIDATION:
                        ok, _ = validate_sql_schema(sql, db_path)
                        if not ok:
                            out[idx] = -0.05
                            continue
                    pred_res = execute_sql_cached_conn(conn, db_path, sql)
                    if pred_res == EXECUTION_ERROR:
                        out[idx] = -0.05
                        continue
                    gold_res = execute_sql_cached_conn(conn, db_path, gold)
                    if gold_res == EXECUTION_ERROR:
                        out[idx] = -0.05
                        continue
                    out[idx] = float(compute_soft_reward(pred_res, gold_res, sample_k=int(sample_k)))
                except Exception:
                    out[idx] = -0.05
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futures = [ex.submit(_worker, db_path, items) for db_path, items in by_db.items()]
        for fut in as_completed(futures):
            fut.result()

    return out
