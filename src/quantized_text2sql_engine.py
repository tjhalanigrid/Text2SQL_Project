from __future__ import annotations

import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from src.quantization_utils import load_quant_artifact
from src.schema_encoder import SchemaEncoder
from src.sql_validator import validate_sql_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"


class QuantizedText2SQLEngine:
    """
    CPU-focused deployment engine:
    - loads a quantized artifact exported by scripts/quantize_export.py
    - supports batched generation
    - executes SQL with a small threadpool for throughput benchmarks
    """

    def __init__(
        self,
        artifact_dir: str,
        *,
        device: str = "cpu",
        use_constrained: bool = False,
        exec_workers: int | None = None,
        default_timeout_s: float = 2.0,
        use_cache: bool = True,
        cache_max_entries: int = 50_000,
    ):
        self.device = device
        self.use_constrained = bool(use_constrained)
        self.tokenizer, self.model, self.meta = load_quant_artifact(artifact_dir, device=device, local_only=True)
        self.schema_encoder = SchemaEncoder(DB_ROOT)
        if exec_workers is None:
            exec_workers = int(os.environ.get("SQL_EXEC_WORKERS", "8"))
        self.exec_pool = ThreadPoolExecutor(max_workers=int(exec_workers))
        self.default_timeout_s = float(default_timeout_s)
        self.use_cache = bool(use_cache)
        self.cache_max_entries = int(cache_max_entries)
        self._cache: "OrderedDict[tuple[str, str], tuple[list, list]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._exec_cache_hits = 0
        self._exec_cache_misses = 0
        self._exec_calls = 0
        self._tls = threading.local()

    def build_prompt(self, question: str, db_id: str) -> str:
        schema = self.schema_encoder.structured_schema(db_id)
        return (
            "You are a SQLite expert.\n\n"
            f"Database: {db_id}\n\n"
            "Schema:\n"
            f"{schema}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "SQL:"
        )

    def generate_sql_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        max_new_tokens: int = 120,
        num_beams: int = 8,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        prompts = [self.build_prompt(q, db_id) for q, db_id in pairs]

        # Constrained decoding is DB-specific; do per-item generation.
        if self.use_constrained:
            from transformers.generation.logits_process import LogitsProcessorList
            from src.constrained_decoding import SchemaConstrainedLogitsProcessor

            sqls: List[str] = []
            for (q, db_id), prompt in zip(pairs, prompts):
                db_path = str(DB_ROOT / db_id / f"{db_id}.sqlite")
                enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                proc = LogitsProcessorList([SchemaConstrainedLogitsProcessor(self.tokenizer, db_path)])
                out = self.model.generate(
                    **enc,
                    max_new_tokens=int(max_new_tokens),
                    num_beams=int(num_beams),
                    repetition_penalty=float(repetition_penalty),
                    logits_processor=proc,
                )
                sqls.append(self.tokenizer.decode(out[0], skip_special_tokens=True).strip())
            return sqls

        # Unconstrained: fast batched generation.
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
        )
        return [self.tokenizer.decode(x, skip_special_tokens=True).strip() for x in out]

    def _get_thread_conn(self, db_path: str) -> sqlite3.Connection:
        conns = getattr(self._tls, "conns", None)
        if conns is None:
            conns = {}
            self._tls.conns = conns
        conn = conns.get(db_path)
        if conn is None:
            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda b: b.decode(errors="ignore")
            conns[db_path] = conn
        return conn

    def _cache_get(self, key: tuple[str, str]) -> tuple[list, list] | None:
        if not self.use_cache:
            return None
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is None:
                return None
            self._cache.move_to_end(key)
            return hit

    def _cache_put(self, key: tuple[str, str], value: tuple[list, list]) -> None:
        if not self.use_cache:
            return
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)

    def _execute_one(self, sql: str, db_path: str, timeout_s: float | None = None):
        timeout_s = float(self.default_timeout_s if timeout_s is None else timeout_s)
        key = (db_path, sql)
        cached = self._cache_get(key)
        with self._stats_lock:
            self._exec_calls += 1
        if cached is not None:
            with self._stats_lock:
                self._exec_cache_hits += 1
            return cached
        with self._stats_lock:
            self._exec_cache_misses += 1

        conn = self._get_thread_conn(db_path)

        # SQLite timeout via progress handler.
        start_t = time.monotonic()

        def handler():
            return 1 if (time.monotonic() - start_t) > timeout_s else 0

        conn.set_progress_handler(handler, 10_000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        out = (rows, cols)
        self._cache_put(key, out)
        return out

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            calls = int(self._exec_calls)
            hits = int(self._exec_cache_hits)
            misses = int(self._exec_cache_misses)
        hit_rate = (hits / calls) if calls else 0.0
        return {
            "exec_calls": calls,
            "exec_cache_hits": hits,
            "exec_cache_misses": misses,
            "exec_cache_hit_rate": float(hit_rate),
            "use_cache": bool(self.use_cache),
            "exec_workers": int(getattr(self.exec_pool, "_max_workers", 0) or 0),
        }

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._exec_calls = 0
            self._exec_cache_hits = 0
            self._exec_cache_misses = 0

    def execute_sql(self, sql: str, db_id: str, *, timeout_s: float | None = None, validate_schema: bool = True):
        db_path = str(DB_ROOT / db_id / f"{db_id}.sqlite")
        if validate_schema:
            try:
                ok, _ = validate_sql_schema(sql, db_path)
            except Exception:
                ok = False
            if not ok:
                raise ValueError("Invalid schema")
        return self._execute_one(sql, db_path, timeout_s=timeout_s)

    def ask(
        self,
        question: str,
        db_id: str,
        *,
        max_new_tokens: int = 120,
        num_beams: int = 8,
        repetition_penalty: float = 1.2,
        timeout_s: float | None = None,
    ) -> Dict[str, Any]:
        sql = self.generate_sql_batch(
            [(question, db_id)],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )[0]
        db_path = str(DB_ROOT / db_id / f"{db_id}.sqlite")

        try:
            ok, _ = validate_sql_schema(sql, db_path)
        except Exception:
            ok = False
        if not ok:
            return {"sql": sql, "rows": [], "columns": [], "error": "Invalid schema"}

        try:
            rows, cols = self._execute_one(sql, db_path, timeout_s=timeout_s)
            return {"sql": sql, "rows": rows, "columns": cols, "error": None}
        except Exception as e:
            return {"sql": sql, "rows": [], "columns": [], "error": str(e)}

    def ask_batch_execute(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
        sqls = self.generate_sql_batch(pairs)
        results: List[Dict[str, Any]] = []

        futures = {}
        for (q, db_id), sql in zip(pairs, sqls):
            db_path = str(DB_ROOT / db_id / f"{db_id}.sqlite")
            futures[self.exec_pool.submit(self._execute_one, sql, db_path)] = (sql, db_id)

        for fut in as_completed(futures):
            sql, db_id = futures[fut]
            try:
                rows, cols = fut.result()
                results.append({"db_id": db_id, "sql": sql, "rows": rows, "columns": cols, "error": None})
            except Exception as e:
                results.append({"db_id": db_id, "sql": sql, "rows": [], "columns": [], "error": str(e)})

        return results
