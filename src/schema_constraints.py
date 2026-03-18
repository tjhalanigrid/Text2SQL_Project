from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from schema_utils import get_table_to_columns


@dataclass(frozen=True)
class ConstraintGraph:
    db_path: str
    fingerprint: str
    tables: Set[str]
    columns_by_table: Dict[str, Set[str]]
    all_columns: Set[str]
    types_by_table_col: Dict[Tuple[str, str], str]
    foreign_keys: List[Tuple[str, str, str, str]]


_LOCK = threading.Lock()
_CACHE: Dict[str, ConstraintGraph] = {}


def _db_state_fingerprint(db_path: str) -> str:
    try:
        st = os.stat(db_path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def build_constraint_graph(db_path: str) -> ConstraintGraph:
    fp = _db_state_fingerprint(db_path)
    with _LOCK:
        cached = _CACHE.get(db_path)
        if cached is not None and cached.fingerprint == fp:
            return cached

    table_to_cols = get_table_to_columns(db_path)  # lowercased
    tables = set(table_to_cols.keys())
    columns_by_table = {t: set(cols) for t, cols in table_to_cols.items()}
    all_columns: Set[str] = set()
    for cols in columns_by_table.values():
        all_columns.update(cols)

    types_by_table_col: Dict[Tuple[str, str], str] = {}
    foreign_keys: List[Tuple[str, str, str, str]] = []

    # Pull types + FKs (keep table/column extraction centralized in schema_utils).
    with _connect_readonly(db_path) as conn:
        for table in tables:
            try:
                cur = conn.execute(f'PRAGMA table_info("{table}")')
                for row in cur.fetchall():
                    if not row or len(row) < 3:
                        continue
                    col = str(row[1]).lower()
                    typ = str(row[2]).lower() if row[2] is not None else ""
                    types_by_table_col[(table, col)] = typ
            except sqlite3.Error:
                continue

            try:
                cur = conn.execute(f'PRAGMA foreign_key_list("{table}")')
                for row in cur.fetchall():
                    # (id, seq, table, from, to, on_update, on_delete, match)
                    if not row or len(row) < 5:
                        continue
                    to_table = str(row[2]).lower()
                    from_col = str(row[3]).lower()
                    to_col = str(row[4]).lower()
                    foreign_keys.append((table, from_col, to_table, to_col))
            except sqlite3.Error:
                continue

    graph = ConstraintGraph(
        db_path=db_path,
        fingerprint=fp,
        tables=tables,
        columns_by_table=columns_by_table,
        all_columns=all_columns,
        types_by_table_col=types_by_table_col,
        foreign_keys=foreign_keys,
    )
    with _LOCK:
        _CACHE[db_path] = graph
    return graph

