# import os
# import sqlite3
# import threading
# from typing import Dict, List, Set, Tuple

# def get_schema(db_path):
#     schema_map = get_table_to_columns(db_path)
#     schema_text = ""
#     for table, col_names in schema_map.items():
#         schema_text += f"{table}({', '.join(col_names)})\n"
#     return schema_text

# _SCHEMA_LOCK = threading.Lock()
# _SCHEMA_CACHE: Dict[str, Tuple[str, Dict[str, List[str]]]] = {}

# def _db_state_fingerprint(db_path: str) -> str:
#     try:
#         st = os.stat(db_path)
#         return f"{st.st_mtime_ns}:{st.st_size}"
#     except OSError:
#         return "missing"

# def _connect_readonly(db_path: str) -> sqlite3.Connection:
#     uri = f"file:{os.path.abspath(db_path)}?mode=ro"
#     conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
#     conn.execute("PRAGMA query_only = ON;")
#     conn.execute("PRAGMA foreign_keys = ON;")
#     return conn

# def get_table_to_columns(db_path: str) -> Dict[str, List[str]]:
#     """
#     Return mapping of table -> column names for the SQLite DB at db_path.
#     Tables and columns are returned lowercased.
#     """
#     fp = _db_state_fingerprint(db_path)
#     with _SCHEMA_LOCK:
#         cached = _SCHEMA_CACHE.get(db_path)
#         if cached is not None and cached[0] == fp:
#             return cached[1]

#     schema: Dict[str, List[str]] = {}
#     with _connect_readonly(db_path) as conn:
#         cur = conn.execute(
#             "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
#         )
#         tables = [r[0] for r in cur.fetchall() if r and isinstance(r[0], str)]
#         for table in tables:
#             table_l = table.lower()
#             try:
#                 cur = conn.execute(f'PRAGMA table_info("{table}")')
#                 cols = [row[1].lower() for row in cur.fetchall() if row and isinstance(row[1], str)]
#                 schema[table_l] = cols
#             except sqlite3.Error:
#                 schema[table_l] = []

#     with _SCHEMA_LOCK:
#         _SCHEMA_CACHE[db_path] = (fp, schema)
#     return schema

# def get_db_tables_and_columns(db_path: str) -> Tuple[Set[str], Set[str]]:
#     schema = get_table_to_columns(db_path)
#     tables = set(schema.keys())
#     columns: Set[str] = set()
#     for cols in schema.values():
#         columns.update(cols)
#     return tables, columns



import os
import sqlite3
import threading
from typing import Dict, List, Set, Tuple


# ===============================
# 🔥 SCHEMA TEXT (for prompting)
# ===============================
def get_schema(db_path):
    schema_map = get_table_to_columns(db_path)
    schema_text = ""

    for table, col_names in schema_map.items():
        schema_text += f"{table}({', '.join(col_names)})\n"

    return schema_text


# ===============================
# 🔥 CACHE + LOCK
# ===============================
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_CACHE: Dict[str, Tuple[str, Dict[str, List[str]]]] = {}


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


# ===============================
# 🔥 CORE: TABLE → COLUMNS
# ===============================
def get_table_to_columns(db_path: str) -> Dict[str, List[str]]:
    """
    Return mapping of table -> column names for the SQLite DB.
    Tables and columns are returned lowercased.
    """
    fp = _db_state_fingerprint(db_path)

    with _SCHEMA_LOCK:
        cached = _SCHEMA_CACHE.get(db_path)
        if cached is not None and cached[0] == fp:
            return cached[1]

    schema: Dict[str, List[str]] = {}

    with _connect_readonly(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )

        tables = [r[0] for r in cur.fetchall() if r and isinstance(r[0], str)]

        for table in tables:
            table_l = table.lower()

            try:
                cur = conn.execute(f'PRAGMA table_info("{table}")')

                cols = []
                for row in cur.fetchall():
                    col_name = row[1].lower()
                    col_type = str(row[2]).lower()

                    # include both name + type (stronger constraint)
                    cols.append(col_name)
                    cols.append(col_type)

                schema[table_l] = list(set(cols))  # remove duplicates

            except sqlite3.Error:
                schema[table_l] = []

    with _SCHEMA_LOCK:
        _SCHEMA_CACHE[db_path] = (fp, schema)

    return schema


# ===============================
# 🔥 TABLE + COLUMN SETS
# ===============================
def get_db_tables_and_columns(db_path: str) -> Tuple[Set[str], Set[str]]:
    schema = get_table_to_columns(db_path)

    tables = set(schema.keys())
    columns: Set[str] = set()

    for cols in schema.values():
        columns.update(cols)

    return tables, columns


# ===============================
# 🔥 NEW: FOREIGN KEYS (IMPORTANT)
# ===============================
def get_foreign_keys(db_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Returns list of foreign key relations:
    (table, column, ref_table, ref_column)
    """
    fks = []

    with _connect_readonly(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [r[0] for r in cur.fetchall()]

        for table in tables:
            try:
                cur = conn.execute(f'PRAGMA foreign_key_list("{table}")')

                for row in cur.fetchall():
                    fks.append((
                        table.lower(),
                        row[3].lower(),  # column
                        row[2].lower(),  # ref table
                        row[4].lower()   # ref column
                    ))

            except sqlite3.Error:
                continue

    return fks


# ===============================
# 🔥 FINAL: CONSTRAINT GRAPH
# ===============================
def get_constraint_graph(db_path: str):
    """
    Build full schema graph:
    - tables
    - columns
    - foreign key relations
    """
    tables, columns = get_db_tables_and_columns(db_path)
    fks = get_foreign_keys(db_path)

    return {
        "tables": tables,
        "columns": columns,
        "foreign_keys": fks
    }