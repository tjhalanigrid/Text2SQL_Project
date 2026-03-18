# import re
# from pathlib import Path
# from typing import Optional, Set, Tuple

# from schema_utils import get_db_tables_and_columns, get_table_to_columns

# class SQLValidator:

#     def __init__(self, db_root):
#         self.db_root = Path(db_root)

#     # ---------------------------
#     # Load schema
#     # ---------------------------
#     def load_schema(self, db_id):
#         db_path = self.db_root / db_id / f"{db_id}.sqlite"
#         return get_table_to_columns(str(db_path))


#     # ---------------------------
#     # Basic syntax check
#     # ---------------------------
#     def basic_structure_valid(self, sql):
#         s = sql.lower()

#         if "select" not in s or "from" not in s:
#             return False, "Missing SELECT or FROM"

#         if len(s.split()) < 4:
#             return False, "Too short to be SQL"

#         return True, None


#     # ---------------------------
#     # Extract identifiers
#     # ---------------------------
#     def extract_identifiers(self, sql):
#         tokens = re.findall(r"[A-Za-z_]+", sql.lower())
#         return set(tokens)


#     # ---------------------------
#     # Table validation
#     # ---------------------------
#     def validate_tables(self, sql, schema):
#         words = self.extract_identifiers(sql)
#         tables = set(schema.keys())

#         used_tables = [w for w in words if w in tables]

#         if not used_tables:
#             return False, "No valid table used"

#         return True, None


#     # ---------------------------
#     # Column validation
#     # ---------------------------
#     def validate_columns(self, sql, schema):
#         words = self.extract_identifiers(sql)

#         valid_columns = set()
#         for cols in schema.values():
#             valid_columns.update(cols)

#         # ignore SQL keywords
#         keywords = {
#             "select","from","where","join","on","group","by",
#             "order","limit","count","sum","avg","min","max",
#             "and","or","in","like","distinct","asc","desc"
#         }

#         invalid = []
#         for w in words:
#             if w not in valid_columns and w not in schema and w not in keywords:
#                 if not w.isdigit():
#                     invalid.append(w)

#         # allow small hallucinations but block many
#         if len(invalid) > 3:
#             return False, f"Too many unknown identifiers: {invalid[:5]}"

#         return True, None


#     # ---------------------------
#     # Dangerous query protection
#     # ---------------------------
#     def block_dangerous(self, sql):
#         bad = ["drop", "delete", "update", "insert", "alter"]

#         s = sql.lower()
#         for b in bad:
#             if b in s:
#                 return False, f"Dangerous keyword detected: {b}"

#         return True, None


#     # ---------------------------
#     # Main validation
#     # ---------------------------
#     def validate(self, sql, db_id):

#         schema = self.load_schema(db_id)

#         checks = [
#             self.block_dangerous(sql),
#             self.basic_structure_valid(sql),
#             self.validate_tables(sql, schema),
#             self.validate_columns(sql, schema),
#         ]

#         for ok, msg in checks:
#             if not ok:
#                 return False, msg

#         return True, None


# _VALIDATION_CACHE = {}
# _VALIDATION_CACHE_MAX = 100_000


# def _db_state_fingerprint(db_path: str) -> str:
#     try:
#         st = Path(db_path).stat()
#         return f"{st.st_mtime_ns}:{st.st_size}"
#     except OSError:
#         return "missing"


# def _extract_referenced_tables(sql: str) -> Set[str]:
#     # Best-effort: FROM/JOIN targets (unquoted identifiers).
#     tokens = re.findall(r"\b(from|join)\s+([a-zA-Z_][\w$]*)", sql, flags=re.I)
#     return {t[1].lower() for t in tokens if t and len(t) > 1}


# def validate_sql_schema(sql: str, db_path: str) -> Tuple[bool, Optional[str]]:
#     """
#     Strict schema validation for reward computation.
#     - References must resolve to real tables/columns in the target DB.
#     - Returns (ok, message). On failure, message is a short reason.
#     """
#     fp = _db_state_fingerprint(db_path)
#     key = f"{fp}|{sql}"
#     cached = _VALIDATION_CACHE.get(key)
#     if cached is not None:
#         return cached

#     valid_tables, valid_columns = get_db_tables_and_columns(db_path)

#     referenced_tables = _extract_referenced_tables(sql)
#     unknown_tables = sorted(t for t in referenced_tables if t not in valid_tables)
#     if unknown_tables:
#         out = (False, f"Unknown table(s): {unknown_tables[:5]}")
#         if len(_VALIDATION_CACHE) >= _VALIDATION_CACHE_MAX:
#             _VALIDATION_CACHE.clear()
#         _VALIDATION_CACHE[key] = out
#         return out

#     # Column-level correctness is hard to do reliably with regex alone; rely on SQLite compilation.
#     # This does not execute the query, but will fail for unknown tables/columns.
#     try:
#         import sqlite3  # local import to keep module lightweight

#         uri = f"file:{Path(db_path).resolve()}?mode=ro"
#         conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
#         try:
#             conn.execute("PRAGMA query_only = ON;")
#             conn.execute("PRAGMA foreign_keys = ON;")
#             conn.execute(f"EXPLAIN QUERY PLAN {sql}")
#         finally:
#             conn.close()
#     except Exception as e:
#         msg = str(e).lower()
#         if "no such table" in msg:
#             out = (False, "Unknown table")
#         elif "no such column" in msg:
#             out = (False, "Unknown column")
#         else:
#             out = (False, "Schema validation failed")

#         if len(_VALIDATION_CACHE) >= _VALIDATION_CACHE_MAX:
#             _VALIDATION_CACHE.clear()
#         _VALIDATION_CACHE[key] = out
#         return out

#     out = (True, None)
#     if len(_VALIDATION_CACHE) >= _VALIDATION_CACHE_MAX:
#         _VALIDATION_CACHE.clear()
#     _VALIDATION_CACHE[key] = out
#     return out





import re
from pathlib import Path
from typing import Optional, Set, Tuple, Dict, List

from src.schema_utils import get_db_tables_and_columns, get_table_to_columns, get_constraint_graph


class SQLValidator:

    def __init__(self, db_root):
        self.db_root = Path(db_root)

    # ---------------------------
    # Load schema
    # ---------------------------
    def load_schema(self, db_id):
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        return get_table_to_columns(str(db_path))

    # ---------------------------
    # Basic syntax check
    # ---------------------------
    def basic_structure_valid(self, sql):
        s = sql.lower()

        if "select" not in s or "from" not in s:
            return False, "Missing SELECT or FROM"

        if len(s.split()) < 4:
            return False, "Too short to be SQL"

        return True, None

    # ---------------------------
    # Extract identifiers
    # ---------------------------
    def extract_identifiers(self, sql):
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql.lower())
        return set(tokens)

    # ---------------------------
    # Table validation
    # ---------------------------
    def validate_tables(self, sql, schema):
        words = self.extract_identifiers(sql)
        tables = set(schema.keys())

        used_tables = [w for w in words if w in tables]

        if not used_tables:
            return False, "No valid table used"

        return True, None

    # ---------------------------
    # Column validation
    # ---------------------------
    def validate_columns(self, sql, schema):
        words = self.extract_identifiers(sql)

        valid_columns = set()
        for cols in schema.values():
            valid_columns.update(cols)

        keywords = {
            "select","from","where","join","on","group","by",
            "order","limit","count","sum","avg","min","max",
            "and","or","in","like","distinct","asc","desc",
            "having","as","inner","left","right","outer"
        }

        invalid = []
        for w in words:
            if (
                w not in valid_columns
                and w not in schema
                and w not in keywords
                and not w.isdigit()
            ):
                invalid.append(w)

        # stricter than before
        if len(invalid) > 2:
            return False, f"Unknown identifiers: {invalid[:5]}"

        return True, None

    # ---------------------------
    # Dangerous query protection
    # ---------------------------
    def block_dangerous(self, sql):
        bad = ["drop", "delete", "update", "insert", "alter"]

        s = sql.lower()
        for b in bad:
            if b in s:
                return False, f"Dangerous keyword detected: {b}"

        return True, None

    # ---------------------------
    # FK-aware JOIN validation (NEW 🔥)
    # ---------------------------
    def validate_joins(self, db_id):
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        graph = get_constraint_graph(str(db_path))

        # not strict enforcement, just check FK existence
        if len(graph["foreign_keys"]) == 0:
            return True, None

        return True, None  # placeholder (safe for now)

    # ---------------------------
    # Main validation
    # ---------------------------
    def validate(self, sql, db_id):

        schema = self.load_schema(db_id)

        checks = [
            self.block_dangerous(sql),
            self.basic_structure_valid(sql),
            self.validate_tables(sql, schema),
            self.validate_columns(sql, schema),
        ]

        for ok, msg in checks:
            if not ok:
                return False, msg

        return True, None


# ===============================
# 🔥 FAST SCHEMA VALIDATION (REWARD)
# ===============================
_VALIDATION_CACHE = {}
_VALIDATION_CACHE_MAX = 100_000


def _db_state_fingerprint(db_path: str) -> str:
    try:
        st = Path(db_path).stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


def _extract_referenced_tables(sql: str) -> Set[str]:
    tokens = re.findall(r"\b(from|join)\s+([a-zA-Z_][\w$]*)", sql, flags=re.I)
    return {t[1].lower() for t in tokens if t and len(t) > 1}


def validate_sql_schema(sql: str, db_path: str) -> Tuple[bool, Optional[str]]:
    """
    STRICT schema validation (Task 3 core)
    """

    fp = _db_state_fingerprint(db_path)
    key = f"{fp}|{sql}"

    cached = _VALIDATION_CACHE.get(key)
    if cached is not None:
        return cached

    valid_tables, valid_columns = get_db_tables_and_columns(db_path)

    # ---------------------------
    # Table validation
    # ---------------------------
    referenced_tables = _extract_referenced_tables(sql)

    unknown_tables = [t for t in referenced_tables if t not in valid_tables]

    if unknown_tables:
        out = (False, f"Unknown table(s): {unknown_tables[:3]}")
        _VALIDATION_CACHE[key] = out
        return out

    # ---------------------------
    # Column validation via SQLite planner
    # ---------------------------
    try:
        import sqlite3

        uri = f"file:{Path(db_path).resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

        try:
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("PRAGMA foreign_keys = ON;")

            # 🔥 Key idea: no execution, only planning
            conn.execute(f"EXPLAIN QUERY PLAN {sql}")

        finally:
            conn.close()

    except Exception as e:
        msg = str(e).lower()

        if "no such table" in msg:
            out = (False, "Unknown table")
        elif "no such column" in msg:
            out = (False, "Unknown column")
        else:
            out = (False, "Invalid SQL")

        _VALIDATION_CACHE[key] = out
        return out

    out = (True, None)
    _VALIDATION_CACHE[key] = out
    return out