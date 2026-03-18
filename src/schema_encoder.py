# import sqlite3


# class SchemaEncoder:

#     def __init__(self, db_root):
#         self.db_root = db_root

#     def get_tables_and_columns(self, db_id):
#         db_path = self.db_root / db_id / f"{db_id}.sqlite"
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         tables = cursor.execute(
#             "SELECT name FROM sqlite_master WHERE type='table';"
#         ).fetchall()

#         schema = {}

#         for (table,) in tables:
#             cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
#             col_names = [c[1] for c in cols]
#             schema[table] = col_names

#         conn.close()
#         return schema

#     # -----------------------------------
#     # Strategy 1: Structured (current)
#     # -----------------------------------
#     def structured_schema(self, db_id):
#         schema = self.get_tables_and_columns(db_id)

#         lines = []
#         for table, cols in schema.items():
#             lines.append(f"{table}({', '.join(cols)})")

#         return "\n".join(lines)

#     # -----------------------------------
#     # Strategy 2: Natural Language
#     # -----------------------------------
#     def natural_language_schema(self, db_id):
#         schema = self.get_tables_and_columns(db_id)

#         lines = []
#         for table, cols in schema.items():
#             col_text = ", ".join(cols)
#             lines.append(f"The table '{table}' contains the columns: {col_text}.")

#         return "\n".join(lines)


import sqlite3
import re

def build_schema_graph(schema_text):
    """
    Parses a structured schema text string into a dictionary graph.
    Matches formats like: table_name(col1, col2, col3)
    """
    tables = {}
    
    for match in re.findall(r'(\w+)\s*\((.*?)\)', schema_text):
        table = match[0]
        # Extracts just the column names, ignoring potential types or constraints
        cols = [c.strip().split()[0] for c in match[1].split(",")]
        tables[table] = cols
    
    return tables


class SchemaEncoder:

    def __init__(self, db_root):
        self.db_root = db_root

    def get_tables_and_columns(self, db_id):
        # Assuming db_root is a pathlib.Path object based on the syntax
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        schema = {}

        for (table,) in tables:
            cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
            col_names = [c[1] for c in cols]
            schema[table] = col_names

        conn.close()
        return schema

    # -----------------------------------
    # Strategy 1: Structured (current)
    # -----------------------------------
    def structured_schema(self, db_id):
        schema = self.get_tables_and_columns(db_id)

        lines = []
        for table, cols in schema.items():
            lines.append(f"{table}({', '.join(cols)})")

        return "\n".join(lines)

    # -----------------------------------
    # Strategy 2: Natural Language
    # -----------------------------------
    def natural_language_schema(self, db_id):
        schema = self.get_tables_and_columns(db_id)

        lines = []
        for table, cols in schema.items():
            col_text = ", ".join(cols)
            lines.append(f"The table '{table}' contains the columns: {col_text}.")

        return "\n".join(lines)