import sqlite3
from pathlib import Path

class SchemaEncoder:
    def __init__(self, db_root):
        self.db_root = Path(db_root)

    def _get_db_path(self, db_id: str) -> Path:
        # Check standard Spider format (subfolder)
        path1 = self.db_root / db_id / f"{db_id}.sqlite"
        # Check flat format (no subfolder)
        path2 = self.db_root / f"{db_id}.sqlite"
        
        if path1.exists():
            return path1
        if path2.exists():
            return path2
            
        raise FileNotFoundError(f"unable to open database file. Looked in:\n1. {path1}\n2. {path2}")

    def structured_schema(self, db_id: str) -> str:
        db_path = self._get_db_path(db_id)
        
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        
        # Get all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall() if r[0] != "sqlite_sequence"]
        
        schema_str = ""
        for table in tables:
            cur.execute(f"PRAGMA table_info(`{table}`);")
            cols = [c[1] for c in cur.fetchall()]
            schema_str += f"{table} ({', '.join(cols)})\n"
            
        conn.close()
        return schema_str.strip()