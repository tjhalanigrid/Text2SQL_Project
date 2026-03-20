import sqlite3
import torch
from transformers.generation.logits_process import LogitsProcessor

# ==============================================================
# SCHEMA CONSTRAINT GRAPH & LOGITS PROCESSOR
# ==============================================================

class SchemaConstraintGraph:
    """Parses DB schema into an allowed vocabulary graph natively via SQLite."""
    def __init__(self, db_path: str):
        self.tables = []
        self.columns = []
        self._parse_sqlite(db_path)

    def _parse_sqlite(self, db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            self.tables = [row[0].lower() for row in cursor.fetchall() if row[0] != "sqlite_sequence"]
            
            for table in self.tables:
                cursor.execute(f"PRAGMA table_info('{table}');")
                self.columns.extend([row[1].lower() for row in cursor.fetchall()])
            
            conn.close()
            self.columns = list(set(self.columns))
        except Exception as e:
            print(f"Schema parsing error: {e}")


class SchemaConstrainedLogitsProcessor(LogitsProcessor):
    """Hooks into Hugging Face generation to block invalid schema tokens for single rollouts."""
    def __init__(self, tokenizer, schema_graph: SchemaConstraintGraph):
        self.tokenizer = tokenizer
        self.schema = schema_graph
        
        self.allowed_table_ids = self._get_token_ids(self.schema.tables)
        self.allowed_column_ids = self._get_token_ids(self.schema.columns)

    def _get_token_ids(self, word_list):
        allowed_ids = set()
        for word in word_list:
            for variant in [word, " " + word]:
                tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                if tokens:
                    allowed_ids.add(tokens[0])
        return allowed_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Safely handle sequence length to prevent index errors
        seq_len = min(5, input_ids.shape[-1])
        generated_text = self.tokenizer.decode(input_ids[0][-seq_len:], skip_special_tokens=True).lower()
        
        if generated_text.endswith("from ") or generated_text.endswith("join "):
            mask = torch.ones_like(scores, dtype=torch.bool)
            for token_id in self.allowed_table_ids:
                if token_id < scores.shape[-1]:
                    mask[0, token_id] = False
            scores = scores.masked_fill(mask, -float("inf"))
            
        elif generated_text.endswith("select ") or generated_text.endswith("where ") or generated_text.endswith("on "):
            mask = torch.ones_like(scores, dtype=torch.bool)
            
            # Allow the * character for COUNT(*) or SELECT *
            star_tokens = self.tokenizer.encode(" *", add_special_tokens=False)
            if star_tokens:
                mask[0, star_tokens[0]] = False
            
            for token_id in self.allowed_column_ids:
                if token_id < scores.shape[-1]:
                    mask[0, token_id] = False
            scores = scores.masked_fill(mask, -float("inf"))

        return scores