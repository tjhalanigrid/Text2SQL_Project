# from __future__ import annotations

# import re
# import threading
# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Sequence, Set

# import torch
# from transformers.generation.logits_process import LogitsProcessor

# from schema_constraints import ConstraintGraph, build_constraint_graph


# def _infer_expected_identifier(prefix_text: str) -> Optional[str]:
#     s = re.sub(r"\s+", " ", prefix_text.lower())
#     last_from = s.rfind(" from ")
#     last_join = s.rfind(" join ")
#     last_select = s.rfind(" select ")
#     last_where = s.rfind(" where ")
#     last_on = s.rfind(" on ")
#     last_group = s.rfind(" group by ")
#     last_order = s.rfind(" order by ")
#     last_having = s.rfind(" having ")

#     last_table_kw = max(last_from, last_join)
#     last_col_kw = max(last_select, last_where, last_on, last_group, last_order, last_having)

#     if last_table_kw < 0 and last_col_kw < 0:
#         return None
#     if last_table_kw > last_col_kw:
#         return "table"
#     if last_col_kw > last_table_kw:
#         return "column"
#     return None


# class _TrieNode:
#     __slots__ = ("children", "terminal")

#     def __init__(self) -> None:
#         self.children: Dict[int, _TrieNode] = {}
#         self.terminal: bool = False

#     def insert(self, token_ids: Sequence[int]) -> None:
#         node: _TrieNode = self
#         for tid in token_ids:
#             tid_i = int(tid)
#             nxt = node.children.get(tid_i)
#             if nxt is None:
#                 nxt = _TrieNode()
#                 node.children[tid_i] = nxt
#             node = nxt
#         node.terminal = True

#     def walk(self, prefix: Sequence[int]) -> Optional["_TrieNode"]:
#         node: _TrieNode = self
#         for tid in prefix:
#             node = node.children.get(int(tid))  # type: ignore[assignment]
#             if node is None:
#                 return None
#         return node


# def _encode_identifier(tokenizer, name: str) -> List[int]:
#     # Leading space encourages word-start markers (e.g. "Ġ" in RoBERTa BPE).
#     return tokenizer.encode(" " + name, add_special_tokens=False)


# def _build_trie(tokenizer, names: Iterable[str]) -> _TrieNode:
#     trie = _TrieNode()
#     for n in names:
#         if not n:
#             continue
#         try:
#             ids = _encode_identifier(tokenizer, n)
#         except Exception:
#             continue
#         if ids:
#             trie.insert(ids)
#     return trie


# def _allow_always_token_ids(tokenizer) -> torch.Tensor:
#     # Allow common delimiters so the model can end an identifier.
#     toks = [",", ")", "(", "\n", ".", ";"]
#     ids: Set[int] = set()
#     for t in toks:
#         try:
#             for tid in tokenizer.encode(t, add_special_tokens=False):
#                 ids.add(int(tid))
#         except Exception:
#             continue
#     return torch.tensor(sorted(ids), dtype=torch.long)


# @dataclass
# class _PerDbTokenSets:
#     fp: str
#     table_trie: _TrieNode
#     column_trie: _TrieNode
#     allow_always: torch.Tensor


# _DB_TOKENSET_LOCK = threading.Lock()
# _DB_TOKENSETS: Dict[str, _PerDbTokenSets] = {}


# def _per_db_tokensets(tokenizer, graph: ConstraintGraph) -> _PerDbTokenSets:
#     with _DB_TOKENSET_LOCK:
#         cached = _DB_TOKENSETS.get(graph.db_path)
#         if cached is not None and cached.fp == graph.fingerprint:
#             return cached

#     out = _PerDbTokenSets(
#         fp=graph.fingerprint,
#         table_trie=_build_trie(tokenizer, graph.tables),
#         column_trie=_build_trie(tokenizer, graph.all_columns),
#         allow_always=_allow_always_token_ids(tokenizer),
#     )
#     with _DB_TOKENSET_LOCK:
#         _DB_TOKENSETS[graph.db_path] = out
#     return out


# class BatchSchemaConstrainedLogitsProcessor(LogitsProcessor):
#     """
#     Schema-aware constrained decoding per item in the generation batch.
#     Uses a tokenizer-based trie so multi-token identifiers can be constrained.
#     """

#     def __init__(self, tokenizer, db_paths: Sequence[str], *, max_prefix_tokens: int = 48):
#         self.tokenizer = tokenizer
#         self.db_paths = list(db_paths)
#         self.max_prefix_tokens = int(max_prefix_tokens)

#         self._graphs = [build_constraint_graph(p) for p in self.db_paths]
#         self._token_sets = [_per_db_tokensets(tokenizer, g) for g in self._graphs]

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if input_ids.dim() != 2 or scores.dim() != 2:
#             return scores

#         batch = input_ids.size(0)
#         if batch != len(self._graphs):
#             return scores

#         for i in range(batch):
#             tail_ids = input_ids[i, -self.max_prefix_tokens :].tolist()
#             prefix_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
#             expected = _infer_expected_identifier(prefix_text)
#             if expected is None:
#                 continue

#             if expected == "table":
#                 m = re.search(r"(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)$", prefix_text, flags=re.I)
#                 partial = m.group(1) if m else None
#                 if partial is None and not re.search(r"(?:from|join)\s*$", prefix_text, flags=re.I):
#                     continue
#                 trie = self._token_sets[i].table_trie
#             else:
#                 m = re.search(
#                     r"(?:select|where|on|group by|order by|having)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)$",
#                     prefix_text,
#                     flags=re.I,
#                 )
#                 partial = m.group(1) if m else None
#                 if partial is None and not re.search(
#                     r"(?:select|where|on|group by|order by|having)\s*$", prefix_text, flags=re.I
#                 ):
#                     continue
#                 trie = self._token_sets[i].column_trie

#             if not partial:
#                 prefix_token_ids: List[int] = []
#             else:
#                 try:
#                     prefix_token_ids = _encode_identifier(self.tokenizer, partial)
#                 except Exception:
#                     continue

#             node = trie.walk(prefix_token_ids)
#             if node is None or node.terminal:
#                 continue

#             allowed_next = sorted(node.children.keys())
#             if not allowed_next:
#                 continue

#             allowed_next_t = torch.tensor(allowed_next, dtype=torch.long, device=scores.device)
#             allow_always = self._token_sets[i].allow_always.to(scores.device)
#             keep = torch.cat([allowed_next_t, allow_always]) if allow_always.numel() else allowed_next_t

#             kept_scores = scores[i, keep].clone()
#             scores[i, :] = -float("inf")
#             scores[i, keep] = kept_scores

#         return scores


# # Backwards-compatible names used elsewhere in the repo.
# class SchemaConstraintGraph:
#     def __init__(self, db_path: str):
#         self._graph = build_constraint_graph(db_path)
#         self.tables = sorted(self._graph.tables)
#         self.columns = sorted(self._graph.all_columns)


# class SchemaConstrainedLogitsProcessor(LogitsProcessor):
#     def __init__(self, tokenizer, schema_graph: SchemaConstraintGraph):
#         self._proc = BatchSchemaConstrainedLogitsProcessor(tokenizer, [schema_graph._graph.db_path])

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         return self._proc(input_ids, scores)




# from __future__ import annotations

# import re
# import threading
# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Sequence, Set

# import torch
# from transformers.generation.logits_process import LogitsProcessor

# from schema_constraints import ConstraintGraph, build_constraint_graph


# # =========================================================
# # 🔍 IDENTIFIER TYPE DETECTION
# # =========================================================
# def _infer_expected_identifier(prefix_text: str) -> Optional[str]:
#     s = re.sub(r"\s+", " ", prefix_text.lower())

#     last_from = s.rfind(" from ")
#     last_join = s.rfind(" join ")
#     last_select = s.rfind(" select ")
#     last_where = s.rfind(" where ")
#     last_on = s.rfind(" on ")
#     last_group = s.rfind(" group by ")
#     last_order = s.rfind(" order by ")
#     last_having = s.rfind(" having ")

#     last_table_kw = max(last_from, last_join)
#     last_col_kw = max(last_select, last_where, last_on, last_group, last_order, last_having)

#     if last_table_kw < 0 and last_col_kw < 0:
#         return None
#     if last_table_kw > last_col_kw:
#         return "table"
#     if last_col_kw > last_table_kw:
#         return "column"
#     return None


# # =========================================================
# # 🌳 TRIE STRUCTURE
# # =========================================================
# class _TrieNode:
#     __slots__ = ("children", "terminal")

#     def __init__(self) -> None:
#         self.children: Dict[int, _TrieNode] = {}
#         self.terminal: bool = False

#     def insert(self, token_ids: Sequence[int]) -> None:
#         node = self
#         for tid in token_ids:
#             tid = int(tid)
#             if tid not in node.children:
#                 node.children[tid] = _TrieNode()
#             node = node.children[tid]
#         node.terminal = True

#     def walk(self, prefix: Sequence[int]) -> Optional["_TrieNode"]:
#         node = self
#         for tid in prefix:
#             node = node.children.get(int(tid))
#             if node is None:
#                 return None
#         return node


# # =========================================================
# # 🔤 TOKEN ENCODING
# # =========================================================
# def _encode_identifier(tokenizer, name: str) -> List[int]:
#     return tokenizer.encode(" " + name, add_special_tokens=False)


# def _build_trie(tokenizer, names: Iterable[str]) -> _TrieNode:
#     trie = _TrieNode()
#     for name in names:
#         try:
#             ids = _encode_identifier(tokenizer, name)
#             if ids:
#                 trie.insert(ids)
#         except Exception:
#             continue
#     return trie


# def _allow_always_token_ids(tokenizer) -> torch.Tensor:
#     tokens = [",", ")", "(", ".", ";", "\n"]
#     ids: Set[int] = set()

#     for t in tokens:
#         try:
#             ids.update(tokenizer.encode(t, add_special_tokens=False))
#         except:
#             pass

#     return torch.tensor(sorted(ids), dtype=torch.long)


# # =========================================================
# # 📦 PER-DB CACHE
# # =========================================================
# @dataclass
# class _PerDbTokenSets:
#     fp: str
#     table_trie: _TrieNode
#     column_trie: _TrieNode
#     allow_always: torch.Tensor


# _DB_CACHE: Dict[str, _PerDbTokenSets] = {}
# _DB_LOCK = threading.Lock()


# def _per_db_tokensets(tokenizer, graph: ConstraintGraph) -> _PerDbTokenSets:
#     with _DB_LOCK:
#         cached = _DB_CACHE.get(graph.db_path)
#         if cached and cached.fp == graph.fingerprint:
#             return cached

#     obj = _PerDbTokenSets(
#         fp=graph.fingerprint,
#         table_trie=_build_trie(tokenizer, graph.tables),
#         column_trie=_build_trie(tokenizer, graph.all_columns),
#         allow_always=_allow_always_token_ids(tokenizer),
#     )

#     with _DB_LOCK:
#         _DB_CACHE[graph.db_path] = obj

#     return obj


# # =========================================================
# # 🚀 MAIN LOGITS PROCESSOR
# # =========================================================
# class BatchSchemaConstrainedLogitsProcessor(LogitsProcessor):
#     def __init__(self, tokenizer, db_paths: Sequence[str], max_prefix_tokens: int = 48):
#         self.tokenizer = tokenizer
#         self.db_paths = list(db_paths)
#         self.max_prefix_tokens = max_prefix_tokens

#         self._graphs = [build_constraint_graph(p) for p in db_paths]
#         self._token_sets = [_per_db_tokensets(tokenizer, g) for g in self._graphs]

#         # 📊 Metrics (IMPORTANT FOR REPORT)
#         self.total_steps = 0
#         self.constrained_steps = 0

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         batch = input_ids.size(0)

#         for i in range(batch):
#             self.total_steps += 1

#             tail_ids = input_ids[i, -self.max_prefix_tokens:].tolist()
#             prefix_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)

#             expected = _infer_expected_identifier(prefix_text)
#             if expected is None:
#                 continue

#             self.constrained_steps += 1

#             # =========================
#             # SELECT TRIE
#             # =========================
#             if expected == "table":
#                 trie = self._token_sets[i].table_trie
#             else:
#                 trie = self._token_sets[i].column_trie

#             # =========================
#             # PARTIAL TOKEN MATCH
#             # =========================
#             match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", prefix_text)
#             partial = match.group(1) if match else ""

#             try:
#                 prefix_ids = _encode_identifier(self.tokenizer, partial) if partial else []
#             except:
#                 continue

#             node = trie.walk(prefix_ids)
#             if node is None or node.terminal:
#                 continue

#             allowed_next = list(node.children.keys())
#             if not allowed_next:
#                 continue

#             allowed_next = torch.tensor(allowed_next, device=scores.device)
#             allow_always = self._token_sets[i].allow_always.to(scores.device)

#             keep = torch.cat([allowed_next, allow_always])

#             kept_scores = scores[i, keep].clone()
#             scores[i, :] = -float("inf")
#             scores[i, keep] = kept_scores

#         return scores

#     # =========================================================
#     # 📊 METRICS FOR REPORT
#     # =========================================================
#     def get_constraint_stats(self):
#         if self.total_steps == 0:
#             return 0
#         return self.constrained_steps / self.total_steps


# # =========================================================
# # 🔁 BACKWARD COMPATIBILITY
# # =========================================================
# class SchemaConstraintGraph:
#     def __init__(self, db_path: str):
#         self._graph = build_constraint_graph(db_path)
#         self.tables = sorted(self._graph.tables)
#         self.columns = sorted(self._graph.all_columns)


# class SchemaConstrainedLogitsProcessor(LogitsProcessor):
#     def __init__(self, tokenizer, schema_graph: SchemaConstraintGraph):
#         self.proc = BatchSchemaConstrainedLogitsProcessor(
#             tokenizer, [schema_graph._graph.db_path]
#         )

#     def __call__(self, input_ids, scores):
#         return self.proc(input_ids, scores)






# from __future__ import annotations

# import re
# import threading
# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Sequence, Set

# import torch
# from transformers.generation.logits_process import LogitsProcessor

# from schema_constraints import ConstraintGraph, build_constraint_graph


# def _infer_expected_identifier(prefix_text: str) -> Optional[str]:
#     s = re.sub(r"\s+", " ", prefix_text.lower())
#     last_from = s.rfind(" from ")
#     last_join = s.rfind(" join ")
#     last_select = s.rfind(" select ")
#     last_where = s.rfind(" where ")
#     last_on = s.rfind(" on ")
#     last_group = s.rfind(" group by ")
#     last_order = s.rfind(" order by ")
#     last_having = s.rfind(" having ")

#     last_table_kw = max(last_from, last_join)
#     last_col_kw = max(last_select, last_where, last_on, last_group, last_order, last_having)

#     if last_table_kw < 0 and last_col_kw < 0:
#         return None
#     if last_table_kw > last_col_kw:
#         return "table"
#     if last_col_kw > last_table_kw:
#         return "column"
#     return None


# class _TrieNode:
#     __slots__ = ("children", "terminal")

#     def __init__(self) -> None:
#         self.children: Dict[int, _TrieNode] = {}
#         self.terminal: bool = False

#     def insert(self, token_ids: Sequence[int]) -> None:
#         node: _TrieNode = self
#         for tid in token_ids:
#             tid_i = int(tid)
#             nxt = node.children.get(tid_i)
#             if nxt is None:
#                 nxt = _TrieNode()
#                 node.children[tid_i] = nxt
#             node = nxt
#         node.terminal = True

#     def walk(self, prefix: Sequence[int]) -> Optional["_TrieNode"]:
#         node: _TrieNode = self
#         for tid in prefix:
#             node = node.children.get(int(tid))  # type: ignore[assignment]
#             if node is None:
#                 return None
#         return node


# def _encode_identifier(tokenizer, name: str) -> List[int]:
#     # Leading space encourages word-start markers (e.g. "Ġ" in RoBERTa BPE).
#     return tokenizer.encode(" " + name, add_special_tokens=False)


# def _build_trie(tokenizer, names: Iterable[str]) -> _TrieNode:
#     trie = _TrieNode()
#     for n in names:
#         if not n:
#             continue
#         try:
#             ids = _encode_identifier(tokenizer, n)
#         except Exception:
#             continue
#         if ids:
#             trie.insert(ids)
#     return trie


# def _allow_always_token_ids(tokenizer) -> torch.Tensor:
#     # Allow common delimiters so the model can end an identifier.
#     toks = [",", ")", "(", "\n", ".", ";"]
#     ids: Set[int] = set()
#     for t in toks:
#         try:
#             for tid in tokenizer.encode(t, add_special_tokens=False):
#                 ids.add(int(tid))
#         except Exception:
#             continue
#     return torch.tensor(sorted(ids), dtype=torch.long)


# @dataclass
# class _PerDbTokenSets:
#     fp: str
#     table_trie: _TrieNode
#     column_trie: _TrieNode
#     allow_always: torch.Tensor


# _DB_TOKENSET_LOCK = threading.Lock()
# _DB_TOKENSETS: Dict[str, _PerDbTokenSets] = {}


# def _per_db_tokensets(tokenizer, graph: ConstraintGraph) -> _PerDbTokenSets:
#     with _DB_TOKENSET_LOCK:
#         cached = _DB_TOKENSETS.get(graph.db_path)
#         if cached is not None and cached.fp == graph.fingerprint:
#             return cached

#     out = _PerDbTokenSets(
#         fp=graph.fingerprint,
#         table_trie=_build_trie(tokenizer, graph.tables),
#         column_trie=_build_trie(tokenizer, graph.all_columns),
#         allow_always=_allow_always_token_ids(tokenizer),
#     )
#     with _DB_TOKENSET_LOCK:
#         _DB_TOKENSETS[graph.db_path] = out
#     return out


# class BatchSchemaConstrainedLogitsProcessor(LogitsProcessor):
#     """
#     Schema-aware constrained decoding per item in the generation batch.
#     Uses a tokenizer-based trie so multi-token identifiers can be constrained.
#     """

#     def __init__(self, tokenizer, db_paths: Sequence[str], *, max_prefix_tokens: int = 48):
#         self.tokenizer = tokenizer
#         self.db_paths = list(db_paths)
#         self.max_prefix_tokens = int(max_prefix_tokens)

#         self._graphs = [build_constraint_graph(p) for p in self.db_paths]
#         self._token_sets = [_per_db_tokensets(tokenizer, g) for g in self._graphs]

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if input_ids.dim() != 2 or scores.dim() != 2:
#             return scores

#         batch = input_ids.size(0)
#         if batch != len(self._graphs):
#             return scores

#         for i in range(batch):
#             tail_ids = input_ids[i, -self.max_prefix_tokens :].tolist()
#             prefix_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
#             expected = _infer_expected_identifier(prefix_text)
#             if expected is None:
#                 continue

#             if expected == "table":
#                 m = re.search(r"(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)$", prefix_text, flags=re.I)
#                 partial = m.group(1) if m else None
#                 if partial is None and not re.search(r"(?:from|join)\s*$", prefix_text, flags=re.I):
#                     continue
#                 trie = self._token_sets[i].table_trie
#             else:
#                 m = re.search(
#                     r"(?:select|where|on|group by|order by|having)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)$",
#                     prefix_text,
#                     flags=re.I,
#                 )
#                 partial = m.group(1) if m else None
#                 if partial is None and not re.search(
#                     r"(?:select|where|on|group by|order by|having)\s*$", prefix_text, flags=re.I
#                 ):
#                     continue
#                 trie = self._token_sets[i].column_trie

#             if not partial:
#                 prefix_token_ids: List[int] = []
#             else:
#                 try:
#                     prefix_token_ids = _encode_identifier(self.tokenizer, partial)
#                 except Exception:
#                     continue

#             node = trie.walk(prefix_token_ids)
#             if node is None or node.terminal:
#                 continue

#             allowed_next = sorted(node.children.keys())
#             if not allowed_next:
#                 continue

#             allowed_next_t = torch.tensor(allowed_next, dtype=torch.long, device=scores.device)
#             allow_always = self._token_sets[i].allow_always.to(scores.device)
#             keep = torch.cat([allowed_next_t, allow_always]) if allow_always.numel() else allowed_next_t

#             kept_scores = scores[i, keep].clone()
#             scores[i, :] = -float("inf")
#             scores[i, keep] = kept_scores

#         return scores


# # Backwards-compatible names used elsewhere in the repo.
# class SchemaConstraintGraph:
#     def __init__(self, db_path: str):
#         self._graph = build_constraint_graph(db_path)
#         self.tables = sorted(self._graph.tables)
#         self.columns = sorted(self._graph.all_columns)


# class SchemaConstrainedLogitsProcessor(LogitsProcessor):
#     def __init__(self, tokenizer, schema_graph: SchemaConstraintGraph):
#         self._proc = BatchSchemaConstrainedLogitsProcessor(tokenizer, [schema_graph._graph.db_path])

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         return self._proc(input_ids, scores)




# from __future__ import annotations

# import re
# import threading
# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Sequence, Set

# import torch
# from transformers.generation.logits_process import LogitsProcessor

# from schema_constraints import ConstraintGraph, build_constraint_graph


# # =========================================================
# # 🔍 IDENTIFIER TYPE DETECTION
# # =========================================================
# def _infer_expected_identifier(prefix_text: str) -> Optional[str]:
#     s = re.sub(r"\s+", " ", prefix_text.lower())

#     last_from = s.rfind(" from ")
#     last_join = s.rfind(" join ")
#     last_select = s.rfind(" select ")
#     last_where = s.rfind(" where ")
#     last_on = s.rfind(" on ")
#     last_group = s.rfind(" group by ")
#     last_order = s.rfind(" order by ")
#     last_having = s.rfind(" having ")

#     last_table_kw = max(last_from, last_join)
#     last_col_kw = max(last_select, last_where, last_on, last_group, last_order, last_having)

#     if last_table_kw < 0 and last_col_kw < 0:
#         return None
#     if last_table_kw > last_col_kw:
#         return "table"
#     if last_col_kw > last_table_kw:
#         return "column"
#     return None


# # =========================================================
# # 🌳 TRIE STRUCTURE
# # =========================================================
# class _TrieNode:
#     __slots__ = ("children", "terminal")

#     def __init__(self) -> None:
#         self.children: Dict[int, _TrieNode] = {}
#         self.terminal: bool = False

#     def insert(self, token_ids: Sequence[int]) -> None:
#         node = self
#         for tid in token_ids:
#             tid = int(tid)
#             if tid not in node.children:
#                 node.children[tid] = _TrieNode()
#             node = node.children[tid]
#         node.terminal = True

#     def walk(self, prefix: Sequence[int]) -> Optional["_TrieNode"]:
#         node = self
#         for tid in prefix:
#             node = node.children.get(int(tid))
#             if node is None:
#                 return None
#         return node


# # =========================================================
# # 🔤 TOKEN ENCODING
# # =========================================================
# def _encode_identifier(tokenizer, name: str) -> List[int]:
#     return tokenizer.encode(" " + name, add_special_tokens=False)


# def _build_trie(tokenizer, names: Iterable[str]) -> _TrieNode:
#     trie = _TrieNode()
#     for name in names:
#         try:
#             ids = _encode_identifier(tokenizer, name)
#             if ids:
#                 trie.insert(ids)
#         except Exception:
#             continue
#     return trie


# def _allow_always_token_ids(tokenizer) -> torch.Tensor:
#     tokens = [",", ")", "(", ".", ";", "\n"]
#     ids: Set[int] = set()

#     for t in tokens:
#         try:
#             ids.update(tokenizer.encode(t, add_special_tokens=False))
#         except:
#             pass

#     return torch.tensor(sorted(ids), dtype=torch.long)


# # =========================================================
# # 📦 PER-DB CACHE
# # =========================================================
# @dataclass
# class _PerDbTokenSets:
#     fp: str
#     table_trie: _TrieNode
#     column_trie: _TrieNode
#     allow_always: torch.Tensor


# _DB_CACHE: Dict[str, _PerDbTokenSets] = {}
# _DB_LOCK = threading.Lock()


# def _per_db_tokensets(tokenizer, graph: ConstraintGraph) -> _PerDbTokenSets:
#     with _DB_LOCK:
#         cached = _DB_CACHE.get(graph.db_path)
#         if cached and cached.fp == graph.fingerprint:
#             return cached

#     obj = _PerDbTokenSets(
#         fp=graph.fingerprint,
#         table_trie=_build_trie(tokenizer, graph.tables),
#         column_trie=_build_trie(tokenizer, graph.all_columns),
#         allow_always=_allow_always_token_ids(tokenizer),
#     )

#     with _DB_LOCK:
#         _DB_CACHE[graph.db_path] = obj

#     return obj


# # =========================================================
# # 🚀 MAIN LOGITS PROCESSOR
# # =========================================================
# class BatchSchemaConstrainedLogitsProcessor(LogitsProcessor):
#     def __init__(self, tokenizer, db_paths: Sequence[str], max_prefix_tokens: int = 48):
#         self.tokenizer = tokenizer
#         self.db_paths = list(db_paths)
#         self.max_prefix_tokens = max_prefix_tokens

#         self._graphs = [build_constraint_graph(p) for p in db_paths]
#         self._token_sets = [_per_db_tokensets(tokenizer, g) for g in self._graphs]

#         # 📊 Metrics (IMPORTANT FOR REPORT)
#         self.total_steps = 0
#         self.constrained_steps = 0

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         batch = input_ids.size(0)

#         for i in range(batch):
#             self.total_steps += 1

#             tail_ids = input_ids[i, -self.max_prefix_tokens:].tolist()
#             prefix_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)

#             expected = _infer_expected_identifier(prefix_text)
#             if expected is None:
#                 continue

#             self.constrained_steps += 1

#             # =========================
#             # SELECT TRIE
#             # =========================
#             if expected == "table":
#                 trie = self._token_sets[i].table_trie
#             else:
#                 trie = self._token_sets[i].column_trie

#             # =========================
#             # PARTIAL TOKEN MATCH
#             # =========================
#             match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", prefix_text)
#             partial = match.group(1) if match else ""

#             try:
#                 prefix_ids = _encode_identifier(self.tokenizer, partial) if partial else []
#             except:
#                 continue

#             node = trie.walk(prefix_ids)
#             if node is None or node.terminal:
#                 continue

#             allowed_next = list(node.children.keys())
#             if not allowed_next:
#                 continue

#             allowed_next = torch.tensor(allowed_next, device=scores.device)
#             allow_always = self._token_sets[i].allow_always.to(scores.device)

#             keep = torch.cat([allowed_next, allow_always])

#             kept_scores = scores[i, keep].clone()
#             scores[i, :] = -float("inf")
#             scores[i, keep] = kept_scores

#         return scores

#     # =========================================================
#     # 📊 METRICS FOR REPORT
#     # =========================================================
#     def get_constraint_stats(self):
#         if self.total_steps == 0:
#             return 0
#         return self.constrained_steps / self.total_steps


# # =========================================================
# # 🔁 BACKWARD COMPATIBILITY
# # =========================================================
# class SchemaConstraintGraph:
#     def __init__(self, db_path: str):
#         self._graph = build_constraint_graph(db_path)
#         self.tables = sorted(self._graph.tables)
#         self.columns = sorted(self._graph.all_columns)


# class SchemaConstrainedLogitsProcessor(LogitsProcessor):
#     def __init__(self, tokenizer, schema_graph: SchemaConstraintGraph):
#         self.proc = BatchSchemaConstrainedLogitsProcessor(
#             tokenizer, [schema_graph._graph.db_path]
#         )

#     def __call__(self, input_ids, scores):
#         return self.proc(input_ids, scores)








# ********* after task 3 

import re
import threading
from functools import lru_cache

import torch
from transformers import LogitsProcessor

from src.schema_utils import get_constraint_graph


_TOKEN_CACHE_LOCK = threading.Lock()
_TOKEN_ID_CACHE = {}  # (id(tokenizer), db_path) -> (allowed_ids_tensor, always_allow_ids_tensor)


def _encode_variants(tokenizer, text: str) -> list[int]:
    ids: list[int] = []
    for variant in (text, " " + text):
        try:
            ids.extend(tokenizer.encode(variant, add_special_tokens=False))
        except Exception:
            continue
    # de-dup while keeping order
    seen = set()
    out = []
    for i in ids:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(int(i))
    return out


def _always_allow_ids(tokenizer) -> list[int]:
    """
    Tokens we should never block, otherwise decoding can get stuck or generate garbage:
    - EOS/PAD
    - punctuation/operators needed for SQL formatting
    - digits/quotes
    """
    ids: list[int] = []
    for special in [getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)]:
        if special is not None:
            ids.append(int(special))

    # Common SQL punctuation/operators
    pieces = [
        " ", "\n", "\t",
        ",", ".", "(", ")", ";",
        "=", "!=", "<>", "<", ">", "<=", ">=",
        "*", "+", "-", "/", "%",
        "'", '"',
    ]
    for p in pieces:
        ids.extend(_encode_variants(tokenizer, p))

    # digits
    for d in "0123456789":
        ids.extend(_encode_variants(tokenizer, d))

    seen = set()
    out = []
    for i in ids:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(int(i))
    return out


def _infer_expected_identifier_tail(tail_text: str):
    """
    Returns ("table"|"column", partial_or_empty) if the tail looks like it's currently
    emitting a table/column identifier. Otherwise returns None.
    """
    t = re.sub(r"\s+", " ", (tail_text or "")).lower()

    m = re.search(r"(?:from|join)\s+([a-z_][a-z0-9_]*)?$", t)
    if m:
        partial = m.group(1) or ""
        # ensure we are actually after keyword (not elsewhere)
        if re.search(r"(?:from|join)\s*$", t) or partial:
            return "table", partial

    m = re.search(
        r"(?:select|where|on|group by|order by|having)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)?$",
        t,
    )
    if m:
        partial = m.group(1) or ""
        if re.search(r"(?:select|where|on|group by|order by|having)\s*$", t) or partial:
            return "column", partial

    return None


class SchemaConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, db_path):
        self.tokenizer = tokenizer

        graph = get_constraint_graph(db_path)

        key = (id(tokenizer), str(db_path))
        with _TOKEN_CACHE_LOCK:
            cached = _TOKEN_ID_CACHE.get(key)
        if cached is None:
            allowed_tokens = set(graph.get("tables", set())) | set(graph.get("columns", set()))

            sql_keywords = {
                "select", "from", "where", "join", "on",
                "group", "by", "order", "limit", "having",
                "and", "or", "desc", "asc",
                "count", "avg", "min", "max", "sum",
                "distinct", "as", "in", "like", "between",
                "is", "null",
            }
            allowed_tokens |= sql_keywords

            allowed_ids: list[int] = []
            for tok in sorted(allowed_tokens):
                allowed_ids.extend(_encode_variants(tokenizer, tok))
            always_ids = _always_allow_ids(tokenizer)

            allowed_ids_t = torch.tensor(sorted(set(allowed_ids)), dtype=torch.long)
            always_ids_t = torch.tensor(sorted(set(always_ids)), dtype=torch.long)
            cached = (allowed_ids_t, always_ids_t)
            with _TOKEN_CACHE_LOCK:
                _TOKEN_ID_CACHE[key] = cached

        self._allowed_ids_t, self._always_ids_t = cached

    def __call__(self, input_ids, scores):
        # Decode only a tail window for speed (beam search calls this a lot).
        try:
            tail_ids = input_ids[0][-128:]
        except Exception:
            tail_ids = input_ids[0]
        tail = self.tokenizer.decode(tail_ids, skip_special_tokens=True)

        inferred = _infer_expected_identifier_tail(tail)
        if inferred is None:
            return scores

        keep = torch.cat([self._allowed_ids_t.to(scores.device), self._always_ids_t.to(scores.device)])
        if keep.numel() == 0:
            return scores

        kept_scores = scores[:, keep].clone()
        scores[:] = -float("inf")
        scores[:, keep] = kept_scores
        return scores
