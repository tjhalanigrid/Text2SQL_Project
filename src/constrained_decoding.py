from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch
from transformers.generation.logits_process import LogitsProcessor

from schema_constraints import ConstraintGraph, build_constraint_graph


def _infer_expected_identifier(prefix_text: str) -> Optional[str]:
    """
    Best-effort grammar state:
    - After FROM/JOIN -> expect table
    - After SELECT/WHERE/ON/GROUP BY/ORDER BY/HAVING -> expect column or expression
    Returns: "table" | "column" | None
    """
    s = prefix_text.lower()
    s = re.sub(r"\s+", " ", s)
    last_from = s.rfind(" from ")
    last_join = s.rfind(" join ")
    last_select = s.rfind(" select ")
    last_where = s.rfind(" where ")
    last_on = s.rfind(" on ")
    last_group = s.rfind(" group by ")
    last_order = s.rfind(" order by ")
    last_having = s.rfind(" having ")

    last_table_kw = max(last_from, last_join)
    last_col_kw = max(last_select, last_where, last_on, last_group, last_order, last_having)

    if last_table_kw < 0 and last_col_kw < 0:
        return None
    if last_table_kw > last_col_kw:
        return "table"
    if last_col_kw > last_table_kw:
        return "column"
    return None


@dataclass
class _PerDbTokenSets:
    fp: str
    table_trie: "_TrieNode"
    column_trie: "_TrieNode"
    allow_always: torch.Tensor


class _TrieNode:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: Dict[int, _TrieNode] = {}
        self.terminal: bool = False

    def insert(self, token_ids: Sequence[int]) -> None:
        node: _TrieNode = self
        for tid in token_ids:
            nxt = node.children.get(tid)
            if nxt is None:
                nxt = _TrieNode()
                node.children[tid] = nxt
            node = nxt
        node.terminal = True

    def walk(self, prefix: Sequence[int]) -> Optional["_TrieNode"]:
        node: _TrieNode = self
        for tid in prefix:
            node = node.children.get(int(tid))  # type: ignore[assignment]
            if node is None:
                return None
        return node


_DB_TOKENSET_LOCK = threading.Lock()
_DB_TOKENSETS: Dict[str, _PerDbTokenSets] = {}


def _encode_identifier(tokenizer, name: str) -> List[int]:
    # Leading space encourages word-start markers (Ġ / ▁) for BPE/SP models.
    return tokenizer.encode(" " + name, add_special_tokens=False)


def _build_trie(tokenizer, names: Iterable[str]) -> _TrieNode:
    trie = _TrieNode()
    for n in names:
        if not n:
            continue
        try:
            ids = _encode_identifier(tokenizer, n)
        except Exception:
            continue
        if not ids:
            continue
        trie.insert(ids)
    return trie


def _allow_always_token_ids(tokenizer) -> torch.Tensor:
    # Tokens we never want to block while constraining identifiers.
    # This allows the model to end an identifier (comma/paren/newline) if needed.
    toks = [",", ")", "(", "\n", ".", ";"]
    ids: Set[int] = set()
    for t in toks:
        try:
            enc = tokenizer.encode(t, add_special_tokens=False)
            for tid in enc:
                ids.add(int(tid))
        except Exception:
            continue
    return torch.tensor(sorted(ids), dtype=torch.long)


def _per_db_tokensets(tokenizer, graph: ConstraintGraph) -> _PerDbTokenSets:
    with _DB_TOKENSET_LOCK:
        cached = _DB_TOKENSETS.get(graph.db_path)
        if cached is not None and cached.fp == graph.fingerprint:
            return cached

    out = _PerDbTokenSets(
        fp=graph.fingerprint,
        table_trie=_build_trie(tokenizer, graph.tables),
        column_trie=_build_trie(tokenizer, graph.all_columns),
        allow_always=_allow_always_token_ids(tokenizer),
    )
    with _DB_TOKENSET_LOCK:
        _DB_TOKENSETS[graph.db_path] = out
    return out


class BatchSchemaConstrainedLogitsProcessor(LogitsProcessor):
    """
    Applies schema-aware constraints per item in the generation batch.
    This is intentionally conservative: it only constrains when the grammar
    state is confident and the schema names are representable as single tokens.
    """

    def __init__(self, tokenizer, db_paths: Sequence[str], *, max_prefix_tokens: int = 48):
        self.tokenizer = tokenizer
        self.db_paths = list(db_paths)
        self.max_prefix_tokens = int(max_prefix_tokens)

        self._graphs = [build_constraint_graph(p) for p in self.db_paths]
        self._token_sets = [_per_db_tokensets(tokenizer, g) for g in self._graphs]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.dim() != 2 or scores.dim() != 2:
            return scores

        batch = input_ids.size(0)
        if batch != len(self._graphs):
            # Safety: if shapes mismatch, avoid corrupting decoding.
            return scores

        for i in range(batch):
            tail_ids = input_ids[i, -self.max_prefix_tokens :].tolist()
            prefix_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
            expected = _infer_expected_identifier(prefix_text)
            if expected is None:
                continue

            # Extract the partial identifier being generated (best-effort).
            # We only constrain when the user appears to be typing an identifier right after a keyword.
            if expected == "table":
                m = re.search(r"(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)$", prefix_text, flags=re.I)
                partial = m.group(1) if m else None
                if partial is None and not re.search(r"(?:from|join)\s*$", prefix_text, flags=re.I):
                    continue
                trie = self._token_sets[i].table_trie
            else:
                m = re.search(
                    r"(?:select|where|on|group by|order by|having)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)$",
                    prefix_text,
                    flags=re.I,
                )
                partial = m.group(1) if m else None
                if partial is None and not re.search(
                    r"(?:select|where|on|group by|order by|having)\s*$",
                    prefix_text,
                    flags=re.I,
                ):
                    continue
                trie = self._token_sets[i].column_trie

            # If identifier hasn't started yet, constrain to valid first tokens.
            if not partial:
                prefix_token_ids: List[int] = []
            else:
                try:
                    prefix_token_ids = _encode_identifier(self.tokenizer, partial)
                except Exception:
                    continue

            node = trie.walk(prefix_token_ids)
            if node is None:
                continue
            if node.terminal:
                continue

            allowed_next = sorted(node.children.keys())
            if not allowed_next:
                continue

            allowed_next_t = torch.tensor(allowed_next, dtype=torch.long, device=scores.device)
            allow_always = self._token_sets[i].allow_always.to(scores.device)

            keep = torch.cat([allowed_next_t, allow_always]) if allow_always.numel() else allowed_next_t
            kept_scores = scores[i, keep].clone()
            scores[i, :] = -float("inf")
            scores[i, keep] = kept_scores

        return scores
