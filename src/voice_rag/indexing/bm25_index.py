"""BM25 sparse retrieval index — in-memory, fast, good for keyword matching.

Tokeniser is Indic-aware: handles Devanagari, Bengali, Tamil and other scripts
by lowercasing and splitting on whitespace/punctuation.  No stopword removal
(it would hurt Indic languages where we don't have good stopword lists for all 14).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Split on any non-word character, including Indic punctuation
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Unicode-aware tokenizer that works for Indic scripts."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """In-memory BM25 index over chunks."""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[Dict] = []
        self._tokenized: List[List[str]] = []

    @property
    def count(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks_data: List[Dict]):
        """Add pre-built chunk dicts (must have 'text' and metadata)."""
        for c in chunks_data:
            self._chunks.append(c)
            self._tokenized.append(tokenize(c.get("text", "")))

    def build(self):
        """Finalise the BM25 index.  Must be called after adding all chunks."""
        if not self._tokenized:
            logger.warning("BM25 index is empty")
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info("BM25 index built with %d documents", len(self._tokenized))

    def query(self, text: str, top_k: int = 10) -> List[Dict]:
        if self._bm25 is None:
            return []
        tokens = tokenize(text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        out = []
        for rank, idx in enumerate(top_idx):
            if scores[idx] <= 0:
                continue
            chunk = self._chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["rank"] = rank
            out.append(chunk)
        return out

    def save(self, path: str):
        """Persist the BM25 index (tokens + metadata) to disk as JSON."""
        import json
        data = {"chunks": self._chunks, "tokens": self._tokenized}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("BM25 index saved to %s (%d docs)", path, len(self._chunks))

    def load(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = data["chunks"]
        self._tokenized = data["tokens"]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        logger.info("BM25 index loaded from %s (%d docs)", path, len(self._chunks))
