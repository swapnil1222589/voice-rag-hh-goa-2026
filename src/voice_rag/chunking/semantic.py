"""Semantic sentence-grouping chunker.

Groups sentences greedily until a token budget is reached, but *also* tries to
keep semantically coherent groups together by measuring sentence-to-sentence
similarity (via embedding cosine) and breaking at low-similarity boundaries.

This is the most sophisticated strategy: it respects meaning, not just size.
For efficiency, similarity is approximated using a lightweight lexical overlap
heuristic (Jaccard on token sets) rather than full embeddings during indexing —
embeddings are computed later at retrieval time anyway.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from .base import Chunk, Chunker, count_tokens
from .recursive import split_sentences


def _token_set(text: str) -> Set[str]:
    """Lowercased word set for Jaccard similarity."""
    return set(text.lower().split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class SemanticChunker(Chunker):
    """Group sentences into topically coherent chunks using similarity breaks.

    Algorithm:
      1. Split text into sentences.
      2. Greedily accumulate sentences into the current chunk.
      3. If adding the next sentence would exceed ``chunk_size`` tokens, OR
         the similarity between the current chunk and the next sentence drops
         below ``similarity_threshold``, close the chunk and start a new one.
      4. The last ``overlap`` tokens of the closed chunk are carried into the
         next one to preserve continuity.
    """
    strategy = "semantic"

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 60,
        min_chunk_size: int = 50,
        similarity_threshold: float = 0.15,
    ):
        super().__init__(chunk_size, overlap, min_chunk_size)
        self.similarity_threshold = similarity_threshold

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sentences = split_sentences(text)
        if not sentences:
            return []

        if count_tokens(text) <= self.chunk_size:
            return [self._make_chunk(text, metadata, 0, 1)]

        chunks: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        current_tokenset: Set[str] = _token_set(sentences[0])

        for i, sent in enumerate(sentences):
            sent_tokens = count_tokens(sent)
            sent_tokenset = _token_set(sent)
            sim = _jaccard(current_tokenset, sent_tokenset)

            should_break = False
            if current:
                # Break if we're over budget
                if current_tokens + sent_tokens > self.chunk_size:
                    should_break = True
                # Break at semantic boundary — low similarity + already have content
                elif sim < self.similarity_threshold and current_tokens >= self.min_chunk_size:
                    should_break = True

            if should_break:
                chunks.append(current)
                # Overlap: carry last sentence(s) up to overlap budget
                overlap_sents: List[str] = []
                ot = 0
                for s in reversed(current):
                    st = count_tokens(s)
                    if ot + st > self.overlap:
                        break
                    overlap_sents.insert(0, s)
                    ot += st
                current = overlap_sents
                current_tokens = ot
                current_tokenset = _token_set(" ".join(current))

            current.append(sent)
            current_tokens += sent_tokens
            current_tokenset |= sent_tokenset

        if current:
            chunks.append(current)

        result: List[Chunk] = []
        for i, sents in enumerate(chunks):
            ct = " ".join(sents)
            if count_tokens(ct) >= self.min_chunk_size:
                result.append(self._make_chunk(ct, metadata, i, len(chunks)))

        for c in result:
            c.total_chunks = len(result)
        return result
