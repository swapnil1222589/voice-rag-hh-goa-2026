"""Recursive structural chunking — split on sentence boundaries first, then words.

Better than fixed-size because it respects sentence boundaries whenever possible.
The overlap is applied at the sentence level: the last N sentences of a chunk
are prepended to the next chunk.

Fallback hierarchy:  paragraph → sentence → word → token
This ensures we never cut mid-sentence if the sentence itself fits in the budget.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .base import Chunk, Chunker, count_tokens

# Indic-aware sentence splitter — handles Devanagari danda (।), Arabic (؟),
# and common punctuation.  This is deliberately simple and language-agnostic.
_SENT_SPLIT = re.compile(r'(?<=[।.!?؟])\s+|\n+')

# Word-level split (works for most scripts)
_WORD_SPLIT = re.compile(r'\s+')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, supporting Indic punctuation."""
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class RecursiveChunker(Chunker):
    strategy = "recursive"

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sentences = split_sentences(text)
        if not sentences:
            return []

        # If the whole text fits, return as single chunk
        if count_tokens(text) <= self.chunk_size:
            return [self._make_chunk(text, metadata, 0, 1)]

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)

            # If a single sentence exceeds chunk_size, split it at word level
            if sent_tokens > self.chunk_size:
                # Flush current buffer first
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0

                # Word-level split for this long sentence
                words = _WORD_SPLIT.split(sent)
                sub_buf: List[str] = []
                sub_tokens = 0
                for w in words:
                    w_tokens = count_tokens(w)
                    if sub_tokens + w_tokens > self.chunk_size and sub_buf:
                        chunks.append(" ".join(sub_buf))
                        # Overlap: keep last ~30% of words
                        keep = max(1, len(sub_buf) // 3)
                        sub_buf = sub_buf[-keep:]
                        sub_tokens = sum(count_tokens(w) for w in sub_buf)
                    sub_buf.append(w)
                    sub_tokens += w_tokens
                if sub_buf:
                    chunks.append(" ".join(sub_buf))
                continue

            # Normal case: add sentence if it fits
            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Overlap: keep last few sentences (~overlap tokens worth)
                overlap_sents: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    st = count_tokens(s)
                    if overlap_tokens + st > self.overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += st
                current_sentences = overlap_sents
                current_tokens = overlap_tokens

            current_sentences.append(sent)
            current_tokens += sent_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # Filter tiny chunks and build Chunk objects
        result: List[Chunk] = []
        for i, ct in enumerate(chunks):
            if count_tokens(ct) >= self.min_chunk_size:
                result.append(self._make_chunk(ct, metadata, i, len(chunks)))

        for c in result:
            c.total_chunks = len(result)
        return result
