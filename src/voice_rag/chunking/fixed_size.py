"""Fixed-size token chunking with overlap — the baseline strategy.

Straightforward: encode the text, slide a window of ``chunk_size`` tokens with
``overlap`` tokens of overlap, decode each window back to text.

This is the simplest strategy and serves as a comparison point for the more
sophisticated ones.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import Chunk, Chunker, get_encoder


class FixedSizeChunker(Chunker):
    strategy = "fixed_size"

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        enc = get_encoder()
        tokens = enc.encode(text)
        if len(tokens) <= self.chunk_size:
            return [self._make_chunk(text, metadata, 0, 1)]

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)
        total = (len(tokens) - self.overlap + step - 1) // step  # ceil for loop count
        idx = 0
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            window = tokens[start:end]
            chunk_text = enc.decode(window)
            if len(window) >= self.min_chunk_size:
                chunks.append(self._make_chunk(chunk_text, metadata, idx, total))
                idx += 1
            if end >= len(tokens):
                break
            start += step

        # Update total
        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks
