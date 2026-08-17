"""Metadata-aware chunker — keeps each passage whole, no splitting.

This strategy treats each passage as an atomic chunk.  It's useful for:
  - Short passages that don't need splitting
  - Preserving all metadata without losing context
  - Serving as a baseline to compare sub-passage chunking against

It also enriches the chunk with *all* available metadata from the example:
query_type, is_selected, language, English passage, etc.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import Chunk, Chunker, count_tokens


class MetadataAwareChunker(Chunker):
    """No splitting — one chunk per passage, with full metadata.

    Falls back to fixed-size splitting only if a passage exceeds 3x the chunk
    size (very rare in MSMARCO).
    """
    strategy = "metadata_aware"

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        # Very long passage — split into a few large pieces but keep metadata
        if count_tokens(text) > self.chunk_size * 3:
            from .fixed_size import FixedSizeChunker
            sub = FixedSizeChunker(
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                min_chunk_size=self.min_chunk_size,
            )
            return sub.chunk_text(text, metadata)

        return [self._make_chunk(text, metadata, 0, 1)]
