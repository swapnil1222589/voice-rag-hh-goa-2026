"""Chunking strategies for MSMARCO-XI passages.

The dataset gives us passages (one per example, ~10 passages each).  We treat
each *passage* as a document and then chunk *within* passages using one of
several strategies.  The choice of strategy is configurable and each strategy
emits Chunk objects with rich metadata so retrieval can filter / boost.

Strategies implemented
----------------------
1. **Fixed-size token chunking** with overlap — the baseline.
2. **Recursive structural chunking** — splits on sentences, then words, with overlap.
3. **Semantic sentence-grouping chunker** — greedily groups sentences until a token budget
   is reached, respecting sentence boundaries so chunks stay topically coherent.
4. **Metadata-aware passage chunker** — keeps each passage whole (no splitting), preserves
   all metadata, used as a comparison baseline and for short passages.

All strategies share a common interface (``chunk(text, metadata) → List[Chunk]``) and
all produce overlapping chunks (where applicable) to avoid losing context at boundaries.
"""
from .base import Chunk, Chunker, ChunkingStrategy
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .metadata_aware import MetadataAwareChunker
from .factory import get_chunker

__all__ = [
    "Chunk", "Chunker", "ChunkingStrategy",
    "FixedSizeChunker", "RecursiveChunker", "SemanticChunker", "MetadataAwareChunker",
    "get_chunker",
]
