"""Factory for chunking strategies."""
from __future__ import annotations

from .base import Chunker, ChunkingStrategy
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .metadata_aware import MetadataAwareChunker

_STRATEGY_MAP = {
    ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
    ChunkingStrategy.RECURSIVE: RecursiveChunker,
    ChunkingStrategy.SEMANTIC: SemanticChunker,
    ChunkingStrategy.METADATA_AWARE: MetadataAwareChunker,
}


def get_chunker(
    strategy: str | ChunkingStrategy = "semantic",
    chunk_size: int = 400,
    overlap: int = 60,
    min_chunk_size: int = 50,
) -> Chunker:
    """Return a chunker for the named strategy.

    Args:
        strategy: One of "fixed_size", "recursive", "semantic", "metadata_aware".
        chunk_size: Target token budget per chunk.
        overlap: Token overlap between adjacent chunks.
        min_chunk_size: Minimum tokens for a chunk to be kept.
    """
    if isinstance(strategy, str):
        strategy = ChunkingStrategy(strategy.lower())
    cls = _STRATEGY_MAP[strategy]
    return cls(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )
