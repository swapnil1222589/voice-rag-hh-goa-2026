"""Core chunking types and shared utilities."""
from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tiktoken

# A multilingual-safe encoder.  tiktoken's cl100k_base handles non-English text
# reasonably well (byte-level BPE).  We cache the encoder.
_ENCODER: Optional[tiktoken.Encoding] = None


def get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Token count via tiktoken (cl100k_base)."""
    if not text:
        return 0
    return len(get_encoder().encode(text))


class ChunkingStrategy(str, enum.Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    METADATA_AWARE = "metadata_aware"


@dataclass
class Chunk:
    """A single chunk of text with rich metadata for retrieval."""
    text: str
    metadata: Dict = field(default_factory=dict)

    # Identity
    chunk_id: str = ""

    # Provenance
    source_doc_id: str = ""
    source_lang: str = ""
    query_type: str = ""         # MSMARCO query_type (DESCRIPTION, LOCATION, etc.)
    is_selected: int = 0         # whether the passage was the selected answer passage
    passage_index: int = 0      # index within the original example's passage list

    # Chunking metadata
    strategy: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0

    # Original (English) text — useful for cross-lingual retrieval
    english_text: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            # Deterministic ID from content hash
            h = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
            self.chunk_id = f"chk_{h}"
        if self.token_count == 0:
            self.token_count = count_tokens(self.text)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "source_doc_id": self.source_doc_id,
            "source_lang": self.source_lang,
            "query_type": self.query_type,
            "is_selected": self.is_selected,
            "passage_index": self.passage_index,
            "strategy": self.strategy,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "english_text": self.english_text,
        }


class Chunker:
    """Base chunker — subclasses implement ``chunk_text``."""

    strategy: str = "base"

    def __init__(self, chunk_size: int = 400, overlap: int = 60, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        raise NotImplementedError

    def _make_chunk(
        self,
        text: str,
        metadata: Optional[Dict],
        chunk_index: int,
        total: int,
    ) -> Chunk:
        md = dict(metadata or {})
        return Chunk(
            text=text.strip(),
            metadata=md,
            source_doc_id=md.get("source_doc_id", ""),
            source_lang=md.get("source_lang", ""),
            query_type=md.get("query_type", ""),
            is_selected=md.get("is_selected", 0),
            passage_index=md.get("passage_index", 0),
            strategy=self.strategy,
            chunk_index=chunk_index,
            total_chunks=total,
            english_text=md.get("english_text", ""),
        )
