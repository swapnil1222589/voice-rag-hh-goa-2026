"""Build the index: load data, chunk it, and index into ChromaDB + BM25."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

from config.settings import get_settings
from ..chunking import Chunk, get_chunker
from ..chunking.base import ChunkingStrategy
from ..data.loader import MSMARCOSource, PassageRecord, load_sample_data
from ..indexing.bm25_index import BM25Index
from ..indexing.vector_db import VectorDB

logger = logging.getLogger(__name__)


def chunk_records(
    records: List[PassageRecord],
    strategy: str = "semantic",
    chunk_size: int = 400,
    overlap: int = 60,
    min_chunk_size: int = 50,
) -> List[Chunk]:
    """Apply the chosen chunking strategy to all passage records."""
    chunker = get_chunker(strategy, chunk_size, overlap, min_chunk_size)
    all_chunks: List[Chunk] = []
    for rec in records:
        # Use translated text; fall back to English if translation is empty
        text = rec.text or rec.english_text
        md = rec.to_chunk_metadata()
        chunks = chunker.chunk_text(text, md)
        all_chunks.extend(chunks)
    logger.info(
        "Chunked %d records → %d chunks (strategy=%s)",
        len(records), len(all_chunks), strategy,
    )
    return all_chunks


def build_index(
    strategy: str = "semantic",
    use_sample: bool = False,
    sample_path: str = "data/sample/sample.jsonl",
    max_examples: Optional[int] = None,
    languages: Optional[List[str]] = None,
    bm25_path: str = "data/bm25_index.json",
):
    """Full index build pipeline.

    Args:
        strategy: Chunking strategy name.
        use_sample: If True, load from local sample file instead of HF.
        sample_path: Path to sample JSONL.
        max_examples: Limit number of examples per language (for testing).
        languages: Override configured languages.
        bm25_path: Where to save the BM25 index.
    """
    settings = get_settings()
    t0 = time.perf_counter()

    # ── Load data ────────────────────────────────────────────
    if use_sample:
        logger.info("Loading sample data from %s", sample_path)
        records = load_sample_data(sample_path)
    else:
        langs = languages or settings.languages
        source = MSMARCOSource(
            languages=langs,
            split="train",
            max_examples=max_examples,
            max_passages_per_example=settings.max_passages_per_example,
        )
        records = list(source)

    logger.info("Loaded %d passage records", len(records))
    if not records:
        logger.error("No records loaded — aborting index build")
        return

    # ── Chunk ────────────────────────────────────────────────
    chunks = chunk_records(
        records,
        strategy=strategy,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
    )

    # ── Index into ChromaDB ─────────────────────────────────
    logger.info("Indexing %d chunks into ChromaDB ...", len(chunks))
    vdb = VectorDB(
        persist_dir=settings.chroma_persist_dir,
        embedding_model_name=settings.embedding_model,
    )
    vdb.add_chunks(chunks)

    # ── Index into BM25 ──────────────────────────────────────
    logger.info("Indexing %d chunks into BM25 ...", len(chunks))
    bm25 = BM25Index()
    bm25.add_chunks([c.to_dict() for c in chunks])
    bm25.build()

    # Save BM25
    os.makedirs(os.path.dirname(bm25_path) or ".", exist_ok=True)
    bm25.save(bm25_path)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Index build complete: %d chunks, %d in ChromaDB, %d in BM25, %.1fs",
        len(chunks), vdb.count, bm25.count, elapsed,
    )
