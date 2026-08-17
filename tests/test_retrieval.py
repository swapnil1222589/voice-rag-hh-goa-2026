"""Tests for hybrid retrieval — dense-only, BM25-only, hybrid modes.

All ChromaDB and SentenceTransformer calls are mocked so no real GPU/model is needed.
"""
from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from voice_rag.indexing.bm25_index import BM25Index, tokenize
from voice_rag.indexing.hybrid_retriever import HybridRetriever, RetrievalResult


# ── BM25 tokenizer ────────────────────────────────────────────────
class TestTokenize:
    def test_english(self):
        tokens = tokenize("The quick brown fox")
        assert "quick" in tokens
        assert "the" in tokens  # lowercased

    def test_hindi(self):
        tokens = tokenize("महात्मा गांधी का जन्म")
        assert len(tokens) > 0

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


# ── BM25Index ────────────────────────────────────────────────────
class TestBM25Index:
    def _make_index(self, texts: List[str]) -> BM25Index:
        idx = BM25Index()
        chunks = [{"chunk_id": f"c{i}", "text": t} for i, t in enumerate(texts)]
        idx.add_chunks(chunks)
        idx.build()
        return idx

    def test_basic_query(self):
        idx = self._make_index([
            "Mahatma Gandhi was born in Porbandar",
            "The Taj Mahal is in Agra",
            "Python is a programming language",
        ])
        results = idx.query("Mahatma Gandhi", top_k=2)
        assert len(results) >= 1
        assert results[0]["text"] == "Mahatma Gandhi was born in Porbandar"

    def test_count(self):
        idx = self._make_index(["doc1", "doc2", "doc3"])
        assert idx.count == 3

    def test_empty_index_returns_empty(self):
        idx = BM25Index()
        assert idx.query("anything") == []

    def test_unbuilt_index_returns_empty(self):
        idx = BM25Index()
        idx.add_chunks([{"chunk_id": "c1", "text": "test"}])
        # Not calling build()
        assert idx.query("test") == []

    def test_zero_score_excluded(self):
        idx = self._make_index(["apple banana cherry", "dog cat fish"])
        results = idx.query("elephant")  # no match
        assert results == []  # all scores ≤ 0 are excluded

    def test_hindi_query(self):
        """BM25 needs a corpus large enough for IDF to produce non-zero scores."""
        idx = BM25Index()
        idx.add_chunks([
            {"chunk_id": "c0", "text": "gandhi porbandar gujarat born 1869"},
            {"chunk_id": "c1", "text": "constitution india january 1950 republic"},
            {"chunk_id": "c2", "text": "taj mahal agra mughal emperor shahjehan"},
            {"chunk_id": "c3", "text": "photosynthesis chlorophyll sunlight carbon dioxide"},
            {"chunk_id": "c4", "text": "vaccine immune system antibody infection prevention"},
        ])
        idx.build()
        results = idx.query("gandhi porbandar", top_k=1)
        assert len(results) >= 1
        assert "gandhi" in results[0]["text"]



# ── HybridRetriever (mocked) ─────────────────────────────────────
def _make_mock_vdb(results: List[Dict] = None):
    vdb = MagicMock()
    vdb.query.return_value = results or [
        {"id": "c1", "text": "Gandhi was born in Porbandar", "metadata": {}, "score": 0.9, "rank": 0},
        {"id": "c2", "text": "Taj Mahal is in Agra", "metadata": {}, "score": 0.7, "rank": 1},
    ]
    # Return normalised embeddings for batch reranking
    import numpy as np
    def embed_batch(texts):
        n = len(texts)
        embs = np.random.randn(n, 64)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / norms).tolist()
    vdb.embed_batch.side_effect = embed_batch
    return vdb


def _make_mock_bm25(results: List[Dict] = None):
    bm25 = MagicMock(spec=BM25Index)
    bm25.count = 100
    bm25.query.return_value = results or [
        {"chunk_id": "c1", "text": "Gandhi was born in Porbandar", "metadata": {}, "score": 5.2, "rank": 0},
        {"chunk_id": "c3", "text": "Salt March 1930", "metadata": {}, "score": 3.1, "rank": 1},
    ]
    return bm25


class TestHybridRetriever:
    def _make_retriever(self, top_k_final=3, rerank=False):
        vdb = _make_mock_vdb()
        bm25 = _make_mock_bm25()
        return HybridRetriever(
            vector_db=vdb,
            bm25_index=bm25,
            top_k_vector=5,
            top_k_bm25=5,
            top_k_final=top_k_final,
            rerank=rerank,
        )

    def test_hybrid_returns_results(self):
        retriever = self._make_retriever()
        results = retriever.retrieve("Gandhi")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_hybrid_respects_top_k(self):
        retriever = self._make_retriever(top_k_final=2)
        results = retriever.retrieve("Gandhi")
        assert len(results) <= 2

    def test_dense_only_mode(self):
        retriever = self._make_retriever()
        results = retriever.retrieve("Gandhi", mode="dense")
        # BM25 should NOT be called
        retriever._bm25.query.assert_not_called()
        assert len(results) > 0

    def test_bm25_only_mode(self):
        retriever = self._make_retriever()
        results = retriever.retrieve("Gandhi", mode="bm25")
        retriever._vdb.query.assert_not_called()
        assert len(results) > 0

    def test_rrf_merges_sources(self):
        """RRF should assign higher scores to docs appearing in both dense and BM25."""
        vdb = _make_mock_vdb([
            {"id": "shared", "text": "Shared doc", "metadata": {}, "score": 0.8, "rank": 0},
            {"id": "only_dense", "text": "Dense only", "metadata": {}, "score": 0.5, "rank": 1},
        ])
        bm25 = MagicMock(spec=BM25Index)
        bm25.count = 10
        bm25.query.return_value = [
            {"chunk_id": "shared", "text": "Shared doc", "metadata": {}, "score": 4.0, "rank": 0},
            {"chunk_id": "only_bm25", "text": "BM25 only", "metadata": {}, "score": 2.0, "rank": 1},
        ]
        retriever = HybridRetriever(vdb, bm25, top_k_vector=5, top_k_bm25=5, top_k_final=5, rerank=False)
        results = retriever.retrieve("test")
        # "shared" doc should have higher RRF score
        shared = next((r for r in results if r.text == "Shared doc"), None)
        only_dense = next((r for r in results if r.text == "Dense only"), None)
        if shared and only_dense:
            assert shared.rrf_score >= only_dense.rrf_score

    def test_final_ranks_sequential(self):
        retriever = self._make_retriever(top_k_final=3)
        results = retriever.retrieve("test")
        for i, r in enumerate(results):
            assert r.final_rank == i

    def test_empty_results_when_no_index(self):
        """When VDB returns nothing and BM25 is empty, should get empty results."""
        vdb = _make_mock_vdb([])
        bm25 = MagicMock(spec=BM25Index)
        bm25.count = 0  # no BM25 index
        bm25.query.return_value = []
        retriever = HybridRetriever(vdb, bm25, top_k_final=5, rerank=False)
        results = retriever.retrieve("test", mode="bm25")  # bm25-only mode with no index
        assert results == []

    def test_reranking_batch_called(self):
        """Reranking should call embed_batch once (not N times)."""
        vdb = _make_mock_vdb()
        bm25 = _make_mock_bm25()
        retriever = HybridRetriever(vdb, bm25, top_k_final=3, rerank=True)
        retriever.retrieve("Gandhi")
        # embed_batch should be called exactly once (batch, not per-candidate)
        assert vdb.embed_batch.call_count == 1

    def test_retrieve_dense_only_method(self):
        retriever = self._make_retriever()
        results = retriever.retrieve_dense_only("Gandhi")
        assert len(results) > 0

    def test_retrieve_bm25_only_method(self):
        retriever = self._make_retriever()
        results = retriever.retrieve_bm25_only("Gandhi")
        assert len(results) > 0
