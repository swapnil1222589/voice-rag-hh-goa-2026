"""Hybrid retriever — combines dense (ChromaDB) + sparse (BM25) via RRF.

Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrievers
without needing calibrated scores.  Formula:
    RRF_score(d) = Σ 1 / (k + rank_i(d))   for each retriever i

After fusion, optional reranking re-scores the top candidates by computing
cosine similarity between the query embedding and each candidate embedding
in a single BATCHED model call (not N sequential calls — that was a bug).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .vector_db import VectorDB
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant


@dataclass
class RetrievalResult:
    """A single retrieved passage with scores from each retriever."""
    text: str
    metadata: Dict = field(default_factory=dict)
    rrf_score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_rank: int = 0
    english_text: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "rrf_score": round(self.rrf_score, 6),
            "vector_score": round(self.vector_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "final_rank": self.final_rank,
            "english_text": self.english_text,
        }


class HybridRetriever:
    """Hybrid dense + sparse retriever with RRF fusion and optional reranking."""

    def __init__(
        self,
        vector_db: VectorDB,
        bm25_index: BM25Index,
        top_k_vector: int = 15,
        top_k_bm25: int = 15,
        top_k_final: int = 5,
        rerank: bool = True,
    ):
        self._vdb = vector_db
        self._bm25 = bm25_index
        self.top_k_vector = top_k_vector
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        self.rerank = rerank

    def retrieve(
        self,
        query: str,
        where: Optional[Dict] = None,
        mode: str = "hybrid",
    ) -> List[RetrievalResult]:
        """Run hybrid retrieval and return top_k_final results.

        Args:
            query: The search query.
            where: Optional ChromaDB metadata filter.
            mode: "hybrid" (default), "dense", or "bm25".

        Returns:
            List of RetrievalResult sorted by final score.
        """
        rrf_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict] = {}

        # ── Dense retrieval ──────────────────────────────────────
        if mode in ("hybrid", "dense"):
            vec_results = self._vdb.query(query, top_k=self.top_k_vector, where=where)
            for rank, r in enumerate(vec_results):
                doc_id = r["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                doc_data.setdefault(doc_id, {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "vector_score": r["score"],
                    "bm25_score": 0.0,
                })
                doc_data[doc_id]["vector_score"] = r["score"]

        # ── Sparse (BM25) retrieval ──────────────────────────────
        if mode in ("hybrid", "bm25") and self._bm25.count > 0:
            bm25_results = self._bm25.query(query, top_k=self.top_k_bm25)
            for rank, r in enumerate(bm25_results):
                doc_id = r.get("chunk_id") or r.get("id") or ""
                if not doc_id:
                    continue
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                doc_data.setdefault(doc_id, {
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", r),
                    "vector_score": 0.0,
                    "bm25_score": r.get("score", 0.0),
                })
                doc_data[doc_id]["bm25_score"] = r.get("score", 0.0)

        if not rrf_scores:
            return []

        # ── RRF sort ─────────────────────────────────────────────
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        overfetch = self.top_k_final * 3  # fetch extra for reranking
        top_ids = sorted_ids[:overfetch]

        results: List[RetrievalResult] = []
        for rank, doc_id in enumerate(top_ids):
            d = doc_data[doc_id]
            md = d["metadata"] if isinstance(d["metadata"], dict) else {}
            results.append(RetrievalResult(
                text=d["text"],
                metadata=md,
                rrf_score=rrf_scores[doc_id],
                vector_score=d["vector_score"],
                bm25_score=d["bm25_score"],
                final_rank=rank,
                english_text=md.get("english_text", ""),
            ))

        # ── Optional reranking — BATCH embed, not per-candidate ──
        if self.rerank and len(results) > 1:
            results = self._rerank_batch(query, results)

        # Final cut
        results = results[: self.top_k_final]
        for i, r in enumerate(results):
            r.final_rank = i

        return results

    def _rerank_batch(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank by cosine similarity using a SINGLE batched embed call.

        Previous implementation called self._vdb.embed() once per candidate
        (N sequential forward passes).  This version batches all texts into
        one encoder call — O(1) forward passes regardless of candidate count.
        """
        try:
            import numpy as np

            texts = [r.text for r in results]
            # Batch encode: query + all candidates in one call
            all_texts = [query] + texts
            all_embs = self._vdb.embed_batch(all_texts)
            q_emb = np.array(all_embs[0])
            c_embs = np.array(all_embs[1:])

            # Cosine similarity via dot product (embeddings are L2-normalised by ST)
            q_norm = np.linalg.norm(q_emb)
            if q_norm < 1e-8:
                return results

            q_emb_n = q_emb / q_norm
            c_norms = np.linalg.norm(c_embs, axis=1, keepdims=True)
            c_norms = np.where(c_norms < 1e-8, 1.0, c_norms)
            c_embs_n = c_embs / c_norms
            cosines = c_embs_n @ q_emb_n

            for i, r in enumerate(results):
                # Blend: 50% RRF rank signal + 50% cosine similarity
                r.rrf_score = 0.5 * r.rrf_score + 0.5 * float(cosines[i])

            results.sort(key=lambda r: r.rrf_score, reverse=True)
        except Exception as exc:
            logger.warning("Reranking failed (%s); using RRF order", exc)

        return results

    def retrieve_dense_only(self, query: str, where: Optional[Dict] = None) -> List[RetrievalResult]:
        """Dense-only retrieval (no BM25, no RRF)."""
        return self.retrieve(query, where=where, mode="dense")

    def retrieve_bm25_only(self, query: str) -> List[RetrievalResult]:
        """BM25-only retrieval (no dense, no RRF)."""
        return self.retrieve(query, mode="bm25")
