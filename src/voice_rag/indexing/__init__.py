"""Vector DB indexing and retrieval — ChromaDB + BM25 hybrid search.

Hybrid retrieval combines:
  - Dense vector search (ChromaDB + sentence-transformers multilingual embeddings)
  - Sparse lexical search (BM25 via rank-bm25)

Results are fused using Reciprocal Rank Fusion (RRF), then optionally reranked
by a cross-encoder-style score (cosine similarity of embeddings).

This hybrid approach is especially important for Indic languages, where
pure-semantic retrieval can miss exact keyword matches that BM25 catches well,
and vice versa.
"""
from .vector_db import VectorDB
from .bm25_index import BM25Index
from .hybrid_retriever import HybridRetriever, RetrievalResult

__all__ = ["VectorDB", "BM25Index", "HybridRetriever", "RetrievalResult"]
