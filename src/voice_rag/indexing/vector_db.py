"""Dense vector store wrapper around ChromaDB.

Uses a multilingual sentence-transformers model so it works across all 14
Indic languages in MSMARCO-XI.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..chunking.base import Chunk

logger = logging.getLogger(__name__)


class VectorDB:
    """ChromaDB-backed dense vector store."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "msmarco_xi",
    ):
        # Lazy imports so tests that don't need VectorDB don't load heavy deps
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import SentenceTransformer

        self._persist_dir = persist_dir
        self._collection_name = collection_name

        logger.info("Loading embedding model: %s", embedding_model_name)
        self._encoder = SentenceTransformer(embedding_model_name)
        self._dim = self._encoder.get_sentence_embedding_dimension()

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_dim": self._dim},
        )
        logger.info(
            "VectorDB ready: collection=%s, dim=%d, count=%d",
            collection_name, self._dim, self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 500) -> int:
        """Add chunks to the vector store.  Returns number added."""
        added = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._encoder.encode(
                texts, batch_size=min(64, len(texts)), show_progress_bar=False
            ).tolist()
            ids = [c.chunk_id for c in batch]
            metadatas = [c.to_dict() for c in batch]
            # Chroma metadata values must be primitives
            clean_meta = []
            for m in metadatas:
                cm = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        cm[k] = v
                    elif v is None:
                        cm[k] = ""
                    else:
                        cm[k] = str(v)
                clean_meta.append(cm)
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=clean_meta,
            )
            added += len(batch)
        return added

    def query(
        self,
        text: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Query the vector store.  Returns list of {text, metadata, score, id}."""
        q_emb = self._encoder.encode([text], show_progress_bar=False).tolist()
        kwargs = dict(
            query_embeddings=q_emb,
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        out = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
                "rank": i,
            })
        return out

    def embed(self, text: str) -> List[float]:
        """Embed a single text.  Prefer embed_batch() for multiple texts."""
        return self._encoder.encode([text], show_progress_bar=False).tolist()[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single batched forward pass.

        This is significantly faster than calling embed() N times because the
        underlying transformer processes all texts in parallel on the GPU/CPU.
        """
        if not texts:
            return []
        return self._encoder.encode(
            texts, batch_size=min(64, len(texts)), show_progress_bar=False
        ).tolist()

    def reset(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", "embedding_dim": self._dim},
        )
