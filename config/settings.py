"""Centralised configuration loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Providers ───────────────────────────────────────────────
    elevenlabs_api_key: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    sarvam_api_key: str = ""
    stt_provider: str = "elevenlabs"   # "elevenlabs" | "sarvam"

    # ── CORS / Frontend ─────────────────────────────────────────
    frontend_url: str = ""             # e.g. https://your-app.vercel.app

    # ── Vector DB / embeddings ──────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ── Chunking ────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 60
    min_chunk_size: int = 50
    chunking_strategy: str = "semantic"  # fixed_size | recursive | semantic | metadata_aware

    # ── Retrieval ───────────────────────────────────────────────
    top_k_vector: int = 15
    top_k_bm25: int = 15
    top_k_final: int = 5
    rerank: bool = True

    # ── Generation ──────────────────────────────────────────────
    max_answer_tokens: int = 256
    generation_temperature: float = 0.0
    generation_timeout: float = 30.0    # seconds

    # ── Guardrails ──────────────────────────────────────────────
    max_query_length: int = 500
    relevance_threshold: float = 0.25

    # ── Server ──────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Dataset ────────────────────────────────────────────────
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_languages: str = "hi,en"
    index_language: str = "all"
    max_passages_per_example: int = 10

    @property
    def languages(self) -> List[str]:
        return [lang.strip() for lang in self.dataset_languages.split(",") if lang.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
