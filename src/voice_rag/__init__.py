"""Voice RAG — voice-enabled Retrieval-Augmented Generation over MSMARCO-XI.

Pipeline:  Voice → STT (ElevenLabs) → Chunking/Retrieval (Chroma + BM25)
            → Guardrails → Answer generation (OpenAI GPT)

Public entry points:
  - harness.run_pipeline(audio_bytes) → PipelineResult
  - api.app                          → FastAPI application
"""

__version__ = "1.0.0"


def run_pipeline(*args, **kwargs):
    """Lazy-imported to avoid heavy startup cost when importing sub-modules in tests."""
    from .harness.pipeline import run_pipeline as _run
    return _run(*args, **kwargs)


def run_text_pipeline(*args, **kwargs):
    from .harness.pipeline import run_text_pipeline as _run
    return _run(*args, **kwargs)


__all__ = ["run_pipeline", "run_text_pipeline", "__version__"]
