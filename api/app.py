"""FastAPI application — REST endpoints for the voice RAG pipeline.

Endpoints:
  GET  /health          — health check
  POST /ask             — text query (JSON) → answer
  POST /ask/voice       — audio file → answer (full voice pipeline)
  GET  /stats           — pipeline / index statistics
  GET  /benchmark       — serve latest benchmark results
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config.settings import get_settings
from voice_rag.harness import run_pipeline, run_text_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

# ── Application ─────────────────────────────────────────────────
app = FastAPI(
    title="Voice-Enabled RAG over MSMARCO-XI",
    description=(
        "Voice → STT (ElevenLabs / Sarvam) → "
        "Chunking / Retrieval (ChromaDB + BM25 RRF) → "
        "Guardrails → Grounded Answer (OpenAI GPT)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────────
# Allow both local dev (any port) and production frontend URL.
_settings = get_settings()
_frontend_url = os.environ.get("FRONTEND_URL", "")
_allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]
if _frontend_url:
    _allowed_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ───────────────────────────────────────────────────
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


# ── Request / Response models ───────────────────────────────────
class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Text query to answer")
    language: Optional[str] = Field(None, description="BCP-47 language hint (optional)")
    stt_provider: Optional[str] = Field(None, description="STT provider override (elevenlabs|sarvam)")


class StageLatency(BaseModel):
    name: str
    latency_ms: float
    success: bool
    error: str = ""


class SourceDocument(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    rrf_score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_rank: int = 0
    english_text: str = ""


class AnswerResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer: str
    transcript: str
    is_refusal: bool = False
    grounded: bool = True
    confidence: float = 0.0
    sources: List[SourceDocument] = []
    stages: List[StageLatency] = []
    total_latency_ms: float = 0.0
    guardrail_flags: List[str] = []
    error: Optional[str] = None
    stt_provider: str = ""
    language_detected: Optional[str] = None


def _pipeline_result_to_response(result, request_id: str) -> AnswerResponse:
    """Convert PipelineResult to AnswerResponse."""
    sources = []
    for ctx in result.context:
        sources.append(SourceDocument(
            text=ctx.get("text", ""),
            metadata=ctx.get("metadata", {}),
            rrf_score=ctx.get("rrf_score", 0.0),
            vector_score=ctx.get("vector_score", 0.0),
            bm25_score=ctx.get("bm25_score", 0.0),
            final_rank=ctx.get("final_rank", 0),
            english_text=ctx.get("english_text", ""),
        ))
    stages = [
        StageLatency(
            name=s["name"],
            latency_ms=s["latency_ms"],
            success=s["success"],
            error=s.get("error", ""),
        )
        for s in result.stages
    ]
    return AnswerResponse(
        request_id=request_id,
        answer=result.answer,
        transcript=result.transcript,
        is_refusal=result.is_refusal,
        grounded=result.grounded,
        confidence=result.confidence,
        sources=sources,
        stages=stages,
        total_latency_ms=result.total_latency_ms,
        guardrail_flags=result.guardrail_flags,
        error=result.error,
    )


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health", tags=["infra"])
async def health():
    """Health check — returns service status and version."""
    return {
        "status": "ok",
        "service": "voice-rag",
        "version": "1.0.0",
        "dataset": "MSMARCO-XI",
    }


@app.post("/ask", response_model=AnswerResponse, tags=["query"])
async def ask_text(req: TextQuery, request: Request):
    """Text query → grounded answer (skips STT — useful for testing and benchmarking)."""
    request_id = str(uuid.uuid4())
    logger.info("[%s] /ask: %r", request_id, req.query[:80])
    try:
        result = await run_in_threadpool(run_text_pipeline, req.query)
        return _pipeline_result_to_response(result, request_id)
    except Exception as exc:
        logger.error("[%s] /ask error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/voice", response_model=AnswerResponse, tags=["query"])
async def ask_voice(request: Request, file: UploadFile = File(...)):
    """Audio file → grounded answer (full voice pipeline including STT)."""
    request_id = str(uuid.uuid4())
    logger.info("[%s] /ask/voice: filename=%s, content_type=%s", request_id, file.filename, file.content_type)

    # Size limit
    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_bytes)} bytes). Max {MAX_AUDIO_SIZE_BYTES} bytes.",
        )
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    mime_type = file.content_type or "audio/wav"

    # Get configured STT provider
    settings = get_settings()
    stt_provider = getattr(settings, "stt_provider", "elevenlabs")

    try:
        result = await run_in_threadpool(run_pipeline, audio_bytes, mime_type, stt_provider)
        resp = _pipeline_result_to_response(result, request_id)
        resp.stt_provider = stt_provider
        return resp
    except Exception as exc:
        logger.error("[%s] /ask/voice error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats", tags=["infra"])
async def stats():
    """Return index statistics from the pipeline singleton."""
    try:
        from voice_rag.harness.pipeline import _PipelineComponents
        components = _PipelineComponents.get()
        retriever = components.get_retriever()
        vdb = components._vdb
        bm25 = components._bm25
        settings = get_settings()
        return {
            "vector_db_count": vdb.count if vdb else 0,
            "bm25_count": bm25.count if bm25 else 0,
            "embedding_model": settings.embedding_model,
            "chunking_strategy": "semantic",
            "dataset": settings.dataset_name,
        }
    except Exception as exc:
        logger.warning("Stats error: %s", exc)
        return {"error": str(exc), "vector_db_count": 0, "bm25_count": 0}


@app.get("/benchmark", tags=["infra"])
async def get_benchmark():
    """Serve the latest benchmark results (generated by scripts/benchmark.py)."""
    benchmark_path = "data/benchmark_results.json"
    if not os.path.exists(benchmark_path):
        return {
            "error": "No benchmark results found. Run: python scripts/benchmark.py",
            "data": None,
        }
    try:
        with open(benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"data": data, "error": None}
    except Exception as exc:
        return {"error": str(exc), "data": None}


# ── Static frontend ──────────────────────────────────────────────
# Serve the frontend from /ui so both API and UI are on the same origin.
# In production the frontend is at http://host:8000/ui/index.html
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    logger.info("Frontend served from /ui (directory: %s)", _frontend_dir)
