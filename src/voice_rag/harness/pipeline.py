"""The main pipeline harness — orchestrates all stages with structure and recovery."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.settings import get_settings
from ..stt import get_stt_provider, STTResult
from ..stt.audio_utils import to_wav, detect_mime
from ..chunking import get_chunker
from ..indexing import VectorDB, BM25Index, HybridRetriever, RetrievalResult
from ..generation import AnswerGenerator, GenerationResult
from ..guardrails import InputGuardrail, OutputGuardrail, GuardrailAction, GuardrailResult

logger = logging.getLogger(__name__)


# ── Structured pipeline state ───────────────────────────────────
@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    name: str
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class PipelineContext:
    """Carries structured data through the pipeline."""
    # Input
    audio_bytes: bytes = b""
    mime_type: str = "audio/wav"

    # STT
    stt_result: Optional[STTResult] = None

    # Guardrail (input)
    input_guard: Optional[GuardrailResult] = None

    # Retrieval
    retrieval_results: List[RetrievalResult] = field(default_factory=list)

    # Guardrail (relevance)
    relevance_guard: Optional[GuardrailResult] = None

    # Generation
    generation_result: Optional[GenerationResult] = None

    # Guardrail (output)
    output_guard: Optional[dict] = None

    # Stages
    stages: List[StageResult] = field(default_factory=list)
    total_latency_ms: float = 0.0


@dataclass
class PipelineResult:
    """Final structured output of the pipeline."""
    answer: str
    transcript: str
    is_refusal: bool = False
    grounded: bool = True
    confidence: float = 0.0
    context: List[dict] = field(default_factory=list)
    stages: List[dict] = field(default_factory=list)
    total_latency_ms: float = 0.0
    guardrail_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "transcript": self.transcript,
            "is_refusal": self.is_refusal,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "context": self.context,
            "stages": self.stages,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "guardrail_flags": self.guardrail_flags,
            "error": self.error,
        }


# ── Pipeline components (lazily initialised singleton) ─────────
class _PipelineComponents:
    """Lazy singleton — initialises heavy components on first use."""
    _instance = None
    _stt_cache: Dict[str, object] = {}
    _retriever = None
    _generator = None
    _input_guard = None
    _output_guard = None
    _vdb = None
    _bm25 = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._init()
        return cls._instance

    def _init(self):
        settings = get_settings()
        self._input_guard = InputGuardrail(
            max_query_length=settings.max_query_length,
        )
        self._output_guard = OutputGuardrail()

    def get_stt(self, provider_name: str = "elevenlabs"):
        """Get a cached STT provider by name."""
        if provider_name not in self._stt_cache:
            self._stt_cache[provider_name] = get_stt_provider(provider_name)
        return self._stt_cache[provider_name]

    def get_retriever(self):
        if self._retriever is None:
            import os
            settings = get_settings()
            self._vdb = VectorDB(
                persist_dir=settings.chroma_persist_dir,
                embedding_model_name=settings.embedding_model,
            )
            self._bm25 = BM25Index()
            bm25_path = "data/bm25_index.json"
            if os.path.exists(bm25_path):
                self._bm25.load(bm25_path)
            else:
                logger.warning("BM25 index not found at %s — BM25 retrieval disabled. Run: python scripts/build_index.py", bm25_path)
            self._retriever = HybridRetriever(
                vector_db=self._vdb,
                bm25_index=self._bm25,
                top_k_vector=settings.top_k_vector,
                top_k_bm25=settings.top_k_bm25,
                top_k_final=settings.top_k_final,
                rerank=settings.rerank,
            )
        return self._retriever

    def get_generator(self):
        if self._generator is None:
            settings = get_settings()
            self._generator = AnswerGenerator(
                model=settings.openai_model,
                max_tokens=settings.max_answer_tokens,
                temperature=settings.generation_temperature,
            )
        return self._generator


def _track_stage(ctx: PipelineContext, name: str, fn):
    """Execute a stage, track latency, and record the result."""
    t0 = time.perf_counter()
    try:
        result = fn()
        latency = (time.perf_counter() - t0) * 1000
        stage = StageResult(name=name, latency_ms=latency, success=True)
        ctx.stages.append(stage)
        return result
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        stage = StageResult(
            name=name, latency_ms=latency, success=False, error=str(exc)[:300]
        )
        ctx.stages.append(stage)
        logger.error("Stage %s failed: %s", name, exc)
        return None


# ── Public entry points ────────────────────────────────────────
def run_pipeline(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    stt_provider: str = "elevenlabs",
) -> PipelineResult:
    """Run the full voice → answer pipeline.

    Args:
        audio_bytes: Raw audio bytes from the user's microphone.
        mime_type: Audio MIME type (auto-detected if wrong).
        stt_provider: "elevenlabs" or "sarvam".

    Returns:
        PipelineResult with answer, transcript, latency, and guardrail info.
    """
    t_total = time.perf_counter()
    settings = get_settings()
    components = _PipelineComponents.get()

    # Use configured provider if not explicitly overridden
    if stt_provider == "elevenlabs":
        stt_provider = getattr(settings, "stt_provider", "elevenlabs")

    ctx = PipelineContext(audio_bytes=audio_bytes, mime_type=mime_type)

    # ── Stage 1: Audio normalisation — ALWAYS normalise to 16 kHz ──
    def _do_audio_norm():
        wav_bytes, wav_mime = to_wav(audio_bytes)
        ctx.audio_bytes = wav_bytes
        ctx.mime_type = wav_mime
        return {"original_size": len(audio_bytes), "normalised_size": len(wav_bytes)}

    _track_stage(ctx, "audio_normalise", _do_audio_norm)
    # Use normalised audio for STT
    normalised_audio = ctx.audio_bytes
    normalised_mime = ctx.mime_type

    # ── Stage 2: Speech-to-text ─────────────────────────────────
    def _do_stt():
        stt = components.get_stt(stt_provider)
        return stt.transcribe(normalised_audio, normalised_mime)

    stt_result = _track_stage(ctx, "stt", _do_stt)

    # Add STT's own internal latency to the stage record
    if stt_result is not None and hasattr(stt_result, "latency_ms"):
        # The STT provider measures its own API round-trip latency —
        # add it as a sub-field on the stage for transparency
        for s in ctx.stages:
            if s.name == "stt":
                s.data["stt_api_latency_ms"] = round(stt_result.latency_ms, 2)

    if stt_result is None or not stt_result.ok:
        ctx.stt_result = stt_result or STTResult(transcript="", error="STT failed", provider=stt_provider)
        return _build_result(ctx, t_total, error="Speech recognition failed")

    ctx.stt_result = stt_result
    query = stt_result.transcript

    return _run_rag_stages(ctx, query, components, settings, t_total)


def run_text_pipeline(query: str) -> PipelineResult:
    """Run the pipeline from a text query (skip STT).

    Useful for benchmarking the retrieval + generation stages without audio.
    """
    t_total = time.perf_counter()
    settings = get_settings()
    components = _PipelineComponents.get()
    ctx = PipelineContext()
    ctx.stt_result = STTResult(transcript=query, provider="text", latency_ms=0.0)

    return _run_rag_stages(ctx, query, components, settings, t_total)


def _run_rag_stages(ctx, query, components, settings, t_total) -> PipelineResult:
    """Shared RAG stages: guardrail → retrieval → relevance → generation → output guard."""

    # ── Stage 3: Input guardrail ─────────────────────────────────
    guard_result = _track_stage(ctx, "input_guard", lambda: components._input_guard.check(query))
    ctx.input_guard = guard_result
    if guard_result and guard_result.action == GuardrailAction.BLOCK:
        return _build_result(ctx, t_total, error=guard_result.reason)

    # ── Stage 4: Hybrid retrieval ────────────────────────────────
    retrieval_results = _track_stage(ctx, "retrieval", lambda: components.get_retriever().retrieve(query))
    if retrieval_results is None:
        retrieval_results = []
    ctx.retrieval_results = retrieval_results

    # ── Stage 5: Relevance guardrail ─────────────────────────────
    rel_guard = _track_stage(
        ctx, "relevance_guard",
        lambda: components._input_guard.check_relevance(retrieval_results, settings.relevance_threshold),
    )
    ctx.relevance_guard = rel_guard

    # Low relevance → still proceed but LLM prompt will instruct refusal
    # Hard block only if no context at all
    if rel_guard and rel_guard.action == GuardrailAction.BLOCK:
        # No context found — generate an abstention without calling LLM
        ctx.generation_result = GenerationResult(
            answer="I don't have enough information to answer this question.",
            latency_ms=0.0,
            model="abstention",
        )
        return _build_result(ctx, t_total)

    # ── Stage 6: Answer generation ───────────────────────────────
    gen_result = _track_stage(ctx, "generation", lambda: components.get_generator().generate(query, retrieval_results))
    if gen_result is None or not gen_result.ok:
        ctx.generation_result = gen_result
        return _build_result(ctx, t_total, error="Answer generation failed")
    ctx.generation_result = gen_result

    # ── Stage 7: Output guardrail (hallucination check) ──────────
    context_texts = [r.text for r in retrieval_results]
    output_check = _track_stage(
        ctx, "output_guard",
        lambda: components._output_guard.check(gen_result.answer, context_texts, query),
    )
    ctx.output_guard = output_check.to_dict() if output_check else None

    return _build_result(ctx, t_total)


# ── Result builder ─────────────────────────────────────────────
def _build_result(ctx: PipelineContext, t0: float, error: Optional[str] = None) -> PipelineResult:
    total = (time.perf_counter() - t0) * 1000
    ctx.total_latency_ms = total

    answer = ""
    is_refusal = False
    grounded = True
    confidence = 0.0   # default — only set if output guard succeeds
    flags = []

    if ctx.generation_result and ctx.generation_result.ok:
        answer = ctx.generation_result.answer

    if ctx.output_guard:
        is_refusal = ctx.output_guard.get("is_refusal", False)
        grounded = ctx.output_guard.get("is_grounded", True)
        confidence = ctx.output_guard.get("confidence", 0.0)
        if not grounded:
            flags.append("hallucination_risk")
    elif ctx.generation_result and ctx.generation_result.ok:
        # Output guard didn't run — use a neutral confidence
        confidence = 0.5

    if ctx.input_guard and ctx.input_guard.flags:
        flags.extend(ctx.input_guard.flags)
    if ctx.relevance_guard and ctx.relevance_guard.flags:
        flags.extend(ctx.relevance_guard.flags)

    if error:
        if not answer:
            answer = f"I'm sorry, I couldn't process your request. {error}"

    context_list = [r.to_dict() for r in ctx.retrieval_results]
    stages = [s.to_dict() for s in ctx.stages]

    return PipelineResult(
        answer=answer,
        transcript=ctx.stt_result.transcript if ctx.stt_result else "",
        is_refusal=is_refusal,
        grounded=grounded,
        confidence=confidence,
        context=context_list,
        stages=stages,
        total_latency_ms=total,
        guardrail_flags=flags,
        error=error,
    )
