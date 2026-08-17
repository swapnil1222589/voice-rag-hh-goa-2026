"""OpenAI GPT answer generator with grounded context."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ..indexing.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. You answer questions based \
strictly on the provided context passages.

Rules:
1. Answer ONLY using the information in the provided context.
2. If the context does not contain enough information to answer, say: \
"I don't have enough information to answer this question."
3. Do not make up facts, numbers, or names not present in the context.
4. Keep answers concise and directly address the question.
5. If the question is in an Indic language, answer in the same language.
6. If the question is in English, answer in English.
"""

CONTEXT_HEADER = "Context passages:\n"
CONTEXT_SEPARATOR = "\n---\n"
QUERY_HEADER = "\n\nQuestion: "


@dataclass
class GenerationResult:
    """Structured output from the answer generator."""
    answer: str
    latency_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_used: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    finish_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer.strip())

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "context_used": self.context_used,
            "error": self.error,
            "finish_reason": self.finish_reason,
        }


class AnswerGenerator:
    """Generates answers from retrieved context using OpenAI GPT."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        if api_key is None:
            from config.settings import get_settings
            api_key = get_settings().openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _build_prompt(self, query: str, context: List[RetrievalResult]) -> str:
        parts = [CONTEXT_HEADER]
        for i, r in enumerate(context, 1):
            text = r.text or r.english_text
            parts.append(f"[{i}] {text}")
        parts.append(QUERY_HEADER + query)
        return CONTEXT_SEPARATOR.join(parts[:-1]) + "".join(parts[-1:])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=3.0),
        reraise=True,
    )
    def _call_openai(self, system: str, user: str) -> tuple:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return resp

    def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> GenerationResult:
        """Generate an answer from the query and retrieved context."""
        t0 = time.perf_counter()

        if not context:
            return GenerationResult(
                answer="I don't have enough information to answer this question.",
                latency_ms=(time.perf_counter() - t0) * 1000,
                model=self._model,
                context_used=[],
            )

        user_prompt = self._build_prompt(query, context)

        try:
            resp = self._call_openai(system_prompt, user_prompt)
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("OpenAI generation error: %s", exc)
            return GenerationResult(
                answer="",
                latency_ms=latency,
                model=self._model,
                error=str(exc)[:300],
            )

        latency = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0]
        answer = choice.message.content.strip()
        usage = resp.usage

        return GenerationResult(
            answer=answer,
            latency_ms=latency,
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            context_used=[r.to_dict() for r in context],
            finish_reason=choice.finish_reason,
        )
