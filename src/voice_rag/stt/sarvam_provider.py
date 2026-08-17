"""Sarvam Aasaar speech-to-text provider (alternative to ElevenLabs).

Provided so the pipeline can switch to Sarvam STT by changing one setting —
STT_PROVIDER=sarvam.  The harness code never needs to change.

Sarvam API docs: https://docs.sarvam.ai/api-reference-docs/endpoints/speech-to-text
Response shape: {"transcript": "...", "language_code": "hi-IN", "request_id": "..."}
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .base import STTProvider, STTResult

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TIMEOUT = 30.0

# Supported Indic languages by Sarvam Aasaar
SARVAM_SUPPORTED_LANGS = {
    "hi", "bn", "gu", "kn", "ml", "mr", "pa", "ta", "te",
    "as", "or", "ne", "ur",
}


class SarvamSTT(STTProvider):
    """Sarvam Aasaar speech-to-text (Indic-optimised)."""

    name = "sarvam"

    def __init__(self, api_key: Optional[str] = None, timeout: float = SARVAM_TIMEOUT):
        if api_key is None:
            from config.settings import get_settings
            api_key = get_settings().sarvam_api_key
        if not api_key:
            raise ValueError("Sarvam API key required. Set SARVAM_API_KEY in .env")
        self._api_key = api_key
        self._timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=3.0),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
        ),
        reraise=True,
    )
    def _post(self, audio_bytes: bytes, lang_code: str = "") -> httpx.Response:
        headers = {"api-subscription-key": self._api_key}
        data = {}
        if lang_code:
            data["language_code"] = lang_code
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                SARVAM_STT_URL,
                headers=headers,
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data=data or None,
            )
            resp.raise_for_status()
            return resp

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> STTResult:
        t0 = time.perf_counter()
        try:
            resp = self._post(audio_bytes)
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("Sarvam STT HTTP %s: %s", exc.response.status_code, exc)
            return STTResult(
                transcript="",
                latency_ms=latency,
                provider=self.name,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("Sarvam STT error: %s", exc)
            return STTResult(
                transcript="",
                latency_ms=latency,
                provider=self.name,
                error=str(exc)[:300],
            )

        latency = (time.perf_counter() - t0) * 1000

        # Sarvam returns {"transcript": "...", "language_code": "hi-IN", ...}
        # Fall back to "text" key for compatibility with older API versions
        transcript = (
            data.get("transcript")
            or data.get("text")
            or ""
        ).strip()

        # language_code is typically "hi-IN" — extract the short code
        lang_raw = data.get("language_code", "").lower()
        language = lang_raw[:2] if len(lang_raw) >= 2 else lang_raw or None

        return STTResult(
            transcript=transcript,
            language=language,
            confidence=float(data.get("confidence", 0.0)),
            latency_ms=latency,
            provider=self.name,
            raw=data,
        )
