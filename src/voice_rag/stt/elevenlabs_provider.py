"""ElevenLabs speech-to-text provider.

Uses the ElevenLabs Scribe API for multilingual transcription, which is well-suited
for Indic languages (MSMARCO-XI covers 14 Indic languages).
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from .base import STTProvider, STTResult

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_TIMEOUT = 30.0
SUPPORTED_LANGS = {
    "hi", "bn", "gu", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur",
    "as", "en", "sa",
}


class ElevenLabsSTT(STTProvider):
    """ElevenLabs Scribe speech-to-text."""

    name = "elevenlabs"

    def __init__(self, api_key: Optional[str] = None, timeout: float = ELEVENLABS_TIMEOUT):
        # Lazy import to avoid hard circular dep on settings.
        if api_key is None:
            from config.settings import get_settings
            api_key = get_settings().elevenlabs_api_key
        if not api_key:
            raise ValueError(
                "ElevenLabs API key is required. Set ELEVENLABS_API_KEY in .env"
            )
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
    def _call_api(self, files_payload: dict, params: dict) -> httpx.Response:
        headers = {"xi-api-key": self._api_key}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                ELEVENLABS_STT_URL,
                headers=headers,
                files=files_payload,
                params=params,
            )
            resp.raise_for_status()
            return resp

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> STTResult:
        import time

        t0 = time.perf_counter()
        filename = "audio.webm" if "webm" in mime_type else "audio.wav"

        # ElevenLabs Scribe accepts language hints — leave it auto-detect
        # since MSMARCO-XI is multilingual, but we set tag_events for richer output.
        params = {
            "tag_events": "false",
            "diarize": "false",
        }

        files_payload = {
            "file": (filename, audio_bytes, mime_type),
            "model_id": (None, "scribe_v1"),
        }

        try:
            resp = self._call_api(files_payload, params)
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("ElevenLabs STT HTTP error %s: %s", exc.response.status_code, exc)
            return STTResult(
                transcript="",
                latency_ms=latency,
                provider=self.name,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("ElevenLabs STT error: %s", exc)
            return STTResult(
                transcript="",
                latency_ms=latency,
                provider=self.name,
                error=str(exc)[:300],
            )

        latency = (time.perf_counter() - t0) * 1000

        # ElevenLabs Scribe returns {"text": "...", "language_code": "hin", ...}
        transcript = data.get("text", "").strip()
        lang_raw = data.get("language_code", "").lower()

        # Normalise ISO-639-3 → short code (hin → hi)
        lang_map = {
            "hin": "hi", "ben": "bn", "guj": "gu", "kan": "kn",
            "mal": "ml", "mar": "mr", "nep": "ne", "ori": "or",
            "pan": "pa", "tam": "ta", "tel": "te", "urd": "ur",
            "asm": "as", "eng": "en", "san": "sa",
        }
        language = lang_map.get(lang_raw, lang_raw[:2] if len(lang_raw) >= 2 else lang_raw)

        return STTResult(
            transcript=transcript,
            language=language or None,
            confidence=float(data.get("confidence", 0.0)),
            latency_ms=latency,
            provider=self.name,
            raw=data,
        )
