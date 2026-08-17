"""Base classes for speech-to-text providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class STTResult:
    """Structured output from every STT provider."""
    transcript: str
    language: Optional[str] = None  # detected language code (e.g. "hi", "en")
    confidence: float = 0.0
    latency_ms: float = 0.0
    provider: str = ""
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.transcript.strip())


class STTProvider(ABC):
    """Common interface implemented by every STT provider."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> STTResult:
        """Transcribe raw audio bytes and return a structured result.

        Args:
            audio_bytes: Raw audio file bytes (wav/mp3/webm/etc.).
            mime_type: MIME type of the audio — passed to providers that need it.

        Returns:
            STTResult with transcript, language, confidence, latency, and any error.
        """
        raise NotImplementedError
