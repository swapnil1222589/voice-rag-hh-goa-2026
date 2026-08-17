"""Speech-to-text abstraction.

A common interface lets you swap Sarvam / ElevenLabs / any future provider
without touching the rest of the pipeline.
"""
from .base import STTProvider, STTResult
from .elevenlabs_provider import ElevenLabsSTT
from .factory import get_stt_provider

__all__ = [
    "STTProvider",
    "STTResult",
    "ElevenLabsSTT",
    "get_stt_provider",
]
