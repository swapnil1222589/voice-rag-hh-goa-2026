"""Factory to pick the configured STT provider at runtime."""
from __future__ import annotations

from .base import STTProvider


def get_stt_provider(provider_name: str = "elevenlabs") -> STTProvider:
    """Return the STT provider named in settings.

    Currently supports:
      - "elevenlabs" → ElevenLabs Scribe (default)
      - "sarvam"     → Sarvam Aasaar (placeholder — implement if switching)
    """
    name = provider_name.lower().strip()
    if name == "elevenlabs":
        from .elevenlabs_provider import ElevenLabsSTT
        return ElevenLabsSTT()
    elif name == "sarvam":
        from .sarvam_provider import SarvamSTT
        return SarvamSTT()
    else:
        raise ValueError(
            f"Unknown STT provider: {provider_name}. Use 'elevenlabs' or 'sarvam'."
        )
