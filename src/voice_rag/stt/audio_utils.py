"""Raw audio utilities — normalisation, resampling, format sniffing.

STT providers have different format expectations.  These helpers keep that
complexity out of the pipeline.

Always normalise to 16 kHz mono WAV before sending to any STT provider,
regardless of the input MIME type (even audio/wav may be 44.1 kHz or 48 kHz
from the browser).
"""
from __future__ import annotations

import io
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Format detection ─────────────────────────────────────────────
_MAGIC = {
    b"RIFF": "audio/wav",
    b"ID3": "audio/mp3",
    b"\xff\xfb": "audio/mp3",
    b"\xff\xf3": "audio/mp3",
    b"\xff\xfa": "audio/mp3",
    b"OggS": "audio/ogg",
    b"EBML": "audio/webm",
    b"fLaC": "audio/flac",
}


def detect_mime(audio_bytes: bytes) -> str:
    """Sniff the audio format from magic bytes."""
    for magic, mime in _MAGIC.items():
        if audio_bytes[: len(magic)] == magic:
            return mime
    # webm sometimes starts with 0x1A 0x45
    if audio_bytes[:2] == b"\x1a\x45":
        return "audio/webm"
    return "audio/wav"  # safe fallback


def to_wav(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[bytes, str]:
    """Convert any supported audio to 16 kHz mono WAV using librosa.

    This is always called — even for audio/wav — because browsers often
    record at 44.1 kHz or 48 kHz which some STT providers reject.

    Returns (wav_bytes, mime_type).  Falls back to original bytes on error.
    """
    try:
        import librosa
        import soundfile as sf

        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        buf = io.BytesIO()
        sf.write(buf, y, target_sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read(), "audio/wav"
    except Exception as exc:
        logger.warning("Audio conversion to 16kHz WAV failed (%s); using original bytes", exc)
        return audio_bytes, detect_mime(audio_bytes)
