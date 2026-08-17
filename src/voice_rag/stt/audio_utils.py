"""Raw audio utilities — normalisation, resampling, format sniffing.

Always normalise to 16 kHz mono WAV before sending to any STT provider.
Two backends are supported (tried in order):
  1. librosa  — preferred on full installs (local / Docker)
  2. resampy + soundfile — lightweight fallback (Vercel / serverless)
"""
from __future__ import annotations

import io
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Format detection ─────────────────────────────────────────────
_MAGIC = {
    b"RIFF": "audio/wav",
    b"ID3":  "audio/mp3",
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
    if audio_bytes[:2] == b"\x1a\x45":
        return "audio/webm"
    return "audio/wav"


def to_wav(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[bytes, str]:
    """Convert any supported audio to 16 kHz mono WAV.

    Tries librosa first (full installs), falls back to soundfile + resampy
    (serverless / Vercel where librosa's torch dep is too heavy).
    Returns (wav_bytes, mime_type).
    """
    # ── Try librosa (preferred) ────────────────────────────────────
    try:
        import librosa
        import soundfile as sf

        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
        buf = io.BytesIO()
        sf.write(buf, y, target_sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read(), "audio/wav"
    except ImportError:
        pass  # librosa not available — try soundfile + resampy
    except Exception as exc:
        logger.warning("librosa conversion failed (%s); trying soundfile fallback", exc)

    # ── Fallback: soundfile + resampy ─────────────────────────────
    try:
        import numpy as np
        import resampy
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True, dtype="float32")
        # Mix down to mono
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        # Resample if needed
        if sr != target_sr:
            data = resampy.resample(data, sr, target_sr)
        buf = io.BytesIO()
        sf.write(buf, data, target_sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read(), "audio/wav"
    except Exception as exc:
        logger.warning("soundfile/resampy conversion failed (%s); using original bytes", exc)

    return audio_bytes, detect_mime(audio_bytes)
