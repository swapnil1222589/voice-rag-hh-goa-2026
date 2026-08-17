"""Unit tests for STT providers — all HTTP calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_rag.stt.base import STTResult


# ── ElevenLabs STT ────────────────────────────────────────────────
class TestElevenLabsSTT:
    def _make_stt(self, api_key: str = "test_key"):
        from voice_rag.stt.elevenlabs_provider import ElevenLabsSTT
        return ElevenLabsSTT(api_key=api_key)

    def test_successful_transcription(self):
        stt = self._make_stt()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "text": "Hello from ElevenLabs",
            "language_code": "en",
        }
        fake_resp.raise_for_status = MagicMock()

        with patch.object(stt, "_call_api", return_value=fake_resp):
            result = stt.transcribe(b"fake_audio_bytes", "audio/wav")

        assert result.ok
        assert result.transcript == "Hello from ElevenLabs"
        assert result.provider == "elevenlabs"
        assert result.error is None

    def test_empty_transcript_returns_not_ok(self):
        stt = self._make_stt()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"text": "", "language_code": "en"}
        fake_resp.raise_for_status = MagicMock()

        with patch.object(stt, "_call_api", return_value=fake_resp):
            result = stt.transcribe(b"fake_audio_bytes", "audio/wav")

        assert not result.ok

    def test_http_error_returns_error_result(self):
        import httpx
        stt = self._make_stt()
        with patch.object(stt, "_call_api", side_effect=httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(status_code=401, text="Unauthorized")
        )):
            result = stt.transcribe(b"fake_audio_bytes", "audio/wav")

        assert not result.ok
        assert result.error is not None
        assert "401" in result.error

    def test_language_normalisation(self):
        stt = self._make_stt()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "text": "Hindi text here",
            "language_code": "hin",   # ISO-639-3
        }
        fake_resp.raise_for_status = MagicMock()

        with patch.object(stt, "_call_api", return_value=fake_resp):
            result = stt.transcribe(b"fake", "audio/wav")

        # Should normalise "hin" → "hi"
        assert result.language == "hi"

    def test_no_api_key_raises(self):
        from voice_rag.stt.elevenlabs_provider import ElevenLabsSTT
        with pytest.raises(ValueError, match="API key"):
            ElevenLabsSTT(api_key="")


# ── Sarvam STT ────────────────────────────────────────────────────
class TestSarvamSTT:
    def _make_stt(self, api_key: str = "test_key"):
        from voice_rag.stt.sarvam_provider import SarvamSTT
        return SarvamSTT(api_key=api_key)

    def test_successful_transcription(self):
        stt = self._make_stt()
        fake_resp = MagicMock()
        # Sarvam returns 'transcript' (not 'text')
        fake_resp.json.return_value = {
            "transcript": "नमस्ते दुनिया",
            "language_code": "hi-IN",
        }
        fake_resp.raise_for_status = MagicMock()

        with patch.object(stt, "_post", return_value=fake_resp):
            result = stt.transcribe(b"fake_audio_bytes", "audio/wav")

        assert result.ok
        assert result.transcript == "नमस्ते दुनिया"
        assert result.language == "hi"
        assert result.provider == "sarvam"

    def test_fallback_to_text_field(self):
        """Older Sarvam API versions returned 'text' instead of 'transcript'."""
        stt = self._make_stt()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "text": "Fallback text",
            "language_code": "en-US",
        }
        fake_resp.raise_for_status = MagicMock()

        with patch.object(stt, "_post", return_value=fake_resp):
            result = stt.transcribe(b"fake_audio_bytes", "audio/wav")

        assert result.ok
        assert result.transcript == "Fallback text"

    def test_http_error_returns_error_result(self):
        import httpx
        stt = self._make_stt()
        with patch.object(stt, "_post", side_effect=httpx.HTTPStatusError(
            "403", request=MagicMock(), response=MagicMock(status_code=403, text="Forbidden")
        )):
            result = stt.transcribe(b"fake_audio_bytes", "audio/wav")

        assert not result.ok
        assert "403" in result.error

    def test_no_api_key_raises(self):
        from voice_rag.stt.sarvam_provider import SarvamSTT
        with pytest.raises(ValueError, match="Sarvam API key"):
            SarvamSTT(api_key="")

    def test_language_code_extraction(self):
        stt = self._make_stt()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "transcript": "বাংলা পাঠ্য",
            "language_code": "bn-BD",
        }
        fake_resp.raise_for_status = MagicMock()

        with patch.object(stt, "_post", return_value=fake_resp):
            result = stt.transcribe(b"fake", "audio/wav")

        assert result.language == "bn"


# ── STTResult ─────────────────────────────────────────────────────
class TestSTTResult:
    def test_ok_true(self):
        r = STTResult(transcript="hello", provider="test")
        assert r.ok

    def test_ok_false_empty(self):
        r = STTResult(transcript="   ", provider="test")
        assert not r.ok

    def test_ok_false_error(self):
        r = STTResult(transcript="hello", error="some error", provider="test")
        assert not r.ok


# ── Audio utils ────────────────────────────────────────────────────
class TestAudioUtils:
    def test_detect_mime_wav(self):
        from voice_rag.stt.audio_utils import detect_mime
        assert detect_mime(b"RIFF\x00\x00\x00\x00WAVE") == "audio/wav"

    def test_detect_mime_webm(self):
        from voice_rag.stt.audio_utils import detect_mime
        assert detect_mime(b"\x1a\x45\xdf\xa3") == "audio/webm"

    def test_detect_mime_mp3(self):
        from voice_rag.stt.audio_utils import detect_mime
        assert detect_mime(b"ID3\x03\x00") == "audio/mp3"

    def test_detect_mime_fallback(self):
        from voice_rag.stt.audio_utils import detect_mime
        assert detect_mime(b"\x00\x00\x00\x00") == "audio/wav"


# ── STT Factory ───────────────────────────────────────────────────
class TestSTTFactory:
    def test_factory_elevenlabs(self):
        from voice_rag.stt import factory
        from voice_rag.stt.elevenlabs_provider import ElevenLabsSTT
        with patch("voice_rag.stt.elevenlabs_provider.ElevenLabsSTT") as mock:
            mock.return_value = MagicMock(spec=ElevenLabsSTT)
            # factory imports ElevenLabsSTT locally, so reload to pick up patch
            import importlib
            importlib.reload(factory)
            # Just verify it doesn't raise with valid name
            from voice_rag.stt.base import STTProvider

    def test_factory_sarvam(self):
        """Test that factory can instantiate Sarvam STT."""
        from voice_rag.stt.factory import get_stt_provider
        from voice_rag.stt.sarvam_provider import SarvamSTT
        # Patch the class itself so no API key is needed
        with patch("voice_rag.stt.sarvam_provider.SarvamSTT.__init__", return_value=None):
            provider = get_stt_provider("sarvam")
            assert provider is not None
            assert isinstance(provider, SarvamSTT)

    def test_factory_unknown_raises(self):
        from voice_rag.stt.factory import get_stt_provider
        with pytest.raises(ValueError, match="Unknown STT provider"):
            get_stt_provider("whisper")
