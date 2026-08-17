"""Integration tests for the FastAPI endpoints.

These tests use a mock pipeline so no real API keys are required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Build a fake PipelineResult ──────────────────────────────────
def _fake_result(query="test"):
    from voice_rag.harness.pipeline import PipelineResult
    return PipelineResult(
        answer="The answer is 42.",
        transcript=query,
        grounded=True,
        confidence=0.85,
        context=[{
            "text": "Context passage here.",
            "metadata": {"source_lang": "en"},
            "rrf_score": 0.05,
            "vector_score": 0.8,
            "bm25_score": 0.3,
            "final_rank": 0,
            "english_text": "",
        }],
        stages=[{"name": "retrieval", "latency_ms": 45.0, "success": True, "error": ""}],
        total_latency_ms=120.0,
        guardrail_flags=[],
    )


@pytest.fixture()
def client():
    # Patch the heavy pipeline functions so no real models are loaded
    with (
        patch("api.app.run_text_pipeline", return_value=_fake_result()),
        patch("api.app.run_pipeline", return_value=_fake_result("voice query")),
    ):
        from api.app import app
        yield TestClient(app)


# ── Health ────────────────────────────────────────────────────────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── /ask ──────────────────────────────────────────────────────────
def test_ask_text_query(client):
    resp = client.post("/ask", json={"query": "What is the capital of India?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "request_id" in data
    assert "stages" in data
    assert "total_latency_ms" in data


def test_ask_empty_query(client):
    """Empty query should fail Pydantic validation (min_length=1)."""
    resp = client.post("/ask", json={"query": ""})
    assert resp.status_code == 422


def test_ask_missing_query(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_has_request_id(client):
    r1 = client.post("/ask", json={"query": "hello"})
    r2 = client.post("/ask", json={"query": "hello"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Each request should have a unique request_id
    assert r1.json()["request_id"] != r2.json()["request_id"]


def test_ask_sources_in_response(client):
    resp = client.post("/ask", json={"query": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["sources"], list)


# ── /ask/voice ────────────────────────────────────────────────────
def test_ask_voice_empty_file(client):
    """Empty audio should return 400."""
    resp = client.post(
        "/ask/voice",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_ask_voice_large_file(client):
    """File over 25MB limit should return 413."""
    large = b"\x00" * (26 * 1024 * 1024)
    resp = client.post(
        "/ask/voice",
        files={"file": ("big.wav", large, "audio/wav")},
    )
    assert resp.status_code == 413


def test_ask_voice_valid_audio(client):
    """Minimal WAV header should be accepted and processed."""
    # 44-byte WAV header (silent audio)
    wav = (
        b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")   # PCM
        + (1).to_bytes(2, "little")   # mono
        + (16000).to_bytes(4, "little")  # sample rate
        + (32000).to_bytes(4, "little")  # byte rate
        + (2).to_bytes(2, "little")   # block align
        + (16).to_bytes(2, "little")  # bits per sample
        + b"data" + (0).to_bytes(4, "little")
    )
    resp = client.post(
        "/ask/voice",
        files={"file": ("audio.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "request_id" in data


# ── /stats ────────────────────────────────────────────────────────
def test_stats(client):
    """Stats endpoint should return a valid JSON response."""
    # The endpoint imports _PipelineComponents from pipeline internally
    # and will gracefully degrade on error — just check it returns 200
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    # Should have either count fields or an error field
    assert "vector_db_count" in data or "error" in data



# ── /benchmark ────────────────────────────────────────────────────
def test_benchmark_no_file(client, tmp_path):
    """When no benchmark_results.json exists, should return error key."""
    import os
    with patch("api.app.os.path.exists", return_value=False):
        resp = client.get("/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["data"] is None


def test_benchmark_with_file(client, tmp_path):
    """When benchmark_results.json exists, should serve it."""
    fake_data = {"aggregate": {"p50_ms": 95.0}, "queries": []}
    with (
        patch("api.app.os.path.exists", return_value=True),
        patch("builtins.open", unittest_mock_open(fake_data)),
    ):
        resp = client.get("/benchmark")
        assert resp.status_code == 200


def unittest_mock_open(data: dict):
    """Helper to mock open() to return JSON data."""
    import io
    from unittest.mock import mock_open
    m = mock_open(read_data=json.dumps(data))
    return m
