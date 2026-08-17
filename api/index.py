"""Vercel serverless entry point for the Voice RAG FastAPI app.

Vercel looks for api/index.py and imports `app` from it.
We simply re-export the FastAPI app object from api/app.py.

Note on limitations vs full server deployment:
- ChromaDB uses a persistent local directory — on Vercel this is ephemeral
  (resets on each cold start). For production with persistence, use Render.com
  which gives a persistent disk. For hackathon demos, the in-memory index
  is rebuilt from the sample data on each cold start (<5s).
- Audio files > 4.5 MB may be rejected by Vercel's payload limit.
  The frontend warns the user and recommends short recordings (<15s).
"""
import os
import sys

# Add src/ to path so voice_rag package is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Also add repo root so config/ is importable
if _root not in sys.path:
    sys.path.insert(0, _root)

from api.app import app  # noqa: F401 — Vercel picks up `app` from this module

__all__ = ["app"]
