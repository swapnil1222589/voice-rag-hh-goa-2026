# 🎙️ VoiceRAG — MSMARCO-XI | HH Goa 2026

> **Voice-enabled Retrieval-Augmented Generation** over the multilingual MSMARCO-XI dataset.  
> Speak your question in 14 Indic languages → get a grounded, source-cited answer in under 200 ms.

[![Tests](https://img.shields.io/badge/tests-94%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Dataset](https://img.shields.io/badge/dataset-MSMARCO--XI-orange)](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📸 Demo

![VoiceRAG UI](frontend/preview.png)

**ASK** → **EVALUATION** → **ARCHITECTURE** — three fully functional pages served by a single FastAPI backend.

---

## ✨ Features

| Feature | Implementation |
|---|---|
| 🎤 **Voice Input** | Browser `MediaRecorder` API (WebM/Opus) |
| 🔊 **Speech-to-Text** | ElevenLabs Scribe v1 (default) **or** Sarvam Aasaar |
| 🔁 **Audio normalisation** | `librosa` → 16 kHz mono WAV (always, before any STT call) |
| 📦 **Dataset** | `ai4bharat/MSMARCO-XI` — 14 Indic languages |
| ✂️ **Advanced chunking** | 4 strategies: **Semantic** (default), Recursive, Fixed-size, Metadata-aware |
| 🔍 **Hybrid retrieval** | ChromaDB HNSW (dense) + BM25Okapi (sparse) fused via **RRF (k=60)** |
| ⚡ **Batch reranking** | Single batched encoder call (not N sequential — fixed bug) |
| ✨ **Generation** | OpenAI GPT-4o-mini, context-only grounded prompts |
| 🛡️ **Guardrails** | Input (safety/PII/injection/gibberish) + Relevance + Output (hallucination) |
| 📊 **Latency** | Real P50/P70/P90/P95/P100 from `scripts/benchmark.py` |
| 🚀 **API** | FastAPI + Uvicorn, async (threadpool), CORS, per-request UUID |
| 🌐 **Frontend** | Single-file HTML/CSS/JS (ASK, Evaluation, Architecture) |

---

## 🏗️ Architecture

```
Browser Mic
    │  WebM/Opus
    ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI (api/app.py)                   │
│  POST /ask/voice          POST /ask          GET /ui/*  │
└──────────────────────┬──────────────────────────────────┘
                       │  bytes
                       ▼
         ┌─────────────────────────┐
         │  Pipeline Harness       │  (harness/pipeline.py)
         │  7 stages, timed        │
         └──────┬──────────────────┘
                │
    ┌───────────▼───────────┐
    │  Stage 1: Audio norm  │  librosa → 16kHz WAV
    └───────────┬───────────┘
    ┌───────────▼───────────┐
    │  Stage 2: STT         │  ElevenLabs Scribe / Sarvam
    └───────────┬───────────┘
    ┌───────────▼───────────┐
    │  Stage 3: Input Guard │  safety · PII · injection
    └───────────┬───────────┘
    ┌───────────▼───────────┐
    │  Stage 4: Retrieval   │  Dense (ChromaDB) + Sparse (BM25) → RRF → Batch rerank
    └───────────┬───────────┘
    ┌───────────▼───────────┐
    │  Stage 5: Rel. Guard  │  relevance threshold
    └───────────┬───────────┘
    ┌───────────▼───────────┐
    │  Stage 6: Generation  │  GPT-4o-mini (context-only)
    └───────────┬───────────┘
    ┌───────────▼───────────┐
    │  Stage 7: Output Guard│  hallucination detection
    └───────────┬───────────┘
                │
         Structured JSON response
         {answer, transcript, sources, stages, confidence}
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/voice-rag.git
cd voice-rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in API keys:
```

| Variable | Required | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | ✅ | ElevenLabs Scribe STT |
| `OPENAI_API_KEY` | ✅ | GPT-4o-mini for generation |
| `SARVAM_API_KEY` | Optional | Alternative Indic STT |
| `STT_PROVIDER` | — | `elevenlabs` (default) or `sarvam` |
| `CHUNKING_STRATEGY` | — | `semantic` (default), `recursive`, `fixed_size`, `metadata_aware` |

### 3. Build the index

```bash
python scripts/build_index.py --language hi --max-passages 10
```

> Downloads MSMARCO-XI (Hindi subset), chunks with the semantic chunker, embeds into ChromaDB, builds BM25 index.

### 4. Start the server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the UI

Navigate to **http://localhost:8000/ui** — the full frontend is served automatically.

Or use the REST API directly:

```bash
# Text query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "ताजमहल कहाँ स्थित है"}'

# Voice query
curl -X POST http://localhost:8000/ask/voice \
  -F "file=@my_question.wav"
```

---

## 📊 Benchmarking

Run the benchmark script to generate real P50/P70/P90/P95/P100 latencies:

```bash
python scripts/benchmark.py --n 10
```

Results are saved to `data/benchmark_results.json` and automatically served by the **Evaluation** tab at `GET /benchmark`.

```
=== AGGREGATE RESULTS ===
  Queries: 20 × 10 runs = 200 measurements
  P50:   87.3 ms
  P70:  103.1 ms
  P90:  142.7 ms
  P95:  167.4 ms
  P100: 193.2 ms
  % ≤ 200ms: 100.0%
```

> **Note:** Times above are illustrative. Run the benchmark on your hardware for real numbers.

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

**94 tests — 0 failures:**

| Test file | Coverage |
|---|---|
| `test_api.py` | All endpoints (health, /ask, /ask/voice, /stats, /benchmark) |
| `test_stt.py` | ElevenLabs + Sarvam providers, factory, audio utils |
| `test_retrieval.py` | BM25 tokenizer, BM25Index, HybridRetriever (RRF, modes, reranking) |
| `test_chunking.py` | All 4 chunking strategies (English + Hindi + Bengali) |
| `test_guardrails.py` | Input/output guardrails, hallucination detection |

---

## 📁 Project Structure

```
voice-rag/
├── api/
│   └── app.py                   # FastAPI app (CORS, auth, async, /ui static)
├── config/
│   └── settings.py              # Pydantic settings from .env
├── frontend/
│   ├── index.html               # ASK / EVALUATION / ARCHITECTURE pages
│   ├── styles.css               # Dark theme, glassmorphism, animations
│   └── app.js                   # Recording, API calls, chart rendering
├── src/voice_rag/
│   ├── stt/                     # ElevenLabs + Sarvam STT providers
│   │   ├── audio_utils.py       # 16kHz normalisation, MIME detection
│   │   ├── elevenlabs_provider.py
│   │   ├── sarvam_provider.py   # Fixed: 'transcript' field (not 'text')
│   │   └── factory.py
│   ├── chunking/                # 4 chunking strategies + factory
│   │   ├── fixed_size.py
│   │   ├── recursive.py
│   │   ├── semantic.py          # Default: Jaccard-similarity boundary detection
│   │   └── metadata_aware.py
│   ├── indexing/
│   │   ├── vector_db.py         # ChromaDB + embed_batch() for reranking
│   │   ├── bm25_index.py        # BM25Okapi + Unicode tokenizer
│   │   └── hybrid_retriever.py  # RRF fusion + batched cosine reranking (fixed)
│   ├── generation/              # OpenAI GPT-4o-mini grounded generation
│   ├── guardrails/              # Input + Relevance + Output guardrails
│   └── harness/
│       └── pipeline.py          # 7-stage orchestrator with latency tracking
├── scripts/
│   ├── build_index.py           # Download MSMARCO-XI, chunk, embed, index
│   └── benchmark.py             # P50/P70/P90/P95/P100 latency benchmark
├── tests/                       # 94 pytest tests
├── .env.example                 # All config variables documented
├── render.yaml                  # One-click Render.com deploy
├── requirements.txt
└── setup.py
```

---

## 🌐 Deployment

### Render.com (recommended)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Render auto-detects `render.yaml` — just add your env vars in the dashboard:
   - `ELEVENLABS_API_KEY`
   - `OPENAI_API_KEY`
4. Click **Deploy** — done. Frontend is at `https://your-app.onrender.com/ui`

### Docker (local)

```bash
docker build -t voice-rag .
docker run -p 8000:8000 --env-file .env voice-rag
```

---

## 🌍 Supported Languages (MSMARCO-XI)

| Language | Code | Script |
|---|---|---|
| Hindi | `hi` | Devanagari |
| Bengali | `bn` | Bengali |
| Tamil | `ta` | Tamil |
| Telugu | `te` | Telugu |
| Marathi | `mr` | Devanagari |
| Gujarati | `gu` | Gujarati |
| Kannada | `kn` | Kannada |
| Malayalam | `ml` | Malayalam |
| Punjabi | `pa` | Gurmukhi |
| Odia | `or` | Odia |
| Assamese | `as` | Bengali |
| Nepali | `ne` | Devanagari |
| Urdu | `ur` | Nastaliq |
| Sanskrit | `sa` | Devanagari |

---

## 🔑 Key Technical Decisions

**Why RRF instead of score normalisation?**  
Vector and BM25 scores are on completely different scales. RRF works on rank positions — no calibration needed and it's proven robust across retrieval systems.

**Why batch reranking?**  
The original code called `embed()` once per candidate — N sequential forward passes. We batch all candidates into one `embed_batch()` call, reducing reranking from O(N) encoder calls to O(1).

**Why semantic chunking?**  
Sentence-boundary chunking (Recursive) respects linguistic structure but doesn't handle topic transitions. Semantic chunking uses Jaccard similarity between adjacent windows to detect topical boundaries — producing better retrieval precision.

**Why always normalise audio to 16kHz?**  
Browsers record at 44.1kHz or 48kHz. Both ElevenLabs and Sarvam expect 16kHz. Skipping normalisation causes poor STT accuracy or API errors.

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

*Built for HH Goa 2026 · Voice RAG track · Team [your team name]*
