# ── Voice RAG Dockerfile ──────────────────────────────────────────
FROM python:3.12-slim

# System deps for audio processing and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model for NLP (used by chunking utilities)
RUN python -m spacy download en_core_web_sm || true

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/chroma_db data/sample data/outputs

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
