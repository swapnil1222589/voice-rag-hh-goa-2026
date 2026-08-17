"""Benchmark the Voice RAG pipeline and report real P50/P70/P90/P95/P100 latencies.

Usage:
  python scripts/benchmark.py                     # text mode, 5 runs each
  python scripts/benchmark.py --n 20              # 20 runs per query
  python scripts/benchmark.py --mode voice --audio-dir data/audio/
  python scripts/benchmark.py --queries q.txt     # one query per line
  python scripts/benchmark.py --retrieval-mode dense  # dense-only, bm25, hybrid

All results are saved to data/benchmark_results.json and served by GET /benchmark.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default multilingual queries ────────────────────────────────
DEFAULT_QUERIES = [
    # Hindi
    "महात्मा गांधी का जन्म कब हुआ था",
    "भारत का संविधान कब लागू हुआ",
    "ताजमहल कहाँ स्थित है",
    "प्रकाश की गति क्या है",
    "टीके कैसे काम करते हैं",
    # Bengali
    "বাংলাদেশের রাজধানী কি",
    "ভারতের জাতীয় পশু কোনটি",
    # Tamil
    "தாஜ்மஹால் எங்கே உள்ளது",
    # English
    "where is the Taj Mahal located",
    "what is the speed of light",
    "how do vaccines work",
    "when was India's constitution adopted",
    "who wrote the Indian National Anthem",
    "what is photosynthesis",
    "what causes earthquakes",
    "who was Mahatma Gandhi",
    "what is the capital of India",
    "how does DNA replication work",
    "what are the main causes of climate change",
    "who invented the telephone",
]


def percentile(data: List[float], p: float) -> float:
    return float(np.percentile(data, p))


def run_text_benchmark(
    queries: List[str],
    n_runs: int = 5,
    retrieval_mode: str = "hybrid",
    output_path: str = "data/benchmark_results.json",
):
    from voice_rag.harness.pipeline import run_text_pipeline, _PipelineComponents

    # Pre-warm (not counted)
    logger.info("Pre-warming pipeline (one uncounted run)...")
    try:
        run_text_pipeline(queries[0])
    except Exception as e:
        logger.warning("Warmup failed: %s", e)

    logger.info("Running benchmark: %d queries × %d runs ...", len(queries), n_runs)
    all_latencies: List[float] = []
    query_results = []
    all_stage_latencies: dict = defaultdict(list)
    refusal_count = 0

    for q in queries:
        q_latencies: List[float] = []
        q_refusals = 0
        t0 = time.perf_counter()

        for run_idx in range(n_runs):
            run_t0 = time.perf_counter()
            try:
                result = run_text_pipeline(q)
                elapsed = (time.perf_counter() - run_t0) * 1000
                q_latencies.append(elapsed)

                # Capture stage latencies from ALL runs (not just first)
                for s in result.stages:
                    all_stage_latencies[s["name"]].append(s["latency_ms"])

                if result.is_refusal or (result.error and not result.answer):
                    q_refusals += 1

            except Exception as exc:
                elapsed = (time.perf_counter() - run_t0) * 1000
                q_latencies.append(elapsed)
                logger.warning("Query %r run %d failed: %s", q[:40], run_idx + 1, exc)

        all_latencies.extend(q_latencies)
        refusal_count += q_refusals

        q_data = {
            "query": q,
            "n_runs": n_runs,
            "latencies_ms": [round(l, 2) for l in q_latencies],
            "p50_ms": round(percentile(q_latencies, 50), 2),
            "p70_ms": round(percentile(q_latencies, 70), 2),
            "p90_ms": round(percentile(q_latencies, 90), 2),
            "p95_ms": round(percentile(q_latencies, 95), 2),
            "p100_ms": round(percentile(q_latencies, 100), 2),
            "mean_ms": round(float(np.mean(q_latencies)), 2),
            "min_ms": round(float(np.min(q_latencies)), 2),
            "max_ms": round(float(np.max(q_latencies)), 2),
            "refusals": q_refusals,
        }
        query_results.append(q_data)
        logger.info(
            "  %s — P50: %.0fms  P70: %.0fms  P90: %.0fms  P100: %.0fms  (n=%d)",
            q[:50], q_data["p50_ms"], q_data["p70_ms"], q_data["p90_ms"], q_data["p100_ms"], n_runs,
        )

    # ── Aggregate stats ──────────────────────────────────────────
    # Average stage latency across all runs and queries
    avg_stage_latencies = {
        name: round(float(np.mean(lats)), 2)
        for name, lats in all_stage_latencies.items()
    }

    pct_under_200 = sum(1 for l in all_latencies if l <= 200) / len(all_latencies) if all_latencies else 0.0

    aggregate = {
        "total_queries": len(queries),
        "runs_per_query": n_runs,
        "total_measurements": len(all_latencies),
        "p50_ms": round(percentile(all_latencies, 50), 2),
        "p70_ms": round(percentile(all_latencies, 70), 2),
        "p90_ms": round(percentile(all_latencies, 90), 2),
        "p95_ms": round(percentile(all_latencies, 95), 2),
        "p100_ms": round(percentile(all_latencies, 100), 2),
        "mean_ms": round(float(np.mean(all_latencies)), 2),
        "min_ms": round(float(np.min(all_latencies)), 2),
        "max_ms": round(float(np.max(all_latencies)), 2),
        "pct_under_200ms": round(pct_under_200, 4),
        "refusal_rate": round(refusal_count / (len(queries) * n_runs), 4) if queries else 0.0,
        "retrieval_mode": retrieval_mode,
        "stage_latencies": avg_stage_latencies,
    }

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "text",
        "aggregate": aggregate,
        "queries": query_results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("\n=== AGGREGATE RESULTS ===")
    logger.info("  Queries: %d × %d runs = %d measurements", len(queries), n_runs, len(all_latencies))
    logger.info("  P50:  %.1f ms", aggregate["p50_ms"])
    logger.info("  P70:  %.1f ms", aggregate["p70_ms"])
    logger.info("  P90:  %.1f ms", aggregate["p90_ms"])
    logger.info("  P95:  %.1f ms", aggregate["p95_ms"])
    logger.info("  P100: %.1f ms", aggregate["p100_ms"])
    logger.info("  Mean: %.1f ms", aggregate["mean_ms"])
    logger.info("  %% ≤ 200ms: %.1f%%", pct_under_200 * 100)
    logger.info("  Results saved to %s", output_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Voice RAG pipeline")
    parser.add_argument("--n", type=int, default=5, help="Runs per query (default: 5)")
    parser.add_argument("--iterations", type=int, default=None, help="Alias for --n")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    parser.add_argument("--queries", type=str, default=None, help="Path to queries file (one per line)")
    parser.add_argument("--audio-dir", type=str, default=None, help="Directory with .wav files (voice mode)")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json")
    args = parser.parse_args()

    n_runs = args.iterations if args.iterations is not None else args.n

    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip()]
    else:
        queries = DEFAULT_QUERIES

    logger.info("Mode: %s | Retrieval: %s | n=%d", args.mode, args.retrieval_mode, n_runs)

    if args.mode == "voice":
        logger.warning("Voice mode requires audio files in %s. Falling back to text mode.", args.audio_dir)
        # TODO: implement voice benchmark when audio files are provided
        run_text_benchmark(queries, n_runs, args.retrieval_mode, args.output)
    else:
        run_text_benchmark(queries, n_runs, args.retrieval_mode, args.output)


if __name__ == "__main__":
    main()
