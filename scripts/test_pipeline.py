"""Quick smoke test — runs the text pipeline against a single query.

Use this after building the index to verify the pipeline works.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_rag.harness.pipeline import run_text_pipeline


def main():
    queries = [
        "महात्मा गांधी का जन्म कब हुआ था",
        "where is the Taj Mahal located",
        "what is the speed of light",
        "how do vaccines work",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        result = run_text_pipeline(q)
        print(f"Answer: {result.answer}")
        print(f"Grounded: {result.grounded}")
        print(f"Confidence: {result.confidence}")
        print(f"Is refusal: {result.is_refusal}")
        print(f"Total latency: {result.total_latency_ms:.2f}ms")
        print(f"Guardrail flags: {result.guardrail_flags}")
        print(f"\nStages:")
        for s in result.stages:
            print(f"  {s['name']:>20s}: {s['latency_ms']:>8.2f}ms  {'✅' if s['success'] else '❌'}")
        if result.context:
            print(f"\nTop context (first 150 chars):")
            for c in result.context[:2]:
                print(f"  → {c['text'][:150]}...")


if __name__ == "__main__":
    main()
