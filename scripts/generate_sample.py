"""Generate a small sample dataset from MSMARCO-XI for local testing.

Downloads ~5 examples from the Hindi config and saves them as JSONL so the
pipeline can be tested without downloading the full 55 GB dataset.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_rag.data.loader import MSMARCOSource


def main():
    out_dir = "data/sample"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample.jsonl")

    # Load a small sample from Hindi + English
    source = MSMARCOSource(
        languages=["hi", "bn", "ta"],
        split="train",
        max_examples=5,  # 5 examples × ~10 passages = ~150 passages
    )

    records = list(source)
    print(f"Loaded {len(records)} passage records")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    print(f"Sample saved to {out_path}")

    # Also save as individual JSON for inspection
    for i, rec in enumerate(records[:5]):
        with open(f"{out_dir}/example_{i}.json", "w", encoding="utf-8") as f:
            json.dump(rec.__dict__, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
