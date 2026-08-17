"""Build the vector index from the sample dataset.

Usage:
  python scripts/build_index.py                  # use local sample
  python scripts/build_index.py --full            # download from HuggingFace
  python scripts/build_index.py --full --lang hi  # specific language
  python scripts/build_index.py --strategy semantic  # chunking strategy
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build the RAG vector index")
    parser.add_argument("--full", action="store_true", help="Download from HuggingFace (55 GB)")
    parser.add_argument("--lang", default="", help="Single language code (e.g. hi, bn)")
    parser.add_argument("--strategy", default="semantic",
                        choices=["fixed_size", "recursive", "semantic", "metadata_aware"],
                        help="Chunking strategy")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max examples per language (for testing)")
    parser.add_argument("--sample-path", default="data/sample/sample.jsonl")
    args = parser.parse_args()

    from voice_rag.data.build_index import build_index

    if args.full:
        languages = [args.lang] if args.lang else None
        build_index(
            strategy=args.strategy,
            use_sample=False,
            max_examples=args.max_examples,
            languages=languages,
        )
    else:
        logger.info("Building index from local sample data")
        build_index(
            strategy=args.strategy,
            use_sample=True,
            sample_path=args.sample_path,
        )

    logger.info("Index build complete!")


if __name__ == "__main__":
    main()
