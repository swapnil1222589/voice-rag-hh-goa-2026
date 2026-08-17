"""Dataset loader for MSMARCO-XI.

Loads the dataset from HuggingFace, extracts passages, and yields them as
chunk-ready records with full metadata.  Supports loading a single language
or multiple languages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Indic language code → HuggingFace config name
LANG_CONFIGS = {
    "as": "as", "bn": "bn", "gu": "gu", "hi": "hi",
    "kn": "kn", "ml": "ml", "mr": "mr", "ne": "ne",
    "or": "or", "pa": "pa", "sa": "sa", "ta": "ta",
    "te": "te", "ur": "ur",
}


@dataclass
class PassageRecord:
    """A single passage extracted from MSMARCO-XI, ready for chunking."""
    text: str
    english_text: str
    source_lang: str       # e.g. "hi"
    query: str             # translated query
    eng_query: str         # original English query
    answer: str            # translated answer
    eng_answer: str        # original English answer
    query_type: str        # e.g. "DESCRIPTION", "LOCATION"
    query_id: int
    is_selected: int       # 1 if this passage is the relevant one
    passage_index: int     # position in the passage list

    @property
    def source_doc_id(self) -> str:
        return f"q{self.query_id}_p{self.passage_index}_{self.source_lang}"

    def to_chunk_metadata(self) -> dict:
        return {
            "source_doc_id": self.source_doc_id,
            "source_lang": self.source_lang,
            "query": self.query,
            "english_text": self.english_text,
            "query_type": self.query_type,
            "query_id": self.query_id,
            "is_selected": self.is_selected,
            "passage_index": self.passage_index,
            "eng_query": self.eng_query,
            "answer": self.answer,
            "eng_answer": self.eng_answer,
        }


class MSMARCOSource:
    """Iterates over MSMARCO-XI passages across one or more languages."""

    def __init__(
        self,
        languages: List[str],
        split: str = "train",
        max_examples: Optional[int] = None,
        max_passages_per_example: int = 10,
    ):
        self.languages = [l for l in languages if l in LANG_CONFIGS]
        if not self.languages:
            raise ValueError(
                f"No valid languages. Supported: {list(LANG_CONFIGS.keys())}"
            )
        self.split = split
        self.max_examples = max_examples
        self.max_passages_per_example = max_passages_per_example

    def __iter__(self) -> Iterator[PassageRecord]:
        from datasets import load_dataset

        for lang in self.languages:
            config = LANG_CONFIGS[lang]
            logger.info("Loading MSMARCO-XI [%s] split=%s ...", config, self.split)
            ds = load_dataset("ai4bharat/MSMARCO-XI", config, split=self.split)

            for count, example in enumerate(ds):
                if self.max_examples and count >= self.max_examples:
                    break

                passages = example.get("passages", {})
                is_selected_list = passages.get("is_selected", [])
                eng_passages = passages.get("English_passages", [])
                trans_passages = passages.get("Translated_passages", [])
                n = min(
                    len(trans_passages),
                    len(eng_passages),
                    len(is_selected_list),
                    self.max_passages_per_example,
                )

                for p_idx in range(n):
                    text = (trans_passages[p_idx] or "").strip()
                    eng = (eng_passages[p_idx] or "").strip()
                    if not text and not eng:
                        continue

                    yield PassageRecord(
                        text=text,
                        english_text=eng,
                        source_lang=lang,
                        query=example.get("query", ""),
                        eng_query=example.get("Eng_Query", ""),
                        answer=example.get("Answer", ""),
                        eng_answer=example.get("Eng_Answer", ""),
                        query_type=example.get("query_type", ""),
                        query_id=example.get("query_id", 0),
                        is_selected=is_selected_list[p_idx] if p_idx < len(is_selected_list) else 0,
                        passage_index=p_idx,
                    )

            logger.info("Finished language: %s", lang)


def load_sample_data(path: str = "data/sample/sample.jsonl") -> List[PassageRecord]:
    """Load a small sample file (for local testing without downloading 55 GB)."""
    import json
    records: List[PassageRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            records.append(PassageRecord(**d))
    return records
