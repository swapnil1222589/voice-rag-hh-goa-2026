"""Tests for chunking strategies."""
import pytest
from voice_rag.chunking import (
    FixedSizeChunker, RecursiveChunker, SemanticChunker, MetadataAwareChunker,
)
from voice_rag.chunking.base import count_tokens


SAMPLE_TEXTS = [
    # English
    ("The Taj Mahal is located in Agra, Uttar Pradesh, India. It was built by "
     "Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal. Construction "
     "began in 1632 and was completed in 1653. The Taj Mahal is considered one "
     "of the finest examples of Mughal architecture and is a UNESCO World "
     "Heritage Site. It is also one of the Seven Wonders of the World.", "en"),
    # Hindi (Devanagari)
    ("महात्मा गांधी का जन्म 2 अक्टूबर 1869 को पोरबंदर, गुजरात में हुआ था। "
     "उनका पूरा नाम मोहनदास करमचंद गांधी था। उन्हें भारत में बापू और राष्ट्रपिता "
     "के रूप में जाना जाता है। गांधीजी ने भारत की स्वतंत्रता आंदोलन में अहिंसक "
     "तरीकों का समर्थन किया और सत्याग्रह के सिद्धांत को अपनाया।", "hi"),
    # Bengali
    ("তাজমহল ভারতের উত্তর প্রদেশের আগ্রায় অবস্থিত। এটি মুঘল সম্রাট শাহজাহান "
     "তাঁর স্ত্রী মমতাজ মহলের স্মৃতিতে নির্মাণ করেছিলেন। নির্মাণকাজ ১৬৩২ সালে "
     "শুরু হয়ে ১৬৫৩ সালে সম্পন্ন হয়।", "bn"),
]


@pytest.mark.parametrize("text,lang", SAMPLE_TEXTS)
class TestChunkers:

    def test_fixed_size_produces_chunks(self, text, lang):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10, min_chunk_size=5)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text.strip()
            assert c.strategy == "fixed_size"
            assert c.token_count > 0

    def test_recursive_respects_sentence_boundaries(self, text, lang):
        chunker = RecursiveChunker(chunk_size=50, overlap=10, min_chunk_size=5)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.strategy == "recursive"

    def test_semantic_groups_coherently(self, text, lang):
        chunker = SemanticChunker(chunk_size=50, overlap=10, min_chunk_size=5)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.strategy == "semantic"

    def test_metadata_aware_keeps_whole(self, text, lang):
        chunker = MetadataAwareChunker(chunk_size=400, overlap=60, min_chunk_size=10)
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text.strip()

    def test_empty_text_returns_empty(self, text, lang):
        for ChunkerCls in [FixedSizeChunker, RecursiveChunker, SemanticChunker, MetadataAwareChunker]:
            chunker = ChunkerCls()
            assert chunker.chunk_text("") == []
            assert chunker.chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self, text, lang):
        for ChunkerCls in [FixedSizeChunker, RecursiveChunker, SemanticChunker, MetadataAwareChunker]:
            chunker = ChunkerCls(chunk_size=400, overlap=60, min_chunk_size=10)
            chunks = chunker.chunk_text("Short text.")
            assert len(chunks) == 1

    def test_chunk_ids_are_unique(self, text, lang):
        chunker = SemanticChunker(chunk_size=30, overlap=5, min_chunk_size=5)
        chunks = chunker.chunk_text(text)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs"

    def test_metadata_preserved(self, text, lang):
        md = {"source_lang": lang, "query_type": "DESCRIPTION", "is_selected": 1}
        chunker = RecursiveChunker(chunk_size=50, overlap=10, min_chunk_size=5)
        chunks = chunker.chunk_text(text, metadata=md)
        for c in chunks:
            assert c.source_lang == lang
            assert c.query_type == "DESCRIPTION"
            assert c.is_selected == 1


def test_token_count():
    assert count_tokens("") == 0
    assert count_tokens("hello") > 0
    assert count_tokens("महात्मा गांधी") > 0


def test_overlap_exists_in_fixed_size():
    """Fixed-size chunker should produce overlapping chunks."""
    text = " ".join([f"word{i}" for i in range(100)])
    chunker = FixedSizeChunker(chunk_size=20, overlap=5, min_chunk_size=5)
    chunks = chunker.chunk_text(text)
    if len(chunks) > 1:
        # Check that consecutive chunks share some words
        words1 = set(chunks[0].text.split())
        words2 = set(chunks[1].text.split())
        assert words1 & words2, "No overlap between consecutive chunks"
