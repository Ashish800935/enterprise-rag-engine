import pytest
from src.services.chunking import recursive_character_chunking

def test_chunking_empty_text():
    assert recursive_character_chunking("") == []
    assert recursive_character_chunking("   \n\n  ") == []

def test_chunking_short_text():
    text = "PostgreSQL is an open-source relational database management system."
    chunks = recursive_character_chunking(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunking_splits_on_boundaries():
    para1 = "Paragraph 1 covers database transactions and ACID guarantees."
    para2 = "Paragraph 2 covers vector search using pgvector and HNSW graphs."
    full_text = f"{para1}\n\n{para2}"
    
    # Force split by setting chunk size smaller than combined length
    chunks = recursive_character_chunking(full_text, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 2
    assert "Paragraph 1" in chunks[0]
    assert "Paragraph 2" in chunks[1]

def test_chunking_respects_overlap():
    text = "WordOne WordTwo WordThree WordFour WordFive WordSix WordSeven WordEight WordNine WordTen"
    chunks = recursive_character_chunking(text, chunk_size=40, chunk_overlap=15)
    assert len(chunks) > 1
