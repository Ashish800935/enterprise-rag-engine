import pytest
from unittest.mock import patch
from src.services.retrieval import RetrievalService

def test_hybrid_search_keyword_boost():
    # Mock vector_search to return deterministic dummy chunks
    mock_candidates = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "server_guide.md",
            "chunk_index": 0,
            "similarity_score": 0.80,
            "content": "General information about server setup and memory."
        },
        {
            "chunk_id": 2,
            "document_id": 1,
            "filename": "troubleshooting.md",
            "chunk_index": 1,
            "similarity_score": 0.75,
            "content": "Handling fatal errors: ERR-404-TIMEOUT occurs when network packet drops."
        }
    ]

    with patch.object(RetrievalService, "vector_search", return_value=mock_candidates):
        # Query containing the exact keyword 'ERR-404-TIMEOUT'
        query = "How to fix ERR-404-TIMEOUT error?"
        results = RetrievalService.hybrid_search(db=None, query=query, top_k=2)

        # Chunk 2 had lower semantic score (0.75 vs 0.80), but contains 'err-404-timeout'
        # With keyword boost (+0.05 per keyword hit), Chunk 2 should be boosted
        boosted_chunk = next(c for c in results if c["chunk_id"] == 2)
        assert boosted_chunk["similarity_score"] > 0.75
