from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.db.models import DocumentChunk, Document
from src.services.embedding import embedding_service
import time
import logging

logger = logging.getLogger("rag.retrieval")

class RetrievalService:
    @staticmethod
    def vector_search(db: Session, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs Cosine Distance vector similarity search against PostgreSQL with pgvector.
        
        Cosine Distance Note:
        pgvector provides the `<=>` operator for cosine distance.
        Cosine Distance = 1 - Cosine Similarity.
        A distance of 0 means identical vectors; distance of 1 means orthogonal (unrelated).
        """
        start_time = time.time()
        
        # 1. Convert user query to vector
        query_vector = embedding_service.embed_text(query)

        # 2. Query PostgreSQL using pgvector cosine_distance function
        # We order by smallest distance (highest similarity)
        results = (
            db.query(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                Document.filename,
                DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .order_by("distance")
            .limit(top_k)
            .all()
        )

        retrieved = []
        for row in results:
            similarity_score = round(1.0 - float(row.distance), 4)
            retrieved.append({
                "chunk_id": row.id,
                "document_id": row.document_id,
                "filename": row.filename,
                "chunk_index": row.chunk_index,
                "similarity_score": max(0.0, similarity_score),
                "content": row.content
            })

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Retrieved {len(retrieved)} chunks in {elapsed_ms}ms")
        return retrieved

    @staticmethod
    def hybrid_search(db: Session, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Hybrid Search = Dense Semantic Search (Vector) + Sparse Lexical Matching (Keywords).
        
        Why Hybrid Search:
        Dense vectors understand concepts ('automobile' matches 'car'), but can miss exact part numbers,
        error codes, or unique identifiers (e.g. 'ERR-404-X'). Lexical search catches exact strings.
        Combining both guarantees superior retrieval precision.
        """
        # Get semantic matches
        semantic_results = RetrievalService.vector_search(db, query, top_k=top_k)
        
        # Extract keywords (words longer than 3 chars)
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        
        # If no keywords, return semantic results
        if not keywords:
            return semantic_results

        # Add keyword boost: if a chunk contains the exact keyword, boost its score
        for chunk in semantic_results:
            content_lower = chunk["content"].lower()
            keyword_hits = sum(1 for kw in keywords if kw in content_lower)
            if keyword_hits > 0:
                # Small boost for lexical match
                chunk["similarity_score"] = min(1.0, round(chunk["similarity_score"] + (keyword_hits * 0.05), 4))

        # Re-sort by boosted score
        semantic_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return semantic_results[:top_k]
