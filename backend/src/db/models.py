from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from src.db.session import Base
from src.config import settings

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    total_chunks = Column(Integer, default=0)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    # Cascading delete: deleting a document automatically cleans up all associated vector chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    
    # Vector column representing semantic dense embeddings (384 dimensions for all-MiniLM-L6-v2)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")

    # Production Indexing: HNSW (Hierarchical Navigable Small World) Index for sub-millisecond vector similarity search
    __table_args__ = (
        Index(
            "idx_document_chunks_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )
