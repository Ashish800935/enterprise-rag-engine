from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DocumentResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    file_size_bytes: int
    total_chunks: int
    uploaded_at: datetime

    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="The natural language question to ask.")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve.")
    use_hybrid: bool = Field(default=True, description="Whether to apply keyword-boosted hybrid retrieval.")

class ChunkSource(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    similarity_score: float
    content: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[ChunkSource]
    retrieval_latency_ms: float
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_model: str
    timestamp: datetime
