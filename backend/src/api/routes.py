from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
import time
import io
from datetime import datetime
from pypdf import PdfReader

from src.db.session import get_db
from src.db.models import Document, DocumentChunk
from src.api.schemas import (
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    ChunkSource,
    HealthResponse
)
from src.services.chunking import recursive_character_chunking
from src.services.embedding import embedding_service
from src.services.retrieval import RetrievalService
from src.services.generator import generate_answer
from src.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(db: Session = Depends(get_db)):
    """Healthcheck endpoint verifying database connectivity and model status."""
    try:
        db.execute(text("SELECT 1;"))
        db_status = "connected (pgvector ready)"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return HealthResponse(
        status="online",
        database=db_status,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        timestamp=datetime.utcnow()
    )

@router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Ingests, chunks, embeds, and stores documents in PostgreSQL (pgvector).
    Supports .txt, .md, and .pdf files.
    """
    filename = file.filename or "unknown.txt"
    content_bytes = await file.read()
    file_size = len(content_bytes)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Extract raw text from file
    raw_text = ""
    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    else:
        # Plain text / Markdown
        try:
            raw_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = content_bytes.decode("latin-1")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract readable text from document.")

    # 2. Chunk text with overlapping sliding window
    chunk_texts = recursive_character_chunking(
        raw_text,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    if not chunk_texts:
        raise HTTPException(status_code=400, detail="Document produced zero chunks.")

    # 3. Batch generate vector embeddings
    vectors = embedding_service.embed_batch(chunk_texts)

    # 4. Save to Database transactionally
    doc = Document(
        filename=filename,
        content_type=file.content_type or "text/plain",
        file_size_bytes=file_size,
        total_chunks=len(chunk_texts)
    )
    db.add(doc)
    db.flush()  # Populates doc.id

    chunks_to_insert = []
    for idx, (text_chunk, vector) in enumerate(zip(chunk_texts, vectors)):
        chunks_to_insert.append(
            DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                content=text_chunk,
                embedding=vector
            )
        )

    db.bulk_save_objects(chunks_to_insert)
    db.commit()
    db.refresh(doc)

    return doc

@router.get("/documents", response_model=List[DocumentResponse], tags=["Ingestion"])
def list_documents(db: Session = Depends(get_db)):
    """Lists all ingested documents in the knowledge base."""
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()

@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Ingestion"])
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Deletes a document and its cascading vector chunks."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    db.delete(doc)
    db.commit()
    return None

@router.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
def query_knowledge_base(request: QueryRequest, db: Session = Depends(get_db)):
    """
    End-to-end RAG Query:
    1. Retrieves top-k semantically relevant chunks from pgvector
    2. Constructs grounded prompt with source citations
    3. Synthesizes verifiable answer
    """
    total_start = time.time()
    
    # Retrieval Phase
    retrieval_start = time.time()
    if request.use_hybrid:
        retrieved_chunks = RetrievalService.hybrid_search(db, request.query, top_k=request.top_k)
    else:
        retrieved_chunks = RetrievalService.vector_search(db, request.query, top_k=request.top_k)
    
    retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

    # Generation Phase
    answer = generate_answer(request.query, retrieved_chunks)
    total_ms = round((time.time() - total_start) * 1000, 2)

    sources = [
        ChunkSource(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            similarity_score=c["similarity_score"],
            content=c["content"]
        )
        for c in retrieved_chunks
    ]

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        retrieval_latency_ms=retrieval_ms,
        total_latency_ms=total_ms
    )
