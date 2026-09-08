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
    HealthResponse,
    StructuredQueryResponse,
    StructuredCitation,
    AgentQueryResponse,
    AgentStepSchema
)
from src.services.chunking import recursive_character_chunking
from src.services.embedding import embedding_service
from src.services.retrieval import RetrievalService
from src.services.generator import generate_answer
from src.services.langchain_rag import run_structured_rag
from src.services.langchain_agent import run_langchain_agent
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
    Ingests, chunks using LangChain's RecursiveCharacterTextSplitter,
    embeds via SentenceTransformers, and stores vectors transactionally in PostgreSQL (pgvector).
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

    # 2. Chunk text with LangChain's RecursiveCharacterTextSplitter
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
    Standard RAG Query (High-performance baseline):
    Retrieves top-k semantically relevant chunks from pgvector and returns answer + sources.
    """
    total_start = time.time()
    
    retrieval_start = time.time()
    if request.use_hybrid:
        retrieved_chunks = RetrievalService.hybrid_search(db, request.query, top_k=request.top_k)
    else:
        retrieved_chunks = RetrievalService.vector_search(db, request.query, top_k=request.top_k)
    
    retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)
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

@router.post("/query/structured", response_model=StructuredQueryResponse, tags=["LangChain Integration"])
def query_structured_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """
    LangChain LCEL Structured Output RAG:
    Executes LCEL chain with Pydantic output parsing.
    Returns: Verified Answer, Confidence Score (0.0 - 1.0), Exact Citations, and 3 Smart Follow-up Questions.
    """
    total_start = time.time()
    
    # 1. Vector Retrieval
    retrieval_start = time.time()
    if request.use_hybrid:
        retrieved_chunks = RetrievalService.hybrid_search(db, request.query, top_k=request.top_k)
    else:
        retrieved_chunks = RetrievalService.vector_search(db, request.query, top_k=request.top_k)
    retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

    # 2. LangChain LCEL Chain Execution
    structured_output = run_structured_rag(request.query, retrieved_chunks)
    total_ms = round((time.time() - total_start) * 1000, 2)

    citations = [
        StructuredCitation(
            filename=c.filename,
            chunk_index=c.chunk_index,
            exact_quote=c.exact_quote
        )
        for c in structured_output.citations
    ]

    return StructuredQueryResponse(
        query=request.query,
        answer=structured_output.answer,
        confidence_score=structured_output.confidence_score,
        citations=citations,
        suggested_followups=structured_output.suggested_followups,
        retrieval_latency_ms=retrieval_ms,
        total_latency_ms=total_ms
    )

@router.post("/query/agent", response_model=AgentQueryResponse, tags=["LangChain Integration"])
def query_agentic_mode(request: QueryRequest, db: Session = Depends(get_db)):
    """
    LangChain Agentic Mode with Tools:
    Autonomous agent equipped with:
    - search_knowledge_base (RAG retriever tool)
    - get_document_stats (Metadata/Stats tool)
    - web_search_fallback (Live Web Search tool)
    Returns step-by-step reasoning steps and final answer.
    """
    total_start = time.time()
    agent_output = run_langchain_agent(db, request.query)
    total_ms = round((time.time() - total_start) * 1000, 2)

    steps = [
        AgentStepSchema(
            step_number=s.step_number,
            thought=s.thought,
            tool_name=s.tool_name,
            tool_input=s.tool_input,
            tool_output=s.tool_output
        )
        for s in agent_output.steps
    ]

    return AgentQueryResponse(
        query=request.query,
        final_answer=agent_output.final_answer,
        tools_used=agent_output.tools_used,
        steps=steps,
        total_steps=agent_output.total_steps,
        latency_ms=total_ms
    )
