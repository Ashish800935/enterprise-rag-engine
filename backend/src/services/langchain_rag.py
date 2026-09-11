from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings
import logging
import json

logger = logging.getLogger("rag.langchain")

# 1. Pydantic Models for Structured Output
class CitationItem(BaseModel):
    filename: str = Field(description="Name of the source document file")
    chunk_index: int = Field(description="Index of the retrieved chunk in the document")
    exact_quote: str = Field(description="A concise, verbatim excerpt from the chunk that justifies the answer")

class StructuredRAGOutput(BaseModel):
    answer: str = Field(description="Clear, fact-grounded answer to the user's question")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (no evidence) to 1.0 (fully supported by context)"
    )
    citations: List[CitationItem] = Field(
        default_factory=list,
        description="List of exact citations proving where information was extracted"
    )
    suggested_followups: List[str] = Field(
        default_factory=list,
        description="3 intelligent, relevant follow-up questions the user might want to explore next"
    )

# 2. Output Parser
structured_parser = PydanticOutputParser(pydantic_object=StructuredRAGOutput)

# 3. LangChain LCEL Prompt Template
STRICT_SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Assistant operating in a strict Retrieval-Augmented Generation (RAG) system.
Your mission is to formulate responses that are 100% factually grounded in the provided CONTEXT.

STRICT OPERATIONAL RULES:
1. Grounding: Answer ONLY using information explicitly stated in the CONTEXT. Never assume or extrapolate.
2. Unanswerable Queries: If the CONTEXT does not contain enough information to answer the question, set confidence_score to 0.0, set answer to "The uploaded documents do not contain sufficient information to answer this question.", and provide empty citations.
3. Citations: For every claim, identify the source filename, chunk index, and provide an exact quote snippet.
4. Follow-up Questions: Propose 3 constructive follow-up questions directly related to the documents.
5. Format: You MUST format your response as valid JSON matching the schema below.

{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", STRICT_SYSTEM_PROMPT),
    ("user", "CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}")
])

def format_context_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved vector chunks into an annotated context block."""
    if not chunks:
        return "No relevant context found."
    
    formatted = []
    for chunk in chunks:
        header = f"--- [File: {chunk['filename']} | Chunk Index: {chunk['chunk_index']} | Similarity: {chunk['similarity_score']}] ---"
        formatted.append(f"{header}\n{chunk['content']}")
    return "\n\n".join(formatted)

def run_structured_rag(query: str, retrieved_chunks: List[Dict[str, Any]]) -> StructuredRAGOutput:
    """
    Executes LangChain LCEL chain with Pydantic Structured Output:
    LCEL Pipeline: (PromptTemplate | ChatGoogleGenerativeAI | PydanticOutputParser)
    """
    if not retrieved_chunks:
        return StructuredRAGOutput(
            answer="No relevant context was found in the uploaded documents. Please upload documents first.",
            confidence_score=0.0,
            citations=[],
            suggested_followups=[
                "How do I upload documents to the knowledge base?",
                "What file formats are supported?",
                "How does pgvector store embeddings?"
            ]
        )

    context_str = format_context_for_llm(retrieved_chunks)

    # 1. Check if Gemini API Key is available
    if settings.GEMINI_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.1,
                max_retries=2
            )

            # Modern LCEL Chain
            lcel_chain = prompt_template | llm | structured_parser
            
            result: StructuredRAGOutput = lcel_chain.invoke({
                "context": context_str,
                "question": query,
                "format_instructions": structured_parser.get_format_instructions()
            })
            return result
        except Exception as e:
            logger.warning(f"LangChain LCEL generation encountered an error: {e}. Falling back to deterministic structured synthesizer.")

    # 2. Deterministic Structured Fallback (Guarantees zero-failure when offline or testing without API key)
    top_chunk = retrieved_chunks[0]
    avg_score = round(sum(c.get("similarity_score", 0.0) for c in retrieved_chunks[:2]) / max(1, len(retrieved_chunks[:2])), 2)
    confidence = min(1.0, max(0.0, avg_score))

    # Extract first 150 chars as quote snippet
    quote = top_chunk['content'][:150].strip() + ("..." if len(top_chunk['content']) > 150 else "")

    return StructuredRAGOutput(
        answer=f"Based on **{top_chunk['filename']}**, the retrieved document states:\n\n> \"{top_chunk['content']}\"",
        confidence_score=confidence,
        citations=[
            CitationItem(
                filename=top_chunk["filename"],
                chunk_index=top_chunk["chunk_index"],
                exact_quote=quote
            )
        ],
        suggested_followups=[
            f"What other details are mentioned in {top_chunk['filename']}?",
            f"Can you summarize all chunks related to '{query}'?",
            "How does hybrid retrieval rank this result against keyword search?"
        ]
    )
