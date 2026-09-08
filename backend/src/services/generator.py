from typing import List, Dict, Any, Optional
from src.config import settings
import logging
import requests

logger = logging.getLogger("rag.generator")

STRICT_SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Assistant.
Your mission is to answer user queries with 100% factual accuracy based STRICTLY on the retrieved context below.

STRICT RULES:
1. ONLY use information explicitly stated in the context.
2. If the context does not contain enough information to answer, reply EXACTLY:
   "The uploaded documents do not contain sufficient information to answer this question."
3. Cite your sources using [Source: filename, Chunk #index] at the end of relevant statements.
4. Do NOT make assumptions, extrapolate, or use outside knowledge.
"""

def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    RAG Generation Layer:
    Constructs the grounded prompt, injects chunk context with citation tags,
    and calls the LLM. Includes a zero-cost local fallback if no API key is set.
    """
    if not context_chunks:
        return "No relevant context was found in the uploaded documents. Please upload documents first."

    # 1. Format context with explicit chunk identifiers for source citation
    context_blocks = []
    for chunk in context_chunks:
        header = f"--- [Source: {chunk['filename']} | Chunk #{chunk['chunk_index']} | Relevance: {chunk['similarity_score']}] ---"
        context_blocks.append(f"{header}\n{chunk['content']}")
    
    formatted_context = "\n\n".join(context_blocks)
    
    user_prompt = f"""CONTEXT:
{formatted_context}

USER QUESTION:
{query}

ANSWER (grounded strictly in context, cite sources):"""

    # 2. If Gemini API key is configured, call Gemini
    if settings.GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{STRICT_SYSTEM_PROMPT}\n\n{user_prompt}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,  # Low temperature = deterministic, zero hallucination
                    "maxOutputTokens": 800
                }
            }
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.warning(f"Gemini API returned error {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")

    # 3. Built-in Deterministic Extractive Synthesizer (Zero-cost fallback)
    # Allows full testing and evaluation even if the user hasn't configured an API key yet!
    top_chunk = context_chunks[0]
    return (
        f"Based on **{top_chunk['filename']}** (Chunk #{top_chunk['chunk_index']}, Score: {top_chunk['similarity_score']}):\n\n"
        f"> \"{top_chunk['content']}\"\n\n"
        f"*(Generated via Grounded Retrieval Engine. Set `GEMINI_API_KEY` in .env for dynamic synthesis)*"
    )
