import os
import gc
from typing import List
from fastapi import HTTPException
from src.config import settings
import logging

logger = logging.getLogger("rag.embedding")

class EmbeddingService:
    _instance = None
    _local_model = None
    _genai_client = None

    def __new__(cls):
        """Singleton instance without blocking startup memory."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    @classmethod
    def _get_genai_client(cls):
        """Lazy-loads and returns the Gemini client if API key is provided."""
        if cls._genai_client is None:
            api_key = (settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
            if api_key:
                try:
                    from google import genai
                    cls._genai_client = genai.Client(api_key=api_key)
                    logger.info("Initialized Gemini Cloud Embedding client (0 MB local RAM overhead).")
                except Exception as e:
                    logger.error(f"Could not initialize Gemini embedding client: {e}")
        return cls._genai_client

    @classmethod
    def _get_local_model(cls):
        """Lazy loader for local PyTorch model only when running completely offline without API key."""
        if cls._local_model is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            import torch
            torch.set_num_threads(1)
            torch.set_grad_enabled(False)
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local fallback model: {settings.EMBEDDING_MODEL_NAME}...")
            cls._local_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
            gc.collect()
        return cls._local_model

    def _call_gemini_embed(self, client, text: str) -> List[float]:
        """Calls Gemini embedding with gemini-embedding-001 (fallback to embedding-001)."""
        from google.genai import types
        models_to_try = ["gemini-embedding-001", "embedding-001"]
        last_err = None
        for m in models_to_try:
            try:
                res = client.models.embed_content(
                    model=m,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIMENSION)
                )
                return res.embeddings[0].values
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("No compatible Gemini embedding model found.")

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single query string into a 384-dimensional vector.
        Uses high-speed Gemini Cloud API if configured (0MB RAM), or local fallback.
        """
        client = self._get_genai_client()
        if client:
            try:
                return self._call_gemini_embed(client, text)
            except Exception as e:
                logger.error(f"Gemini embedding call failed: {e}")
                if os.getenv("RENDER"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Gemini API error during query embedding: {str(e)}. Please verify your GEMINI_API_KEY."
                    )

        # Local fallback (for local development)
        import torch
        model = self._get_local_model()
        with torch.inference_mode():
            embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embeds multiple chunks simultaneously.
        Uses gemini-embedding-001 on Google Cloud (0MB RAM) when API key is provided,
        or lightweight local CPU SentenceTransformer fallback on local machines.
        """
        if not texts:
            return []

        client = self._get_genai_client()
        if client:
            try:
                results = []
                for chunk in texts:
                    vec = self._call_gemini_embed(client, chunk)
                    results.append(vec)
                return results
            except Exception as e:
                logger.error(f"Gemini batch embedding call failed: {e}")
                if os.getenv("RENDER"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Gemini embedding API error: {str(e)}. Please check your GEMINI_API_KEY in Render Environment Variables."
                    )

        # If running on Render's 512MB free tier without GEMINI_API_KEY, prevent kernel OOM SIGKILL
        if os.getenv("RENDER"):
            raise HTTPException(
                status_code=400,
                detail="GEMINI_API_KEY is not configured in Render. On Render's 512MB free tier, cloud embedding via GEMINI_API_KEY is required to prevent memory limit crashes. Please add GEMINI_API_KEY in Render Dashboard -> Environment Variables."
            )

        # Local fallback with low memory footprint (for local machines with sufficient RAM)
        import torch
        model = self._get_local_model()
        gc.collect()
        with torch.inference_mode():
            embeddings = model.encode(texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
        gc.collect()
        return [emb.tolist() for emb in embeddings]

# Global instance for dependency injection
embedding_service = EmbeddingService()
