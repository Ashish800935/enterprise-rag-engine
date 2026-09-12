import os
import gc
from typing import List
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
            if settings.GEMINI_API_KEY:
                try:
                    from google import genai
                    cls._genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                    logger.info("Initialized Gemini Cloud Embedding client (zero local RAM overhead).")
                except Exception as e:
                    logger.warning(f"Could not initialize Gemini embedding client: {e}")
        return cls._instance

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

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single query string into a 384-dimensional vector.
        Uses high-speed Gemini Cloud API if configured (0MB RAM), or local fallback.
        """
        if self._genai_client:
            try:
                from google.genai import types
                res = self._genai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIMENSION)
                )
                return res.embeddings[0].values
            except Exception as e:
                logger.warning(f"Gemini embedding call failed, falling back to local model: {e}")

        # Local fallback
        import torch
        model = self._get_local_model()
        with torch.inference_mode():
            embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embeds multiple chunks simultaneously.
        """
        if not texts:
            return []

        if self._genai_client:
            try:
                from google.genai import types
                res = self._genai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIMENSION)
                )
                return [emb.values for emb in res.embeddings]
            except Exception as e:
                logger.warning(f"Gemini batch embedding call failed, falling back to local model: {e}")

        # Local fallback
        import torch
        model = self._get_local_model()
        with torch.inference_mode():
            embeddings = model.encode(texts, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
        gc.collect()
        return [emb.tolist() for emb in embeddings]

# Global instance for dependency injection
embedding_service = EmbeddingService()
