import os
import gc
from typing import List

# Restrict multi-threading memory overhead in constrained cloud environments (e.g. Render 512MB)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

from src.config import settings
import logging

logger = logging.getLogger("rag.embedding")

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        """Singleton instance without blocking startup memory."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        """Lazy loader: loads model weights only when first needed, keeping boot memory under 100MB."""
        if cls._model is None:
            logger.info(f"Lazy loading embedding model: {settings.EMBEDDING_MODEL_NAME} (CPU mode)...")
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
            gc.collect()
            logger.info("Embedding model loaded successfully.")
        return cls._model

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single query string into a 384-dimensional normalized vector.
        """
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embeds multiple chunks simultaneously.
        Vectorized batching is 5x to 10x faster than looping one chunk at a time.
        """
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

# Global instance for dependency injection
embedding_service = EmbeddingService()
