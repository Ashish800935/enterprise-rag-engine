from typing import List
from sentence_transformers import SentenceTransformer
from src.config import settings
import logging

logger = logging.getLogger("rag.embedding")

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        """Singleton pattern: ensures the heavy embedding model weights are loaded only once into memory."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single query string into a 384-dimensional normalized vector.
        """
        if self._model is None:
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embeds multiple chunks simultaneously.
        Vectorized batching is 5x to 10x faster than looping one chunk at a time.
        """
        if not texts:
            return []
        if self._model is None:
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

# Global instance for dependency injection
embedding_service = EmbeddingService()
