from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Hybrid-RAG Engine"
    API_V1_STR: str = "/api/v1"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres_password@localhost:5432/rag_db"
    
    # Embedding Configuration (Default: all-MiniLM-L6-v2 -> 384 dimensions)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Ingestion Hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # LLM Provider Configuration
    GEMINI_API_KEY: Optional[str] = None
    
    model_config = SettingsConfigDict(env_file=[".env", "../.env"], env_file_encoding="utf-8", extra="ignore")

settings = Settings()
