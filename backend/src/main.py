from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.config import settings
from src.api.routes import router as api_router
from src.db.session import init_db
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes database and models upon app startup."""
    logger.info("Initializing Enterprise RAG Engine database & pgvector extension...")
    try:
        init_db()
        logger.info("Database schema & vector extension verified successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL on startup (will retry on requests): {e}")
    yield
    logger.info("Shutting down RAG Engine...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Production-grade Hybrid-RAG API with PostgreSQL, pgvector, and FastAPI.",
    lifespan=lifespan
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
def root():
    return {
        "project": settings.PROJECT_NAME,
        "docs_url": "/docs",
        "api_prefix": settings.API_V1_STR,
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
