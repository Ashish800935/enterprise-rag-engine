from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    FastAPI dependency yielding a transactional database session per request.
    Automatically closes session on request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Bootstrap database:
    1. Ensures pgvector extension is enabled
    2. Creates all tables defined in models.py
    """
    with engine.connect() as connection:
        # Enable pgvector extension inside PostgreSQL
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        connection.commit()
    
    # Create all tables
    import src.db.models  # Ensure models are registered with Base
    Base.metadata.create_all(bind=engine)
