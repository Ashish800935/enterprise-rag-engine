from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import settings

raw_url = str(settings.DATABASE_URL).strip().strip('"').strip("'")
# Strip potential environment variable key prefixes if copied from .env file
if "=" in raw_url and not raw_url.startswith("postgresql") and not raw_url.startswith("postgres"):
    raw_url = raw_url.split("=", 1)[1].strip().strip('"').strip("'")

db_url = raw_url
if db_url.startswith("postgresql://") and not db_url.startswith("postgresql+"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
elif db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)

is_remote_db = any(domain in db_url for domain in ["supabase.com", "pooler", "amazonaws.com", "render.com"])

connect_args = {}
if is_remote_db:
    connect_args = {
        "sslmode": "require",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        "connect_timeout": 15
    }

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=60,
    pool_size=5,
    max_overflow=10
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
