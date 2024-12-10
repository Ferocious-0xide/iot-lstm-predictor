# app/utils/db_utils.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging
from config.settings import get_settings
import re

logger = logging.getLogger(__name__)
settings = get_settings()

def get_db_url():
    """Format database URL properly for SQLAlchemy"""
    db_url = settings.DATABASE_URL
    
    # Handle Heroku's DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    # Ensure proper format for SQLAlchemy
    if not any(db_url.startswith(prefix) for prefix in ['postgresql://', 'postgresql+psycopg2://']):
        db_url = f"postgresql://{db_url}"
    
    return db_url

# Create database engine
engine = create_engine(
    get_db_url(),
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)

# Create sessionmaker
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create base class for declarative models
Base = declarative_base()

def init_db():
    """Initialize database, creating all tables"""
    try:
        logger.info("Initializing database...")
        Base.metadata.create_all(bind=engine)
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

@contextmanager
def get_db() -> Session:
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()