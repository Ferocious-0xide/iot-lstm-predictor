# app/utils/db_utils.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
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
            conn.execute(text("SELECT 1"))
            conn.commit()
            
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def get_db():
    """FastAPI dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()