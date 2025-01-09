# app/utils/db_utils.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.orm import Session
from contextlib import contextmanager
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

# Create a single engine instance for reuse
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()

def get_db():
    """Get a database session - use this as FastAPI dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    """Context manager for database sessions - use this in with statements"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Maintain old function names for backward compatibility
get_sensor_db = get_db
get_prediction_db = get_db
get_sensor_db_context = get_db_context