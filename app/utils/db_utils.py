# app/utils/db_utils.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.orm import Session
from contextlib import contextmanager
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

__all__ = [
    'Base', 
    'get_db', 
    'get_db_context', 
    'get_sensor_db', 
    'get_prediction_db',
    'get_sensor_db_context',
    'get_prediction_db_context'
]

# Create separate engines for sensor and prediction databases
sensor_engine = create_engine(settings.DATABASE_URL.replace("postgres://", "postgresql://"))
prediction_engine = create_engine(settings.PREDICTION_DATABASE_URL.replace("postgres://", "postgresql://"))

# Create session factories for each database
SensorSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sensor_engine)
PredictionSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=prediction_engine)

# Base class for SQLAlchemy models
Base = declarative_base()

def get_sensor_db():
    """Get a database session for the read-only sensor database"""
    db = SensorSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_prediction_db():
    """Get a database session for the read-write prediction database"""
    db = PredictionSessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_sensor_db_context():
    """Context manager for sensor database sessions"""
    db = SensorSessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_prediction_db_context():
    """Context manager for prediction database sessions"""
    db = PredictionSessionLocal()
    try:
        yield db
    finally:
        db.close()

# For backward compatibility and places where we only need predictions
get_db = get_prediction_db
get_db_context = get_prediction_db_context