# app/utils/db_utils.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.orm import Session
from contextlib import contextmanager
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

# Create separate engines for sensor and prediction databases
sensor_engine = create_engine(settings.DATABASE_URL)
prediction_engine = create_engine(settings.PREDICTION_DATABASE_URL)

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

# For backward compatibility and places where we only need predictions
get_db = get_prediction_db