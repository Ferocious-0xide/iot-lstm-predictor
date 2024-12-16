# app/utils/db_utils.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import logging
from typing import Generator
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create engines for both databases
prediction_engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)

sensor_engine = create_engine(
    settings.SENSOR_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_timeout=30
)

# Create session factories
PredictionSession = sessionmaker(bind=prediction_engine)
SensorSession = sessionmaker(bind=sensor_engine)

Base = declarative_base()

@contextmanager
def get_prediction_db() -> Generator[Session, None, None]:
    """Get session for our prediction database"""
    session = PredictionSession()
    try:
        yield session
    finally:
        session.close()

@contextmanager
def get_sensor_db() -> Generator[Session, None, None]:
    """Get session for the external sensor database"""
    session = SensorSession()
    try:
        yield session
    finally:
        session.close()