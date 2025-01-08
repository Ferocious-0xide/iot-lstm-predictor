from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base  # Updated import
from sqlalchemy.orm import Session
from contextlib import contextmanager
from config.settings import get_settings

settings = get_settings()

Base = declarative_base()

def create_db_engine(db_url: str):
    return create_engine(db_url)

def get_sensor_db():
    engine = create_engine(settings.sensor_database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_prediction_db():
    engine = create_engine(settings.prediction_database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_sensor_db_context():
    engine = create_engine(settings.sensor_database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()