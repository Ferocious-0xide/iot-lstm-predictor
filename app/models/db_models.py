# app/models/db_models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String)
    temperature = Column(Float)
    humidity = Column(Float)
    timestamp = Column(DateTime)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String)
    temperature_prediction = Column(Float)
    humidity_prediction = Column(Float)
    prediction_timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(String)
    metrics = Column(String)  # JSON string of model metrics

class TrainedModel(Base):
    __tablename__ = "trained_models"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True)
    model_data = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)