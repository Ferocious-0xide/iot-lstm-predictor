from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary, JSON, Boolean
from sqlalchemy.orm import declarative_base, relationship  # Updated import
from datetime import datetime

Base = declarative_base()

class Sensor(Base):
    __tablename__ = "sensor"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    location = Column(String)
    
    # Relationships
    trained_models = relationship("TrainedModel", back_populates="sensor")
    predictions = relationship("Prediction", back_populates="sensor")

class TrainedModel(Base):
    __tablename__ = "trained_model"
    id = Column(Integer, primary_key=True)
    sensor_id = Column(Integer, ForeignKey('sensor.id'), nullable=False)
    model_data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_metrics = Column(JSON)
    model_type = Column(String, nullable=False, default='lstm')
    is_active = Column(Boolean, default=True)
    
    # Relationships
    sensor = relationship("Sensor", back_populates="trained_models")
    predictions = relationship("Prediction", back_populates="model")

class Prediction(Base):
    __tablename__ = "prediction"
    id = Column(Integer, primary_key=True)
    sensor_id = Column(Integer, ForeignKey('sensor.id'), nullable=False)
    model_id = Column(Integer, ForeignKey('trained_model.id'), nullable=False)
    prediction_value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    confidence_score = Column(Float)
    
    # Relationships
    sensor = relationship("Sensor", back_populates="predictions")
    model = relationship("TrainedModel", back_populates="predictions")