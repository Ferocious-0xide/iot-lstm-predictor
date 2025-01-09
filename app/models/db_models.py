# app/models/db_models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary, JSON, Boolean, Table
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from app.utils.db_utils import Base

# For reading sensor data from the shared database
class SensorReading(Base):
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True)
    sensor_id = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<SensorReading(sensor_id={self.sensor_id}, temp={self.temperature}°C, humidity={self.humidity}%, time={self.timestamp})>"

# For our prediction system
class Sensor(Base):
    __tablename__ = "sensor"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    location = Column(String)
    
    # Relationships
    trained_models = relationship("TrainedModel", back_populates="sensor")
    predictions = relationship("Prediction", back_populates="sensor")

    def __repr__(self):
        return f"<Sensor(id={self.id}, name={self.name}, location={self.location})>"

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

    def __repr__(self):
        return f"<TrainedModel(id={self.id}, sensor_id={self.sensor_id}, type={self.model_type}, active={self.is_active})>"

class Prediction(Base):
    __tablename__ = "prediction"
    
    id = Column(Integer, primary_key=True)
    sensor_id = Column(Integer, ForeignKey('sensor.id'), nullable=False)
    model_id = Column(Integer, ForeignKey('trained_model.id'), nullable=False)
    prediction_value = Column(Float, nullable=False)
    prediction_type = Column(String, nullable=False)  # 'temperature' or 'humidity'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    confidence_score = Column(Float)
    
    # Relationships
    sensor = relationship("Sensor", back_populates="predictions")
    model = relationship("TrainedModel", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(sensor_id={self.sensor_id}, type={self.prediction_type}, value={self.prediction_value}, time={self.created_at})>"