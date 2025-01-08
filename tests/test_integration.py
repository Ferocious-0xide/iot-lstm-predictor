import pytest
import tensorflow as tf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.db_models import Base, Sensor, TrainedModel, Prediction
from app.utils.model_persistence import ModelPersistence
from app.services.training import ModelTrainingService
from app.services.prediction import PredictionService
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def db_session():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    # Create a test sensor
    sensor = Sensor(id=1, name="Test Sensor", location="Test Location")
    session.add(sensor)
    session.commit()
    
    yield session
    session.close()

@pytest.fixture
def sample_data():
    # Create some sample sensor readings
    timestamps = [datetime.utcnow() - timedelta(minutes=5*i) for i in range(24)]
    readings = [
        {
            "sensor_id": 1,
            "temperature": float(20 + np.sin(i/4)),
            "humidity": float(50 + np.cos(i/4)),
            "timestamp": ts.isoformat()
        }
        for i, ts in enumerate(timestamps)
    ]
    return readings

def create_test_model():
    # Input layer
    inputs = tf.keras.layers.Input(shape=(24, 2))
    
    # LSTM layers
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32)(x)
    
    # Split into two heads for temperature and humidity
    temp_output = tf.keras.layers.Dense(1, name='temperature')(x)
    humid_output = tf.keras.layers.Dense(1, name='humidity')(x)
    
    # Create model with multiple outputs
    model = tf.keras.Model(inputs=inputs, outputs=[temp_output, humid_output])
    model.compile(optimizer='adam', loss='mse')
    
    return model

def test_end_to_end_flow(db_session, sample_data):
    """Test the entire flow from model creation to prediction"""
    
    # 1. Create and save a model
    model = create_test_model()
    
    # Create sample training data
    X = np.random.random((100, 24, 2))
    y_temp = np.random.random((100, 1))
    y_humid = np.random.random((100, 1))
    
    # Train model
    history = model.fit(
        X, [y_temp, y_humid],
        epochs=1,
        verbose=0
    )
    
    # Save model
    persistence = ModelPersistence(db_session)
    model_record = persistence.save_model(
        model=model,
        sensor_id=1,
        metrics={'loss': float(history.history['loss'][-1])},
        is_active=True
    )
    
    assert model_record is not None
    assert model_record.sensor_id == 1
    
    # 2. Test prediction service
    prediction_service = PredictionService(db_session)
    
    # Make predictions synchronously
    predictions = prediction_service.predict_sync(
        sensor_id=1,
        readings=sample_data,
        steps_ahead=3
    )
    
    assert len(predictions) == 3
    for pred in predictions:
        assert 'temperature' in pred
        assert 'humidity' in pred
        assert 'timestamp' in pred

def test_model_versioning(db_session):
    """Test that model versioning works correctly"""
    
    persistence = ModelPersistence(db_session)
    
    # Create two versions of a model
    model1 = create_test_model()
    model2 = create_test_model()
    
    # Save both models
    record1 = persistence.save_model(model1, sensor_id=1, is_active=True)
    record2 = persistence.save_model(model2, sensor_id=1, is_active=True)
    
    # Verify only the second model is active
    db_session.refresh(record1)
    db_session.refresh(record2)
    
    assert not record1.is_active
    assert record2.is_active

def test_prediction_storage(db_session, sample_data):
    """Test that predictions are properly stored"""
    
    prediction_service = PredictionService(db_session)
    
    # Create and save a model first
    model = create_test_model()
    model.compile(optimizer='adam', loss='mse')
    
    persistence = ModelPersistence(db_session)
    model_record = persistence.save_model(model, sensor_id=1, is_active=True)
    
    # Make and store predictions
    predictions = prediction_service.predict_sync(1, sample_data, steps_ahead=3)
    
    # Store predictions
    for pred in predictions:
        new_prediction = Prediction(
            sensor_id=1,
            model_id=model_record.id,
            prediction_value=pred['temperature'],
            created_at=datetime.utcnow()
        )
        db_session.add(new_prediction)
    db_session.commit()
    
    # Verify predictions were stored
    stored_predictions = db_session.query(Prediction).filter_by(sensor_id=1).all()
    assert len(stored_predictions) == 3
    assert all(p.model_id == model_record.id for p in stored_predictions)