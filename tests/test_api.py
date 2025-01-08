from fastapi.testclient import TestClient
from app.main import app
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.db_models import Base, Sensor, TrainedModel
import tensorflow as tf
from app.utils.model_persistence import ModelPersistence

client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    # Create test database
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    
    # Create test data
    db = TestingSessionLocal()
    sensor = Sensor(id=1, name="Test Sensor", location="Test Location")
    db.add(sensor)
    
    # Create and save a test model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(24, 2)),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    persistence = ModelPersistence(db)
    persistence.save_model(model, sensor_id=1, is_active=True)
    
    db.commit()
    
    yield db
    
    db.close()

def test_get_predictions(test_db):
    response = client.get("/api/v1/sensors/1/predict")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3  # default steps_ahead

def test_get_sensor_readings(test_db):
    response = client.get("/api/v1/sensors/1/readings")
    assert response.status_code == 200
    data = response.json()
    assert "sensor_id" in data
    assert "temperature" in data
    assert "humidity" in data

def test_get_sensor_stats(test_db):
    response = client.get("/api/v1/sensors/stats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)