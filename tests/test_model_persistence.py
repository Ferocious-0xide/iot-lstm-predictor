import pytest
import tensorflow as tf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.db_models import Base, Sensor, TrainedModel
from app.utils.model_persistence import ModelPersistence

@pytest.fixture
def db_session():
    # Create in-memory SQLite database for testing
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture
def sample_model():
    # Create a simple model for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def test_save_and_load_model(db_session, sample_model):
    # Create a test sensor
    sensor = Sensor(name="Test Sensor", location="Test Location")
    db_session.add(sensor)
    db_session.commit()
    
    persistence = ModelPersistence(db_session)
    
    # Save the model
    metrics = {'loss': 0.1, 'val_loss': 0.2}
    saved_model = persistence.save_model(
        model=sample_model,
        sensor_id=sensor.id,
        metrics=metrics,
        model_type='lstm',
        is_active=True
    )
    
    assert saved_model.sensor_id == sensor.id
    assert saved_model.is_active == True
    assert saved_model.model_metrics == metrics
    
    # Load the model
    loaded_model = persistence.load_model(sensor_id=sensor.id)
    assert loaded_model is not None
    assert isinstance(loaded_model, tf.keras.Model)
    
    # Verify model architecture
    assert len(loaded_model.layers) == len(sample_model.layers)

def test_model_activation(db_session, sample_model):
    # Create a test sensor
    sensor = Sensor(name="Test Sensor", location="Test Location")
    db_session.add(sensor)
    db_session.commit()
    
    persistence = ModelPersistence(db_session)
    
    # Save multiple models
    model1 = persistence.save_model(sample_model, sensor.id, is_active=True)
    model2 = persistence.save_model(sample_model, sensor.id, is_active=False)
    
    # Verify initial state
    assert model1.is_active == True
    assert model2.is_active == False
    
    # Test activation
    success = persistence.set_model_active(model2.id, sensor.id)
    assert success == True
    
    # Verify new state
    db_session.refresh(model1)
    db_session.refresh(model2)
    assert model1.is_active == False
    assert model2.is_active == True

def test_list_models(db_session, sample_model):
    # Create test sensors
    sensor1 = Sensor(name="Sensor 1", location="Location 1")
    sensor2 = Sensor(name="Sensor 2", location="Location 2")
    db_session.add_all([sensor1, sensor2])
    db_session.commit()
    
    persistence = ModelPersistence(db_session)
    
    # Save models for different sensors
    persistence.save_model(sample_model, sensor1.id)
    persistence.save_model(sample_model, sensor1.id)
    persistence.save_model(sample_model, sensor2.id)
    
    # Test listing
    sensor1_models = persistence.list_models(sensor1.id)
    all_models = persistence.list_models()
    
    assert len(sensor1_models) == 2
    assert len(all_models) == 3