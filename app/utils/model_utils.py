# app/utils/model_utils.py
import io
import tensorflow as tf
from app.models.db_models import TrainedModel  # Add this import
import tempfile
import os

def serialize_model(model):
    """Serialize a TensorFlow model to bytes"""
    buffer = io.BytesIO()
    tf.keras.models.save_model(model, buffer, save_format='h5')
    return buffer.getvalue()

def deserialize_model(model_bytes):
    """Deserialize bytes back to a TensorFlow model"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'temp_model.h5')
        # Save bytes to temporary file
        with open(temp_model_path, 'wb') as f:
            f.write(model_bytes)
        # Load model from file
        return tf.keras.models.load_model(temp_model_path)

async def store_model_in_db(db, sensor_id: str, model):
    """Store a model in the database"""
    try:
        model_bytes = serialize_model(model)
        trained_model = TrainedModel(
            sensor_id=sensor_id,
            model_data=model_bytes
        )
        db.add(trained_model)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e

async def load_model_from_db(db, sensor_id: str):
    """Load a model from the database"""
    try:
        model_record = db.query(TrainedModel).filter(
            TrainedModel.sensor_id == sensor_id
        ).first()
        if model_record:
            return deserialize_model(model_record.model_data)
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None