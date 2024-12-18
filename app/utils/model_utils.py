# app/utils/model_utils.py
import io
import tensorflow as tf

def serialize_model(model):
    """Serialize a TensorFlow model to bytes"""
    buffer = io.BytesIO()
    tf.keras.models.save_model(model, buffer, save_format='h5')
    return buffer.getvalue()

def deserialize_model(model_bytes):
    """Deserialize bytes back to a TensorFlow model"""
    buffer = io.BytesIO(model_bytes)
    return tf.keras.models.load_model(buffer)

async def store_model_in_db(db, sensor_id: str, model):
    """Store a model in the database"""
    model_bytes = serialize_model(model)
    trained_model = TrainedModel(
        sensor_id=sensor_id,
        model_data=model_bytes
    )
    db.add(trained_model)
    db.commit()

async def load_model_from_db(db, sensor_id: str):
    """Load a model from the database"""
    model_record = db.query(TrainedModel).filter(
        TrainedModel.sensor_id == sensor_id
    ).first()
    if model_record:
        return deserialize_model(model_record.model_data)
    return None