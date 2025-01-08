# scripts/store_models.py
import tensorflow as tf
from app.utils.db_utils import get_prediction_db_context
from app.models.db_models import Base, TrainedModel
from app.utils.model_utils import store_model_in_db
import asyncio
import logging

logger = logging.getLogger(__name__)

async def store_existing_models():
    """Store existing models in the database"""
    try:
        for sensor_id in range(1, 6):
            model_path = f"models/sensor_{sensor_id}_model"
            if tf.io.gfile.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                model = tf.keras.models.load_model(model_path)
                
                with get_prediction_db_context() as db:
                    await store_model_in_db(db, f"sensor_{sensor_id}", model)
                    logger.info(f"Stored model for sensor {sensor_id}")
    except Exception as e:
        logger.error(f"Error storing models: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        asyncio.run(store_existing_models())
        print("Successfully stored all models")
    except Exception as e:
        print(f"Error: {str(e)}")