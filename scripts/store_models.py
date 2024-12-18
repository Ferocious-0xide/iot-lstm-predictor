# scripts/store_models.py
import tensorflow as tf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.db_models import Base, TrainedModel
from app.utils.model_utils import store_model_in_db
from config.settings import get_settings

async def store_existing_models():
    """Store existing models in the database"""
    settings = get_settings()
    engine = create_engine(settings.prediction_database_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        for sensor_id in range(1, 6):
            model_path = f"models/sensor_{sensor_id}_model"
            if tf.io.gfile.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                await store_model_in_db(db, f"sensor_{sensor_id}", model)
                print(f"Stored model for sensor {sensor_id}")
    finally:
        db.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(store_existing_models())