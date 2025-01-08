# app/services/prediction.py
from fastapi import HTTPException
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.data_prep import SensorDataLoader
from app.models.db_models import Prediction, Sensor
from app.utils.db_utils import get_sensor_db, get_prediction_db
from app.utils.model_persistence import ModelPersistence
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, db: Session):
        self.db = db
        self.data_loader = SensorDataLoader()
        self.model_persistence = ModelPersistence(db)
        self.models = {}  # Cache for loaded models
    
    def load_model(self, sensor_id: int) -> tf.keras.Model:
        """Load model for a specific sensor"""
        try:
            if sensor_id not in self.models:
                model = self.model_persistence.load_model(sensor_id)
                if model is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No active model found for sensor {sensor_id}"
                    )
                self.models[sensor_id] = model
            return self.models[sensor_id]
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    async def initialize_sensors(self):
        """Initialize sensors in the database if they don't exist"""
        try:
            for i in range(1, 6):
                sensor = self.db.query(Sensor).filter(Sensor.id == i).first()
                if not sensor:
                    sensor = Sensor(
                        id=i,
                        name=f"Sensor {i}",
                        location=f"Location {i}"
                    )
                    self.db.add(sensor)
            self.db.commit()
            logger.info("Sensors initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sensors: {e}")
            raise
    
    def prepare_sequence(self, readings: List[Dict], sensor_id: int):
        """Prepare sequence for prediction"""
        try:
            temps = np.array([r['temperature'] for r in readings])
            humids = np.array([r['humidity'] for r in readings])
            
            temps_norm = self.data_loader.normalize_data(temps, str(sensor_id), 'temperature')
            humids_norm = self.data_loader.normalize_data(humids, str(sensor_id), 'humidity')
            
            sequence = np.column_stack((temps_norm, humids_norm))
            sequence = np.expand_dims(sequence, axis=0)
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error preparing sequence: {str(e)}")
            raise

    async def predict(
        self,
        sensor_id: int,
        readings: List[Dict],
        steps_ahead: int = 1
    ) -> List[Dict]:
        """Make predictions for a sensor"""
        try:
            if len(readings) < 24:
                raise HTTPException(
                    status_code=400,
                    detail="Need at least 24 readings for prediction"
                )
            
            # Load model
            model = self.load_model(sensor_id)
            
            # Prepare sequence
            current_sequence = self.prepare_sequence(readings[-24:], sensor_id)
            
            # Make predictions
            predictions = []
            
            for step in range(steps_ahead):
                pred = model.predict(current_sequence, verbose=0)
                temp_pred, humid_pred = pred[0][0], pred[1][0]
                
                temp_denorm = self.data_loader.denormalize_data(
                    temp_pred, str(sensor_id), 'temperature'
                )
                humid_denorm = self.data_loader.denormalize_data(
                    humid_pred, str(sensor_id), 'humidity'
                )
                
                last_timestamp = datetime.fromisoformat(readings[-1]['timestamp'])
                pred_timestamp = last_timestamp + timedelta(minutes=5 * (step + 1))
                
                predictions.append({
                    'sensor_id': sensor_id,
                    'timestamp': pred_timestamp.isoformat(),
                    'temperature': float(temp_denorm),
                    'humidity': float(humid_denorm),
                    'is_prediction': True
                })
                
                new_point = np.array([[temp_pred, humid_pred]])
                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],
                    new_point.reshape(1, 1, 2)
                ], axis=1)
            
            return predictions
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )

    async def get_latest_readings(self, sensor_id: int, lookback_minutes: int = 120) -> List[Dict]:
        """Get latest readings from sensor database"""
        try:
            with next(get_sensor_db()) as db:
                threshold = datetime.utcnow() - timedelta(minutes=lookback_minutes)
                
                query = text("""
                    SELECT sensor_id, temperature, humidity, timestamp 
                    FROM sensor_readings 
                    WHERE sensor_id = :sensor_id 
                    AND timestamp > :threshold 
                    ORDER BY timestamp DESC 
                    LIMIT 24
                """)
                
                result = db.execute(query, {
                    "sensor_id": sensor_id,
                    "threshold": threshold
                })
                
                readings = [
                    {
                        "sensor_id": row.sensor_id,
                        "temperature": row.temperature,
                        "humidity": row.humidity,
                        "timestamp": row.timestamp.isoformat()
                    }
                    for row in result
                ]
                
                return readings[::-1]
        except Exception as e:
            logger.error(f"Error fetching sensor readings: {str(e)}")
            raise

    async def store_predictions(self, predictions: List[Dict], model_id: int):
        """Store predictions in the database"""
        try:
            for pred in predictions:
                new_prediction = Prediction(
                    sensor_id=pred['sensor_id'],
                    model_id=model_id,
                    prediction_value=pred['temperature'],  # Store temperature prediction
                    created_at=datetime.utcnow(),
                    confidence_score=0.95
                )
                self.db.add(new_prediction)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error storing predictions: {str(e)}")
            raise

    async def predict_and_store(self, sensor_id: int, steps_ahead: int = 1) -> List[Dict]:
        """Get latest data, make predictions, and store them"""
        try:
            # Get latest readings
            readings = await self.get_latest_readings(sensor_id)
            
            if not readings:
                raise HTTPException(
                    status_code=404,
                    detail=f"No recent readings found for sensor {sensor_id}"
                )
            
            # Load model and get its ID
            model = self.load_model(sensor_id)
            model_record = (
                self.db.query(ModelPersistence.TrainedModel)
                .filter_by(sensor_id=sensor_id, is_active=True)
                .first()
            )
            
            # Make predictions
            predictions = await self.predict(sensor_id, readings, steps_ahead)
            
            # Store predictions with model ID
            await self.store_predictions(predictions, model_record.id)
            
            return predictions
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error in predict and store pipeline: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction pipeline error: {str(e)}"
            )