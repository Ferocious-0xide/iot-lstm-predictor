# app/services/prediction.py
from fastapi import HTTPException
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.data_prep import SensorDataLoader
from app.models.db_models import SensorReading, Prediction, ModelMetadata
from app.utils.db_utils import get_sensor_db, get_prediction_db
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.data_loader = SensorDataLoader()
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            for sensor_id in ['1', '2', '3', '4', '5']:
                model_path = self.model_dir / f"sensor_{sensor_id}_model"
                if model_path.exists():
                    self.models[sensor_id] = tf.keras.models.load_model(str(model_path))
                    logger.info(f"Loaded model for sensor {sensor_id}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def prepare_sequence(self, readings: List[Dict], sensor_id: str):
        """Prepare sequence for prediction"""
        try:
            # Convert readings to numpy arrays
            temps = np.array([r['temperature'] for r in readings])
            humids = np.array([r['humidity'] for r in readings])
            
            logger.info(f"Original temps shape: {temps.shape}")
            logger.info(f"Original humids shape: {humids.shape}")
            
            # Normalize data
            temps_norm = self.data_loader.normalize_data(temps, sensor_id, 'temperature')
            humids_norm = self.data_loader.normalize_data(humids, sensor_id, 'humidity')
            
            logger.info(f"Normalized temps shape: {temps_norm.shape}")
            logger.info(f"Normalized humids shape: {humids_norm.shape}")
            
            # Reshape to match the model's expected input shape
            # Stack features first
            sequence = np.column_stack((temps_norm, humids_norm))  # Should be shape (24, 2)
            logger.info(f"Sequence shape after stack: {sequence.shape}")
            
            # Add batch dimension
            sequence = np.expand_dims(sequence, axis=0)  # Shape becomes (1, 24, 2)
            logger.info(f"Final sequence shape: {sequence.shape}")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error preparing sequence: {str(e)}")
            raise
    
    async def predict(
        self,
        sensor_id: str,
        readings: List[Dict],
        steps_ahead: int = 1
    ) -> List[Dict]:
        """Make predictions for a sensor"""
        try:
            if sensor_id not in self.models:
                raise HTTPException(
                    status_code=404,
                    detail=f"No trained model found for sensor {sensor_id}"
                )
            
            if len(readings) < 24:  # Minimum sequence length
                raise HTTPException(
                    status_code=400,
                    detail="Need at least 24 readings for prediction"
                )
            
            # Get model
            model = self.models[sensor_id]
            
            # Prepare sequence
            current_sequence = self.prepare_sequence(readings[-24:], sensor_id)
            logger.info(f"Initial sequence shape: {current_sequence.shape}")
            
            # Make predictions
            predictions = []
            
            for step in range(steps_ahead):
                # Get prediction
                pred = model.predict(current_sequence, verbose=0)
                temp_pred, humid_pred = pred[0][0], pred[1][0]  # Get first (and only) prediction
                
                logger.info(f"Prediction shape - temp: {temp_pred.shape}, humid: {humid_pred.shape}")
                
                # Denormalize predictions
                temp_denorm = self.data_loader.denormalize_data(
                    temp_pred, sensor_id, 'temperature'
                )
                humid_denorm = self.data_loader.denormalize_data(
                    humid_pred, sensor_id, 'humidity'
                )
                
                # Create prediction timestamp
                last_timestamp = datetime.fromisoformat(readings[-1]['timestamp'])
                pred_timestamp = last_timestamp + timedelta(minutes=5 * (step + 1))
                
                predictions.append({
                    'sensor_id': f"sensor_{sensor_id}",
                    'timestamp': pred_timestamp.isoformat(),
                    'temperature': float(temp_denorm),
                    'humidity': float(humid_denorm),
                    'is_prediction': True
                })
                
                # Update sequence for next prediction
                # Reshape predictions to match sequence format
                new_point = np.array([[temp_pred, humid_pred]])  # Shape: (1, 2)
                
                # Remove oldest point and add new prediction
                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],  # Remove first timestep
                    new_point.reshape(1, 1, 2)   # Add new point with correct shape
                ], axis=1)
                
                logger.info(f"Updated sequence shape: {current_sequence.shape}")
            
            return predictions
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )

    async def get_latest_readings(self, sensor_id: str, lookback_minutes: int = 120) -> List[Dict]:
        """Get latest readings from sensor database"""
        try:
            with next(get_sensor_db()) as db:
                # Calculate timestamp threshold
                threshold = datetime.utcnow() - timedelta(minutes=lookback_minutes)
            
            # Query latest readings - modified to use integer sensor_id
            query = text("""
                SELECT sensor_id, temperature, humidity, timestamp 
                FROM sensor_readings 
                WHERE sensor_id = :sensor_id 
                AND timestamp > :threshold 
                ORDER BY timestamp DESC 
                LIMIT 24
            """)
            
            # Convert sensor_id to integer
            sensor_id_int = int(sensor_id)  # Convert '1' to 1
            
            result = db.execute(query, {
                "sensor_id": sensor_id_int,  # Pass integer instead of string
                "threshold": threshold
            })
            
            readings = [
                {
                    "sensor_id": str(row.sensor_id),  # Convert back to string for consistency
                    "temperature": row.temperature,
                    "humidity": row.humidity,
                    "timestamp": row.timestamp.isoformat()
                }
                for row in result
            ]
            
            return readings[::-1]  # Reverse to get chronological order
        except Exception as e:
            logger.error(f"Error fetching sensor readings: {str(e)}")
            raise

    async def store_predictions(self, predictions: List[Dict]):
        """Store predictions in the database"""
        try:
            with next(get_prediction_db()) as db:
                for pred in predictions:
                    new_prediction = Prediction(
                        sensor_id=pred['sensor_id'],
                        temperature_prediction=pred['temperature'],
                        humidity_prediction=pred['humidity'],
                        prediction_timestamp=datetime.fromisoformat(pred['timestamp']),
                        created_at=datetime.utcnow(),
                        confidence_score=0.95  # You might want to calculate this
                    )
                    db.add(new_prediction)
                db.commit()
        except Exception as e:
            logger.error(f"Error storing predictions: {str(e)}")
            raise

    async def predict_and_store(self, sensor_id: str, steps_ahead: int = 1) -> List[Dict]:
        """Get latest data, make predictions, and store them"""
        try:
            # Get latest readings
            readings = await self.get_latest_readings(sensor_id)
            
            if not readings:
                raise HTTPException(
                    status_code=404,
                    detail=f"No recent readings found for sensor {sensor_id}"
                )
            
            # Make predictions
            predictions = await self.predict(sensor_id, readings, steps_ahead)
            
            # Store predictions
            await self.store_predictions(predictions)
            
            return predictions
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error in predict and store pipeline: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction pipeline error: {str(e)}"
            )