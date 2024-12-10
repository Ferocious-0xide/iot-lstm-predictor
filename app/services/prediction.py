# app/services/prediction.py
from fastapi import HTTPException
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from app.utils.data_prep import SensorDataLoader
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
            
            # Normalize data
            temps_norm = self.data_loader.normalize_data(temps, sensor_id, 'temperature')
            humids_norm = self.data_loader.normalize_data(humids, sensor_id, 'humidity')
            
            # Create sequence
            sequence = np.column_stack((temps_norm, humids_norm))
            return np.expand_dims(sequence, axis=0)  # Add batch dimension
            
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
            sequence = self.prepare_sequence(readings[-24:], sensor_id)
            
            # Make predictions
            predictions = []
            current_sequence = sequence
            
            for step in range(steps_ahead):
                # Get prediction
                temp_pred, humid_pred = model.predict(current_sequence, verbose=0)
                
                # Denormalize predictions
                temp_denorm = self.data_loader.denormalize_data(
                    temp_pred, sensor_id, 'temperature'
                )[0]
                humid_denorm = self.data_loader.denormalize_data(
                    humid_pred, sensor_id, 'humidity'
                )[0]
                
                # Create prediction timestamp
                last_timestamp = datetime.fromisoformat(readings[-1]['timestamp'])
                pred_timestamp = last_timestamp + timedelta(minutes=5 * (step + 1))
                
                predictions.append({
                    'sensor_id': sensor_id,
                    'timestamp': pred_timestamp.isoformat(),
                    'temperature': float(temp_denorm),
                    'humidity': float(humid_denorm),
                    'is_prediction': True
                })
                
                # Update sequence for next prediction
                new_point = np.column_stack((temp_pred, humid_pred))
                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],
                    new_point
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