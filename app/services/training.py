# app/services/training.py
from app.models.lstm_model import TempHumidityPredictor
from app.utils.advanced_data_prep import AdvancedSensorDataLoader
from app.utils.model_persistence import ModelPersistence
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

class ModelTrainingService:
    def __init__(self, db: Session):
        self.data_loader = AdvancedSensorDataLoader()
        self.model = None
        self.model_persistence = ModelPersistence(db)
    
    async def train_model(self, sensor_id: int, csv_path: str):
        """Train model with enhanced data preparation and persistence"""
        try:
            # Use enhanced data loader
            data = self.data_loader.prepare_data(csv_path, sensor_id)
            
            self.model = TempHumidityPredictor()
            history = self.model.train(
                train_data=data['train'],
                val_data=data['val']
            )
            
            # Calculate final metrics
            final_metrics = {
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'temp_loss': history.history.get('temp_output_loss', [-1])[-1],
                'humid_loss': history.history.get('humid_output_loss', [-1])[-1],
                'epochs': len(history.history['loss'])
            }
            
            # Save model to database
            trained_model = self.model_persistence.save_model(
                model=self.model,
                sensor_id=sensor_id,
                metrics=final_metrics,
                model_type='lstm',
                is_active=True  # Make this the active model for the sensor
            )
            
            logger.info(f"Model saved to database with ID: {trained_model.id}")
            return {
                'model_id': trained_model.id,
                'history': history.history,
                'metrics': final_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    async def get_model_metrics(self, model_id: int):
        """Get metrics for a specific model"""
        try:
            metrics = self.model_persistence.get_model_metrics(model_id)
            if not metrics:
                raise ValueError(f"No metrics found for model {model_id}")
            return metrics
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            raise