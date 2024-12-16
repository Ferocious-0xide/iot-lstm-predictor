# app/services/training.py
from app.models.lstm_model import TempHumidityPredictor
from app.utils.data_prep import SensorDataLoader
import logging

logger = logging.getLogger(__name__)

class ModelTrainingService:
    def __init__(self):
        self.data_loader = SensorDataLoader()
        self.model = None
    
    async def train_model(self, sensor_id: str, csv_path: str):
        """Train model for a specific sensor"""
        try:
            data = self.data_loader.prepare_data(csv_path, sensor_id)
            
            self.model = TempHumidityPredictor()
            history = self.model.train(
                train_data=data['train'],
                val_data=data['val']
            )
            
            return history
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise