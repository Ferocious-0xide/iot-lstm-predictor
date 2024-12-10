# app/services/training.py
from typing import Dict, Optional, Tuple
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.lstm_model import TempHumidityPredictor
from app.models.db_models import ModelVersion, TrainingJob, SensorReading
from app.utils.db_utils import get_db

logger = logging.getLogger(__name__)

class ModelTrainingService:
    def __init__(self, db: Session):
        self.db = db
        
    async def start_training_job(
        self,
        sensor_id: str,
        hyperparameters: Optional[Dict] = None
    ) -> TrainingJob:
        """Initialize a new training job"""
        try:
            # Create training job record
            job = TrainingJob(
                status="pending",
                parameters={
                    "sensor_id": sensor_id,
                    "hyperparameters": hyperparameters or {}
                },
                start_time=datetime.utcnow()
            )
            self.db.add(job)
            self.db.commit()
            
            # Start async training process
            await self._train_model(job.id, sensor_id, hyperparameters)
            
            return job
            
        except Exception as e:
            logger.error(f"Error starting training job: {str(e)}")
            raise

    async def _train_model(
        self,
        job_id: int,
        sensor_id: str,
        hyperparameters: Optional[Dict] = None
    ) -> None:
        """Execute model training"""
        try:
            # Update job status
            job = self.db.query(TrainingJob).get(job_id)
            job.status = "running"
            self.db.commit()
            
            # Get training data
            data = pd.read_sql(
                self.db.query(SensorReading)
                .filter(SensorReading.sensor_id == sensor_id)
                .statement,
                self.db.bind
            )
            
            # Initialize and train model
            model = TempHumidityPredictor(**(hyperparameters or {}))
            history = model.train(
                train_data=data,
                model_path=f"models/{sensor_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Create new model version
            model_version = ModelVersion(
                version=f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                sensor_id=sensor_id,
                hyperparameters=hyperparameters,
                metrics=history,
                training_date=datetime.utcnow(),
                model_path=model.model_path
            )
            
            # Deactivate old versions
            self.db.query(ModelVersion)\
                .filter(ModelVersion.sensor_id == sensor_id)\
                .update({"is_active": False})
            
            # Set new version as active
            model_version.is_active = True
            self.db.add(model_version)
            
            # Update job status
            job.status = "completed"
            job.end_time = datetime.utcnow()
            job.model_version_id = model_version.id
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            job = self.db.query(TrainingJob).get(job_id)
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.utcnow()
            self.db.commit()
            raise

    async def get_training_status(self, job_id: int) -> Dict:
        """Get status of a training job"""
        job = self.db.query(TrainingJob).get(job_id)
        return {
            "status": job.status,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "error_message": job.error_message
        }

# app/services/prediction.py
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.lstm_model import TempHumidityPredictor
from app.models.db_models import ModelVersion, Prediction, SensorReading

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, db: Session):
        self.db = db
        self._models = {}  # Cache for loaded models
        
    def get_predictions(
        self,
        sensor_id: str,
        sequence_length: int = 24,
        prediction_steps: int = 1
    ) -> List[Dict]:
        """Generate predictions for a sensor"""
        try:
            # Get active model version
            model_version = self.db.query(ModelVersion)\
                .filter(ModelVersion.sensor_id == sensor_id, 
                       ModelVersion.is_active == True)\
                .first()
            
            if not model_version:
                raise ValueError(f"No active model found for sensor {sensor_id}")
            
            # Load model if not in cache
            if sensor_id not in self._models:
                predictor = TempHumidityPredictor()
                predictor.load_model(model_version.model_path)
                self._models[sensor_id] = predictor
            
            # Get latest readings
            latest_readings = pd.read_sql(
                self.db.query(SensorReading)
                .filter(SensorReading.sensor_id == sensor_id)
                .order_by(SensorReading.timestamp.desc())
                .limit(sequence_length)
                .statement,
                self.db.bind
            )
            
            predictions = []
            current_data = latest_readings.copy()
            
            # Generate predictions for specified steps
            for step in range(prediction_steps):
                temp_pred, humidity_pred = self._models[sensor_id].predict(current_data)
                
                prediction = Prediction(
                    sensor_id=sensor_id,
                    model_version_id=model_version.id,
                    timestamp=latest_readings.timestamp.max() + timedelta(hours=step+1),
                    temperature_prediction=temp_pred,
                    humidity_prediction=humidity_pred,
                    created_at=datetime.utcnow()
                )
                
                self.db.add(prediction)
                predictions.append({
                    "timestamp": prediction.timestamp,
                    "temperature": temp_pred,
                    "humidity": humidity_pred
                })
                
                # Update current_data for next prediction
                new_row = pd.DataFrame({
                    'timestamp': [prediction.timestamp],
                    'temperature': [temp_pred],
                    'humidity': [humidity_pred]
                })
                current_data = pd.concat([current_data[1:], new_row])
            
            self.db.commit()
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            self.db.rollback()
            raise

    def get_model_performance(
        self,
        sensor_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Get model performance metrics"""
        try:
            # Get predictions and actual values
            query = self.db.query(Prediction, SensorReading)\
                .join(SensorReading, 
                      (Prediction.sensor_id == SensorReading.sensor_id) & 
                      (Prediction.timestamp == SensorReading.timestamp))\
                .filter(Prediction.sensor_id == sensor_id)
            
            if start_date:
                query = query.filter(Prediction.timestamp >= start_date)
            if end_date:
                query = query.filter(Prediction.timestamp <= end_date)
            
            results = query.all()
            
            # Calculate metrics
            temp_mae = sum(abs(p.temperature_prediction - r.temperature) 
                          for p, r in results) / len(results)
            humid_mae = sum(abs(p.humidity_prediction - r.humidity) 
                           for p, r in results) / len(results)
            
            return {
                "temperature_mae": temp_mae,
                "humidity_mae": humid_mae,
                "predictions_count": len(results),
                "date_range": {
                    "start": min(p.timestamp for p, _ in results),
                    "end": max(p.timestamp for p, _ in results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {str(e)}")
            raise

# Example usage functions
async def train_new_model(sensor_id: str) -> Dict:
    """Utility function to train a new model"""
    with get_db() as db:
        training_service = ModelTrainingService(db)
        job = await training_service.start_training_job(sensor_id)
        return {"job_id": job.id}

def get_latest_predictions(sensor_id: str, steps: int = 24) -> List[Dict]:
    """Utility function to get predictions"""
    with get_db() as db:
        prediction_service = PredictionService(db)
        return prediction_service.get_predictions(sensor_id, prediction_steps=steps)