import pickle
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.db_models import TrainedModel, Sensor
import tensorflow as tf
import json

class ModelPersistence:
    def __init__(self, db: Session):
        self.db = db

    def save_model(
        self, 
        model: tf.keras.Model,
        sensor_id: int,
        metrics: Optional[Dict[str, Any]] = None,
        model_type: str = 'lstm',
        is_active: bool = True
    ) -> TrainedModel:
        """
        Saves a trained model to the database
        
        Args:
            model: The trained TensorFlow model
            sensor_id: ID of the sensor this model is trained for
            metrics: Optional dictionary of model metrics
            model_type: Type of the model (default: 'lstm')
            is_active: Whether this should be the active model for this sensor
        
        Returns:
            TrainedModel instance
        """
        # Serialize the model
        model_bytes = pickle.dumps(model)
        
        # If this is to be the active model, deactivate other models for this sensor
        if is_active:
            self.db.query(TrainedModel)\
                .filter(TrainedModel.sensor_id == sensor_id)\
                .update({'is_active': False})
        
        # Create new model record
        trained_model = TrainedModel(
            sensor_id=sensor_id,
            model_data=model_bytes,
            model_metrics=metrics,
            model_type=model_type,
            is_active=is_active,
            created_at=datetime.utcnow()
        )
        
        self.db.add(trained_model)
        self.db.commit()
        self.db.refresh(trained_model)
        
        return trained_model

    def load_model(self, sensor_id: int, model_id: Optional[int] = None) -> Optional[tf.keras.Model]:
        """
        Loads a model from the database
        
        Args:
            sensor_id: ID of the sensor to load model for
            model_id: Optional specific model ID to load. If None, loads the active model
            
        Returns:
            TensorFlow model or None if no model found
        """
        query = self.db.query(TrainedModel).filter(TrainedModel.sensor_id == sensor_id)
        
        if model_id is not None:
            model_record = query.filter(TrainedModel.id == model_id).first()
        else:
            model_record = query.filter(TrainedModel.is_active == True).first()
            
        if not model_record:
            return None
            
        try:
            model = pickle.loads(model_record.model_data)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_model_metrics(self, model_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves metrics for a specific model
        
        Args:
            model_id: ID of the model to get metrics for
            
        Returns:
            Dictionary of metrics or None if not found
        """
        model_record = self.db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
        return model_record.model_metrics if model_record else None

    def set_model_active(self, model_id: int, sensor_id: int) -> bool:
        """
        Sets a specific model as the active one for a sensor
        
        Args:
            model_id: ID of the model to activate
            sensor_id: ID of the sensor this model is for
            
        Returns:
            Boolean indicating success
        """
        # Deactivate all models for this sensor
        self.db.query(TrainedModel)\
            .filter(TrainedModel.sensor_id == sensor_id)\
            .update({'is_active': False})
            
        # Activate the specified model
        model = self.db.query(TrainedModel)\
            .filter(TrainedModel.id == model_id)\
            .filter(TrainedModel.sensor_id == sensor_id)\
            .first()
            
        if not model:
            return False
            
        model.is_active = True
        self.db.commit()
        return True

    def list_models(self, sensor_id: Optional[int] = None) -> list:
        """
        Lists all models in the database, optionally filtered by sensor
        
        Args:
            sensor_id: Optional sensor ID to filter by
            
        Returns:
            List of model records
        """
        query = self.db.query(TrainedModel)
        if sensor_id is not None:
            query = query.filter(TrainedModel.sensor_id == sensor_id)
            
        return query.all()