# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.db_utils import get_db, get_db_context
from app.models.db_models import Sensor, Prediction, TrainedModel, SensorReading
from app.utils.model_utils import load_model_from_db
from app.utils.model_persistence import ModelPersistence
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import tensorflow as tf
import numpy as np
from app.utils.data_prep import SensorDataLoader

# Initialize router and logging
router = APIRouter()
logger = logging.getLogger(__name__)
data_loader = SensorDataLoader()
models = {}  # Cache for loaded models

async def load_model(sensor_id: int, db: Session) -> Optional[tf.keras.Model]:
    """Load model from database if not already in memory"""
    str_id = str(sensor_id)
    if str_id not in models:
        try:
            model_persistence = ModelPersistence(db)
            loaded_model = model_persistence.load_model(sensor_id)
            
            if loaded_model:
                models[str_id] = loaded_model
                logger.info(f"Loaded model for sensor {sensor_id} from database")
            else:
                logger.error(f"No model found for sensor {sensor_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading model from database: {e}")
            return None
    return models.get(str_id)

@router.get("/api/v1/sensors/{sensor_id}/readings")
async def get_sensor_readings(
    sensor_id: int,
    limit: int = 24,
    db: Session = Depends(get_db)
):
    """Get recent readings for a sensor from the shared database"""
    try:
        logger.info(f"Fetching readings for sensor {sensor_id}")
        
        # Query the actual sensor readings table
        readings = (
            db.query(SensorReading)
            .filter(SensorReading.sensor_id == sensor_id)
            .order_by(SensorReading.timestamp.desc())
            .limit(limit)
            .all()
        )
        
        if not readings:
            logger.warning(f"No readings found for sensor {sensor_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No readings found for sensor {sensor_id}"
            )
        
        # Get most recent reading
        latest = readings[0]
        return {
            "sensor_id": str(latest.sensor_id),
            "temperature": latest.temperature,
            "humidity": latest.humidity,
            "timestamp": latest.timestamp.isoformat(),
            "historical": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "temperature": r.temperature,
                    "humidity": r.humidity
                }
                for r in readings[1:]  # Exclude the latest reading
            ]
        }
    
    except Exception as e:
        logger.error(f"Error fetching sensor readings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor readings: {str(e)}"
        )

@router.get("/api/v1/sensors/{sensor_id}/predict")
async def get_predictions(
    sensor_id: int,
    steps_ahead: int = 3,
    db: Session = Depends(get_db)
):
    """Get predictions for a sensor"""
    try:
        logger.info(f"Starting predictions for sensor {sensor_id}")
        
        # Get latest reading
        latest_reading = (
            db.query(SensorReading)
            .filter(SensorReading.sensor_id == sensor_id)
            .order_by(SensorReading.timestamp.desc())
            .first()
        )
        
        if not latest_reading:
            raise HTTPException(
                status_code=404,
                detail="No sensor readings found"
            )
        
        readings = {
            "sensor_id": str(latest_reading.sensor_id),
            "temperature": latest_reading.temperature,
            "humidity": latest_reading.humidity,
            "timestamp": latest_reading.timestamp.isoformat()
        }
        
        # Load and verify model
        model = await load_model(sensor_id, db)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for sensor {sensor_id}"
            )
        
        # Prepare data
        temps = np.array([readings['temperature']])
        humids = np.array([readings['humidity']])
        
        # Normalize
        temps_norm = data_loader.normalize_data(temps, str(sensor_id), 'temperature')
        humids_norm = data_loader.normalize_data(humids, str(sensor_id), 'humidity')
        
        # Create sequence and make predictions
        sequence = np.expand_dims(np.column_stack((temps_norm, humids_norm)), axis=0)
        predictions = []
        current_sequence = sequence
        
        for step in range(steps_ahead):
            pred = model.predict(current_sequence, verbose=0)
            temp_pred, humid_pred = pred[0][0], pred[1][0]
            
            # Denormalize
            temp_denorm = data_loader.denormalize_data(temp_pred, str(sensor_id), 'temperature')
            humid_denorm = data_loader.denormalize_data(humid_pred, str(sensor_id), 'humidity')
            
            # Calculate timestamp
            pred_timestamp = latest_reading.timestamp + timedelta(minutes=5 * (step + 1))
            
            predictions.append({
                'sensor_id': sensor_id,
                'timestamp': pred_timestamp.isoformat(),
                'temperature': float(temp_denorm),
                'humidity': float(humid_denorm)
            })
            
            # Update sequence
            new_point = np.array([[temp_pred, humid_pred]])
            current_sequence = np.concatenate([
                current_sequence[:, 1:, :],
                new_point.reshape(1, 1, 2)
            ], axis=1)
        
        # Store predictions
        try:
            active_model = (
                db.query(TrainedModel)
                .filter_by(sensor_id=sensor_id, is_active=True)
                .first()
            )
            
            if active_model:
                for pred in predictions:
                    # Store temperature prediction
                    temp_prediction = Prediction(
                        sensor_id=pred['sensor_id'],
                        model_id=active_model.id,
                        prediction_value=pred['temperature'],
                        prediction_type='temperature',
                        created_at=datetime.fromisoformat(pred['timestamp']),
                        confidence_score=0.95
                    )
                    db.add(temp_prediction)
                    
                    # Store humidity prediction
                    humid_prediction = Prediction(
                        sensor_id=pred['sensor_id'],
                        model_id=active_model.id,
                        prediction_value=pred['humidity'],
                        prediction_type='humidity',
                        created_at=datetime.fromisoformat(pred['timestamp']),
                        confidence_score=0.95
                    )
                    db.add(humid_prediction)
                
                db.commit()
                logger.info(f"Stored {len(predictions) * 2} predictions successfully")
        except Exception as db_error:
            logger.error(f"Error storing predictions: {str(db_error)}")
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error storing predictions: {str(db_error)}"
            )
        
        return {
            "status": "success",
            "predictions": predictions,
            "latest_reading": readings
        }
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return {"status": "error", "detail": str(e)}

@router.get("/api/v1/sensors/stats")
async def get_sensor_stats(db: Session = Depends(get_db)):
    """Get statistics for all sensors"""
    try:
        # Get actual readings stats
        readings_query = text("""
            SELECT 
                sensor_id,
                COUNT(*) as reading_count,
                AVG(temperature) as avg_temperature,
                AVG(humidity) as avg_humidity,
                MAX(timestamp) as latest_reading
            FROM sensor_readings
            GROUP BY sensor_id
            ORDER BY sensor_id
        """)
        
        # Get predictions stats
        predictions_query = text("""
            WITH prediction_stats AS (
                SELECT 
                    sensor_id,
                    prediction_type,
                    COUNT(*) as prediction_count,
                    AVG(prediction_value) as avg_value,
                    MAX(created_at) as latest_prediction
                FROM prediction
                GROUP BY sensor_id, prediction_type
            )
            SELECT 
                s.id as sensor_id,
                s.name,
                s.location,
                pt.prediction_count as temp_predictions,
                ph.prediction_count as humid_predictions,
                pt.avg_value as avg_temp_prediction,
                ph.avg_value as avg_humid_prediction
            FROM sensor s
            LEFT JOIN prediction_stats pt ON s.id = pt.sensor_id AND pt.prediction_type = 'temperature'
            LEFT JOIN prediction_stats ph ON s.id = ph.sensor_id AND ph.prediction_type = 'humidity'
            ORDER BY s.id
        """)
        
        readings_stats = db.execute(readings_query).fetchall()
        prediction_stats = db.execute(predictions_query).fetchall()
        
        # Combine stats
        stats = []
        for r_stat in readings_stats:
            p_stat = next(
                (p for p in prediction_stats if p.sensor_id == r_stat.sensor_id), 
                None
            )
            
            stats.append({
                "sensor_id": r_stat.sensor_id,
                "reading_count": r_stat.reading_count,
                "avg_temperature": round(float(r_stat.avg_temperature), 2),
                "avg_humidity": round(float(r_stat.avg_humidity), 2),
                "latest_reading": r_stat.latest_reading.isoformat(),
                "prediction_counts": {
                    "temperature": p_stat.temp_predictions if p_stat else 0,
                    "humidity": p_stat.humid_predictions if p_stat else 0
                } if p_stat else None,
                "location": p_stat.location if p_stat else None
            })
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching sensor stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor stats: {str(e)}"
        )