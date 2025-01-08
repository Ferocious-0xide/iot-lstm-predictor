# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.db_utils import get_sensor_db, get_prediction_db, get_sensor_db_context
from app.models.db_models import SensorReading, Prediction, ModelMetadata, TrainedModel
from app.utils.model_utils import load_model_from_db
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import tensorflow as tf
import numpy as np
import io
import os
import tempfile
from app.utils.data_prep import SensorDataLoader

# Initialize the router
router = APIRouter()
logger = logging.getLogger(__name__)

data_loader = SensorDataLoader()
models = {}  # Cache for loaded models

async def load_model(sensor_id: str, db: Session):
    """Load model from database if not already in memory"""
    if sensor_id not in models:
        try:
            loaded_model = await load_model_from_db(db, f"sensor_{sensor_id}")
            if loaded_model:
                models[sensor_id] = loaded_model
                logger.info(f"Loaded model for sensor {sensor_id} from database")
            else:
                logger.error(f"No model found in database for sensor {sensor_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading model from database: {e}")
            return None
    return models.get(sensor_id)

@router.get("/api/v1/sensors/{sensor_id}/readings")
async def get_sensor_readings(
    sensor_id: str,
    limit: int = 24,
    db: Session = Depends(get_sensor_db)
):
    """Get recent readings for a sensor from the external database"""
    try:
        logger.info(f"Attempting to fetch readings for sensor {sensor_id}")
        
        query = text("""
            SELECT sensor_id, temperature, humidity, timestamp
            FROM sensor_readings
            WHERE sensor_id = :sensor_id
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        
        # Using the context manager for database operations
        with get_sensor_db_context() as session:
            result = session.execute(
                query,
                {
                    "sensor_id": int(sensor_id),
                    "limit": limit
                }
            ).fetchall()
            
            if not result:
                logger.warning(f"No readings found for sensor {sensor_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"No readings found for sensor {sensor_id}"
                )
            
            most_recent = result[0]
            return {
                "sensor_id": str(most_recent[0]),
                "temperature": most_recent[1],
                "humidity": most_recent[2],
                "timestamp": most_recent[3].isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error fetching sensor readings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor readings: {str(e)}"
        )

@router.get("/api/v1/sensors/{sensor_id}/predict")
async def get_predictions(
    sensor_id: str,
    steps_ahead: int = 3,
    db: Session = Depends(get_prediction_db)
):
    """Get predictions for a sensor"""
    try:
        logger.info(f"Starting predictions for sensor {sensor_id}")
        
        # Get latest readings using a new database session
        logger.info("Fetching latest readings...")
        with get_sensor_db_context() as sensor_db:
            query = text("""
                SELECT sensor_id, temperature, humidity, timestamp
                FROM sensor_readings
                WHERE sensor_id = :sensor_id
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            result = sensor_db.execute(
                query,
                {"sensor_id": int(sensor_id)}
            ).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="No sensor readings found")
            
            readings = {
                "sensor_id": str(result[0]),
                "temperature": result[1],
                "humidity": result[2],
                "timestamp": result[3].isoformat()
            }
        
        logger.info(f"Got readings data: {readings}")
        
        # Load model from database if needed
        logger.info("Loading model...")
        model = await load_model(sensor_id, db)
        if not model:
            logger.error(f"No model found for sensor {sensor_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No model found for sensor {sensor_id}"
            )
        logger.info("Model loaded successfully")
        
        # Prepare data
        logger.info("Preparing data for prediction...")
        temps = np.array([readings['temperature']])
        humids = np.array([readings['humidity']])
        logger.info(f"Temperature data: {temps}")
        logger.info(f"Humidity data: {humids}")
        
        # Normalize
        logger.info("Normalizing data...")
        temps_norm = data_loader.normalize_data(temps, sensor_id, 'temperature')
        humids_norm = data_loader.normalize_data(humids, sensor_id, 'humidity')
        
        # Create sequence
        logger.info("Creating prediction sequence...")
        sequence = np.column_stack((temps_norm, humids_norm))
        sequence = np.expand_dims(sequence, axis=0)
        
        # Make predictions
        predictions = []
        current_sequence = sequence
        
        for step in range(steps_ahead):
            logger.info(f"Making prediction {step + 1}/{steps_ahead}")
            pred = model.predict(current_sequence, verbose=0)
            temp_pred, humid_pred = pred[0][0], pred[1][0]
            
            # Denormalize
            temp_denorm = data_loader.denormalize_data(temp_pred, sensor_id, 'temperature')
            humid_denorm = data_loader.denormalize_data(humid_pred, sensor_id, 'humidity')
            
            # Create timestamp
            last_timestamp = datetime.fromisoformat(readings['timestamp'])
            pred_timestamp = last_timestamp + timedelta(minutes=5 * (step + 1))
            
            predictions.append({
                'sensor_id': f"sensor_{sensor_id}",
                'timestamp': pred_timestamp.isoformat(),
                'temperature': float(temp_denorm),
                'humidity': float(humid_denorm)
            })
            
            # Update sequence for next prediction
            new_point = np.array([[temp_pred, humid_pred]])
            current_sequence = np.concatenate([
                current_sequence[:, 1:, :],
                new_point.reshape(1, 1, 2)
            ], axis=1)
            
            logger.info(f"Prediction {step + 1} complete: Temp={float(temp_denorm):.2f}, Humidity={float(humid_denorm):.2f}")
        
        logger.info("Storing predictions in database...")
        try:
            for pred in predictions:
                new_prediction = Prediction(
                    sensor_id=pred['sensor_id'],
                    temperature_prediction=pred['temperature'],
                    humidity_prediction=pred['humidity'],
                    prediction_timestamp=datetime.fromisoformat(pred['timestamp']),
                    created_at=datetime.utcnow(),
                    confidence_score=0.95
                )
                db.add(new_prediction)
            db.commit()
            logger.info("Predictions stored successfully")
        except Exception as db_error:
            logger.error(f"Error storing predictions: {str(db_error)}")
            db.rollback()
        
        logger.info(f"Returning {len(predictions)} predictions")
        return {"status": "success", "predictions": predictions}
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return {"status": "error", "detail": str(e)}

@router.get("/api/v1/sensors/stats")
async def get_sensor_stats(
    db: Session = Depends(get_prediction_db)
):
    """Get statistics for all sensors"""
    try:
        stats_query = text("""
            WITH sensor_stats AS (
                SELECT 
                    p.sensor_id,
                    COUNT(*) as prediction_count,
                    MAX(p.prediction_timestamp) as latest_prediction,
                    AVG(p.temperature_prediction) as avg_temp,
                    AVG(p.humidity_prediction) as avg_humidity
                FROM predictions p
                GROUP BY p.sensor_id
            )
            SELECT 
                s.sensor_id,
                s.prediction_count,
                s.latest_prediction,
                s.avg_temp,
                s.avg_humidity
            FROM sensor_stats s
            ORDER BY s.sensor_id
        """)
        
        result = db.execute(stats_query).fetchall()
        
        stats = [{
            "sensor_id": row[0],
            "prediction_count": row[1],
            "latest_prediction": row[2].isoformat() if row[2] else None,
            "avg_temperature": round(float(row[3]), 2) if row[3] else None,
            "avg_humidity": round(float(row[4]), 2) if row[4] else None
        } for row in result]
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching sensor stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor stats: {str(e)}"
        )