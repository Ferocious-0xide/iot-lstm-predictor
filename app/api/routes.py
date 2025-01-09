# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.db_utils import get_sensor_db, get_prediction_db, get_sensor_db_context
from app.models.db_models import Sensor, Prediction, TrainedModel  # Updated imports
from app.utils.model_utils import load_model_from_db
from app.utils.model_persistence import ModelPersistence
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import tensorflow as tf
import numpy as np
import io
import os
import tempfile
from app.utils.data_prep import SensorDataLoader

# Rest of the imports remain the same
router = APIRouter()
logger = logging.getLogger(__name__)
data_loader = SensorDataLoader()
models = {} 

@router.get("/api/v1/sensors/{sensor_id}/readings")
async def get_sensor_readings(
    sensor_id: str,
    limit: int = 24,
    db: Session = Depends(get_sensor_db)
):
    """Get recent readings for a sensor from the external database"""
    try:
        logger.info(f"Attempting to fetch readings for sensor {sensor_id}")
        
        # Modified query to use new schema
        query = text("""
            SELECT s.id as sensor_id, t.prediction_value as temperature, 
                   h.prediction_value as humidity, COALESCE(t.created_at, h.created_at) as timestamp
            FROM sensor s
            LEFT JOIN prediction t ON s.id = t.sensor_id 
            LEFT JOIN prediction h ON s.id = h.sensor_id
            WHERE s.id = :sensor_id
            ORDER BY COALESCE(t.created_at, h.created_at) DESC
            LIMIT :limit
        """)
        
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

# Update the prediction storage in get_predictions route
try:
    model_persistence = ModelPersistence(db)
    active_model = (
        db.query(TrainedModel)
        .filter_by(sensor_id=int(sensor_id), is_active=True)
        .first()
    )
    
    if active_model:
        for pred in predictions:
            new_prediction = Prediction(
                sensor_id=int(sensor_id),  # Convert string ID to integer
                model_id=active_model.id,
                prediction_value=pred['temperature'],  # Store temperature prediction
                created_at=datetime.fromisoformat(pred['timestamp']),
                confidence_score=0.95
            )
            db.add(new_prediction)
            
            # Add separate prediction for humidity
            humidity_prediction = Prediction(
                sensor_id=int(sensor_id),
                model_id=active_model.id,
                prediction_value=pred['humidity'],  # Store humidity prediction
                created_at=datetime.fromisoformat(pred['timestamp']),
                confidence_score=0.95
            )
            db.add(humidity_prediction)
        db.commit()
        logger.info("Predictions stored successfully")
except Exception as db_error:
    logger.error(f"Error storing predictions: {str(db_error)}")
    db.rollback()

# Update the stats route query
@router.get("/api/v1/sensors/stats")
async def get_sensor_stats(
    db: Session = Depends(get_prediction_db)
):
    """Get statistics for all sensors"""
    try:
        stats_query = text("""
            WITH sensor_stats AS (
                SELECT 
                    s.id as sensor_id,
                    COUNT(p.id) as prediction_count,
                    MAX(p.created_at) as latest_prediction,
                    AVG(p.prediction_value) as avg_prediction
                FROM sensor s
                LEFT JOIN prediction p ON s.id = p.sensor_id
                GROUP BY s.id
            )
            SELECT 
                s.sensor_id,
                s.prediction_count,
                s.latest_prediction,
                s.avg_prediction
            FROM sensor_stats s
            ORDER BY s.sensor_id
        """)
        
        result = db.execute(stats_query).fetchall()
        
        stats = [{
            "sensor_id": row[0],
            "prediction_count": row[1],
            "latest_prediction": row[2].isoformat() if row[2] else None,
            "avg_prediction": round(float(row[3]), 2) if row[3] else None
        } for row in result]
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching sensor stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor stats: {str(e)}"
        )