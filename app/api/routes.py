# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.db_utils import get_sensor_db, get_prediction_db
from app.services.prediction import PredictionService
import logging
from datetime import datetime, timedelta
from typing import List, Dict

router = APIRouter()
logger = logging.getLogger(__name__)
prediction_service = PredictionService()

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
        
        result = db.execute(
            query,
            {"sensor_id": sensor_id, "limit": limit}
        ).fetchall()
        
        if not result:
            logger.warning(f"No readings found for sensor {sensor_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No readings found for sensor {sensor_id}"
            )
        
        readings = [{
            "sensor_id": row[0],
            "temperature": row[1],
            "humidity": row[2],
            "timestamp": row[3].isoformat()
        } for row in result]
        
        return readings
        
    except Exception as e:
        logger.error(f"Error fetching sensor readings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor readings: {str(e)}"
        )

@router.get("/api/v1/sensors/{sensor_id}/predict")
async def get_predictions(
    sensor_id: str,
    steps: int = 3
):
    """Get predictions for a sensor"""
    try:
        predictions = await prediction_service.predict_and_store(sensor_id, steps)
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )

@router.get("/api/v1/sensors/stats")
async def get_sensor_stats(
    db: Session = Depends(get_prediction_db)
):
    """Get statistics for all sensors"""
    try:
        # Query for latest readings and predictions
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