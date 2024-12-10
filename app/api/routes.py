# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.utils.db_utils import get_db
from app.models.db_models import SensorReading
from app.services.prediction import PredictionService
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize prediction service
prediction_service = PredictionService()

@router.get("/sensors/stats")
async def get_sensor_stats(db: Session = Depends(get_db)):
    """Get sensor reading statistics"""
    try:
        stats = db.query(
            SensorReading.sensor_id,
            func.count().label('total_readings'),
            func.avg(SensorReading.temperature).label('avg_temperature'),
            func.avg(SensorReading.humidity).label('avg_humidity'),
            func.min(SensorReading.timestamp).label('first_reading'),
            func.max(SensorReading.timestamp).label('latest_reading')
        ).group_by(SensorReading.sensor_id).all()
        
        return [{
            "sensor_id": stat[0],
            "total_readings": stat[1],
            "avg_temperature": round(float(stat[2]), 2),
            "avg_humidity": round(float(stat[3]), 2),
            "first_reading": stat[4],
            "latest_reading": stat[5]
        } for stat in stats]
    except Exception as e:
        logger.error(f"Error getting sensor stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensors/{sensor_id}/readings")
async def get_sensor_readings(
    sensor_id: str,
    limit: int = 24,
    db: Session = Depends(get_db)
):
    """Get recent readings for a sensor"""
    try:
        readings = db.query(SensorReading)\
            .filter(SensorReading.sensor_id == sensor_id)\
            .order_by(desc(SensorReading.timestamp))\
            .limit(limit)\
            .all()
        
        return [{
            "sensor_id": reading.sensor_id,
            "timestamp": reading.timestamp.isoformat(),
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "is_prediction": False
        } for reading in reversed(readings)]  # Reverse to get chronological order
    except Exception as e:
        logger.error(f"Error getting sensor readings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensors/{sensor_id}/predict")
async def predict_sensor_values(
    sensor_id: str,
    steps: int = 1,
    db: Session = Depends(get_db)
):
    """Get predictions for a sensor"""
    try:
        # Get recent readings
        readings = await get_sensor_readings(sensor_id, limit=24, db=db)
        
        if not readings:
            raise HTTPException(
                status_code=404,
                detail=f"No readings found for sensor {sensor_id}"
            )
        
        # Make predictions
        predictions = await prediction_service.predict(
            sensor_id,
            readings,
            steps_ahead=steps
        )
        
        # Combine actual readings and predictions
        return {
            "sensor_id": sensor_id,
            "latest_readings": readings,
            "predictions": predictions
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))