# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.utils.db_utils import get_db
from app.models.db_models import SensorReading
from datetime import datetime
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

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
            "sensor_id": stat.sensor_id,
            "total_readings": stat.total_readings,
            "avg_temperature": round(stat.avg_temperature, 2),
            "avg_humidity": round(stat.avg_humidity, 2),
            "first_reading": stat.first_reading,
            "latest_reading": stat.latest_reading
        } for stat in stats]
    except Exception as e:
        logger.error(f"Error getting sensor stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))