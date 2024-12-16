# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.db_utils import get_sensor_db
import logging
from datetime import datetime
from typing import List, Dict

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/sensors/{sensor_id}/readings")
async def get_sensor_readings(
    sensor_id: str,
    limit: int = 24,
    db: Session = Depends(get_sensor_db)
):
    """Get recent readings for a sensor from the external database"""
    try:
        # Log attempt to fetch data
        logger.info(f"Attempting to fetch readings for sensor {sensor_id}")
        
        # Query the sensor database (follower)
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
        
        # Format the results
        readings = [{
            "sensor_id": row[0],
            "temperature": row[1],
            "humidity": row[2],
            "timestamp": row[3].isoformat()
        } for row in result]
        
        logger.info(f"Successfully fetched {len(readings)} readings for sensor {sensor_id}")
        return readings
        
    except Exception as e:
        logger.error(f"Error fetching sensor readings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor readings: {str(e)}"
        )