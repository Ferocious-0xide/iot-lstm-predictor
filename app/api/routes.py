# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.db_utils import get_prediction_db, get_sensor_db
from app.models.db_models import Sensor, Prediction, TrainedModel, SensorReading
# ... (other imports remain the same)

router = APIRouter()
logger = logging.getLogger(__name__)
data_loader = SensorDataLoader()
models = {}  # Cache for loaded models

async def load_model(sensor_id: int, db: Session = Depends(get_prediction_db)) -> Optional[tf.keras.Model]:
    """Load model from database if not already in memory"""
    # ... (function remains the same)

@router.get("/api/v1/sensors/{sensor_id}/readings")
async def get_sensor_readings(
    sensor_id: int,
    limit: int = 24,
    sensor_db: Session = Depends(get_sensor_db)  # Use sensor database
):
    """Get recent readings for a sensor from the shared database"""
    try:
        logger.info(f"Fetching readings for sensor {sensor_id}")
        readings = (
            sensor_db.query(SensorReading)
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
                for r in readings[1:]
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
    sensor_db: Session = Depends(get_sensor_db),  # For reading sensor data
    pred_db: Session = Depends(get_prediction_db)  # For storing predictions
):
    """Get predictions for a sensor"""
    try:
        logger.info(f"Starting predictions for sensor {sensor_id}")
        
        # Get latest reading from sensor database
        latest_reading = (
            sensor_db.query(SensorReading)
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
        
        # Load and verify model using prediction database
        model = await load_model(sensor_id, pred_db)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for sensor {sensor_id}"
            )
        
        # ... (prediction calculation remains the same)
        
        # Store predictions in prediction database
        try:
            active_model = (
                pred_db.query(TrainedModel)
                .filter_by(sensor_id=sensor_id, is_active=True)
                .first()
            )
            
            if active_model:
                for pred in predictions:
                    temp_prediction = Prediction(
                        sensor_id=pred['sensor_id'],
                        model_id=active_model.id,
                        prediction_value=pred['temperature'],
                        prediction_type='temperature',
                        created_at=datetime.fromisoformat(pred['timestamp']),
                        confidence_score=0.95
                    )
                    pred_db.add(temp_prediction)
                    
                    humid_prediction = Prediction(
                        sensor_id=pred['sensor_id'],
                        model_id=active_model.id,
                        prediction_value=pred['humidity'],
                        prediction_type='humidity',
                        created_at=datetime.fromisoformat(pred['timestamp']),
                        confidence_score=0.95
                    )
                    pred_db.add(humid_prediction)
                
                pred_db.commit()
                logger.info(f"Stored {len(predictions) * 2} predictions successfully")
        except Exception as db_error:
            logger.error(f"Error storing predictions: {str(db_error)}")
            pred_db.rollback()
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
async def get_sensor_stats(
    sensor_db: Session = Depends(get_sensor_db),
    pred_db: Session = Depends(get_prediction_db)
):
    """Get statistics for all sensors"""
    try:
        # Get actual readings stats from sensor database
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
        
        # Get predictions stats from prediction database
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
                prediction_stats.sensor_id,
                prediction_stats.prediction_type,
                prediction_stats.prediction_count,
                prediction_stats.avg_value,
                prediction_stats.latest_prediction
            FROM prediction_stats
            ORDER BY sensor_id
        """)
        
        readings_stats = sensor_db.execute(readings_query).fetchall()
        prediction_stats = pred_db.execute(predictions_query).fetchall()
        
        # ... (stats combination logic remains the same)
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching sensor stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching sensor stats: {str(e)}"
        )