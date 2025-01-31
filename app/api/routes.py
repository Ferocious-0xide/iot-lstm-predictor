# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.db_utils import get_db, get_sensor_db, get_prediction_db
from app.models.db_models import Prediction, TrainedModel, SensorReading
from app.utils.model_utils import load_model_from_db
from app.utils.model_persistence import ModelPersistence
import logging
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
from app.utils.data_prep import SensorDataLoader

# Initialize router and logging
router = APIRouter()
logger = logging.getLogger(__name__)
data_loader = SensorDataLoader()
models = {}

async def load_model(sensor_id: int, db: Session) -> tf.keras.Model:
    str_id = str(sensor_id)
    if str_id not in models:
        try:
            model_persistence = ModelPersistence(db)
            loaded_model = model_persistence.load_model(sensor_id)
            if loaded_model:
                models[str_id] = loaded_model
                logger.info(f"Loaded model for sensor {sensor_id}")
            else:
                logger.error(f"No model found for sensor {sensor_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    return models.get(str_id)

@router.get("/api/v1/sensors/{sensor_id}/readings")
async def get_sensor_readings(
    sensor_id: int,
    limit: int = 24,
    db: Session = Depends(get_sensor_db)
):
    """Get recent readings for a sensor"""
    try:
        logger.info(f"Fetching readings for sensor {sensor_id}")
        
        query = text("""
            SELECT sensor_id, temperature, humidity, timestamp
            FROM sensor_readings
            WHERE sensor_id = :sensor_id
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        
        readings = db.execute(query, {"sensor_id": sensor_id, "limit": limit}).fetchall()
        
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
    sensor_db: Session = Depends(get_sensor_db),
    pred_db: Session = Depends(get_prediction_db)
):
    """Get predictions for a sensor"""
    try:
        logger.info(f"Starting predictions for sensor {sensor_id}")
        
        # Get latest reading from sensor database
        latest_query = text("""
            SELECT sensor_id, temperature, humidity, timestamp
            FROM sensor_readings
            WHERE sensor_id = :sensor_id
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        latest_reading = sensor_db.execute(latest_query, {"sensor_id": sensor_id}).first()
        
        if not latest_reading:
            raise HTTPException(status_code=404, detail="No sensor readings found")
            
        readings = {
            "sensor_id": str(latest_reading.sensor_id),
            "temperature": latest_reading.temperature,
            "humidity": latest_reading.humidity,
            "timestamp": latest_reading.timestamp.isoformat()
        }
        
        # Load model from prediction database
        model = await load_model(sensor_id, pred_db)
        if not model:
            raise HTTPException(status_code=404, detail=f"No model found for sensor {sensor_id}")
        
        # Make predictions
        temps = np.array([readings['temperature']])
        humids = np.array([readings['humidity']])
        
        temps_norm = data_loader.normalize_data(temps, str(sensor_id), 'temperature')
        humids_norm = data_loader.normalize_data(humids, str(sensor_id), 'humidity')
        
        sequence = np.expand_dims(np.column_stack((temps_norm, humids_norm)), axis=0)
        predictions = []
        current_sequence = sequence
        
        for step in range(steps_ahead):
            pred = model.predict(current_sequence, verbose=0)
            temp_pred, humid_pred = pred[0][0], pred[1][0]
            
            temp_denorm = data_loader.denormalize_data(temp_pred, str(sensor_id), 'temperature')
            humid_denorm = data_loader.denormalize_data(humid_pred, str(sensor_id), 'humidity')
            
            pred_timestamp = latest_reading.timestamp + timedelta(minutes=5 * (step + 1))
            
            predictions.append({
                'sensor_id': sensor_id,
                'timestamp': pred_timestamp.isoformat(),
                'temperature': float(temp_denorm),
                'humidity': float(humid_denorm)
            })
            
            new_point = np.array([[temp_pred, humid_pred]])
            current_sequence = np.concatenate([current_sequence[:, 1:, :], new_point.reshape(1, 1, 2)], axis=1)
        
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
            raise HTTPException(status_code=500, detail=str(db_error))
        
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
            SELECT 
                p.sensor_id,
                p.prediction_type,
                COUNT(*) as prediction_count,
                AVG(p.prediction_value) as avg_value,
                MAX(p.created_at) as latest_prediction
            FROM prediction p
            GROUP BY p.sensor_id, p.prediction_type
            ORDER BY p.sensor_id
        """)
        
        readings_stats = sensor_db.execute(readings_query).fetchall()
        prediction_stats = pred_db.execute(predictions_query).fetchall()
        
        stats = []
        for r_stat in readings_stats:
            # Find matching prediction stats
            temp_pred = next(
                (p for p in prediction_stats 
                 if p.sensor_id == r_stat.sensor_id and p.prediction_type == 'temperature'),
                None
            )
            humid_pred = next(
                (p for p in prediction_stats 
                 if p.sensor_id == r_stat.sensor_id and p.prediction_type == 'humidity'),
                None
            )
            
            stats.append({
                "sensor_id": r_stat.sensor_id,
                "reading_count": r_stat.reading_count,
                "avg_temperature": round(float(r_stat.avg_temperature), 2),
                "avg_humidity": round(float(r_stat.avg_humidity), 2),
                "latest_reading": r_stat.latest_reading.isoformat(),
                "predictions": {
                    "temperature": {
                        "count": temp_pred.prediction_count if temp_pred else 0,
                        "avg_value": round(float(temp_pred.avg_value), 2) if temp_pred else None
                    },
                    "humidity": {
                        "count": humid_pred.prediction_count if humid_pred else 0,
                        "avg_value": round(float(humid_pred.avg_value), 2) if humid_pred else None
                    }
                }
            })
        
        return stats
    except Exception as e:
        logger.error(f"Error fetching sensor stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))