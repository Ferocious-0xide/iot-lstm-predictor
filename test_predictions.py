# test_predictions.py
import asyncio
from app.services.prediction import PredictionService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_prediction_pipeline():
    try:
        # Initialize prediction service
        predictor = PredictionService()
        
        # Test for one sensor first
        sensor_id = "1"
        logger.info(f"Testing predictions for sensor {sensor_id}...")
        
        # Try to make and store predictions
        predictions = await predictor.predict_and_store(sensor_id, steps_ahead=3)
        
        # Log the results
        logger.info("Predictions generated:")
        for pred in predictions:
            logger.info(f"Time: {pred['timestamp']}, Temp: {pred['temperature']:.2f}, Humidity: {pred['humidity']:.2f}")
            
    except Exception as e:
        logger.error(f"Error during prediction test: {e}")

if __name__ == "__main__":
    asyncio.run(test_prediction_pipeline())