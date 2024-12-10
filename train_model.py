# train_model.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from app.utils.data_prep import SensorDataLoader
from app.models.lstm_model import SensorLSTM
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_training_history(history, sensor_id):
    """Plot training metrics"""
    plt.figure(figsize=(12, 8))
    
    # Temperature metrics
    plt.subplot(2, 1, 1)
    plt.plot(history['temperature_loss'], label='Temperature Loss')
    plt.plot(history['val_temperature_loss'], label='Val Temperature Loss')
    plt.title(f'Training Metrics - Sensor {sensor_id}')
    plt.ylabel('Loss')
    plt.legend()
    
    # Humidity metrics
    plt.subplot(2, 1, 2)
    plt.plot(history['humidity_loss'], label='Humidity Loss')
    plt.plot(history['val_humidity_loss'], label='Val Humidity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plt.savefig(f'training_history_sensor_{sensor_id}.png')
    plt.close()

def train_sensor_model(
    csv_path: str,
    sensor_id: str,
    sequence_length: int = 24,
    prediction_steps: int = 1,
    epochs: int = 50,
    batch_size: int = 32
):
    """Train LSTM model for a specific sensor"""
    try:
        logger.info(f"Starting training for sensor {sensor_id}")
        
        # Initialize data loader
        data_loader = SensorDataLoader(
            sequence_length=sequence_length,
            prediction_steps=prediction_steps
        )
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data = data_loader.prepare_data(csv_path, sensor_id)
        
        logger.info(f"Training data shape: {data['train']['X'].shape}")
        logger.info(f"Validation data shape: {data['val']['X'].shape}")
        
        # Initialize model
        model = SensorLSTM(
            sequence_length=sequence_length,
            n_features=2,  # temperature and humidity
            lstm_units=64,
            dropout_rate=0.2
        )
        
        # Train model
        logger.info("Starting model training...")
        history = model.train(
            train_data=data['train'],
            val_data=data['val'],
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Plot training history
        plot_training_history(history, sensor_id)
        
        # Save model
        model_path = f'models/sensor_{sensor_id}_model'
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Test prediction
        test_sequence = data['val']['X'][:1]
        temp_pred, humid_pred = model.predict(test_sequence)
        
        # Denormalize predictions
        temp_actual = data_loader.denormalize_data(
            data['val']['y_temp'][0],
            sensor_id,
            'temperature'
        )
        humid_actual = data_loader.denormalize_data(
            data['val']['y_humid'][0],
            sensor_id,
            'humidity'
        )
        
        temp_pred = data_loader.denormalize_data(
            np.array([temp_pred]),
            sensor_id,
            'temperature'
        )
        humid_pred = data_loader.denormalize_data(
            np.array([humid_pred]),
            sensor_id,
            'humidity'
        )
        
        logger.info(f"""
        Test Prediction Results:
        Temperature: Actual = {temp_actual[0]:.2f}, Predicted = {temp_pred[0]:.2f}
        Humidity: Actual = {humid_actual[0]:.2f}, Predicted = {humid_pred[0]:.2f}
        """)
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Train models for each sensor
    csv_path = "dataclips_gtwalycvdjkbrsuztimvktqzmvfs.csv"
    sensor_ids = ['1', '2', '3', '4', '5']
    
    for sensor_id in sensor_ids:
        logger.info(f"\nTraining model for sensor {sensor_id}")
        model, history = train_sensor_model(csv_path, sensor_id)
        logger.info(f"Completed training for sensor {sensor_id}\n")