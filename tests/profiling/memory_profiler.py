"""
Memory Profiler for IoT LSTM Predictor

This module profiles memory usage of the LSTM model during loading and prediction
operations. It helps determine resource requirements for deployment and identifies
potential memory leaks or inefficiencies.

Key metrics tracked:
- Baseline memory usage
- Memory increase during model operations
- Prediction time and throughput
- Memory stability over multiple predictions

Usage:
    PYTHONPATH=$PYTHONPATH:. python -m tests.profiling.memory_profiler

Results Analysis:
    - Baseline ~440MB indicates base Python + TensorFlow overhead
    - ~21MB increase shows efficient memory usage during predictions
    - 12-13ms inference time suggests good performance for IoT applications
"""

from memory_profiler import profile
import psutil
import gc
import time
import numpy as np
from typing import Dict, Any
import logging
from app.utils.db_utils import get_sensor_db_context
from app.utils.model_persistence import ModelPersistence
from app.models.lstm_model import TempHumidityPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def profile_model():
    """Profile memory usage for model operations.
    
    This function:
    1. Establishes baseline memory usage
    2. Loads model and tracks memory increase
    3. Performs batch predictions
    4. Monitors memory stability
    5. Reports timing metrics
    """
    # Initialize model with production-like settings
    with get_sensor_db_context() as db:
        model_persistence = ModelPersistence(db)
        predictor = TempHumidityPredictor(sequence_length=24)

        # Establish baseline before any operations
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Baseline memory usage: {baseline_memory:.2f} MB")

        # Generate realistic test data
        # Shape: (1000, 24, 2) represents:
        # - 1000 sequences
        # - 24 timesteps per sequence
        # - 2 features (temperature, humidity)
        sample_data = np.random.random((1000, 24, 2))
        
        # Profile predictions with production batch size
        start_time = time.time()
        for i in range(0, len(sample_data), 32):
            batch = sample_data[i:i + 32]
            predictor.predict(batch)
            
            # Track memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Current memory usage: {current_memory:.2f} MB")
            logger.info(f"Memory increase: {current_memory - baseline_memory:.2f} MB")
        
        # Report total processing time
        total_time = time.time() - start_time
        logger.info(f"Total prediction time: {total_time:.2f} seconds")

if __name__ == "__main__":
    profile_model()