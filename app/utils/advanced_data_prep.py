import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta

class AdvancedSensorDataLoader:
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.scalers = {}
    
    def normalize_data(self, data: np.ndarray, sensor_id: str, feature: str) -> np.ndarray:
        """Normalize data using min-max scaling"""
        key = f"{sensor_id}_{feature}"
        if key not in self.scalers:
            min_val = np.min(data)
            max_val = np.max(data)
            self.scalers[key] = {'min': min_val, 'max': max_val}
        
        min_val = self.scalers[key]['min']
        max_val = self.scalers[key]['max']
        
        return (data - min_val) / (max_val - min_val + 1e-10)
    
    def denormalize_data(self, data: np.ndarray, sensor_id: str, feature: str) -> np.ndarray:
        """Denormalize data using stored scaling parameters"""
        key = f"{sensor_id}_{feature}"
        if key not in self.scalers:
            raise ValueError(f"No scaling parameters found for {key}")
        
        min_val = self.scalers[key]['min']
        max_val = self.scalers[key]['max']
        
        return data * (max_val - min_val) + min_val
    
    def create_sequences(self, data: np.ndarray, target_steps: int = 1) -> Dict[str, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - target_steps + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length:i + self.sequence_length + target_steps])
        
        return {
            'X': np.array(X),
            'y': np.array(y)
        }
    
    def prepare_data(self, csv_path: str, sensor_id: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare data for training from CSV file"""
        # Load data
        df = pd.read_csv(csv_path)
        
        # Ensure datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Extract features
        temps = df['temperature'].values
        humids = df['humidity'].values
        
        # Normalize data
        temps_norm = self.normalize_data(temps, sensor_id, 'temperature')
        humids_norm = self.normalize_data(humids, sensor_id, 'humidity')
        
        # Combine features
        combined_data = np.column_stack((temps_norm, humids_norm))
        
        # Create sequences
        sequences = self.create_sequences(combined_data)
        
        # Split into train and validation (80/20)
        split_idx = int(len(sequences['X']) * 0.8)
        
        train_data = {
            'X': sequences['X'][:split_idx],
            'y': sequences['y'][:split_idx]
        }
        
        val_data = {
            'X': sequences['X'][split_idx:],
            'y': sequences['y'][split_idx:]
        }
        
        return {
            'train': train_data,
            'val': val_data
        }
    
    def prepare_prediction_sequence(
        self,
        recent_data: Dict[str, Any],
        sensor_id: str
    ) -> np.ndarray:
        """Prepare a sequence for prediction"""
        temps = np.array([reading['temperature'] for reading in recent_data])
        humids = np.array([reading['humidity'] for reading in recent_data])
        
        temps_norm = self.normalize_data(temps, sensor_id, 'temperature')
        humids_norm = self.normalize_data(humids, sensor_id, 'humidity')
        
        sequence = np.column_stack((temps_norm, humids_norm))
        return np.expand_dims(sequence[-self.sequence_length:], axis=0)