# app/utils/data_prep.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SensorDataLoader:
    def __init__(
        self,
        sequence_length: int = 24,
        prediction_steps: int = 1,
        train_split: float = 0.8
    ):
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.train_split = train_split
        self.scalers = {}

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare sensor data from CSV"""
        try:
            # Load the CSV data
            df = pd.read_csv(csv_path)
            
            # Convert timestamp columns to datetime
            df['first_reading'] = pd.to_datetime(df['first_reading'])
            df['latest_reading'] = pd.to_datetime(df['latest_reading'])
            
            # Generate synthetic time series based on averages
            expanded_data = []
            
            for _, row in df.iterrows():
                # Calculate time delta between readings
                total_duration = (row['latest_reading'] - row['first_reading']).total_seconds()
                interval = total_duration / row['total_readings']
                
                # Generate timestamps
                timestamps = [
                    row['first_reading'] + timedelta(seconds=i*interval)
                    for i in range(int(row['total_readings']))
                ]
                
                # Add some random variation around the average
                temp_std = 0.1  # Standard deviation for temperature
                humid_std = 0.5  # Standard deviation for humidity
                
                temperatures = np.random.normal(
                    row['avg_temperature'], 
                    temp_std, 
                    len(timestamps)
                )
                
                humidities = np.random.normal(
                    row['avg_humidity'], 
                    humid_std, 
                    len(timestamps)
                )
                
                # Create expanded dataframe
                expanded_df = pd.DataFrame({
                    'sensor_id': row['sensor_id'],
                    'timestamp': timestamps,
                    'temperature': temperatures,
                    'humidity': humidities
                })
                
                expanded_data.append(expanded_df)
            
            # Combine all expanded data
            expanded_df = pd.concat(expanded_data, ignore_index=True)
            return expanded_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def normalize_data(self, data: np.ndarray, sensor_id: str, feature_name: str) -> np.ndarray:
        """Normalize data using min-max scaling"""
        if f"{sensor_id}_{feature_name}" not in self.scalers:
            self.scalers[f"{sensor_id}_{feature_name}"] = {
                'min': np.min(data),
                'max': np.max(data)
            }
        
        scaler = self.scalers[f"{sensor_id}_{feature_name}"]
        normalized = (data - scaler['min']) / (scaler['max'] - scaler['min'])
        return normalized

    def denormalize_data(self, data: np.ndarray, sensor_id: str, feature_name: str) -> np.ndarray:
        """Denormalize data back to original scale"""
        scaler = self.scalers[f"{sensor_id}_{feature_name}"]
        return data * (scaler['max'] - scaler['min']) + scaler['min']

    def create_sequences(
        self,
        data: pd.DataFrame,
        sensor_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Extract features
            temp_data = self.normalize_data(data['temperature'].values, sensor_id, 'temperature')
            humid_data = self.normalize_data(data['humidity'].values, sensor_id, 'humidity')
            
            X, y_temp, y_humid = [], [], []
            
            for i in range(len(data) - self.sequence_length - self.prediction_steps + 1):
                # Input sequence
                temp_seq = temp_data[i:(i + self.sequence_length)]
                humid_seq = humid_data[i:(i + self.sequence_length)]
                
                # Stack features
                sequence = np.column_stack((temp_seq, humid_seq))
                
                # Target values
                temp_target = temp_data[i + self.sequence_length:i + self.sequence_length + self.prediction_steps]
                humid_target = humid_data[i + self.sequence_length:i + self.sequence_length + self.prediction_steps]
                
                X.append(sequence)
                y_temp.append(temp_target)
                y_humid.append(humid_target)
            
            return np.array(X), np.array(y_temp), np.array(y_humid)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise

    def prepare_data(
        self,
        csv_path: str,
        sensor_id: str
    ) -> Dict[str, np.ndarray]:
        """Prepare data for training and validation"""
        try:
            # Load and expand data
            df = self.load_data(csv_path)
            
            # Filter for specific sensor
            sensor_data = df[df['sensor_id'] == int(sensor_id)]
            
            # Create sequences
            X, y_temp, y_humid = self.create_sequences(sensor_data, sensor_id)
            
            # Split into train and validation
            split_idx = int(len(X) * self.train_split)
            
            train_data = {
                'X': X[:split_idx],
                'y_temp': y_temp[:split_idx],
                'y_humid': y_humid[:split_idx]
            }
            
            val_data = {
                'X': X[split_idx:],
                'y_temp': y_temp[split_idx:],
                'y_humid': y_humid[split_idx:]
            }
            
            return {'train': train_data, 'val': val_data}
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise