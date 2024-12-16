import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import logging
import scipy.stats as stats

logger = logging.getLogger(__name__)

class AdvancedSensorDataLoader:
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
        
        # Enhanced variability parameters
        self.noise_models = {
            'temperature': {
                'drift_rate': 0.05,      # Gradual drift tendency
                'outlier_prob': 0.02,    # Probability of extreme readings
                'failure_prob': 0.01     # Probability of sensor failure
            },
            'humidity': {
                'drift_rate': 0.1,       # Higher drift for humidity
                'outlier_prob': 0.03,    # More frequent outliers
                'failure_prob': 0.02     # Slightly higher failure rate
            }
        }

    def generate_realistic_noise(
        self, 
        base_value: float, 
        std: float, 
        length: int, 
        feature: str
    ) -> np.ndarray:
        """
        Generate noise with multiple realistic characteristics
        - Gaussian base noise
        - Occasional drift
        - Rare but significant outliers
        - Potential sensor failures
        """
        noise_config = self.noise_models[feature]
        
        # Base Gaussian noise
        base_noise = np.random.normal(0, std, length)
        
        # Gradual drift
        drift = np.cumsum(np.random.normal(0, noise_config['drift_rate'], length))
        
        # Outlier generation (using heavy-tailed distribution)
        outliers = np.zeros(length)
        outlier_mask = np.random.random(length) < noise_config['outlier_prob']
        outliers[outlier_mask] = np.random.normal(
            loc=base_value * 2, 
            scale=std * 3, 
            size=np.sum(outlier_mask)
        )
        
        # Combine noise components
        combined_noise = base_noise + drift + outliers
        
        # Simulate occasional sensor failures (replace with NaN)
        failure_mask = np.random.random(length) < noise_config['failure_prob']
        combined_noise[failure_mask] = np.nan
        
        return combined_noise

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare sensor data from CSV with enhanced realism"""
        try:
            # Load the CSV data
            df = pd.read_csv(csv_path)
            
            # Convert timestamp columns to datetime
            df['first_reading'] = pd.to_datetime(df['first_reading'])
            df['latest_reading'] = pd.to_datetime(df['latest_reading'])
            
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
                
                # Enhanced noise model for temperature and humidity
                temperatures = row['avg_temperature'] + self.generate_realistic_noise(
                    base_value=row['avg_temperature'], 
                    std=0.5,  # Increased variability 
                    length=len(timestamps),
                    feature='temperature'
                )
                
                humidities = row['avg_humidity'] + self.generate_realistic_noise(
                    base_value=row['avg_humidity'], 
                    std=1.0,  # Higher variability for humidity
                    length=len(timestamps),
                    feature='humidity'
                )
                
                # Create expanded dataframe
                expanded_df = pd.DataFrame({
                    'sensor_id': row['sensor_id'],
                    'timestamp': timestamps,
                    'temperature': temperatures,
                    'humidity': humidities
                })
                
                # Handle NaN values (sensor failures)
                expanded_df = self._handle_sensor_failures(expanded_df)
                
                expanded_data.append(expanded_df)
            
            # Combine all expanded data
            expanded_df = pd.concat(expanded_data, ignore_index=True)
            return expanded_df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _handle_sensor_failures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent handling of sensor failures:
        - Interpolate short gaps
        - Flag extended failures
        """
        # Interpolate small gaps (up to 3 consecutive NaNs)
        df['temperature'] = df['temperature'].interpolate(limit=3, method='linear')
        df['humidity'] = df['humidity'].interpolate(limit=3, method='linear')
        
        # Flag extended failures (more than 3 consecutive NaNs)
        df['sensor_status'] = np.where(
            df['temperature'].isna() | df['humidity'].isna(), 
            'FAILURE', 
            'OPERATIONAL'
        )
        
        return df

    # Rest of the methods remain the same as in the original implementation
    # (normalize_data, denormalize_data, create_sequences, prepare_data)