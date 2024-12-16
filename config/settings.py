# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # Our app's database for predictions
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # External sensor data (follower from other app)
    SENSOR_DATABASE_URL: str = os.getenv("SENSOR_DATABASE_URL", "")
    
    # Model settings
    MODEL_PATH: str = "./models"
    SEQUENCE_LENGTH: int = 24
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()