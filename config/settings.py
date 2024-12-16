# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from typing import List

class Settings(BaseSettings):
    # Database URLs
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    SENSOR_DATABASE_URL: str = os.getenv("SENSOR_DATABASE_URL", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-123")
    
    # Model settings
    MODEL_PATH: str = "./models"
    SEQUENCE_LENGTH: int = 24
    
    # Optional: CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()