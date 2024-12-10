# config/settings.py
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
    
    # Redis
    REDIS_URL: str = os.environ.get("REDIS_URL", "")
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Model Settings
    MODEL_PATH: str = "./models"
    SEQUENCE_LENGTH: int = 24
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings"""
    return Settings()