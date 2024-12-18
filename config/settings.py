# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # Database URLs from environment variables
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    HEROKU_POSTGRESQL_PURPLE_URL: str = os.getenv("HEROKU_POSTGRESQL_PURPLE_URL", "")
    
    # Application settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-123")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Model settings
    MODEL_PATH: str = "./models"
    SEQUENCE_LENGTH: int = 24

    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow extra fields to prevent validation errors
        extra = 'allow'

    @property
    def sensor_database_url(self) -> str:
        """Get the sensor database URL (follower database)"""
        return self.DATABASE_URL

    @property
    def prediction_database_url(self) -> str:
        """Get the predictor app's database URL"""
        return self.HEROKU_POSTGRESQL_PURPLE_URL

@lru_cache()
def get_settings() -> Settings:
    return Settings()