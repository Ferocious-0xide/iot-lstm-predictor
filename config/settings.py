# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path

class Settings(BaseSettings):
    # Create data directory if it doesn't exist
    data_dir: Path = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Database URLs from environment variables with SQLite fallbacks
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{data_dir}/sensor.db")
    HEROKU_POSTGRESQL_PURPLE_URL: str = os.getenv("HEROKU_POSTGRESQL_PURPLE_URL", 
                                                 f"sqlite:///{data_dir}/predictions.db")
    
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

    def _fix_postgres_url(self, url: str) -> str:
        """Convert postgres:// to postgresql:// if needed"""
        if url and url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql://", 1)
        return url

    @property
    def sensor_database_url(self) -> str:
        """Get the sensor database URL (follower database)"""
        return self._fix_postgres_url(self.DATABASE_URL)

    @property
    def prediction_database_url(self) -> str:
        """Get the predictor app's database URL"""
        # First try HEROKU_POSTGRESQL_PURPLE_URL, fall back to DATABASE_URL
        url = self.HEROKU_POSTGRESQL_PURPLE_URL or self.DATABASE_URL
        return self._fix_postgres_url(url)

@lru_cache()
def get_settings() -> Settings:
    return Settings()