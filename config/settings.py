# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path

class Settings(BaseSettings):
    # Create data directory if it doesn't exist
    data_dir: Path = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Database URLs
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{data_dir}/sensor.db")
    PREDICTION_DATABASE_URL: str = os.getenv("HEROKU_POSTGRESQL_PURPLE_URL", DATABASE_URL)
    
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
        extra = 'allow'

    def _fix_postgres_url(self, url: str) -> str:
        """Convert postgres:// to postgresql:// if needed"""
        if url and url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql://", 1)
        return url

    @property
    def sensor_database_url(self) -> str:
        """Get the read-only sensor database URL"""
        return self._fix_postgres_url(self.DATABASE_URL)

    @property
    def prediction_database_url(self) -> str:
        """Get the read-write prediction database URL"""
        return self._fix_postgres_url(self.PREDICTION_DATABASE_URL)

@lru_cache()
def get_settings() -> Settings:
    return Settings()