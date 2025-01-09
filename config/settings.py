# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path

class Settings(BaseSettings):
    # Create data directory if it doesn't exist
    data_dir: Path = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Single database URL for all operations
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{data_dir}/sensor.db")
    
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
    def database_url(self) -> str:
        """Get the database URL with proper protocol"""
        return self._fix_postgres_url(self.DATABASE_URL)

@lru_cache()
def get_settings() -> Settings:
    return Settings()