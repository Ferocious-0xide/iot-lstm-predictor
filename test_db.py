# test_db.py
from sqlalchemy import create_engine, text
from config.settings import get_settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    settings = get_settings()
    logger.info(f"Testing connection to database...")
    
    try:
        # Create test engine
        engine = create_engine(settings.DATABASE_URL)
        
        # Try to connect and execute a simple query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info(f"Connection successful! Result: {result.scalar()}")
            return True
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()