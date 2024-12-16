# test_db_connection.py
from app.utils.db_utils import get_sensor_db
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sensor_db_connection():
    """Test connection to sensor database"""
    try:
        with get_sensor_db() as db:
            # Try to get table information
            result = db.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            logger.info(f"Found tables: {tables}")
            
            # Try to get a sample reading
            result = db.execute(text("""
                SELECT COUNT(*) 
                FROM sensor_readings
            """))
            count = result.scalar()
            logger.info(f"Total sensor readings: {count}")
            
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_sensor_db_connection()