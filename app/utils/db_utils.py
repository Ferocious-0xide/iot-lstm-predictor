# app/utils/db_utils.py
from sqlalchemy import create_engine, text  # Added text to this line
from sqlalchemy.orm import sessionmaker, Session
import logging
from config.settings import get_settings
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


logger = logging.getLogger(__name__)
settings = get_settings()

def create_db_engine(url: str, **kwargs):
    """Create database engine, handling Heroku's postgres:// URLs"""
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return create_engine(url, **kwargs)

# Create engines using environment-based settings
prediction_engine = create_db_engine(
    settings.prediction_database_url,  # Now using HEROKU_POSTGRESQL_PURPLE_URL
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_timeout=30
)

sensor_engine = create_db_engine(
    settings.sensor_database_url,  # Using DATABASE_URL (follower database)
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_timeout=30,
    # Adding read-only settings since this is a follower database
    execution_options={'readonly': True}
)

# Create session factories
PredictionSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=prediction_engine
)

SensorSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=sensor_engine
)

def get_prediction_db():
    """Get a database session for storing predictions"""
    db = PredictionSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_sensor_db():
    """Get a read-only database session for sensor data"""
    db = SensorSessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_database_connections():
    """Test both database connections and verify tables"""
    try:
        # Test sensor database
        with next(get_sensor_db()) as session:
            result = session.execute(text("SELECT current_timestamp, current_database()"))
            timestamp, database = result.first()
            logger.info(f"Successfully connected to sensor database: {database}")
            
        # Test prediction database and verify new tables
        with next(get_prediction_db()) as session:
            result = session.execute(text("SELECT current_timestamp, current_database()"))
            timestamp, database = result.first()
            logger.info(f"Successfully connected to prediction database: {database}")
            
            # Check if prediction tables exist
            tables = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            table_names = [table[0] for table in tables]
            logger.info(f"Available tables in prediction database: {table_names}")
            
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False