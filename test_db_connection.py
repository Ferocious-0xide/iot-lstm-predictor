import logging
from app.utils.db_utils import test_database_connections
from config.settings import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    logger.info("Testing database connections...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Test connections
    if test_database_connections():
        logger.info("✅ All database connections successful!")
    else:
        logger.error("❌ Database connection test failed")

if __name__ == "__main__":
    main()