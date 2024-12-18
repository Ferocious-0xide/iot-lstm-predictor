# migrations/create_prediction_tables.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.db_models import Base
from app.utils.db_utils import create_db_engine
from config.settings import get_settings

def create_prediction_tables():
    settings = get_settings()
    engine = create_db_engine(settings.prediction_database_url)
    
    try:
        Base.metadata.create_all(engine)
        print("Successfully created prediction tables!")
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    create_prediction_tables()