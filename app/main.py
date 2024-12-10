# app/main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
from app.utils.db_utils import init_db, get_db
from app.api.routes import router as sensor_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Include routers
app.include_router(sensor_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialization complete")

@app.get("/")
async def root():
    return {"message": "IoT LSTM Predictor API"}

@app.get("/health")
async def health(db: Session = Depends(get_db)):
    try:
        # Test database connection
        result = db.execute(text("SELECT 1")).scalar()
        db_status = "connected" if result == 1 else "error"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "error"
    
    return {
        "status": "healthy",
        "database": db_status
    }