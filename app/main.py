# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import tensorflow as tf
import uvicorn

from app.api.routes import router as api_router
from app.utils.db_utils import init_db, engine
from app.models.db_models import Base
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        logger.info("Starting up application...")
        
        # Initialize database
        logger.info("Initializing database...")
        Base.metadata.create_all(bind=engine)
        
        # Configure TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')
        logger.info("TensorFlow configured for CPU usage")
        
        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    
    # Shutdown
    try:
        logger.info("Shutting down application...")
        # Add any cleanup code here
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise

def create_application() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title="IoT LSTM Predictor",
        description="Real-time IoT sensor data prediction using LSTM neural networks",
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(
        api_router,
        prefix="/api/v1"
    )

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "timestamp": datetime.now().isoformat(),
                "status": exc.status_code,
                "error": exc.detail,
                "path": request.url.path
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "timestamp": datetime.now().isoformat(),
                "status": 500,
                "error": "Internal Server Error",
                "path": request.url.path,
                "detail": str(exc) if settings.ENVIRONMENT == "development" else "Internal Server Error"
            },
        )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "IoT LSTM Predictor API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health": "healthy",
            "timestamp": datetime.now().isoformat()
        }

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "environment": settings.ENVIRONMENT
        }

    return app

app = create_application()

# Run the application directly if this file is executed
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if get_settings().ENVIRONMENT == "development" else False
    )