# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.services.training import ModelTrainingService
from app.services.prediction import PredictionService
from app.utils.db_utils import get_db

router = APIRouter()

# Pydantic models for request/response validation
class TrainingRequest(BaseModel):
    sensor_id: str
    hyperparameters: Optional[dict] = None

class TrainingResponse(BaseModel):
    job_id: int
    status: str
    start_time: datetime

class TrainingStatus(BaseModel):
    job_id: int
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str]

class PredictionRequest(BaseModel):
    sensor_id: str
    steps: Optional[int] = 24

class Prediction(BaseModel):
    timestamp: datetime
    temperature: float
    humidity: float

class ModelPerformance(BaseModel):
    temperature_mae: float
    humidity_mae: float
    predictions_count: int
    date_range: dict

# Training endpoints
@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a new model training job"""
    try:
        training_service = ModelTrainingService(db)
        job = await training_service.start_training_job(
            request.sensor_id,
            request.hyperparameters
        )
        
        return {
            "job_id": job.id,
            "status": job.status,
            "start_time": job.start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/train/{job_id}", response_model=TrainingStatus)
async def get_training_status(
    job_id: int,
    db: Session = Depends(get_db)
):
    """Get the status of a training job"""
    try:
        training_service = ModelTrainingService(db)
        status = await training_service.get_training_status(job_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# Prediction endpoints
@router.post("/predict", response_model=List[Prediction])
async def get_predictions(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Get predictions for a sensor"""
    try:
        prediction_service = PredictionService(db)
        predictions = prediction_service.get_predictions(
            request.sensor_id,
            prediction_steps=request.steps
        )
        return predictions
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{sensor_id}", response_model=ModelPerformance)
async def get_model_performance(
    sensor_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get model performance metrics"""
    try:
        prediction_service = PredictionService(db)
        performance = prediction_service.get_model_performance(
            sensor_id,
            start_date,
            end_date
        )
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@router.get("/models/{sensor_id}")
async def get_model_versions(
    sensor_id: str,
    db: Session = Depends(get_db)
):
    """Get all model versions for a sensor"""
    try:
        query = db.query(ModelVersion)\
            .filter(ModelVersion.sensor_id == sensor_id)\
            .order_by(ModelVersion.training_date.desc())\
            .all()
        
        return [{
            "version": model.version,
            "training_date": model.training_date,
            "is_active": model.is_active,
            "metrics": model.metrics
        } for model in query]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{sensor_id}/{version_id}/activate")
async def activate_model_version(
    sensor_id: str,
    version_id: int,
    db: Session = Depends(get_db)
):
    """Activate a specific model version"""
    try:
        # Deactivate current active model
        db.query(ModelVersion)\
            .filter(ModelVersion.sensor_id == sensor_id)\
            .update({"is_active": False})
        
        # Activate specified version
        model = db.query(ModelVersion).get(version_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model version not found")
        
        model.is_active = True
        db.commit()
        
        return {"message": f"Activated model version {model.version}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow()
    }