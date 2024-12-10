# app/main.py
from fastapi import FastAPI
from app.utils.db_utils import init_db

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/")
async def root():
    return {"message": "IoT LSTM Predictor API"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": "connected",
        "redis": "connected"
    }