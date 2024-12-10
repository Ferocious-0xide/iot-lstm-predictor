# app/main.py
from fastapi import FastAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "IoT LSTM Predictor API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}