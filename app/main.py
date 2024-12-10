# app/main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "IoT LSTM Predictor API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}