"""
api/main.py — STEP 9: FastAPI backend for the Flight Delay Prediction Engine.
Run from project root:  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Open:  http://127.0.0.1:8000/docs  (interactive API docs)
"""
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.predict import predict_flight

app = FastAPI(
    title="Predictive Aviation Analytics — Flight Delay API",
    description="Predicts P(arrival delay > 15 min) and estimated delay minutes.",
    version="1.0.0",
)

class FlightInput(BaseModel):
    """Pydantic validates types & ranges automatically (bad input -> HTTP 422)."""
    airline: str = Field(..., example="UA", description="IATA airline code, e.g. UA, DL, AA, WN")
    origin: str = Field(..., example="ORD")
    dest: str = Field(..., example="LAX")
    date: str = Field(..., example="2015-07-04", description="YYYY-MM-DD")
    sched_dep: int = Field(..., example=1830, ge=0, le=2359, description="scheduled departure HHMM")
    sched_arr: int = Field(..., example=2110, ge=0, le=2359)
    sched_time: int = Field(..., example=280, ge=10, le=1200, description="scheduled flight duration, minutes")
    distance: float = Field(None, description="optional km; auto-estimated if omitted")

@app.get("/")
def home():
    return {"message": "Flight Delay Prediction API is running",
            "docs": "/docs", "predict_endpoint": "POST /predict"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/airlines")
def airlines():
    a = pd.read_parquet("data/processed/airlines.parquet")
    return a.to_dict(orient="records")

@app.get("/airports")
def airports():
    a = pd.read_parquet("data/processed/airports.parquet")
    return a[["IATA_CODE", "AIRPORT", "CITY", "STATE"]].to_dict(orient="records")

@app.post("/predict")
def predict(f: FlightInput):
    return predict_flight(
        airline=f.airline.upper(), origin=f.origin.upper(), dest=f.dest.upper(),
        date=f.date, sched_dep=f.sched_dep, sched_arr=f.sched_arr,
        sched_time=f.sched_time, distance=f.distance,
    )