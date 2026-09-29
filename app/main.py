import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.registry import load_latest_model  # noqa: E402

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
DB = os.getenv("PREDICTION_DB", "predictions.db")
REGISTERED_MODEL = "credit-default"
# Credit portfolios default at a low rate, so the naive 0.5 cut-off flags almost
# nobody. The threshold analysis in MLflow (and src/train.py) shows where F1 or
# the business cut-off peaks; set it here.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

class Customer(BaseModel):
    age: int
    income: float
    loan_amount: float
    tenure: int

def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT,
            age REAL,
            income REAL,
            loan_amount REAL,
            tenure REAL,
            prediction INTEGER,
            probability REAL,
            model_version TEXT
        )
        ''')

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Credit Default MLOps API", lifespan=lifespan)

def get_model():
    """Serve the newest registered version of the credit-default model."""
    return load_latest_model(REGISTERED_MODEL)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: Customer):
    model, version = get_model()
    data = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= DECISION_THRESHOLD)

    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                customer.age,
                customer.income,
                customer.loan_amount,
                customer.tenure,
                prediction,
                probability,
                version,
            ),
        )

    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": version,
    }
