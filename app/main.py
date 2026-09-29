import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
DB = os.getenv("PREDICTION_DB", "predictions.db")

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
    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='credit-default'")
    if not versions:
        raise RuntimeError("No registered model. Run training first.")
    version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    # Load the scikit-learn estimator (not a pyfunc wrapper) so predict_proba
    # is available for the default probability.
    model = mlflow.sklearn.load_model(version.source)
    return model, str(version.version)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: Customer):
    model, version = get_model()
    data = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

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
