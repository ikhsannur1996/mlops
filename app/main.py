import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/champion.joblib"))
FEATURES = ["age", "monthly_income", "loan_amount", "tenure_months"]

app = FastAPI(title="Simple MLOps Education")

# Champion model saved by src/pipeline.py
_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
_model = _bundle["model"] if _bundle else None
_champion_name = _bundle["name"] if _bundle else None


class Applicant(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    monthly_income: int = Field(..., gt=0, description="Monthly income in IDR",
                                examples=[12_000_000])
    loan_amount: int = Field(..., gt=0, description="Requested loan amount in IDR",
                             examples=[80_000_000])
    tenure_months: int = Field(..., gt=0, le=120, description="Loan tenure in months",
                               examples=[36])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 34,
                "monthly_income": 12_000_000,
                "loan_amount": 80_000_000,
                "tenure_months": 36
            }]
        }
    }


class Prediction(BaseModel):
    prediction: str = Field(..., description="Good or Risk")
    probability_risk: float = Field(..., description="P(Risk)")
    model: str = Field(..., description="Model used for the prediction")


@app.get("/")
def home():
    return {
        "message": "Simple MLOps Education",
        "mlflow": "http://localhost:5001",
        "evidently": "http://localhost:8001",
        "api": "http://localhost:8000"
    }


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Run the pipeline first."
        )
    X = pd.DataFrame([[applicant.age, applicant.monthly_income,
                       applicant.loan_amount, applicant.tenure_months]],
                     columns=FEATURES)
    proba = _model.predict_proba(X)[0]
    p_risk = float(proba[list(_model.classes_).index("Risk")])
    return Prediction(
        prediction="Risk" if p_risk >= 0.5 else "Good",
        probability_risk=round(p_risk, 4),
        model=_champion_name
    )
