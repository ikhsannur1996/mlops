"""FastAPI serving contract: health, prediction and prediction persistence."""

import importlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.train import main as train_main

VALID_PAYLOAD = {"age": 35, "income": 10000000, "loan_amount": 50000000, "tenure": 24}


@pytest.fixture
def api():
    """A TestClient serving a freshly registered model from the isolated backend."""
    train_main()

    import app.main as api_module

    api_module = importlib.reload(api_module)  # pick up the isolated environment

    with TestClient(api_module.app) as client:
        yield client, api_module.DB


def test_health_endpoint(api):
    client, _ = api
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_prediction_and_the_model_version(api):
    client, _ = api
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "1"


def test_predict_persists_the_request_for_monitoring(api):
    client, db_path = api
    client.post("/predict", json=VALID_PAYLOAD)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT age, income, loan_amount, tenure FROM predictions").fetchall()

    assert rows == [(35, 10000000.0, 50000000.0, 24)]


def test_predict_rejects_an_invalid_payload():
    import app.main as api_module

    with TestClient(importlib.reload(api_module).app) as client:
        assert client.post("/predict", json={"age": "not-a-number"}).status_code == 422


def test_a_lower_decision_threshold_flags_more_applicants(monkeypatch):
    """DECISION_THRESHOLD is the operational cut-off, so the API must honour it."""
    train_main()
    monkeypatch.setenv("DECISION_THRESHOLD", "0.0")

    import app.main as api_module

    with TestClient(importlib.reload(api_module).app) as client:
        body = client.post("/predict", json=VALID_PAYLOAD).json()

    assert body["prediction"] == 1
    assert body["probability"] >= 0.0
