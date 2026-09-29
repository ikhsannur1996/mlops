"""Shared fixtures: every test runs against an isolated MLflow backend and temp dir."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """Point MLflow, the prediction log and generated reports at a temp directory."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("PREDICTION_DB", str(tmp_path / "predictions.db"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.delenv("AUTO_RETRAIN", raising=False)
    monkeypatch.chdir(tmp_path)

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    yield tmp_path


@pytest.fixture
def repository_root():
    """Absolute path of the repository root."""
    return ROOT


@pytest.fixture
def train_frame(repository_root):
    """The committed training dataset."""
    return pd.read_csv(repository_root / "data" / "train.csv")
