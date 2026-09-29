"""Drift-gated retraining: which action follows a monitoring result."""

import os

import pytest

from src import retrain
from tests.helpers import make_prediction_rows, seed_predictions

SHIFTED_INCOME = 500_000_000.0


@pytest.fixture
def retrain_calls(monkeypatch):
    """Replace the expensive train/evaluate steps with cheap recorders."""
    calls = []

    monkeypatch.setattr(retrain.train, "main", lambda: calls.append("train"))
    monkeypatch.setattr(retrain.evaluate, "main", lambda: calls.append("evaluate"))

    return calls


def test_retrain_skips_without_production_data(retrain_calls):
    assert retrain.main() == retrain.ACTION_NO_DATA
    assert retrain_calls == []


def test_retrain_skips_when_there_is_no_drift(train_frame, retrain_calls):
    seed_predictions(os.environ["PREDICTION_DB"], make_prediction_rows(train_frame))

    assert retrain.main() == retrain.ACTION_NO_DRIFT
    assert retrain_calls == []


def test_retrain_waits_for_the_auto_retrain_flag(train_frame, retrain_calls):
    seed_predictions(
        os.environ["PREDICTION_DB"],
        make_prediction_rows(train_frame, income=SHIFTED_INCOME),
    )

    assert retrain.main() == retrain.ACTION_DRIFT_LOCKED
    assert retrain_calls == []


def test_retrain_retrains_after_drift_when_enabled(train_frame, monkeypatch, retrain_calls):
    seed_predictions(
        os.environ["PREDICTION_DB"],
        make_prediction_rows(train_frame, income=SHIFTED_INCOME),
    )
    monkeypatch.setenv("AUTO_RETRAIN", "1")

    assert retrain.main() == retrain.ACTION_RETRAINED
    assert retrain_calls == ["train", "evaluate"]
