"""Detailed evaluation: the registered model scored on the holdout split."""

import os
from pathlib import Path

import mlflow
import pytest

from src.evaluate import EXPERIMENT
from src.evaluate import main as evaluate_main
from src.train import main as train_main

EXPECTED_REPORTS = [
    "confusion_matrix.png",
    "roc_curve.png",
    "precision_recall_curve.png",
    "calibration_curve.png",
    "feature_importance.png",
    "prediction_distribution.png",
    "threshold_analysis.png",
    "threshold_analysis.csv",
]


def test_evaluation_scores_the_registered_model_and_logs_every_report():
    training_metrics = train_main()
    metrics = evaluate_main()
    reports_dir = Path(os.environ["REPORTS_DIR"]) / "evaluation"

    # Scoring the registered version reproduces the training holdout metrics,
    # which proves evaluation loads the served model instead of refitting one.
    assert metrics == pytest.approx(training_metrics)

    for name in EXPECTED_REPORTS:
        assert (reports_dir / name).is_file(), name

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 1
    assert runs[0].data.params["model_version"] == "1"
    assert runs[0].data.params["dataset"] == "test.csv"
    assert set(runs[0].data.metrics) >= set(metrics)


def test_threshold_analysis_sweeps_the_probability_range():
    train_main()
    evaluate_main()

    thresholds = Path(os.environ["REPORTS_DIR"]) / "evaluation" / "threshold_analysis.csv"
    rows = thresholds.read_text().strip().splitlines()

    assert rows[0] == "threshold,precision,recall,f1"
    assert len(rows) == 10
