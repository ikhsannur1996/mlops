"""Detailed evaluation: metrics, report artifacts and the MLflow run."""

import os
from pathlib import Path

import mlflow

from src.evaluate import EXPERIMENT, main

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


def test_evaluation_produces_metrics_reports_and_a_run():
    metrics = main()
    reports_dir = Path(os.environ["REPORTS_DIR"]) / "evaluation"

    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"} <= set(metrics)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, name

    for name in EXPECTED_REPORTS:
        assert (reports_dir / name).is_file(), name

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 1
    assert set(runs[0].data.metrics) >= set(metrics)
