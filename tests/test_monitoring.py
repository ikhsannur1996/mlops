"""Production monitoring: the PSI helper and the drift report."""

import os
from pathlib import Path

import pytest

from src import monitor
from tests.helpers import make_prediction_rows, seed_predictions


def test_psi_is_zero_for_matching_distributions():
    values = list(range(100))
    assert monitor.psi(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_shifted_distribution():
    expected = list(range(100))
    actual = [value + 100 for value in expected]
    assert monitor.psi(expected, actual) > monitor.DEFAULT_DRIFT_THRESHOLD


def test_psi_returns_zero_when_bins_cannot_be_formed():
    assert monitor.psi([5, 5, 5, 5], [1, 2, 3, 4]) == 0.0


def test_monitoring_reports_ok_when_there_is_no_drift(train_frame):
    seed_predictions(os.environ["PREDICTION_DB"], make_prediction_rows(train_frame))

    report = monitor.main()
    reports_dir = Path(os.environ["REPORTS_DIR"]) / "monitoring"

    assert report["status"] == "OK"
    assert report["prediction_count"] == len(train_frame)
    assert report["threshold"] == monitor.DEFAULT_DRIFT_THRESHOLD
    assert (reports_dir / "feature_drift.png").is_file()
    assert (reports_dir / "production_prediction_distribution.png").is_file()


def test_monitoring_detects_a_drifted_population(train_frame):
    seed_predictions(os.environ["PREDICTION_DB"], make_prediction_rows(train_frame, drift="all"))

    report = monitor.main()

    assert report["status"] == "DRIFT"
    assert report["max_psi"] > report["threshold"]


def test_monitoring_returns_none_without_a_prediction_log():
    assert monitor.main() is None


def test_monitoring_returns_none_with_too_few_predictions(train_frame):
    seed_predictions(os.environ["PREDICTION_DB"], make_prediction_rows(train_frame.head(3)))

    assert monitor.main() is None
