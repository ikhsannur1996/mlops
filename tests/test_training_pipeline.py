"""Training pipeline: metrics, logged params and Model Registry versioning."""

import mlflow
import pytest

from src.train import EXPERIMENT, REGISTERED_MODEL, main


@pytest.fixture
def trained():
    """Train once against the isolated backend and return the metrics."""
    return main()


def test_training_returns_bounded_metrics(trained):
    assert set(trained) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    for name, value in trained.items():
        assert 0.0 <= value <= 1.0, name


def test_training_logs_params_and_metrics(trained):
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 1

    params = runs[0].data.params
    assert params["model"] == "RandomForestClassifier"
    assert params["n_estimators"] == "150"
    assert params["max_depth"] == "6"
    assert params["dataset"] == "train.csv"
    assert set(runs[0].data.metrics) >= set(trained)


def test_training_registers_a_new_model_version(trained):
    versions = mlflow.MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL}'")

    assert len(versions) == 1
    assert versions[0].name == REGISTERED_MODEL
    assert str(versions[0].version) == "1"
