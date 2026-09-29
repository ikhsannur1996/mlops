"""Training pipeline: holdout metrics, logged params and registry versions."""

import mlflow
import pytest

from src.train import EXPERIMENT, REGISTERED_MODEL, main

MIN_ROC_AUC = 0.6
MIN_PR_AUC = 0.15


@pytest.fixture
def trained():
    """Train once against the isolated backend and return the holdout metrics."""
    return main()


def test_training_returns_bounded_metrics(trained):
    assert set(trained) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    for name, value in trained.items():
        assert 0.0 <= value <= 1.0, name


def test_model_beats_random_on_the_holdout(trained):
    """Quality gate: a scorecard that cannot rank risk is not shippable."""
    assert trained["roc_auc"] > MIN_ROC_AUC
    assert trained["pr_auc"] > MIN_PR_AUC


def test_training_logs_holdout_metadata(trained):
    experiment = mlflow.MlflowClient().get_experiment_by_name(EXPERIMENT)
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])

    assert len(runs) == 1

    params = runs[0].data.params
    assert params["model"] == "RandomForestClassifier"
    assert params["n_estimators"] == "150"
    assert params["max_depth"] == "6"
    assert params["train_dataset"] == "train.csv"
    assert params["holdout_dataset"] == "test.csv"
    assert int(params["train_rows"]) >= 1000
    assert int(params["holdout_rows"]) >= 250
    assert set(runs[0].data.metrics) >= set(trained)
    assert 0.0 <= runs[0].data.metrics["best_f1"] <= 1.0


def test_best_f1_sits_below_the_naive_cutoff(trained):
    """With a realistic bad rate the F1-optimal cut-off is well below 0.5."""
    experiment = mlflow.MlflowClient().get_experiment_by_name(EXPERIMENT)
    run = mlflow.MlflowClient().search_runs([experiment.experiment_id])[0]

    assert float(run.data.params["decision_threshold"]) < 0.5


def test_training_registers_a_new_model_version(trained):
    versions = mlflow.MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL}'")

    assert len(versions) == 1
    assert versions[0].name == REGISTERED_MODEL
    assert str(versions[0].version) == "1"
