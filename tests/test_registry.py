"""Model Registry loading, shared by the API and the evaluation pipeline."""

import mlflow
import pytest

from src import registry
from src.train import REGISTERED_MODEL
from src.train import main as train_main


def test_loading_without_a_registered_model_fails_clearly():
    with pytest.raises(RuntimeError, match="Run training first"):
        registry.load_latest_model(REGISTERED_MODEL)


def test_latest_version_returns_the_newest_registration():
    train_main()
    train_main()

    model, version = registry.load_latest_model(REGISTERED_MODEL)

    assert version == "2"
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "feature_importances_")


def test_latest_version_reports_what_is_registered():
    train_main()
    version = registry.latest_version(mlflow.MlflowClient(), REGISTERED_MODEL)

    assert version.name == REGISTERED_MODEL
    assert str(version.version) == "1"
