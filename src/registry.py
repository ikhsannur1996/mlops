"""Model Registry helpers shared by the FastAPI service and the evaluation pipeline."""

import mlflow
import mlflow.sklearn


def latest_version(client, name):
    """Return the newest registered version of a model.

    Raises:
        RuntimeError: when the model has never been registered.
    """
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No registered model named '{name}'. Run training first.")
    return sorted(versions, key=lambda version: int(version.version), reverse=True)[0]


def load_latest_model(name):
    """Load the newest registered version of a model as a scikit-learn estimator.

    The estimator (rather than an MLflow pyfunc wrapper) is returned so callers
    keep predict_proba and feature_importances_.

    Returns:
        tuple: ``(estimator, version string)``
    """
    version = latest_version(mlflow.MlflowClient(), name)
    return mlflow.sklearn.load_model(version.source), str(version.version)
