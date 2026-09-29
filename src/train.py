"""Train the credit-default model and register it in the MLflow Model Registry."""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(BASE, "data", "train.csv")

FEATURES = ["age", "income", "loan_amount", "tenure"]
TARGET = "default"
EXPERIMENT = "credit-default"
REGISTERED_MODEL = "credit-default"
MODEL_PARAMS = {"n_estimators": 150, "max_depth": 6, "random_state": 42}
TEST_SIZE = 0.25
RANDOM_STATE = 42


def main():
    """Fit the model, log params/metrics/artifacts and register a new version."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)

    data_path = os.getenv("TRAIN_DATA", DEFAULT_DATA)
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    model = RandomForestClassifier(**MODEL_PARAMS)

    with mlflow.start_run(run_name="training"):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob),
        }

        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "feature_count": len(FEATURES),
                "dataset": os.path.basename(data_path),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                **MODEL_PARAMS,
            }
        )
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="credit_default_model",
            registered_model_name=REGISTERED_MODEL,
            # The MLflow 3 default ("skops") refuses the sklearn.tree._tree.Tree
            # type that every RandomForest model contains, so log with cloudpickle.
            serialization_format="cloudpickle",
        )

        print("Training metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    main()

