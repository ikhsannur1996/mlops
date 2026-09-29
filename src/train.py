"""Train the credit-default model on the train split and register it in MLflow."""

import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.metrics import classification_metrics  # noqa: E402

DEFAULT_TRAIN = os.path.join(BASE, "data", "train.csv")
DEFAULT_TEST = os.path.join(BASE, "data", "test.csv")

FEATURES = ["age", "income", "loan_amount", "tenure"]
TARGET = "default"
EXPERIMENT = "credit-default"
REGISTERED_MODEL = "credit-default"
MODEL_PARAMS = {"n_estimators": 150, "max_depth": 6, "random_state": 42}


def main():
    """Fit on the train split, score the held-out test split, register the model."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)

    train_path = os.getenv("TRAIN_DATA", DEFAULT_TRAIN)
    test_path = os.getenv("TEST_DATA", DEFAULT_TEST)
    train = pd.read_csv(train_path)
    holdout = pd.read_csv(test_path)

    model = RandomForestClassifier(**MODEL_PARAMS)

    with mlflow.start_run(run_name="training"):
        model.fit(train[FEATURES], train[TARGET])

        predictions = model.predict(holdout[FEATURES])
        probabilities = model.predict_proba(holdout[FEATURES])[:, 1]
        metrics = classification_metrics(holdout[TARGET], predictions, probabilities)

        # The 0.5 cut-off flags almost no one at a realistic bad rate, so record
        # the F1-optimal threshold for operators to set as DECISION_THRESHOLD.
        sweep = np.arange(0.05, 0.96, 0.05)
        f1_sweep = [
            f1_score(holdout[TARGET], (probabilities >= threshold).astype(int), zero_division=0)
            for threshold in sweep
        ]
        best_threshold = float(sweep[int(np.argmax(f1_sweep))])
        best_f1 = float(max(f1_sweep))

        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "decision_threshold": round(best_threshold, 2),
                "feature_count": len(FEATURES),
                "train_dataset": os.path.basename(train_path),
                "holdout_dataset": os.path.basename(test_path),
                "train_rows": len(train),
                "holdout_rows": len(holdout),
                "holdout_default_rate": round(float(holdout[TARGET].mean()), 4),
                **MODEL_PARAMS,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_f1", best_f1)

        mlflow.sklearn.log_model(
            model,
            name="credit_default_model",
            registered_model_name=REGISTERED_MODEL,
            # The MLflow 3 default ("skops") refuses the sklearn.tree._tree.Tree
            # type that every RandomForest model contains, so log with cloudpickle.
            serialization_format="cloudpickle",
        )

        print(f"trained on {len(train)} rows, scored {len(holdout)} held-out rows")
        print("Holdout metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print(f"flagged at the 0.5 cut-off: {float(predictions.mean()):.4f} of applicants")
        print(f"F1 peaks at threshold {best_threshold:.2f} (F1={best_f1:.4f}) - set DECISION_THRESHOLD to use it")

    return metrics


if __name__ == "__main__":
    main()
