"""Production monitoring: prediction volume, feature drift (PSI) and quality metrics."""

import os
import sqlite3

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected before pyplot)
import mlflow  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(BASE, "data", "train.csv")
DEFAULT_DB = os.path.join(BASE, "predictions.db")

FEATURES = ["age", "income", "loan_amount", "tenure"]
EXPERIMENT = "production-monitoring"
MIN_PREDICTIONS = 10
DEFAULT_DRIFT_THRESHOLD = 0.20


def psi(expected, actual, bins=10):
    """Population Stability Index between the training and production distribution."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    """Compare production predictions against training data and log the drift report.

    Returns ``None`` when there is not enough production data yet, otherwise a
    report dict with the per-feature PSI, the maximum PSI and the drift status.
    """
    reports_dir = os.path.join(os.getenv("REPORTS_DIR", os.path.join(BASE, "reports")), "monitoring")
    os.makedirs(reports_dir, exist_ok=True)

    data_path = os.getenv("TRAIN_DATA", DEFAULT_DATA)
    db_path = os.getenv("PREDICTION_DB", DEFAULT_DB)
    threshold = float(os.getenv("DRIFT_THRESHOLD", DEFAULT_DRIFT_THRESHOLD))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)

    if not os.path.exists(db_path):
        print("No predictions database found. Make predictions first.")
        return None

    train = pd.read_csv(data_path)
    with sqlite3.connect(db_path) as conn:
        prod = pd.read_sql_query("SELECT * FROM predictions", conn)

    if len(prod) < MIN_PREDICTIONS:
        print(f"Need at least {MIN_PREDICTIONS} predictions for monitoring.")
        return None

    drifts = {f"psi_{feature}": psi(train[feature], prod[feature]) for feature in FEATURES}
    max_psi = max(drifts.values())
    status = "DRIFT" if max_psi > threshold else "OK"

    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.log_metrics(
            {
                **drifts,
                "max_psi": max_psi,
                "prediction_count": len(prod),
                "default_prediction_rate": float(prod["prediction"].mean()),
                "average_probability": float(prod["probability"].mean()),
            }
        )
        mlflow.log_params({"drift_threshold": threshold, "feature_count": len(FEATURES)})

        # Drift chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(list(drifts.keys()), list(drifts.values()))
        ax.axhline(threshold, linestyle="--", label="PSI threshold")
        ax.set_title("Feature Drift - PSI")
        ax.set_ylabel("PSI")
        ax.legend()
        plt.xticks(rotation=30)
        fig.tight_layout()
        path = os.path.join(reports_dir, "feature_drift.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "monitoring")

        # Prediction distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(prod["probability"], bins=10)
        ax.set_title("Production Prediction Distribution")
        ax.set_xlabel("Default Probability")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = os.path.join(reports_dir, "production_prediction_distribution.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "monitoring")

        mlflow.set_tag("monitoring_status", status)

    print("Monitoring complete.")
    print(drifts)
    print("STATUS:", status)

    return {
        "status": status,
        "max_psi": max_psi,
        "drift": drifts,
        "prediction_count": len(prod),
        "threshold": threshold,
    }


if __name__ == "__main__":
    main()
