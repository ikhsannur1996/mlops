"""Evaluate the registered model on the held-out test split and log the reports."""

import os
import sys

import matplotlib
import mlflow
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.metrics import classification_metrics  # noqa: E402
from src.registry import load_latest_model  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected before pyplot)
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

try:  # scikit-learn >= 1.9 exposes the calibration curve in sklearn.calibration
    from sklearn.calibration import calibration_curve
except ImportError:  # pragma: no cover - scikit-learn < 1.9
    from sklearn.metrics import calibration_curve

DEFAULT_TEST = os.path.join(BASE, "data", "test.csv")

FEATURES = ["age", "income", "loan_amount", "tenure"]
TARGET = "default"
EXPERIMENT = "credit-default-evaluation"
REGISTERED_MODEL = "credit-default"


def main():
    """Score the registered model on the holdout split and log every report."""
    reports_dir = os.path.join(os.getenv("REPORTS_DIR", os.path.join(BASE, "reports")), "evaluation")
    os.makedirs(reports_dir, exist_ok=True)

    test_path = os.getenv("TEST_DATA", DEFAULT_TEST)
    holdout = pd.read_csv(test_path)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)

    # Evaluate the model that is actually serving, not a freshly refit one.
    model, model_version = load_latest_model(REGISTERED_MODEL)
    predictions = model.predict(holdout[FEATURES])
    probabilities = model.predict_proba(holdout[FEATURES])[:, 1]
    y_true = holdout[TARGET]

    metrics = classification_metrics(y_true, predictions, probabilities)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "model_version": model_version,
                "dataset": os.path.basename(test_path),
                "rows": len(holdout),
                "default_rate": round(float(y_true.mean()), 4),
            }
        )

        # 1. Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_true, predictions)).plot(ax=ax)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        path = os.path.join(reports_dir, "confusion_matrix.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        # 2. ROC curve
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(reports_dir, "roc_curve.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        # 3. Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision, label=f"PR-AUC={metrics['pr_auc']:.3f}")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(reports_dir, "precision_recall_curve.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        # 4. Calibration curve
        frac_pos, mean_pred = calibration_curve(y_true, probabilities, n_bins=5)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mean_pred, frac_pos, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        ax.set_title("Calibration Curve")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction Positive")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(reports_dir, "calibration_curve.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        # 5. Feature importance
        importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        importance.plot(kind="barh", ax=ax)
        ax.set_title("Feature Importance")
        fig.tight_layout()
        path = os.path.join(reports_dir, "feature_importance.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        # 6. Prediction distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(probabilities, bins=20)
        ax.set_title("Prediction Probability Distribution")
        ax.set_xlabel("Default Probability")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = os.path.join(reports_dir, "prediction_distribution.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        # 7. Threshold analysis
        rows = []
        for threshold in np.arange(0.1, 0.91, 0.1):
            flagged = (probabilities >= threshold).astype(int)
            rows.append(
                {
                    "threshold": round(float(threshold), 2),
                    "precision": precision_score(y_true, flagged, zero_division=0),
                    "recall": recall_score(y_true, flagged, zero_division=0),
                    "f1": f1_score(y_true, flagged, zero_division=0),
                }
            )
        threshold_frame = pd.DataFrame(rows)
        threshold_csv = os.path.join(reports_dir, "threshold_analysis.csv")
        threshold_frame.to_csv(threshold_csv, index=False)
        mlflow.log_artifact(threshold_csv, "evaluation")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(threshold_frame["threshold"], threshold_frame["precision"], marker="o", label="Precision")
        ax.plot(threshold_frame["threshold"], threshold_frame["recall"], marker="o", label="Recall")
        ax.plot(threshold_frame["threshold"], threshold_frame["f1"], marker="o", label="F1")
        ax.set_title("Threshold Analysis")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(reports_dir, "threshold_analysis.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        print(f"evaluated model version {model_version} on {len(holdout)} held-out rows")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    main()
