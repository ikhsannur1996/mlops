"""Detailed model evaluation: metrics, curves and threshold analysis in MLflow."""

import os

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:  # scikit-learn >= 1.9 exposes the calibration curve in sklearn.calibration
    from sklearn.calibration import calibration_curve
except ImportError:  # pragma: no cover - scikit-learn < 1.9
    from sklearn.metrics import calibration_curve

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected before pyplot)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(BASE, "data", "train.csv")

FEATURES = ["age", "income", "loan_amount", "tenure"]
TARGET = "default"
EXPERIMENT = "credit-default-evaluation"
MODEL_PARAMS = {"n_estimators": 150, "max_depth": 6, "random_state": 42}
TEST_SIZE = 0.25
RANDOM_STATE = 42


def main():
    """Compute evaluation metrics and log every report as an MLflow artifact."""
    reports_dir = os.path.join(os.getenv("REPORTS_DIR", os.path.join(BASE, "reports")), "evaluation")
    os.makedirs(reports_dir, exist_ok=True)

    data_path = os.getenv("TRAIN_DATA", DEFAULT_DATA)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    model = RandomForestClassifier(**MODEL_PARAMS)
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

    with mlflow.start_run(run_name="detailed-evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "dataset": os.path.basename(data_path),
                "test_rows": len(X_test),
                **MODEL_PARAMS,
            }
        )

        # 1. Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        path = os.path.join(reports_dir, "confusion_matrix.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        # 2. ROC curve
        fpr, tpr, _ = roc_curve(y_test, prob)
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
        precision, recall, _ = precision_recall_curve(y_test, prob)
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
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=5)
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
        ax.hist(prob, bins=10)
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
            predicted = (prob >= threshold).astype(int)
            rows.append(
                {
                    "threshold": round(float(threshold), 2),
                    "precision": precision_score(y_test, predicted, zero_division=0),
                    "recall": recall_score(y_test, predicted, zero_division=0),
                    "f1": f1_score(y_test, predicted, zero_division=0),
                }
            )
        threshold_df = pd.DataFrame(rows)
        threshold_csv = os.path.join(reports_dir, "threshold_analysis.csv")
        threshold_df.to_csv(threshold_csv, index=False)
        mlflow.log_artifact(threshold_csv, "evaluation")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
        ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
        ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
        ax.set_title("Threshold Analysis")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(reports_dir, "threshold_analysis.png")
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, "evaluation")

        print("Detailed evaluation logged to MLflow.")

    return metrics


if __name__ == "__main__":
    main()
