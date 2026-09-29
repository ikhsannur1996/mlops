import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

FEATURES = ["age", "monthly_income", "loan_amount", "tenure_months"]
TARGET = "credit_status"

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("MLOps-Education")

np.random.seed(42)
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# -------------------------------------------------
# 1. Synthetic Indonesian-style credit dataset
# -------------------------------------------------
n = 3000
age = np.random.randint(21, 61, n)
income = np.random.randint(4_000_000, 25_000_000, n)
loan = np.random.randint(10_000_000, 180_000_000, n)
tenure = np.random.choice([12, 24, 36, 48, 60], n)

risk = (
    loan / income
    + (age < 25) * 0.5
    + (age > 55) * 0.2
    + (tenure >= 48) * 0.3
    + np.random.normal(0, 0.35, n)
)

status = np.where(risk > 8.0, "Risk", "Good")

df = pd.DataFrame({
    "age": age,
    "monthly_income": income,
    "loan_amount": loan,
    "tenure_months": tenure,
    "credit_status": status
})

df.to_csv("data/train.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET],
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET]
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

# -------------------------------------------------
# 2. Train + evaluate 3 models
# -------------------------------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, list(model.classes_).index("Risk")]

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred, pos_label="Risk"),
            "roc_auc": roc_auc_score((y_test == "Risk").astype(int), prob)
        }

        mlflow.log_metrics(metrics)
        mlflow.log_param("model", name)
        mlflow.sklearn.log_model(
            model, name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        results[name] = metrics

# Champion = highest ROC-AUC
ranking = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
champion = ranking[0][0]
challenger = ranking[1][0]
loser = ranking[2][0]

with open("reports/model_roles.txt", "w") as f:
    f.write(f"Champion: {champion}\n")
    f.write(f"Challenger: {challenger}\n")
    f.write(f"Loser: {loser}\n")

# -------------------------------------------------
# 3. Simple A/B testing
# -------------------------------------------------
champion_model = models[champion]
challenger_model = models[challenger]

traffic = df.sample(1000, random_state=7).reset_index(drop=True)
a = traffic.iloc[::2]
b = traffic.iloc[1::2]

pa = champion_model.predict_proba(a[FEATURES])[:, list(champion_model.classes_).index("Risk")]
pb = challenger_model.predict_proba(b[FEATURES])[:, list(challenger_model.classes_).index("Risk")]

with mlflow.start_run(run_name="A-B-Test"):
    mlflow.log_metrics({
        "champion_requests": len(a),
        "challenger_requests": len(b),
        "champion_avg_risk": float(pa.mean()),
        "challenger_avg_risk": float(pb.mean())
    })

# -------------------------------------------------
# 4. Simulate production drift
# -------------------------------------------------
production = df.sample(1000, random_state=9).copy()
production["monthly_income"] *= 1.4
production["loan_amount"] *= 1.3
production["age"] += 5

production.to_csv("data/production.csv", index=False)

report = Report(metrics=[
    DataQualityPreset(),
    DataDriftPreset()
])

report.run(
    reference_data=df[FEATURES],
    current_data=production[FEATURES]
)

report.save_html("reports/evidently.html")

# -------------------------------------------------
# 5. Simple PSI drift check
# -------------------------------------------------
def psi(ref, cur, bins=10):
    cuts = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0
    r = np.histogram(ref, bins=cuts)[0] / len(ref)
    c = np.histogram(cur, bins=cuts)[0] / len(cur)
    r = np.clip(r, 1e-6, None)
    c = np.clip(c, 1e-6, None)
    return float(np.sum((c-r) * np.log(c/r)))

psi_values = {
    f: psi(df[f].values, production[f].values)
    for f in FEATURES
}

max_drift = max(psi_values.values())
DRIFT_THRESHOLD = 0.20

# -------------------------------------------------
# 6. Automatic retraining if drift detected
# -------------------------------------------------
if max_drift > DRIFT_THRESHOLD:
    retrain_model = models[champion]
    retrain_model.fit(df[FEATURES], df[TARGET])

    with mlflow.start_run(run_name="Automatic-Retraining"):
        mlflow.log_param("trigger", "data_drift")
        mlflow.log_param("previous_champion", champion)
        mlflow.log_metric("max_psi", max_drift)
        mlflow.sklearn.log_model(
            retrain_model, name="retrained_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

    retrained = True
else:
    retrained = False

with mlflow.start_run(run_name="Monitoring"):
    for feature, value in psi_values.items():
        mlflow.log_metric(f"psi_{feature}", value)
    mlflow.log_metric("max_psi", max_drift)
    mlflow.set_tag("drift_status", "DRIFT" if max_drift > DRIFT_THRESHOLD else "OK")
    mlflow.set_tag("automatic_retraining", str(retrained))

print("\n=== MLOps Education Demo ===")
print("Champion  :", champion)
print("Challenger:", challenger)
print("Loser     :", loser)
print("Max PSI   :", round(max_drift, 3))
print("Drift     :", max_drift > DRIFT_THRESHOLD)
print("Retrained :", retrained)
print("============================")
