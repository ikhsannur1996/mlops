# End-to-End MLOps with MLflow

[![CI](https://github.com/ikhsannur1996/mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/ikhsannur1996/mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracking%20%2B%20registry-0194E2)

A simple but complete MLOps portfolio project for credit-default prediction: data,
training, evaluation, MLflow tracking and Model Registry, FastAPI serving,
production monitoring, drift detection and drift-gated retraining — with linted,
tested GitHub Actions CI.

## Lifecycle

Data
→ Training
→ Evaluation
→ MLflow Tracking
→ Model Registry
→ FastAPI Serving
→ Prediction Logging
→ Production Monitoring
→ Drift Detection
→ Retraining
→ CI/CD (lint, tests, Docker build)

## MLflow is the central ML platform

MLflow stores:

- training parameters
- model metrics
- evaluation metrics
- model artifacts
- confusion matrix
- ROC curve
- Precision-Recall curve
- calibration curve
- threshold analysis
- feature importance
- prediction distribution
- data-quality metrics
- PSI / drift metrics
- monitoring charts
- registered model versions

## Monitoring coverage

### Model evaluation
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Calibration Curve
- Threshold Analysis
- Feature Importance
- Prediction Distribution

### Production monitoring
- Prediction volume
- Default prediction rate
- Feature missing rate
- Feature drift using PSI
- Prediction drift
- Production performance when labels are available
- Performance trend

## Quality gates

```bash
make install-dev
make lint     # ruff check .
make test     # pytest -q
```

The suite covers data quality, training (metrics, params and Model Registry
versions), evaluation artifacts, PSI-based drift monitoring, the FastAPI serving
contract and the retraining decision logic. Every test runs against a throwaway
SQLite MLflow backend, so no running server is required.

GitHub Actions runs the same gates plus a Docker build and compose validation on
every push and pull request — see `docs/09-cicd.md`.

## Run with Docker

```bash
docker compose up --build
```

Open:
- MLflow: http://localhost:5000
- FastAPI docs: http://localhost:8000/docs

## Important first step

Run training after MLflow is available:

```bash
python src/train.py
```

Then make predictions:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":35,"income":10000000,"loan_amount":50000000,"tenure":24}'
```

Make at least 10 predictions before monitoring.

## Run evaluation manually

```bash
python src/evaluate.py
```

## Run monitoring

```bash
python src/monitor.py
```

## Retraining

```bash
make retrain        # or: AUTO_RETRAIN=1 python src/retrain.py
```

`src/retrain.py` monitors production first and retrains only when the maximum
feature PSI exceeds `DRIFT_THRESHOLD` (default `0.20`) **and** `AUTO_RETRAIN=1`
is set. It reports the action it took:

```text
no-data        not enough production predictions yet
no-drift       drift below the threshold, nothing to do
drift-locked   drift detected but AUTO_RETRAIN is not enabled
retrained      training and evaluation were re-run
```

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev          # runtime + test and lint dependencies
bash scripts/start-mlflow.sh
```

Without make:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

In another terminal:

```bash
python src/train.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Public VM

For a simple VM deployment, expose:
- TCP 5000 for MLflow
- TCP 8000 for FastAPI

For real production, use HTTPS, authentication, secrets, a reverse proxy, and private storage.

## Configuration

Every path and threshold is configurable through environment variables. Copy
`.env.example` to `.env` and adjust:

| Variable | Default | Used by |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | train, evaluate, monitor, API |
| `PREDICTION_DB` | `predictions.db` | API, monitor |
| `TRAIN_DATA` | `data/train.csv` | train, evaluate, monitor |
| `REPORTS_DIR` | `reports` | evaluate, monitor |
| `DRIFT_THRESHOLD` | `0.20` | monitor, retrain |
| `AUTO_RETRAIN` | `0` | retrain |

## Project layout

```text
app/main.py          FastAPI service (predict + prediction logging)
src/train.py         training, metrics and Model Registry registration
src/evaluate.py      detailed evaluation reports logged to MLflow
src/monitor.py       production monitoring and PSI drift detection
src/retrain.py       drift-gated retraining orchestration
scripts/             MLflow bootstrap and training helpers
tests/               pytest suite used by CI
docs/                step-by-step guides (01-09)
.github/workflows/   GitHub Actions pipeline
```

## Make targets

```bash
make help       # list every target
make up         # docker compose up -d --build
make train      # python src/train.py
make evaluate   # python src/evaluate.py
make monitor    # python src/monitor.py
make retrain    # AUTO_RETRAIN=1 python src/retrain.py
make test       # pytest -q
make lint       # ruff check .
make build      # docker build
make clean      # drop caches and local artifacts
```

## Architecture

```text
                  ┌───────────────┐
                  │   train.csv   │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │   train.py    │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │  evaluate.py  │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │    MLflow     │
                  │ Tracking      │
                  │ Evaluation    │
                  │ Registry      │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │    FastAPI    │
                  └───────┬───────┘
                          ↓
                    Predictions
                          ↓
                  ┌───────────────┐
                  │    SQLite     │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │   monitor.py  │
                  └───────┬───────┘
                          ↓
                    Drift / Quality
                          ↓
                  ┌───────────────┐
                  │  retrain.py   │
                  └───────┬───────┘
                          └────→ MLflow
```
