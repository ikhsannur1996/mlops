# End-to-End MLOps with MLflow

[![CI](https://github.com/ikhsannur1996/mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/ikhsannur1996/mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracking%20%2B%20registry-0194E2)

A simple but complete MLOps portfolio project for credit-default prediction: a
realistic generated dataset with a held-out test split, training, evaluation,
MLflow tracking and Model Registry, FastAPI serving, simulated production
traffic, production monitoring, drift detection and drift-gated retraining —
with linted, tested GitHub Actions CI.

## Lifecycle

Data generation
→ Training
→ Evaluation
→ MLflow Tracking
→ Model Registry
→ FastAPI Serving
→ Traffic Simulation
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

The suite covers the data generator and data-quality gates, training (holdout
metrics, params, decision threshold and Model Registry versions), evaluation
artifacts, PSI-based drift monitoring, the FastAPI serving contract, the traffic
simulator (payloads, drift modes, reports and a live end-to-end batch) and the
retraining decision logic. Every test runs against a throwaway SQLite MLflow
backend, so no running server is required.

GitHub Actions runs the same gates plus a Docker build and compose validation on
every push and pull request — see `docs/09-cicd.md`.

## Data

`data/train.csv` (8000 rows) and `data/test.csv` (2000 rows) are produced by
`src/generate_data.py`: log-normal incomes, affordability-based loan sizing and a
latent risk model calibrated to a 12% default rate, split into two disjoint,
duplicate-free populations.

```bash
make data     # regenerate both splits — same seed, identical files
```

`train.csv` fits the model; `test.csv` is the holdout that evaluation scores and
that the simulator samples from. See `docs/10-data.md`.

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

## Simulate production traffic

Instead of clicking through Swagger, generate a whole batch of realistic
predictions:

```bash
make simulate                      # 200 requests sampled from data/test.csv
make simulate ARGS="--count 500"   # heavier batch
make simulate-drift                # shifted population -> DRIFT in monitoring
```

The simulator validates every response against the serving contract, reports
latency and throughput, and writes `reports/simulation/simulation-<drift>.json`.
See `docs/11-api-simulation.md`.

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
| `MLFLOW_SERVER_ALLOWED_HOSTS` | *(unset → localhost + private IPs)* | MLflow UI — Host headers allowed past the DNS-rebinding guard |
| `PREDICTION_DB` | `predictions.db` | API, monitor |
| `TRAIN_DATA` | `data/train.csv` | train, monitor |
| `TEST_DATA` | `data/test.csv` | train, evaluate, simulate |
| `REPORTS_DIR` | `reports` | evaluate, monitor |
| `DECISION_THRESHOLD` | `0.5` | API — cut-off applied to `probability` |
| `DRIFT_THRESHOLD` | `0.20` | monitor, retrain |
| `AUTO_RETRAIN` | `0` | retrain |
| `N_TRAIN`, `N_TEST`, `RANDOM_SEED`, `TARGET_DEFAULT_RATE` | `8000`, `2000`, `42`, `0.12` | generate_data |
| `API_BASE_URL`, `SIMULATION_REQUESTS`, `SIMULATION_WORKERS`, `SIMULATION_DRIFT` | `http://localhost:8000`, `200`, `4`, `none` | simulate |

## Project layout

```text
app/main.py          FastAPI service (predict + prediction logging)
src/generate_data.py realistic train/test data generator
src/train.py         training, holdout metrics and Model Registry registration
src/evaluate.py      evaluation of the registered model, artifacts to MLflow
src/monitor.py       production monitoring and PSI drift detection
src/retrain.py       drift-gated retraining orchestration
src/simulate.py      production traffic simulator for /predict
src/registry.py      load the newest registered model version
src/metrics.py       shared metric computation
scripts/             MLflow bootstrap, training and simulation helpers
tests/               pytest suite used by CI
docs/                step-by-step guides (01-11)
.github/workflows/   GitHub Actions pipeline
```

## Make targets

```bash
make help       # list every target
make up         # docker compose up -d --build
make up-public  # same, but MLFLOW_SERVER_ALLOWED_HOSTS=* (public demo)
make mlflow-public  # local MLflow accepting any Host header
make data       # regenerate data/train.csv and data/test.csv
make train      # python src/train.py
make evaluate   # python src/evaluate.py
make monitor    # python src/monitor.py
make retrain    # AUTO_RETRAIN=1 python src/retrain.py
make simulate   # python src/simulate.py (ARGS="--count 500")
make simulate-drift  # python src/simulate.py --drift all
make test       # pytest -q
make lint       # ruff check .
make build      # docker build
make clean      # drop caches and local artifacts
```

## Architecture

```text
                  ┌───────────────┐
                  │ generate_data │
                  │  train.csv    │
                  │  test.csv     │
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
   ┌────────────┐ ┌───────────────┐
   │  simulate  │→│    FastAPI    │
   └────────────┘ └───────┬───────┘
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
