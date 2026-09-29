# 09 - CI/CD and Testing

The repository ships with a GitHub Actions pipeline and a pytest suite so every
change is linted, tested and buildable before it reaches `main`.

## Pipeline

`.github/workflows/ci.yml` runs on every push to `main`, every pull request and
on manual dispatch.

### Job 1 - Lint and test

| Step | Command |
| --- | --- |
| Install | `pip install -r requirements.txt -r requirements-dev.txt` |
| Lint | `ruff check .` |
| Test | `pytest -q` |
| Publish | `reports/` uploaded as a build artifact |

The tests never need a running MLflow server. `MLFLOW_TRACKING_URI` is set to a
throwaway SQLite backend (`sqlite:///mlflow-ci.db`) so training, evaluation and
monitoring runs are written to a database that disappears with the runner.

### Job 2 - Docker build

1. `docker compose config --quiet` validates `docker-compose.yml`.
2. `docker build -t credit-default-api:ci .` builds the API image.
3. A smoke test imports the FastAPI application inside the image.

## Test suite

| File | Coverage |
| --- | --- |
| `tests/test_data_quality.py` | Schema, dtypes, ranges, target balance, duplicates |
| `tests/test_training_pipeline.py` | `src/train.py` metrics, params, registered model version |
| `tests/test_evaluation.py` | `src/evaluate.py` metrics plus all evaluation artifacts |
| `tests/test_monitoring.py` | PSI helper behaviour and the monitoring run |
| `tests/test_api.py` | `/health`, `/predict`, prediction persistence |

Every test runs with `MLFLOW_TRACKING_URI`, `PREDICTION_DB` and `REPORTS_DIR`
pointed at a temporary directory, so the suite never mutates local state.

## Run the pipeline locally

```bash
make install-dev
make lint
make test
make build
```

## Deployment flow

```text
push / pull request
        ↓
   ruff check .
        ↓
   pytest (train + evaluate + monitor + api)
        ↓
   docker compose config
        ↓
   docker build
        ↓
   merge to main
        ↓
   docker compose up -d --build on the VM
```
