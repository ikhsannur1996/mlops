# Public VM Quick Start

## 1. Install
Read `docs/01-installation.md`.

## 2. Configure
Read `docs/02-configuration.md`.

## 3. Start
```bash
docker compose up -d --build
```

## 4. Train
```bash
docker compose exec api python src/train.py
docker compose exec api python src/evaluate.py
```

## 5. Browser
```text
http://PUBLIC_IP:5000
http://PUBLIC_IP:8000/docs
```

## 6. Generate predictions
Use Swagger `/predict`.

## 7. Monitor
```bash
docker compose exec api python src/monitor.py
```

## 8. Retrain
```bash
docker compose exec -e AUTO_RETRAIN=1 api python src/retrain.py
```

Monitoring runs first. Training is only re-run when the maximum feature PSI
exceeds `DRIFT_THRESHOLD` (default `0.20`) and `AUTO_RETRAIN=1` is set.

## 9. Run the quality gates
```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest -q
```

The same gates run in GitHub Actions on every push and pull request, see
`docs/09-cicd.md`.

For detailed configuration, firewall, public access, MLflow, monitoring, and troubleshooting, read the files in `docs/`.
