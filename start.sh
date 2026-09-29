#!/bin/bash
set -e

mlflow server --host 0.0.0.0 --port 5001   --backend-store-uri sqlite:///app/mlflow.db   --default-artifact-root /app/mlruns > /app/reports/mlflow.log 2>&1 &

sleep 8

python src/pipeline.py

uvicorn app.report:app --host 0.0.0.0 --port 8001 &

uvicorn app.main:app --host 0.0.0.0 --port 8000
