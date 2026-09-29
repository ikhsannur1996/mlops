#!/usr/bin/env bash
set -e
mkdir -p mlruns

# MLflow 3 validates the Host header to block DNS rebinding attacks. By
# default it only accepts localhost and private-network hosts, so browsing
# the UI through a public IP, domain, .local name or VS Code forwarded port
# fails with "Invalid Host header - possible DNS rebinding attack detected".
# List the host(s) you actually browse with, comma separated, or use "*"
# to accept any Host header (demo only, disables the protection):
#   MLFLOW_SERVER_ALLOWED_HOSTS="203.0.113.10,mlflow.example.com" make mlflow
if [ -n "${MLFLOW_SERVER_ALLOWED_HOSTS:-}" ]; then
  mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --allowed-hosts "$MLFLOW_SERVER_ALLOWED_HOSTS"
else
  mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
fi
