PYTHON ?= python3
IMAGE ?= credit-default-api:local

.DEFAULT_GOAL := help

.PHONY: help install install-dev data mlflow mlflow-public up up-public down logs train evaluate monitor retrain simulate simulate-drift test lint build clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install runtime and development dependencies
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

data: ## Generate the realistic train and test datasets
	$(PYTHON) src/generate_data.py

mlflow: ## Start a local MLflow server on port 5000
	bash scripts/start-mlflow.sh

mlflow-public: ## Start local MLflow accepting any Host header (public demo)
	MLFLOW_SERVER_ALLOWED_HOSTS='*' bash scripts/start-mlflow.sh

up: ## Start MLflow and the API with Docker Compose
	docker compose up -d --build

up-public: ## Start the stack accepting any Host header (public demo)
	MLFLOW_SERVER_ALLOWED_HOSTS='*' docker compose up -d --build

down: ## Stop the Docker Compose stack
	docker compose down

logs: ## Follow Docker Compose logs
	docker compose logs -f

train: ## Train the model and register it in MLflow
	$(PYTHON) src/train.py

evaluate: ## Run the detailed evaluation and log artifacts to MLflow
	$(PYTHON) src/evaluate.py

monitor: ## Run production monitoring (drift, volume, quality)
	$(PYTHON) src/monitor.py

retrain: ## Monitor drift and retrain when the threshold is exceeded
	AUTO_RETRAIN=1 $(PYTHON) src/retrain.py

simulate: ## Send simulated production traffic to the API (ARGS="--count 500")
	$(PYTHON) src/simulate.py $(ARGS)

simulate-drift: ## Send drifted traffic so monitoring reports drift
	$(PYTHON) src/simulate.py --drift all $(ARGS)

test: ## Run the test suite
	$(PYTHON) -m pytest -q

lint: ## Lint the codebase with ruff
	$(PYTHON) -m ruff check .

build: ## Build the API Docker image
	docker build -t $(IMAGE) .

clean: ## Remove caches and local run artifacts
	rm -rf .pytest_cache .ruff_cache mlruns artifacts reports/evaluation reports/monitoring reports/simulation
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
