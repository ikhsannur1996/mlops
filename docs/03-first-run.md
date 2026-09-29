# 03 - First Run

## 1. Start MLflow

With Docker:

```bash
docker compose up -d mlflow
```

Verify:

```bash
docker compose logs mlflow
```

Open:

```text
http://PUBLIC_IP:5000
```

## 2. Train the model

If Python dependencies are installed on the host:

```bash
python src/train.py
python src/evaluate.py
```

If you want to use the application container:

```bash
docker compose exec api python src/train.py
docker compose exec api python src/evaluate.py
```

## 3. Check MLflow

Open the MLflow UI.

You should see experiments such as:

```text
credit-default
credit-default-evaluation
```

The run contains:

- parameters
- metrics
- model
- evaluation artifacts

## 4. Check Model Registry

The training script registers:

```text
credit-default
```

The registered model version is used by FastAPI.

## 5. Start API

```bash
docker compose up -d api
```

Open:

```text
http://PUBLIC_IP:8000/docs
```

## 6. Test prediction

From Swagger:

1. Open `/predict`
2. Click `Try it out`
3. Enter:

```json
{
  "age": 35,
  "income": 10000000,
  "loan_amount": 50000000,
  "tenure": 24
}
```

4. Execute.

The API stores the prediction in:

```text
predictions.db
```

## 7. Generate enough predictions

Monitoring requires at least 10 predictions in this demo.

Repeat the request with different values.

## 8. Run monitoring

```bash
docker compose exec api python src/monitor.py
```

The monitoring run is visible in MLflow.

It records:

- PSI per feature
- maximum PSI
- prediction count
- default prediction rate
- average probability
- monitoring status

## 9. Run retraining

Dry run (monitoring only, no retraining):

```bash
docker compose exec api python src/retrain.py
```

Automatic retraining mode:

```bash
docker compose exec -e AUTO_RETRAIN=1 api python src/retrain.py
```

The command performs monitoring first and retrains only when the maximum feature
PSI exceeds `DRIFT_THRESHOLD` (default `0.20`).

It prints the action it took:

```text
no-data        not enough production predictions yet
no-drift       drift below the threshold, nothing to do
drift-locked   drift detected but AUTO_RETRAIN is not enabled
retrained      training and detailed evaluation were re-run
```

Set `DRIFT_THRESHOLD` in the container environment to change the sensitivity:

```bash
docker compose exec -e AUTO_RETRAIN=1 -e DRIFT_THRESHOLD=0.10 api python src/retrain.py
```
