# 11 - API Simulation

`src/simulate.py` sends realistic production-like traffic to the FastAPI
`/predict` endpoint. It is the bridge between a freshly deployed model and the
monitoring/retraining loop: it fills `predictions.db` with a representative
sample of scored applicants, then measures what the service actually does under
load.

It uses only the standard library (`urllib` + `concurrent.futures`), so it runs
anywhere the project runs — no extra dependency to install.

## Quick start

```bash
# start the API first
uvicorn app.main:app --port 8000

make simulate                      # 200 requests (SIMULATION_REQUESTS)
make simulate ARGS="--count 500"   # heavier batch
make simulate-drift                # deliberately drifted traffic
```

Or directly:

```bash
python src/simulate.py --count 300 --workers 8 --base-url http://localhost:8000
```

## Options

| Flag | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `--base-url` | `API_BASE_URL` | `http://localhost:8000` | where the API lives |
| `--count` | `SIMULATION_REQUESTS` | `200` | number of requests |
| `--workers` | `SIMULATION_WORKERS` | `4` | concurrent threads |
| `--drift` | `SIMULATION_DRIFT` | `none` | `none`, `income`, `loan_amount`, `tenure`, `age`, `all` |
| `--data` | `TEST_DATA` | `data/test.csv` | population to sample from |
| `--seed` | `RANDOM_SEED` | `42` | makes sampling reproducible |
| `--timeout` | — | `10` | per-request timeout in seconds |
| `--report-dir` | `SIMULATION_REPORT_DIR` | `reports/simulation` | where the JSON report is written |
| `--skip-health-check` | — | off | send traffic even if `/health` fails |

If `/health` is unreachable or does not answer `200`, the script exits with code
`2` before sending anything — useful in pipelines where a stopped service should
fail loudly instead of producing a wall of connection errors. Note that `/health`
only proves the process is up; a missing registered model only shows up as
failed `/predict` calls.

## How requests are built

Applicants are sampled from `data/test.csv` (the holdout split, see
`docs/10-data.md`), so simulated traffic has the same shape as real customers:
log-normal income, affordability-based loan sizing, a ~12% default mix. Payloads
carry exactly the four fields `/predict` accepts:

```json
{"age": 35, "income": 10000000.0, "loan_amount": 50000000.0, "tenure": 24}
```

If `--count` exceeds the number of rows in the file, applicants are reused with
replacement, so any count works.

## Drift scenarios

`--drift` shifts the payload before it is sent, simulating a population change:

| Mode | Effect |
| --- | --- |
| `income` | income × 2.5 (a book of suddenly richer applicants) |
| `loan_amount` | loan amount × 4.0 |
| `tenure` | tenure × 2, capped at 60 months |
| `age` | age + 10 years, capped at 75 |
| `all` | all of the above at once |

`make simulate-drift` uses `all`, which is strong enough for `src/monitor.py` to
report `DRIFT` — that is the intended way to demo the drift → retrain path.

## Output

Every response is validated against the serving contract
(`prediction` ∈ {0, 1}, `probability` ∈ [0, 1], non-empty `model_version`); a
response that violates it is counted as a contract violation rather than a
successful prediction.

The script prints a summary:

```text
target               http://localhost:8000  drift=none  data=test.csv
requests             200
succeeded            200
failed               0 (rejected: 0, unreachable: 0)
contract violations  0
throughput           312.4 req/s over 0.64s
latency (ms)         min=3.1 p50=4.1 p95=11.7 max=30.2
default rate         0.115
probability          mean=0.087 min=0.002 max=0.61
model versions       1
```

and writes a machine-readable report to
`reports/simulation/simulation-<drift>.json` containing the same numbers plus
the target URL and source file (request counts, latency min/mean/p50/p95/max,
throughput, prediction rate, probability stats, model versions and up to three
distinct errors).

## End-to-end demo

```bash
# 1. train + register a model (once MLflow is up)
make train

# 2. serve it
uvicorn app.main:app --port 8000

# 3. normal production traffic -> 200 scored applicants in predictions.db
make simulate

# 4. monitoring sees an in-distribution population: status OK
python src/monitor.py

# 5. drifted traffic -> monitoring now reports DRIFT
make simulate-drift
python src/monitor.py

# 6. drift-gated retraining
make retrain
```

Steps 3–5 are covered by the test suite (`tests/test_simulate.py`), which runs
the real simulator against a `TestClient`-backed poster, so the loop
`simulate → monitor` is verified without needing a running server.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | every request succeeded and satisfied the contract |
| `1` | some requests failed, were unreachable, or broke the contract |
| `2` | the health check failed (`/health` unreachable or not `200`) |
