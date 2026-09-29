"""Simulate production traffic against the running FastAPI service.

The simulator replays real applicant profiles from the holdout split, so every
payload looks like the portfolio the model was validated on, and it checks the
response contract of each call. Use --drift to send a shifted population, which
is what makes src/monitor.py report drift and src/retrain.py retrain.

Only the standard library is used for HTTP, so the simulator has no extra
dependencies and can be pointed at any deployment.

Usage:
    python src/simulate.py
    python src/simulate.py --count 500 --workers 8 --base-url http://10.0.0.5:8000
    python src/simulate.py --drift all
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA = os.path.join(BASE, "data", "test.csv")
DEFAULT_URL = "http://localhost:8000"
DEFAULT_COUNT = 200
DEFAULT_WORKERS = 4
DEFAULT_SEED = 42
DEFAULT_TIMEOUT = 10.0
DEFAULT_REPORT_DIR = os.path.join(BASE, "reports", "simulation")

FEATURES = ["age", "income", "loan_amount", "tenure"]
INTEGER_FEATURES = ("age", "tenure")

DRIFT_MODES = ("none", "income", "loan_amount", "tenure", "age", "all")

# A campaign that attracts bigger, longer loans from older applicants, plus a
# salary market adjustment. Applied on top of the sampled population.
DRIFT_INCOME_FACTOR = 2.5
DRIFT_LOAN_FACTOR = 4.0
DRIFT_TENURE_FACTOR = 2
DRIFT_AGE_SHIFT = 10
MAX_TENURE = 60
MAX_AGE = 75


def load_applicants(path, count, seed=DEFAULT_SEED):
    """Sample applicant payloads from a CSV of real-looking profiles."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} contains no applicants")

    sample = frame.sample(n=count, replace=count > len(frame), random_state=seed)
    return [
        {
            feature: int(row[feature]) if feature in INTEGER_FEATURES else float(row[feature])
            for feature in FEATURES
        }
        for _, row in sample.iterrows()
    ]


def apply_drift(payload, mode):
    """Return a copy of the payload shifted the way a changed population looks."""
    if mode not in DRIFT_MODES:
        raise ValueError(f"unknown drift mode '{mode}', expected one of {DRIFT_MODES}")

    shifted = dict(payload)
    if mode in ("income", "all"):
        shifted["income"] = round(payload["income"] * DRIFT_INCOME_FACTOR)
    if mode in ("loan_amount", "all"):
        shifted["loan_amount"] = round(payload["loan_amount"] * DRIFT_LOAN_FACTOR)
    if mode in ("tenure", "all"):
        shifted["tenure"] = int(min(MAX_TENURE, payload["tenure"] * DRIFT_TENURE_FACTOR))
    if mode in ("age", "all"):
        shifted["age"] = int(min(MAX_AGE, payload["age"] + DRIFT_AGE_SHIFT))
    return shifted


def build_payloads(path, count, drift="none", seed=DEFAULT_SEED):
    """Sample applicants and apply the requested population shift."""
    return [apply_drift(applicant, drift) for applicant in load_applicants(path, count, seed)]


def http_post(url, payload, timeout=DEFAULT_TIMEOUT):
    """POST one JSON payload and return a normalised result dictionary."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {
                "ok": True,
                "status": response.status,
                "body": body,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "error": None,
            }
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        return {
            "ok": False,
            "status": error.code,
            "body": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": f"HTTP {error.code}: {detail}",
        }
    except Exception as error:  # URLError, timeouts, invalid JSON
        return {
            "ok": False,
            "status": None,
            "body": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": f"{type(error).__name__}: {error}"[:300],
        }


def make_http_poster(base_url, timeout=DEFAULT_TIMEOUT):
    """Return a callable that posts one payload to the prediction endpoint."""
    url = base_url.rstrip("/") + "/predict"
    return lambda payload: http_post(url, payload, timeout)


def server_is_ready(base_url, timeout=DEFAULT_TIMEOUT):
    """Return True when GET /health answers."""
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:  # unreachable, refused or timed out
        return False


def check_contract(body):
    """Return the list of contract violations found in a prediction response."""
    if not isinstance(body, dict):
        return ["response is not a JSON object"]

    problems = []
    if body.get("prediction") not in (0, 1):
        problems.append(f"prediction={body.get('prediction')!r} is not 0 or 1")

    probability = body.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        problems.append(f"probability={probability!r} is not numeric")
    elif not 0.0 <= float(probability) <= 1.0:
        problems.append(f"probability={probability!r} is outside [0, 1]")

    if not str(body.get("model_version") or "").strip():
        problems.append("model_version is missing")

    return problems

def _percentile(ordered_values, fraction):
    """Nearest-rank percentile of an already sorted list."""
    position = round(fraction * (len(ordered_values) - 1))
    return ordered_values[min(len(ordered_values) - 1, max(0, position))]


def _latency_stats(latencies):
    """min/p50/p95/max/mean in milliseconds, or None when there is no data."""
    if not latencies:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(latencies)
    return {
        "min": round(ordered[0], 2),
        "p50": round(_percentile(ordered, 0.50), 2),
        "p95": round(_percentile(ordered, 0.95), 2),
        "max": round(ordered[-1], 2),
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def summarise_results(results, duration_seconds, drift="none"):
    """Turn raw call results into the summary that gets printed and stored."""
    successful = [result for result in results if result["ok"]]
    rejected = [result for result in results if not result["ok"] and result["status"] is not None]
    unreachable = [result for result in results if not result["ok"] and result["status"] is None]

    violations = 0
    predictions = []
    probabilities = []
    versions = set()
    for result in successful:
        problems = check_contract(result["body"])
        violations += len(problems)
        if not problems:
            predictions.append(int(result["body"]["prediction"]))
            probabilities.append(float(result["body"]["probability"]))
        version = str(result["body"].get("model_version") or "").strip()
        if version:
            versions.add(version)

    return {
        "drift": drift,
        "requests": len(results),
        "succeeded": len(successful),
        "failed": len(results) - len(successful),
        "rejected": len(rejected),
        "unreachable": len(unreachable),
        "contract_violations": violations,
        "duration_seconds": round(duration_seconds, 3),
        "throughput_per_second": round(len(results) / duration_seconds, 2) if duration_seconds else 0.0,
        "latency_ms": _latency_stats([result["latency_ms"] for result in results]),
        "prediction_rate": round(sum(predictions) / len(predictions), 4) if predictions else None,
        "probability": {
            "mean": round(sum(probabilities) / len(probabilities), 4) if probabilities else None,
            "min": round(min(probabilities), 4) if probabilities else None,
            "max": round(max(probabilities), 4) if probabilities else None,
        },
        "model_versions": sorted(versions),
        "errors": sorted({result["error"] for result in results if result["error"]})[:3],
    }


def send(poster, payloads, workers=DEFAULT_WORKERS):
    """Send every payload through the poster and time the whole run."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(poster, payloads))
    return results, time.perf_counter() - started

def run(
    count=DEFAULT_COUNT,
    data_path=DEFAULT_DATA,
    base_url=DEFAULT_URL,
    drift="none",
    workers=DEFAULT_WORKERS,
    seed=DEFAULT_SEED,
    timeout=DEFAULT_TIMEOUT,
    poster=None,
):
    """Simulate `count` predictions and return the summary report."""
    payloads = build_payloads(data_path, count, drift, seed)
    if poster is None:
        poster = make_http_poster(base_url, timeout)

    results, duration = send(poster, payloads, workers)
    summary = summarise_results(results, duration, drift)
    summary["base_url"] = base_url
    summary["data_path"] = os.path.basename(data_path)
    return summary


def print_report(summary):
    """Print the simulation summary in a readable layout."""
    latency = summary["latency_ms"]
    probability = summary["probability"]

    print(f"target               {summary['base_url']}  drift={summary['drift']}  data={summary['data_path']}")
    print(f"requests             {summary['requests']}")
    print(f"succeeded            {summary['succeeded']}")
    print(
        f"failed               {summary['failed']}"
        f" (rejected: {summary['rejected']}, unreachable: {summary['unreachable']})"
    )
    print(f"contract violations  {summary['contract_violations']}")
    print(f"throughput           {summary['throughput_per_second']} req/s over {summary['duration_seconds']}s")

    if summary["succeeded"]:
        print(
            f"latency (ms)         min={latency['min']} p50={latency['p50']}"
            f" p95={latency['p95']} max={latency['max']}"
        )
        print(f"default rate         {summary['prediction_rate']}")
        print(f"probability          mean={probability['mean']} min={probability['min']} max={probability['max']}")
        print(f"model versions       {', '.join(summary['model_versions']) or 'unknown'}")
    else:
        print("no successful responses - is a model registered? try: make train")

    for error in summary["errors"]:
        print(f"error sample         {error}")


def write_report(summary, report_dir=DEFAULT_REPORT_DIR):
    """Persist the summary as JSON and return the path it was written to."""
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"simulation-{summary['drift']}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return path


def main(argv=None):
    """Run the simulation from the command line."""
    parser = argparse.ArgumentParser(
        description="Simulate production traffic against the credit-default API."
    )
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", DEFAULT_URL))
    parser.add_argument("--count", type=int, default=int(os.getenv("SIMULATION_REQUESTS", DEFAULT_COUNT)))
    parser.add_argument("--workers", type=int, default=int(os.getenv("SIMULATION_WORKERS", DEFAULT_WORKERS)))
    parser.add_argument("--drift", choices=DRIFT_MODES, default=os.getenv("SIMULATION_DRIFT", "none"))
    parser.add_argument("--data", default=os.getenv("TEST_DATA", DEFAULT_DATA))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", DEFAULT_SEED)))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--report-dir", default=os.getenv("SIMULATION_REPORT_DIR", DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="do not probe GET /health before sending traffic",
    )
    args = parser.parse_args(argv)

    if not args.skip_health_check and not server_is_ready(args.base_url, args.timeout):
        print(f"api at {args.base_url} is not reachable.")
        print("start it with: docker compose up -d api")
        return 2

    summary = run(
        count=args.count,
        data_path=args.data,
        base_url=args.base_url,
        drift=args.drift,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )
    print_report(summary)
    print(f"\nreport written to {write_report(summary, args.report_dir)}")

    if summary["failed"] or summary["contract_violations"]:
        print("simulation finished with failures")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
