"""API simulation: payloads, drift scenarios, contract checks and the live loop."""

import importlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import monitor, simulate
from src.train import main as train_main

APPLICANTS_CSV = """age,income,loan_amount,tenure,default
25,6000000,20000000,12,0
28,7000000,25000000,18,0
31,9000000,30000000,24,0
35,10000000,50000000,24,0
45,15000000,70000000,36,0
50,18000000,90000000,48,0
29,5000000,40000000,24,1
33,6000000,45000000,36,1
38,7000000,60000000,48,1
42,8000000,70000000,48,1
"""


@pytest.fixture
def applicant_csv(tmp_path):
    """A small stand-in for the applicant population."""
    path = tmp_path / "applicants.csv"
    path.write_text(APPLICANTS_CSV)
    return path


def _always_ok(payload):
    """A poster that never touches the network."""
    return {
        "ok": True,
        "status": 200,
        "body": {"prediction": 0, "probability": 0.11, "model_version": "1"},
        "latency_ms": 4.0,
        "error": None,
    }


def test_payloads_match_the_api_contract(applicant_csv):
    payloads = simulate.build_payloads(str(applicant_csv), count=4, seed=1)

    assert len(payloads) == 4
    for payload in payloads:
        assert set(payload) == set(simulate.FEATURES)
        assert isinstance(payload["age"], int)
        assert isinstance(payload["tenure"], int)
        assert isinstance(payload["income"], float)
        assert isinstance(payload["loan_amount"], float)


def test_payload_sampling_is_reproducible(applicant_csv):
    first = simulate.build_payloads(str(applicant_csv), count=5, seed=4)
    second = simulate.build_payloads(str(applicant_csv), count=5, seed=4)

    assert first == second


def test_count_larger_than_the_file_reuses_applicants(applicant_csv):
    assert len(simulate.build_payloads(str(applicant_csv), count=25, seed=2)) == 25


def test_drift_shifts_only_the_requested_feature(applicant_csv):
    plain = simulate.build_payloads(str(applicant_csv), count=1, seed=3)[0]
    drifted = simulate.apply_drift(plain, "income")

    assert drifted["income"] == plain["income"] * simulate.DRIFT_INCOME_FACTOR
    assert drifted["age"] == plain["age"]
    assert drifted["loan_amount"] == plain["loan_amount"]
    assert drifted["tenure"] == plain["tenure"]


def test_all_drift_moves_every_feature(applicant_csv):
    plain = simulate.build_payloads(str(applicant_csv), count=1, seed=3)[0]
    drifted = simulate.apply_drift(plain, "all")

    assert drifted["income"] > plain["income"]
    assert drifted["loan_amount"] > plain["loan_amount"]
    assert drifted["tenure"] >= plain["tenure"]
    assert drifted["age"] >= plain["age"]


def test_unknown_drift_mode_is_rejected(applicant_csv):
    payload = simulate.build_payloads(str(applicant_csv), count=1, seed=3)[0]

    with pytest.raises(ValueError, match="unknown drift mode"):
        simulate.apply_drift(payload, "seasonal")


def test_contract_check_accepts_a_valid_response():
    assert simulate.check_contract({"prediction": 0, "probability": 0.2, "model_version": "1"}) == []


def test_contract_check_flags_a_broken_response():
    problems = simulate.check_contract({"prediction": 7, "probability": 1.4, "model_version": ""})

    assert len(problems) == 3
    assert simulate.check_contract("not-an-object") == ["response is not a JSON object"]


def test_summary_counts_failures_and_latency():
    results = [
        {
            "ok": True,
            "status": 200,
            "body": {"prediction": 1, "probability": 0.9, "model_version": "2"},
            "latency_ms": 10.0,
            "error": None,
        },
        {"ok": False, "status": 500, "body": None, "latency_ms": 30.0, "error": "HTTP 500: boom"},
    ]

    summary = simulate.summarise_results(results, duration_seconds=0.5)

    assert summary["requests"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["rejected"] == 1
    assert summary["unreachable"] == 0
    assert summary["contract_violations"] == 0
    assert summary["prediction_rate"] == 1.0
    assert summary["model_versions"] == ["2"]
    assert summary["latency_ms"]["max"] == 30.0
    assert summary["errors"] == ["HTTP 500: boom"]


def test_summary_counts_unreachable_requests_separately():
    results = [{"ok": False, "status": None, "body": None, "latency_ms": 5.0, "error": "URLError: refused"}]

    summary = simulate.summarise_results(results, duration_seconds=0.1)

    assert summary["unreachable"] == 1
    assert summary["rejected"] == 0
    assert summary["failed"] == 1
    assert summary["prediction_rate"] is None
    assert summary["probability"]["mean"] is None
    assert summary["latency_ms"]["p50"] == 5.0


def test_summary_counts_contract_violations():
    results = [
        {
            "ok": True,
            "status": 200,
            "body": {"prediction": 9, "probability": 0.1, "model_version": "1"},
            "latency_ms": 1.0,
            "error": None,
        }
    ]

    summary = simulate.summarise_results(results, duration_seconds=0.1)

    assert summary["succeeded"] == 1
    assert summary["contract_violations"] == 1
    # A response that breaks the contract is not a usable prediction.
    assert summary["prediction_rate"] is None


def test_run_reports_a_successful_batch(applicant_csv):
    summary = simulate.run(
        count=25, data_path=str(applicant_csv), poster=_always_ok, workers=4, base_url="injected"
    )

    assert summary["requests"] == 25
    assert summary["succeeded"] == 25
    assert summary["failed"] == 0
    assert summary["contract_violations"] == 0
    assert summary["prediction_rate"] == 0.0
    assert summary["probability"]["mean"] == 0.11
    assert summary["model_versions"] == ["1"]


def test_report_is_written_as_json(applicant_csv, tmp_path):
    summary = simulate.run(
        count=3, data_path=str(applicant_csv), poster=_always_ok, workers=2, base_url="injected"
    )
    path = simulate.write_report(summary, str(tmp_path / "reports"))

    stored = json.loads(Path(path).read_text())
    assert stored["requests"] == 3
    assert stored["drift"] == "none"
    assert path.endswith("simulation-none.json")


def test_main_fails_fast_when_the_api_is_unreachable(applicant_csv, tmp_path):
    exit_code = simulate.main(
        [
            "--base-url",
            "http://127.0.0.1:9",
            "--data",
            str(applicant_csv),
            "--count",
            "1",
            "--report-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 2


def _poster_for(client):
    """Adapt the TestClient to the poster interface the simulator expects."""

    def post(payload):
        started = time.perf_counter()
        response = client.post("/predict", json=payload)
        ok = response.status_code == 200
        return {
            "ok": ok,
            "status": response.status_code,
            "body": response.json() if ok else None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": None if ok else f"HTTP {response.status_code}",
        }

    return post


@pytest.fixture
def live_api():
    """A TestClient with a freshly registered model behind it."""
    train_main()

    import app.main as api_module

    api_module = importlib.reload(api_module)  # pick up the isolated environment

    with TestClient(api_module.app) as client:
        yield client


def test_simulation_drives_the_serving_api(live_api, repository_root):
    summary = simulate.run(
        count=50,
        data_path=str(repository_root / "data" / "test.csv"),
        poster=_poster_for(live_api),
        workers=4,
        base_url="testclient",
    )

    assert summary["requests"] == 50
    assert summary["succeeded"] == 50
    assert summary["failed"] == 0
    assert summary["contract_violations"] == 0
    assert summary["model_versions"] == ["1"]
    assert summary["probability"]["mean"] is not None
    assert summary["latency_ms"]["p95"] is not None


def test_simulated_holdout_traffic_feeds_monitoring_without_drift(live_api, repository_root):
    simulate.run(
        count=200,
        data_path=str(repository_root / "data" / "test.csv"),
        poster=_poster_for(live_api),
        workers=4,
        base_url="testclient",
    )

    report = monitor.main()

    assert report["prediction_count"] == 200
    assert report["status"] == "OK"


def test_drifted_simulation_makes_monitoring_report_drift(live_api, repository_root):
    simulate.run(
        count=200,
        data_path=str(repository_root / "data" / "test.csv"),
        drift="all",
        poster=_poster_for(live_api),
        workers=4,
        base_url="testclient",
    )

    report = monitor.main()

    assert report["status"] == "DRIFT"
    assert report["max_psi"] > report["threshold"]
