"""Drift-gated retraining: monitor production data, then retrain when PSI is too high."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.evaluate as evaluate  # noqa: E402
import src.monitor as monitor  # noqa: E402
import src.train as train  # noqa: E402

ACTION_NO_DATA = "no-data"
ACTION_NO_DRIFT = "no-drift"
ACTION_DRIFT_LOCKED = "drift-locked"
ACTION_RETRAINED = "retrained"


def main():
    """Run monitoring, then retrain only when drift is detected and enabled.

    Returns the action that was taken so schedulers and tests can react to it.
    """
    report = monitor.main()

    if report is None:
        print("Not enough production data to evaluate drift. Retraining skipped.")
        return ACTION_NO_DATA

    if report["status"] != "DRIFT":
        print(f"No drift detected (max PSI {report['max_psi']:.4f}). Retraining skipped.")
        return ACTION_NO_DRIFT

    print(f"Drift detected (max PSI {report['max_psi']:.4f} > {report['threshold']}).")

    if os.getenv("AUTO_RETRAIN", "0") != "1":
        print("AUTO_RETRAIN is disabled.")
        print("Set AUTO_RETRAIN=1 to execute retraining.")
        return ACTION_DRIFT_LOCKED

    train.main()
    evaluate.main()

    print("Retraining and evaluation complete. Check MLflow for the new run/model version.")
    return ACTION_RETRAINED


if __name__ == "__main__":
    main()

