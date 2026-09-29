"""Helpers shared by the test modules."""

import sqlite3

from src import simulate

PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    timestamp TEXT,
    age REAL,
    income REAL,
    loan_amount REAL,
    tenure REAL,
    prediction INTEGER,
    probability REAL,
    model_version TEXT
)
"""


def seed_predictions(db_path, rows):
    """Create the prediction table and insert the given prediction rows."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(PREDICTIONS_SCHEMA)
        conn.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)", rows)


def _timestamp(index):
    """A valid UTC timestamp string for a served request."""
    hours, remainder = divmod(index, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"2026-01-01T{hours % 24:02d}:{minutes:02d}:{seconds:02d}+00:00"


def make_prediction_rows(frame, probability=0.35, model_version="1", drift=None):
    """Turn a customer frame into served-prediction rows.

    When ``drift`` is given the features are shifted exactly the way
    ``src/simulate.py`` shifts a drifted population, so the monitoring tests
    exercise the same scenario the simulator produces in production.
    """
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        payload = {feature: row[feature] for feature in simulate.FEATURES}
        if drift is not None:
            payload = simulate.apply_drift(payload, drift)
        rows.append(
            (
                _timestamp(index),
                float(payload["age"]),
                float(payload["income"]),
                float(payload["loan_amount"]),
                float(payload["tenure"]),
                int(row["default"]),
                probability,
                model_version,
            )
        )
    return rows
