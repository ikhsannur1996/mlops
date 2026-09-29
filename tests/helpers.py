"""Helpers shared by the test modules."""

import sqlite3

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


def make_prediction_rows(frame, probability=0.35, model_version="1", **overrides):
    """Turn a training frame into prediction rows, optionally shifting features."""
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        values = {
            "age": float(row["age"]),
            "income": float(row["income"]),
            "loan_amount": float(row["loan_amount"]),
            "tenure": float(row["tenure"]),
        }
        values.update(overrides)
        rows.append(
            (
                f"2026-01-01T00:00:{index:02d}+00:00",
                values["age"],
                values["income"],
                values["loan_amount"],
                values["tenure"],
                int(row["default"]),
                probability,
                model_version,
            )
        )
    return rows
