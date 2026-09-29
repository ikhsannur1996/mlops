"""Data quality gates for the training dataset."""

from pathlib import Path

import pandas as pd

from src.train import FEATURES, TARGET

EXPECTED_COLUMNS = [*FEATURES, TARGET]


def test_dataset_has_expected_columns(train_frame):
    assert list(train_frame.columns) == EXPECTED_COLUMNS


def test_dataset_has_no_missing_values(train_frame):
    assert not train_frame.isna().to_numpy().any()


def test_dataset_columns_are_numeric(train_frame):
    assert all(pd.api.types.is_numeric_dtype(train_frame[column]) for column in EXPECTED_COLUMNS)


def test_target_is_binary(train_frame):
    assert set(train_frame[TARGET].unique()) == {0, 1}


def test_target_is_balanced_enough_for_stratification(train_frame):
    assert 0.2 <= train_frame[TARGET].mean() <= 0.8
    assert train_frame[TARGET].value_counts().min() >= 2


def test_features_are_within_business_ranges(train_frame):
    assert train_frame["age"].between(18, 100).all()
    assert (train_frame["income"] > 0).all()
    assert (train_frame["loan_amount"] > 0).all()
    assert train_frame["tenure"].between(1, 120).all()


def test_dataset_has_no_duplicate_rows(train_frame):
    assert not train_frame.duplicated().any()


def test_dataset_has_enough_rows(train_frame):
    assert len(train_frame) >= 20


def test_committed_dataset_is_what_training_reads_by_default(train_frame, repository_root):
    from src.train import DEFAULT_DATA

    assert Path(DEFAULT_DATA) == repository_root / "data" / "train.csv"
    assert len(train_frame) == len(pd.read_csv(DEFAULT_DATA))
