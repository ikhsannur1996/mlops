"""Data quality gates for the generated train and test splits."""

from pathlib import Path

import pandas as pd

from src.train import DEFAULT_TEST, DEFAULT_TRAIN, FEATURES, TARGET

EXPECTED_COLUMNS = [*FEATURES, TARGET]
MIN_TRAIN_ROWS = 1000
MIN_TEST_ROWS = 250
# A realistic consumer-lending bad rate is a few percent to the mid teens,
# never the ~50% that a balanced toy dataset would produce.
MIN_BAD_RATE = 0.03
MAX_BAD_RATE = 0.30


def test_dataset_has_expected_columns(train_frame):
    assert list(train_frame.columns) == EXPECTED_COLUMNS


def test_dataset_has_no_missing_values(train_frame, test_frame):
    assert not train_frame.isna().to_numpy().any()
    assert not test_frame.isna().to_numpy().any()


def test_dataset_columns_are_numeric(train_frame, test_frame):
    for frame in (train_frame, test_frame):
        assert all(pd.api.types.is_numeric_dtype(frame[column]) for column in EXPECTED_COLUMNS)


def test_dataset_is_large_enough_to_be_meaningful(train_frame, test_frame):
    assert len(train_frame) >= MIN_TRAIN_ROWS
    assert len(test_frame) >= MIN_TEST_ROWS


def test_target_is_binary(train_frame, test_frame):
    for frame in (train_frame, test_frame):
        assert set(frame[TARGET].unique()) == {0, 1}


def test_bad_rate_is_realistic(train_frame, test_frame):
    for frame in (train_frame, test_frame):
        assert MIN_BAD_RATE <= frame[TARGET].mean() <= MAX_BAD_RATE


def test_both_splits_are_free_of_duplicates(train_frame, test_frame):
    assert not train_frame.duplicated().any()
    assert not test_frame.duplicated().any()


def test_no_applicant_appears_in_both_splits(train_frame, test_frame):
    shared = train_frame.merge(test_frame, how="inner", on=EXPECTED_COLUMNS)
    assert shared.empty


def test_train_and_test_come_from_the_same_population(train_frame, test_frame):
    assert abs(train_frame[TARGET].mean() - test_frame[TARGET].mean()) < 0.03
    for feature in FEATURES:
        median = float(train_frame[feature].median())
        assert abs(float(test_frame[feature].median()) - median) / median < 0.25, feature


def test_features_are_within_business_ranges(train_frame, test_frame):
    for frame in (train_frame, test_frame):
        assert frame["age"].between(18, 100).all()
        assert (frame["income"] > 0).all()
        assert (frame["loan_amount"] > 0).all()
        assert frame["tenure"].between(1, 120).all()


def test_money_columns_are_right_skewed_like_a_real_portfolio(train_frame):
    assert train_frame["income"].mean() > train_frame["income"].median()
    assert train_frame["loan_amount"].mean() > train_frame["loan_amount"].median()


def test_committed_splits_are_what_the_pipeline_reads_by_default(repository_root):
    assert Path(DEFAULT_TRAIN) == repository_root / "data" / "train.csv"
    assert Path(DEFAULT_TEST) == repository_root / "data" / "test.csv"
    assert len(pd.read_csv(DEFAULT_TRAIN)) >= MIN_TRAIN_ROWS
    assert len(pd.read_csv(DEFAULT_TEST)) >= MIN_TEST_ROWS
