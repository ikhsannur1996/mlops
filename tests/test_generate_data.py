"""The dataset generator must be reproducible, calibrated and realistic."""

import pandas as pd
import pytest

from src import generate_data


def test_generation_is_reproducible():
    first_train, first_test = generate_data.generate(n_train=500, n_test=200, seed=7)
    second_train, second_test = generate_data.generate(n_train=500, n_test=200, seed=7)

    assert first_train.equals(second_train)
    assert first_test.equals(second_test)


def test_requested_split_sizes_survive_deduplication():
    train, test = generate_data.generate(n_train=800, n_test=200, seed=11)

    assert len(train) == 800
    assert len(test) == 200
    assert not train.duplicated().any()
    assert not test.duplicated().any()


def test_splits_do_not_share_applicants():
    train, test = generate_data.generate(n_train=800, n_test=200, seed=11)
    columns = [*generate_data.FEATURES, generate_data.TARGET]

    assert train.merge(test, how="inner", on=columns).empty


def test_bad_rate_is_calibrated_to_the_requested_value():
    train, _ = generate_data.generate(n_train=4000, n_test=500, seed=3, target_rate=0.18)

    assert train[generate_data.TARGET].mean() == pytest.approx(0.18, abs=0.03)


def test_generated_features_stay_inside_the_product_ranges():
    train, _ = generate_data.generate(n_train=1000, n_test=200, seed=5)

    assert train["age"].between(generate_data.AGE_MIN, generate_data.AGE_MAX).all()
    assert train["income"].between(generate_data.INCOME_MIN, generate_data.INCOME_MAX).all()
    assert train["loan_amount"].between(generate_data.LOAN_MIN, generate_data.LOAN_MAX).all()
    assert set(train["tenure"].unique()) <= set(generate_data.TENURE_CHOICES.tolist())


def test_risk_drivers_point_the_right_way():
    """Heavier debt service and lower income mean more defaults."""
    train, _ = generate_data.generate(n_train=4000, n_test=500, seed=9)
    debt_service = (train["loan_amount"] / train["tenure"]) / train["income"]
    defaulted = train[generate_data.TARGET] == 1

    assert debt_service[defaulted].mean() > debt_service[~defaulted].mean()
    assert train["income"].corr(train[generate_data.TARGET]) < 0


def test_describe_reports_the_correlation_with_the_target():
    train, _ = generate_data.generate(n_train=500, n_test=100, seed=17)
    stats = generate_data.describe(train)

    assert stats["rows"] == 500
    assert set(stats["features"]) == set(generate_data.FEATURES)
    assert stats["features"]["loan_amount"]["correlation_with_target"] > 0


def test_write_datasets_creates_both_splits(tmp_path):
    train, test = generate_data.generate(n_train=300, n_test=100, seed=13)
    train_path, test_path = generate_data.write_datasets(train, test, str(tmp_path))

    assert len(pd.read_csv(train_path)) == 300
    assert len(pd.read_csv(test_path)) == 100
    assert list(pd.read_csv(train_path).columns) == [*generate_data.FEATURES, generate_data.TARGET]
