"""Generate a realistic synthetic credit-default dataset with train and test splits.

The generator imitates how a consumer-lending portfolio actually looks:

* monthly income follows a log-normal distribution (a small tail of high earners)
* applicants are 21-70 years old, centred around 38
* loan tenors are standard product terms (6-60 months) with realistic shares
* the loan amount follows an affordability rule - the monthly instalment is a
  share of income, so the amount is income x debt-service ratio x tenor
* the label is a Bernoulli draw from a latent risk model driven by the
  debt-service ratio, income, age and tenor, calibrated to a realistic bad rate

The data is synthetic on purpose: it carries portfolio-like statistics without
exposing any real customer. Real files can be dropped in instead by pointing
TRAIN_DATA and TEST_DATA at them.

Usage:
    python src/generate_data.py
    N_TRAIN=20000 N_TEST=5000 TARGET_DEFAULT_RATE=0.15 python src/generate_data.py
"""

import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(BASE, "data")

FEATURES = ["age", "income", "loan_amount", "tenure"]
TARGET = "default"

DEFAULT_N_TRAIN = 8000
DEFAULT_N_TEST = 2000
DEFAULT_SEED = 42
DEFAULT_TARGET_RATE = 0.12

AGE_MEAN = 38.0
AGE_SD = 11.0
AGE_MIN = 21
AGE_MAX = 70

INCOME_MEDIAN = 8_000_000.0          # IDR per month
INCOME_SIGMA = 0.55
INCOME_MIN = 2_500_000.0
INCOME_MAX = 75_000_000.0

TENURE_CHOICES = np.array([6, 12, 18, 24, 36, 48, 60])
TENURE_WEIGHTS = np.array([0.08, 0.18, 0.10, 0.26, 0.24, 0.09, 0.05])

# Monthly instalment as a share of income. Banks approve debt-service ratios
# between roughly 3% and 45%.
DEBT_SERVICE_ALPHA = 2.2
DEBT_SERVICE_BETA = 6.0
DEBT_SERVICE_SCALE = 0.6
DEBT_SERVICE_MIN = 0.03
DEBT_SERVICE_MAX = 0.45

LOAN_MIN = 3_000_000.0
LOAN_MAX = 500_000_000.0
IDR_ROUNDING = 100_000.0             # money is quoted to the nearest 100k IDR

# Latent risk model coefficients, applied to standardised features. A higher
# debt-service ratio is the dominant driver; higher income and age are
# protective; a longer tenor means longer exposure to default.
RISK_DEBT_SERVICE = 1.45
RISK_INCOME = 0.55
RISK_AGE = 0.35
RISK_TENURE = 0.25

def _round_idr(values):
    """Round money to realistic IDR figures (nearest 100 thousand)."""
    return np.round(np.asarray(values, dtype=float) / IDR_ROUNDING) * IDR_ROUNDING


def _zscore(values):
    """Standardise a numeric array so coefficients are comparable."""
    values = np.asarray(values, dtype=float)
    spread = values.std()
    return (values - values.mean()) / (spread if spread > 0 else 1.0)


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def sample_population(size, rng):
    """Draw realistic applicants: demographics plus an affordable loan size."""
    age = np.clip(rng.normal(AGE_MEAN, AGE_SD, size), AGE_MIN, AGE_MAX).round()
    income = np.clip(
        rng.lognormal(np.log(INCOME_MEDIAN), INCOME_SIGMA, size), INCOME_MIN, INCOME_MAX
    )
    tenure = rng.choice(TENURE_CHOICES, size=size, p=TENURE_WEIGHTS)
    debt_service = np.clip(
        rng.beta(DEBT_SERVICE_ALPHA, DEBT_SERVICE_BETA, size) * DEBT_SERVICE_SCALE,
        DEBT_SERVICE_MIN,
        DEBT_SERVICE_MAX,
    )
    loan_amount = np.clip(income * debt_service * tenure, LOAN_MIN, LOAN_MAX)

    return pd.DataFrame(
        {
            "age": age.astype(int),
            "income": _round_idr(income),
            "loan_amount": _round_idr(loan_amount),
            "tenure": tenure.astype(int),
        }
    )


def risk_score(customers):
    """Standardised linear predictor of default risk for a customer frame."""
    debt_service = (customers["loan_amount"] / customers["tenure"]) / customers["income"]
    return (
        RISK_DEBT_SERVICE * _zscore(np.log(debt_service))
        - RISK_INCOME * _zscore(np.log(customers["income"]))
        - RISK_AGE * _zscore(customers["age"])
        + RISK_TENURE * _zscore(customers["tenure"])
    )


def calibrate_intercept(scores, target_rate):
    """Find the intercept that makes the average predicted risk hit target_rate."""
    low, high = -40.0, 40.0
    for _ in range(120):
        middle = (low + high) / 2
        if _sigmoid(scores + middle).mean() < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def generate(n_train=DEFAULT_N_TRAIN, n_test=DEFAULT_N_TEST, seed=DEFAULT_SEED, target_rate=DEFAULT_TARGET_RATE):
    """Build duplicate-free, disjoint train and test splits from one population."""
    rng = np.random.default_rng(seed)
    requested = n_train + n_test

    # Draw a small margin so dropping duplicate applicant records cannot shrink
    # the requested split sizes.
    customers = sample_population(int(requested * 1.05) + 10, rng).drop_duplicates().reset_index(drop=True)
    if len(customers) < requested:
        raise ValueError(f"only {len(customers)} unique applicants for {requested} requested rows")

    scores = risk_score(customers)
    intercept = calibrate_intercept(scores, target_rate)
    customers[TARGET] = (
        rng.random(len(customers)) < _sigmoid(scores + intercept)
    ).astype(int)

    shuffled = customers.iloc[rng.permutation(len(customers))].reset_index(drop=True)
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    test = shuffled.iloc[n_train : n_train + n_test].reset_index(drop=True)
    return train, test


def describe(frame):
    """Data-card style statistics for one split."""
    summary = {
        "rows": int(len(frame)),
        "default_rate": float(frame[TARGET].mean()),
        "features": {},
    }
    for feature in FEATURES:
        values = frame[feature].to_numpy(dtype=float)
        summary["features"][feature] = {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "correlation_with_target": float(np.corrcoef(values, frame[TARGET])[0, 1]),
        }
    return summary


def write_datasets(train, test, data_dir):
    """Write data/train.csv and data/test.csv and return their paths."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def main():
    """Generate the datasets and print a data card for each split."""
    data_dir = os.getenv("DATA_DIR", DEFAULT_DIR)
    n_train = int(os.getenv("N_TRAIN", DEFAULT_N_TRAIN))
    n_test = int(os.getenv("N_TEST", DEFAULT_N_TEST))
    seed = int(os.getenv("RANDOM_SEED", DEFAULT_SEED))
    target_rate = float(os.getenv("TARGET_DEFAULT_RATE", DEFAULT_TARGET_RATE))

    train, test = generate(n_train, n_test, seed, target_rate)
    train_path, test_path = write_datasets(train, test, data_dir)

    print(f"seed={seed} requested train={n_train} test={n_test} target rate={target_rate}")
    for name, frame, path in (("train", train, train_path), ("test", test, test_path)):
        stats = describe(frame)
        print(f"\n{name}: {path}")
        print(f"  rows={stats['rows']} default_rate={stats['default_rate']:.4f}")
        for feature, values in stats["features"].items():
            print(
                f"  {feature:12s} min={values['min']:>12,.0f}"
                f" median={values['median']:>12,.0f}"
                f" max={values['max']:>12,.0f}"
                f" corr_with_target={values['correlation_with_target']:+.3f}"
            )

    return {"train": len(train), "test": len(test)}


if __name__ == "__main__":
    main()
