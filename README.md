# Simple MLOps Education

A deliberately small end-to-end MLOps demo.

## One command

```bash
docker compose up --build
```

Then open:

- MLflow: http://localhost:5001
- Evidently: http://localhost:8001
- API: http://localhost:8000

## What it demonstrates

```text
Synthetic Indonesian Credit Data
            ↓
       3 ML Models
            ↓
        Evaluation
            ↓
 Champion / Challenger / Loser
            ↓
        A/B Testing
            ↓
    Simulated Production Data
            ↓
        Data Drift
            ↓
         Evidently
            ↓
   Drift threshold exceeded?
          /       \
        No         Yes
        |           |
     Continue    Retrain
```

## Dataset

Only 5 columns:

```text
age
monthly_income
loan_amount
tenure_months
credit_status
```

`credit_status` contains:

```text
Good
Risk
```

The data is synthetic and intended only for education.

## Three models

- Logistic Regression
- Random Forest
- Gradient Boosting

The model with the highest ROC-AUC is assigned:

```text
Champion
```

Second:

```text
Challenger
```

Third:

```text
Loser
```

These roles are generated automatically from the actual run.

## A/B testing

The demo simulates:

```text
50% Champion
50% Challenger
```

and logs simple traffic metrics to MLflow.

## Data drift

Production data is intentionally shifted:

- income
- loan amount
- age

Evidently generates an interactive report.

## Automatic retraining

A simple PSI threshold is used:

```text
PSI <= 0.20  → no retraining
PSI >  0.20  → automatic retraining
```

This is intentionally simplified for teaching.

Important concept:

> Data drift is a signal for investigation. It does not universally mean that a production model must be retrained.

## Stop

```bash
docker compose down
```
