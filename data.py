"""Synthetic loan applications (inspired by public credit-default datasets)."""
import numpy as np
import pandas as pd

FEATURES = ["credit_score", "annual_income_k", "loan_amount_k", "debt_to_income", "employment_years",
            "missed_payments_2y"]
TARGET = "defaulted"


def make_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    credit = np.clip(rng.normal(670, 70, n_rows), 300, 850)             # credit score
    income = np.clip(rng.lognormal(4.0, 0.45, n_rows), 12, 400)          # thousands per year
    loan = np.clip(income * rng.uniform(0.1, 0.8, n_rows), 1, 200)       # thousands
    dti = np.clip(rng.normal(0.30, 0.12, n_rows), 0, 0.95)               # debt-to-income ratio
    employment = np.clip(rng.exponential(6, n_rows), 0, 45)              # years in current job
    missed = rng.poisson(0.4 + 2.0 * (credit < 600), n_rows).clip(0, 12) # missed payments, last 2 years

    score = (
        -1.6
        - 0.012 * (credit - 670)
        + 3.0 * (dti - 0.30)
        + 1.2 * (loan / income - 0.45)
        - 0.05 * employment
        + 0.45 * missed
    )
    default = rng.random(n_rows) < 1 / (1 + np.exp(-score))
    return pd.DataFrame({
        "credit_score": credit.round(0),
        "annual_income_k": income.round(1),
        "loan_amount_k": loan.round(1),
        "debt_to_income": dti.round(3),
        "employment_years": employment.round(1),
        "missed_payments_2y": missed,
        TARGET: default.astype(int),
    })
