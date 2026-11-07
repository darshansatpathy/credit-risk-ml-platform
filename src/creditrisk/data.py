"""Load, validate and split the credit card default data.

The data is the UCI "Default of Credit Card Clients" set (Yeh, 2009; CC BY 4.0): 30,000 Taiwanese
card holders, six months of billing and repayment history, and whether they defaulted the
following month.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from creditrisk.resources import resource

TARGET = "default_next_month"
PAY_STATUS = [f"pay_{i}" for i in range(1, 7)]
BILLS = [f"bill_amt{i}" for i in range(1, 7)]
PAYMENTS = [f"pay_amt{i}" for i in range(1, 7)]
PROTECTED = ["sex", "age", "marriage", "education"]
COLUMNS = ["id", "limit_bal", *PROTECTED, *PAY_STATUS, *BILLS, *PAYMENTS, TARGET]


class DataError(ValueError):
    pass


def load(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or resource("data") / "credit_card_default.csv.gz")
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Reject data the models were not built for, instead of scoring it silently."""
    problems: list[str] = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {', '.join(missing)}")
    if df[COLUMNS].isna().any().any():
        problems.append("null values: " + ", ".join(c for c in COLUMNS if df[c].isna().any()))
    if (df["limit_bal"] <= 0).any():
        problems.append("limit_bal must be positive")
    if not df[PAY_STATUS].isin(range(-2, 10)).all().all():
        problems.append("repayment status outside -2..9")
    if (df[PAYMENTS] < 0).any().any():
        problems.append("negative payment amounts")
    if not df["age"].between(18, 100).all():
        problems.append("age outside 18..100")
    if TARGET in df and not df[TARGET].isin([0, 1]).all():
        problems.append(f"{TARGET} must be 0 or 1")
    if df["id"].duplicated().any():
        problems.append("duplicate ids")
    if problems:
        raise DataError("; ".join(problems))


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split(df: pd.DataFrame, seed: int = 42, valid: float = 0.2, test: float = 0.2) -> Split:
    """Stratified train / validation (calibration, early stopping) / test (reporting only)."""
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    for _, group in df.groupby(TARGET):
        idx = rng.permutation(group.index.to_numpy())
        n_test, n_valid = round(len(idx) * test), round(len(idx) * valid)
        parts["test"].append(df.loc[idx[:n_test]])
        parts["valid"].append(df.loc[idx[n_test : n_test + n_valid]])
        parts["train"].append(df.loc[idx[n_test + n_valid :]])
    out = {k: pd.concat(v).sort_values("id").reset_index(drop=True) for k, v in parts.items()}
    return Split(out["train"], out["valid"], out["test"])
