"""Model features, each with a direction a credit officer would expect.

Protected and proxy attributes (sex, age, marriage, education) are never model inputs; they are
kept beside the scores for fairness testing only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditrisk.data import BILLS, PAY_STATUS, PAYMENTS


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    direction: int  # +1: higher means riskier, -1: higher means safer, 0: no constraint
    reason: str  # wording for an adverse-action reason when this feature raises risk


FEATURES = [
    Feature("limit_bal", "Credit limit", -1, "Low credit limit relative to other customers"),
    Feature("recent_status", "Repayment status last month (-2 no use, -1 paid in full, 0 revolving, n months late)", 1, "Recent payment delinquency"),
    Feature("max_delinquency_6m", "Worst months-late status in the last six months", 1, "Serious delinquency in the last six months"),
    Feature("months_delinquent_6m", "Months with a late payment in the last six months", 1, "Frequent late payments"),
    Feature("months_paid_full_6m", "Months paid in full in the last six months", -1, "Few months paid in full"),
    Feature("utilization", "Last bill as a share of the limit", 1, "High credit utilization"),
    Feature("avg_utilization_6m", "Average bill as a share of the limit over six months", 1, "Sustained high utilization"),
    Feature("utilization_trend", "Change in utilization over six months", 1, "Rising balances"),
    Feature("payment_ratio", "Last payment as a share of the previous bill", -1, "Low payments relative to balance"),
    Feature("avg_payment_ratio_6m", "Average payment as a share of the previous bill", -1, "Persistently low payments"),
    Feature("paid_to_limit_6m", "Total paid in six months as a share of the limit", -1, "Little repaid relative to the limit"),
    Feature("bill_amt1", "Last bill amount", 0, "Large outstanding balance"),
]
NAMES = [f.name for f in FEATURES]
MONOTONE = [f.direction for f in FEATURES]
BY_NAME = {f.name: f for f in FEATURES}


def build(df: pd.DataFrame) -> pd.DataFrame:
    status = df[PAY_STATUS].to_numpy()
    bills = df[BILLS].to_numpy(dtype=float)
    pays = df[PAYMENTS].to_numpy(dtype=float)
    limit = df["limit_bal"].to_numpy(dtype=float)
    util = np.clip(bills / limit[:, None], -1.0, 3.0)
    # Payment in month i settles the bill of month i+1 (the columns run newest to oldest).
    prev_bill = np.maximum(bills[:, 1:], 0.0)
    ratio = np.where(prev_bill > 0, pays[:, :-1] / np.where(prev_bill > 0, prev_bill, 1.0), 1.0)
    ratio = np.clip(ratio, 0.0, 2.0)
    out = pd.DataFrame(
        {
            "limit_bal": limit,
            "recent_status": status[:, 0],
            "max_delinquency_6m": np.maximum(status.max(axis=1), 0),
            "months_delinquent_6m": (status > 0).sum(axis=1),
            "months_paid_full_6m": (status == -1).sum(axis=1),
            "utilization": util[:, 0],
            "avg_utilization_6m": util.mean(axis=1),
            "utilization_trend": util[:, 0] - util[:, 5],
            "payment_ratio": ratio[:, 0],
            "avg_payment_ratio_6m": ratio.mean(axis=1),
            "paid_to_limit_6m": np.clip(pays.sum(axis=1) / limit, 0.0, 5.0),
            "bill_amt1": bills[:, 0],
        },
        index=df.index,
    )
    return out[NAMES]
