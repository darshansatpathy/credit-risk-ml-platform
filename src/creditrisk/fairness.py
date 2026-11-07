"""Outcome fairness at the decision cutoff, by protected attribute.

The attributes are never model inputs. They are joined back to the decisions to test whether
the outcomes differ by group, which is what fair-lending reviews (disparate impact) look at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditrisk.data import TARGET

AGE_BANDS = [(18, 25, "under 25"), (25, 35, "25-34"), (35, 50, "35-49"), (50, 120, "50 and over")]
LABELS = {
    "sex": {1: "male", 2: "female"},
    "education": {1: "graduate school", 2: "university", 3: "high school"},
    "marriage": {1: "married", 2: "single"},
}


def groups(df: pd.DataFrame, attribute: str) -> pd.Series:
    if attribute == "age_band":
        out = pd.Series("", index=df.index)
        for lo, hi, label in AGE_BANDS:
            out[(df["age"] >= lo) & (df["age"] < hi)] = label
        return out
    labels = LABELS[attribute]
    return df[attribute].map(lambda v: labels.get(int(v), "other"))


@dataclass
class GroupStats:
    group: str
    count: int
    approval_rate: float
    good_approval_rate: float  # approval rate among those who did not default (equal opportunity)
    mean_pd: float
    default_rate: float


@dataclass
class AttributeReport:
    attribute: str
    groups: list[GroupStats]
    adverse_impact_ratio: float
    equal_opportunity_gap: float


def assess(df: pd.DataFrame, pd_: np.ndarray, approved: np.ndarray, attributes: list[str], min_group: int) -> list[AttributeReport]:
    good = df[TARGET].to_numpy() == 0
    reports = []
    for attribute in attributes:
        labels = groups(df, attribute).to_numpy()
        stats = []
        for g in sorted(set(labels)):
            m = labels == g
            if m.sum() < min_group:
                continue
            stats.append(
                GroupStats(
                    str(g), int(m.sum()), float(approved[m].mean()), float(approved[m & good].mean()), float(pd_[m].mean()), float((~good[m]).mean())
                )
            )
        rates = [s.approval_rate for s in stats]
        goods = [s.good_approval_rate for s in stats]
        reports.append(AttributeReport(attribute, stats, min(rates) / max(rates), max(goods) - min(goods)))
    return reports
