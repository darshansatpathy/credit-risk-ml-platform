"""Population stability of the score and of every feature, against the training population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditrisk.data import BILLS, PAY_STATUS, PAYMENTS
from creditrisk.features import NAMES
from creditrisk.metrics import psi


@dataclass
class Stability:
    name: str
    psi: float
    status: str  # stable, warn, alert


def stability(reference: pd.DataFrame, current: pd.DataFrame, ref_pd: np.ndarray, cur_pd: np.ndarray, warn: float, alert: float) -> list[Stability]:
    def status(v: float) -> str:
        return "alert" if v >= alert else "warn" if v >= warn else "stable"

    rows = [Stability("score (PD)", psi(ref_pd, cur_pd), "")]
    rows += [Stability(n, psi(reference[n].to_numpy(dtype=float), current[n].to_numpy(dtype=float)), "") for n in NAMES]
    for r in rows:
        r.status = status(r.psi)
    return rows


def stress(df: pd.DataFrame, seed: int = 3, severity: float = 1.0) -> pd.DataFrame:
    """An economic-downturn population: higher balances, smaller payments, more delinquency."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[BILLS] = (out[BILLS] * (1 + 0.25 * severity)).round()
    out[PAYMENTS] = (out[PAYMENTS] * (1 - 0.35 * severity)).round()
    hit = rng.random(len(out)) < 0.20 * severity
    for col in PAY_STATUS[:3]:
        out.loc[hit, col] = np.clip(out.loc[hit, col].clip(lower=0) + 1, -2, 8)
    return out
