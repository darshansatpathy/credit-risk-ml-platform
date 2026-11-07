"""Discrimination, calibration and stability measures, written out so they can be audited."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score


def auc(y: ArrayLike, p: ArrayLike) -> float:
    return float(roc_auc_score(y, p))


def gini(y: ArrayLike, p: ArrayLike) -> float:
    return 2 * auc(y, p) - 1


def ks(y: ArrayLike, p: ArrayLike) -> float:
    """Largest gap between the cumulative score distributions of defaulters and non-defaulters."""
    y_arr, p_arr = np.asarray(y), np.asarray(p, dtype=float)
    thresholds = np.unique(p_arr)
    bad = np.searchsorted(np.sort(p_arr[y_arr == 1]), thresholds, side="right") / max((y_arr == 1).sum(), 1)
    good = np.searchsorted(np.sort(p_arr[y_arr == 0]), thresholds, side="right") / max((y_arr == 0).sum(), 1)
    return float(np.max(np.abs(good - bad)))


def brier(y: ArrayLike, p: ArrayLike) -> float:
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p_arr - y_arr) ** 2))


def ece(y: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins of predicted probability."""
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p_arr, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(p_arr[chunk].mean() - y_arr[chunk].mean())
    return float(total / len(p_arr))


def psi(expected: ArrayLike, actual: ArrayLike, bins: int = 10) -> float:
    """Population stability index, with bins from the expected (reference) distribution."""
    e, a = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # a nearly constant feature: compare the share at each distinct value
        values = np.unique(np.concatenate([e, a]))
        e_share = np.array([(e == v).mean() for v in values])
        a_share = np.array([(a == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e_share = np.histogram(e, edges)[0] / len(e)
        a_share = np.histogram(a, edges)[0] / len(a)
    e_share, a_share = np.clip(e_share, 1e-4, None), np.clip(a_share, 1e-4, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float


def bootstrap_auc(y: ArrayLike, p: ArrayLike, n: int = 500, seed: int = 0) -> Interval:
    y_arr, p_arr = np.asarray(y), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if y_arr[idx].min() != y_arr[idx].max():
            stats.append(roc_auc_score(y_arr[idx], p_arr[idx]))
    low, high = np.quantile(stats, [0.025, 0.975])
    return Interval(auc(y_arr, p_arr), float(low), float(high))


def deciles(y: ArrayLike, p: ArrayLike) -> list[dict[str, float]]:
    """Ten equal-count risk bands, riskiest first: predicted and observed default rates."""
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(-p_arr, kind="stable")
    rows = []
    for i, chunk in enumerate(np.array_split(order, 10), start=1):
        rows.append({"band": i, "count": len(chunk), "predicted": float(p_arr[chunk].mean()), "observed": float(y_arr[chunk].mean())})
    return rows
