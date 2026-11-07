from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve

from creditrisk import data, features, metrics


def test_validation_rejects_bad_rows() -> None:
    df = data.load().head(50).copy()
    with pytest.raises(data.DataError, match="missing columns"):
        data.validate(df.drop(columns=["pay_3"]))
    bad = df.copy()
    bad.loc[0, "pay_1"] = 12
    bad.loc[1, "pay_amt2"] = -5
    bad.loc[2, "age"] = 12
    bad.loc[3, "limit_bal"] = 0
    bad.loc[4, data.TARGET] = 2
    bad.loc[5, "id"] = bad.loc[6, "id"]
    bad["bill_amt3"] = bad["bill_amt3"].astype(float)
    bad.loc[7, "bill_amt3"] = np.nan
    with pytest.raises(data.DataError) as err:
        data.validate(bad)
    for expected in ["repayment status", "negative payment", "age outside", "limit_bal", data.TARGET, "duplicate ids", "null values"]:
        assert expected in str(err.value)


def test_split_is_stratified_disjoint_and_deterministic(split: data.Split) -> None:
    ids = [set(p["id"]) for p in (split.train, split.valid, split.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == 30_000
    rates = [p[data.TARGET].mean() for p in (split.train, split.valid, split.test)]
    assert max(rates) - min(rates) < 0.002
    again = data.split(data.load())
    assert again.test["id"].tolist() == split.test["id"].tolist()


def test_features_by_hand() -> None:
    row = {"id": 1, "limit_bal": 10000, "sex": 2, "age": 30, "marriage": 1, "education": 2, data.TARGET: 0}
    row |= dict(zip(data.PAY_STATUS, [2, 0, -1, -1, 1, -2], strict=True))
    row |= dict(zip(data.BILLS, [5000, 4000, 3000, 2000, 1000, 0], strict=True))
    row |= dict(zip(data.PAYMENTS, [400, 3000, 0, 0, 0, 0], strict=True))
    f = features.build(pd.DataFrame([row])).iloc[0]
    assert list(f.index) == features.NAMES and not set(data.PROTECTED) & set(f.index)
    assert f["recent_status"] == 2 and f["max_delinquency_6m"] == 2 and f["months_delinquent_6m"] == 2
    assert f["months_paid_full_6m"] == 2 and f["utilization"] == 0.5 and f["utilization_trend"] == 0.5
    assert f["payment_ratio"] == pytest.approx(0.1) and f["paid_to_limit_6m"] == pytest.approx(0.34)
    # The oldest month has no previous bill; a zero previous bill counts as paid (ratio 1).
    assert f["avg_payment_ratio_6m"] == pytest.approx(np.mean([0.1, 1.0, 0, 0, 1.0]))


def test_metrics() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    fpr, tpr, _ = roc_curve(y, p)
    assert metrics.ks(y, p) == pytest.approx(np.max(tpr - fpr), abs=1e-9)
    assert metrics.gini(y, p) == pytest.approx(2 * metrics.auc(y, p) - 1)
    calibrated = rng.random(20000)
    outcome = (rng.random(20000) < calibrated).astype(int)
    assert metrics.ece(outcome, calibrated) < 0.02 and metrics.ece(outcome, np.clip(calibrated + 0.2, 0, 1)) > 0.15
    assert metrics.brier([0, 1], [0.0, 1.0]) == 0
    assert metrics.psi(calibrated, calibrated) == pytest.approx(0, abs=1e-12)
    assert metrics.psi(calibrated, calibrated**2) > 0.25
    assert metrics.psi(np.zeros(100), np.r_[np.zeros(50), np.ones(50)]) > 1
    ci = metrics.bootstrap_auc(y, p, n=200)
    assert ci.low < ci.value < ci.high
    bands = metrics.deciles(y, p)
    assert len(bands) == 10 and sum(b["count"] for b in bands) == 2000 and bands[0]["predicted"] > bands[-1]["predicted"]
