"""The policy gate passes the trained champion and fails models with planted problems."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk import data, fairness, models
from creditrisk.data import TARGET, Split
from creditrisk.evaluation import evaluate
from creditrisk.features import build
from creditrisk.pipeline import RunResult, choose
from creditrisk.policy import Policy


def failed(ev: object) -> set[str]:
    return {c.name.split(":")[0] for c in ev.checks if not c.passed}  # type: ignore[attr-defined]


def test_champion_passes_and_scorecard_is_the_benchmark(run: RunResult) -> None:
    assert run.champion == "lightgbm"
    assert run.evaluation("lightgbm").passed and run.evaluation("xgboost").passed
    assert failed(run.evaluation("scorecard")) == {"discrimination"}
    champ = run.evaluation("lightgbm")
    assert 0.79 < champ.decisions["approval_rate"] < 0.82
    assert champ.decisions["bad_rate_declined"] > 3 * champ.decisions["bad_rate_approved"]


def test_unconstrained_model_fails_conceptual_soundness(split: Split, pol: Policy) -> None:
    X, Xv = build(split.train), build(split.valid)
    m = models.train_lightgbm(X, split.train[TARGET].to_numpy(), Xv, split.valid[TARGET].to_numpy(), monotone=False, name="free")
    m.calibrate(Xv, split.valid[TARGET].to_numpy())
    assert "conceptual soundness" in failed(evaluate(m, split, pol))


def test_miscalibrated_model_fails_calibration(run: RunResult, split: Split, pol: Policy) -> None:
    m = run.candidates[1]
    original = m.calibrator
    try:
        m.calibrator = models.Calibrator([-6.0, 3.0], [0.15, 0.95])  # monotone, but probabilities far too high
        assert failed(evaluate(m, split, pol)) >= {"calibration"}
    finally:
        m.calibrator = original


def test_leaked_target_fails_leakage(split: Split, pol: Policy) -> None:
    def leak(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["pay_1"] = np.where(out[TARGET] == 1, 8, -1)
        return out

    leaky = Split(leak(split.train), leak(split.valid), leak(split.test))
    X, Xv = build(leaky.train), build(leaky.valid)
    m = models.train_lightgbm(X, leaky.train[TARGET].to_numpy(), Xv, leaky.valid[TARGET].to_numpy(), name="leaky")
    m.calibrate(Xv, leaky.valid[TARGET].to_numpy())
    assert "leakage" in failed(evaluate(m, leaky, pol))


def test_fairness_detects_disparate_outcomes(split: Split) -> None:
    df = split.test
    approved = (df["sex"] == 1).to_numpy() | (np.arange(len(df)) % 3 == 0)
    report = fairness.assess(df, np.full(len(df), 0.2), approved, ["sex", "age_band", "education", "marriage"], 200)
    sex = report[0]
    assert sex.adverse_impact_ratio < 0.5 and {g.group for g in sex.groups} == {"male", "female"}
    assert all(g.count >= 200 for r in report for g in r.groups)


def test_no_reasons_fails_explainability(run: RunResult, split: Split, pol: Policy) -> None:
    m = run.candidates[2]
    original = m.contributions
    m.contributions = lambda X: np.zeros((len(X), 12))  # type: ignore[method-assign]
    try:
        assert "explainability" in failed(evaluate(m, split, pol))
    finally:
        m.contributions = original  # type: ignore[method-assign]


def test_choose_requires_an_eligible_model(run: RunResult) -> None:
    assert choose([run.evaluation("scorecard")]) is None
    assert choose(run.evaluations) == "lightgbm"


def test_data_module_constants() -> None:
    assert len(data.COLUMNS) == 25 and data.TARGET in data.COLUMNS
