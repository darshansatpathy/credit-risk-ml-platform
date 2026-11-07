"""Everything the model-risk policy asks of a model, measured on data it was not fitted on."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from creditrisk import explain, fairness, metrics, monitoring
from creditrisk.data import TARGET, Split
from creditrisk.features import FEATURES, build
from creditrisk.models import RiskModel, score_from_pd
from creditrisk.policy import Policy


@dataclass
class Check:
    name: str
    value: float
    threshold: str
    passed: bool


@dataclass
class ModelEvaluation:
    name: str
    kind: str
    train_auc: float
    valid_auc: float
    test: dict[str, float]
    auc_interval: tuple[float, float]
    cutoff: float  # on the raw model score: decisions are made on it because calibrated PDs have ties
    pd_cutoff: float
    decisions: dict[str, float]
    deciles: list[dict[str, float]]
    fairness: list[fairness.AttributeReport]
    stability: list[monitoring.Stability]
    monotonicity: dict[str, int]
    reason_coverage: float
    importance: list[tuple[str, float]]
    examples: list[dict[str, Any]]
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def monotonicity(model: RiskModel, X: pd.DataFrame, points: int = 15) -> dict[str, int]:
    """For each constrained feature, sweep it across its range for every sampled applicant and
    count applicants whose PD moves against the declared direction."""
    violations = {}
    for f in FEATURES:
        if f.direction == 0:
            continue
        grid = np.unique(np.quantile(X[f.name], np.linspace(0.01, 0.99, points)))
        curves = []
        for v in grid:
            probe = X.copy()
            probe[f.name] = v
            curves.append(model.predict_pd(probe))
        steps = np.diff(np.column_stack(curves), axis=1) * f.direction
        violations[f.name] = int((steps < -1e-9).any(axis=1).sum())
    return violations


def evaluate(model: RiskModel, split: Split, policy: Policy, sample: int = 300, seed: int = 0) -> ModelEvaluation:
    X, Xv, Xt = build(split.train), build(split.valid), build(split.test)
    y, yv, yt = (s[TARGET].to_numpy() for s in (split.train, split.valid, split.test))
    p, pv, pt = model.predict_pd(X), model.predict_pd(Xv), model.predict_pd(Xt)

    # The cutoff is set on the raw score, which ranks applicants without ties; the isotonic PD is
    # a step function, and a cutoff on it would approve every applicant on a step or none of them.
    cutoff = float(np.quantile(model.raw(Xv), policy.decision.approval_rate))
    approved = model.raw(Xt) < cutoff
    interval = metrics.bootstrap_auc(yt, pt, seed=seed)
    th = policy.thresholds

    declined = ~approved
    contributions = model.contributions(Xt)
    reason_lists = explain.reasons(contributions[declined])
    coverage = float(np.mean([bool(r) for r in reason_lists])) if reason_lists else 1.0
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.where(declined)[0], size=min(3, int(declined.sum())), replace=False)
    examples = [
        {
            "id": int(split.test["id"].iloc[i]),
            "pd": round(float(pt[i]), 4),
            "score": int(score_from_pd(pt[i : i + 1])[0]),
            "reasons": explain.reasons(contributions[i : i + 1])[0],
        }
        for i in sorted(pick)
    ]

    fair = fairness.assess(split.test, pt, approved, policy.fairness.reported, policy.fairness.min_group_size)
    stab = monitoring.stability(X, Xt, p, pt, policy.monitoring.psi_warn, policy.monitoring.psi_alert)
    mono = monotonicity(model, Xt.sample(n=min(sample, len(Xt)), random_state=seed))

    ev = ModelEvaluation(
        name=model.name,
        kind=model.kind,
        train_auc=metrics.auc(y, p),
        valid_auc=metrics.auc(yv, pv),
        test={
            "auc": interval.value,
            "gini": metrics.gini(yt, pt),
            "ks": metrics.ks(yt, pt),
            "brier": metrics.brier(yt, pt),
            "ece": float(metrics.ece(yt, pt)),
        },
        auc_interval=(interval.low, interval.high),
        cutoff=cutoff,
        pd_cutoff=float(model.calibrator(np.array([cutoff]))[0]),
        decisions={
            "approval_rate": float(approved.mean()),
            "bad_rate_approved": float(yt[approved].mean()),
            "bad_rate_declined": float(yt[declined].mean()) if declined.any() else 0.0,
        },
        deciles=metrics.deciles(yt, pt),
        fairness=fair,
        stability=stab,
        monotonicity=mono,
        reason_coverage=coverage,
        importance=explain.global_importance(contributions),
        examples=examples,
    )
    gated = [r for r in fair if r.attribute in policy.fairness.gated]
    air = min(r.adverse_impact_ratio for r in gated) if gated else 1.0
    ev.checks = [
        Check("discrimination: test AUC", ev.test["auc"], f">= {th.auc_min}", ev.test["auc"] >= th.auc_min),
        Check("discrimination: test KS", ev.test["ks"], f">= {th.ks_min}", ev.test["ks"] >= th.ks_min),
        Check("calibration: test ECE", ev.test["ece"], f"<= {th.ece_max}", ev.test["ece"] <= th.ece_max),
        Check(
            "overfitting: train AUC - test AUC",
            ev.train_auc - ev.test["auc"],
            f"<= {th.overfit_auc_gap_max}",
            ev.train_auc - ev.test["auc"] <= th.overfit_auc_gap_max,
        ),
        Check("leakage: test AUC not implausibly high", ev.test["auc"], f"<= {th.leakage_auc_max}", ev.test["auc"] <= th.leakage_auc_max),
        Check("stability: score PSI train vs test", stab[0].psi, f"<= {th.score_psi_max}", stab[0].psi <= th.score_psi_max),
        Check("conceptual soundness: monotonic in every constrained feature", float(sum(mono.values())), "== 0", sum(mono.values()) == 0),
        Check(
            "fairness: lowest adverse impact ratio (" + ", ".join(policy.fairness.gated) + ")",
            air,
            f">= {th.adverse_impact_min}",
            air >= th.adverse_impact_min,
        ),
        Check("explainability: declines with a reason code", coverage, f">= {th.reason_coverage_min}", coverage >= th.reason_coverage_min),
    ]
    return ev
