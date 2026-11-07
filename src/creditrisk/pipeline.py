"""Train the candidates, evaluate them against the policy, choose a champion, record everything."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from creditrisk import data, models
from creditrisk.data import TARGET, Split
from creditrisk.evaluation import ModelEvaluation, evaluate
from creditrisk.features import build
from creditrisk.models import RiskModel
from creditrisk.policy import Policy


@dataclass
class RunResult:
    split: Split
    candidates: list[RiskModel]
    evaluations: list[ModelEvaluation]
    champion: str | None
    run_ids: dict[str, str]

    def evaluation(self, name: str) -> ModelEvaluation:
        return next(e for e in self.evaluations if e.name == name)


def train_candidates(split: Split, seed: int = 7) -> list[RiskModel]:
    X, Xv = build(split.train), build(split.valid)
    y, yv = split.train[TARGET].to_numpy(), split.valid[TARGET].to_numpy()
    candidates: list[RiskModel] = [
        models.train_scorecard(X, y),
        models.train_lightgbm(X, y, Xv, yv, seed=seed),
        models.train_xgboost(X, y, Xv, yv, seed=seed),
    ]
    for m in candidates:
        m.calibrate(Xv, yv)
    return candidates


def choose(evaluations: list[ModelEvaluation]) -> str | None:
    """The champion is the eligible model with the best validation AUC. The test split is for
    reporting, never for choosing."""
    eligible = [e for e in evaluations if e.passed]
    return max(eligible, key=lambda e: e.valid_auc).name if eligible else None


def run(policy: Policy, out: Path, tracking_uri: str | None = None, seed: int = 42) -> RunResult:
    split = data.split(data.load(), seed=seed)
    candidates = train_candidates(split)
    evaluations = [evaluate(m, split, policy) for m in candidates]
    champion = choose(evaluations)
    out.mkdir(parents=True, exist_ok=True)
    for m in candidates:
        m.save(out / "models" / m.name)
    summary = {"champion": champion, "evaluations": [e.to_dict() for e in evaluations]}
    (out / "evaluation.json").write_text(json.dumps(summary, indent=1, default=_json))
    run_ids: dict[str, str] = {}
    if tracking_uri:
        from creditrisk.tracking import log_run

        run_ids = log_run(tracking_uri, out, candidates, evaluations, champion, policy)
    return RunResult(split, candidates, evaluations, champion, run_ids)


def _json(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)
