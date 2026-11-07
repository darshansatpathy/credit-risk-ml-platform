"""The model-risk policy file."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from creditrisk.resources import resource


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Decision(_Strict):
    approval_rate: float = Field(gt=0, lt=1)


class Thresholds(_Strict):
    auc_min: float
    ks_min: float
    ece_max: float
    overfit_auc_gap_max: float
    leakage_auc_max: float
    score_psi_max: float
    adverse_impact_min: float
    reason_coverage_min: float


class Fairness(_Strict):
    gated: list[str]
    reported: list[str]
    min_group_size: int


class Monitoring(_Strict):
    psi_warn: float
    psi_alert: float


class Policy(_Strict):
    decision: Decision
    thresholds: Thresholds
    fairness: Fairness
    monitoring: Monitoring


def load(path: Path | None = None) -> Policy:
    return Policy.model_validate(yaml.safe_load((path or resource("policy") / "model_risk.yaml").read_text()))
