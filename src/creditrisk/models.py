"""Three model families behind one interface: a points scorecard, LightGBM and XGBoost.

All three are monotone in the direction each feature declares, are calibrated on the validation
split, and are saved as plain text or JSON (no pickles), so a stored model can be inspected and
loaded without executing anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditrisk.features import FEATURES, MONOTONE, NAMES

PDO = 20.0  # points to double the odds
BASE_SCORE, BASE_ODDS = 600.0, 50.0  # 600 points at 50:1 good:bad
FACTOR = PDO / math.log(2)
OFFSET = BASE_SCORE - FACTOR * math.log(BASE_ODDS)


def score_from_pd(pd_: np.ndarray) -> np.ndarray:
    p = np.clip(pd_, 1e-6, 1 - 1e-6)
    return np.asarray(np.round(OFFSET + FACTOR * np.log((1 - p) / p)), dtype=int)


# ---- calibration ---------------------------------------------------------------------------


@dataclass
class Calibrator:
    """Isotonic map from a raw model score to a probability of default, stored as its knots."""

    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, raw: np.ndarray, target: np.ndarray) -> Calibrator:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(raw, target)
        return cls([float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_])

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        if not self.x:
            return raw
        return np.asarray(np.interp(raw, self.x, self.y), dtype=float)


# ---- the common interface ---------------------------------------------------------------------


class RiskModel:
    kind = "base"

    def __init__(self, name: str) -> None:
        self.name = name
        self.calibrator = Calibrator()

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature contributions to risk (log-odds or points), positive means riskier."""
        raise NotImplementedError

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator(self.raw(X[NAMES]))

    def calibrate(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.calibrator = Calibrator.fit(self.raw(X[NAMES]), y)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "calibration.json").write_text(json.dumps({"x": self.calibrator.x, "y": self.calibrator.y}))
        (directory / "model.json").write_text(json.dumps({"kind": self.kind, "name": self.name, "features": NAMES}))
        self._save(directory)

    def _save(self, directory: Path) -> None:
        raise NotImplementedError


def load(directory: Path) -> RiskModel:
    meta = json.loads((directory / "model.json").read_text())
    if meta["features"] != NAMES:
        raise ValueError(f"model {meta['name']} was trained on different features")
    model: RiskModel
    if meta["kind"] == "lightgbm":
        model = LightGBMModel(meta["name"], lgb.Booster(model_file=str(directory / "booster.txt")))
    elif meta["kind"] == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(directory / "booster.json"))
        model = XGBoostModel(meta["name"], booster)
    elif meta["kind"] == "scorecard":
        model = Scorecard.from_dict(meta["name"], json.loads((directory / "scorecard.json").read_text()))
    else:
        raise ValueError(f"unknown model kind {meta['kind']}")
    cal = json.loads((directory / "calibration.json").read_text())
    model.calibrator = Calibrator(cal["x"], cal["y"])
    return model


# ---- gradient boosting -----------------------------------------------------------------------


class LightGBMModel(RiskModel):
    kind = "lightgbm"

    def __init__(self, name: str, booster: lgb.Booster) -> None:
        super().__init__(name)
        self.booster = booster

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(X[NAMES], raw_score=True), dtype=float)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        # TreeSHAP values in log-odds; the last column is the expected value.
        return np.asarray(self.booster.predict(X[NAMES], pred_contrib=True), dtype=float)[:, :-1]

    def _save(self, directory: Path) -> None:
        self.booster.save_model(str(directory / "booster.txt"))


def train_lightgbm(
    X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray, monotone: bool = True, seed: int = 7, name: str = "lightgbm"
) -> LightGBMModel:
    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 2,
        "verbose": -1,
    }
    if monotone:
        params |= {"monotone_constraints": MONOTONE, "monotone_constraints_method": "advanced"}
    booster = lgb.train(
        params,
        lgb.Dataset(X[NAMES], y),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xv[NAMES], yv)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return LightGBMModel(name, booster)


class XGBoostModel(RiskModel):
    kind = "xgboost"

    def __init__(self, name: str, booster: xgb.Booster) -> None:
        super().__init__(name)
        self.booster = booster

    def _matrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X[NAMES], feature_names=NAMES)

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(self._matrix(X), output_margin=True, iteration_range=self._range()), dtype=float)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(self._matrix(X), pred_contribs=True, iteration_range=self._range()), dtype=float)[:, :-1]

    def _range(self) -> tuple[int, int]:
        best = self.booster.attributes().get("best_iteration")
        return (0, int(best) + 1) if best is not None else (0, 0)

    def _save(self, directory: Path) -> None:
        self.booster.save_model(str(directory / "booster.json"))


def train_xgboost(X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray, seed: int = 7, name: str = "xgboost") -> XGBoostModel:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 5.0,
        "tree_method": "hist",
        "seed": seed,
        "nthread": 2,
        "monotone_constraints": "(" + ",".join(str(m) for m in MONOTONE) + ")",
    }
    dtrain = xgb.DMatrix(X[NAMES], y, feature_names=NAMES)
    dvalid = xgb.DMatrix(Xv[NAMES], yv, feature_names=NAMES)
    booster = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dvalid, "valid")], early_stopping_rounds=100, verbose_eval=False)
    return XGBoostModel(name, booster)


# ---- scorecard ---------------------------------------------------------------------------------


@dataclass
class Characteristic:
    feature: str
    edges: list[float]  # upper bounds of each bin except the last (which is open)
    woe: list[float]
    points: list[float] = field(default_factory=list)

    def bin_of(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges), values, side="left")


class Scorecard(RiskModel):
    """Binned features, weight of evidence, a logistic regression, and points per bin."""

    kind = "scorecard"

    def __init__(self, name: str, characteristics: list[Characteristic], intercept: float, coefficients: list[float]) -> None:
        super().__init__(name)
        self.characteristics = characteristics
        self.intercept = intercept
        self.coefficients = coefficients

    def _woe(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.characteristics:
            cols.append(np.asarray(c.woe)[c.bin_of(X[c.feature].to_numpy(dtype=float))])
        return np.column_stack(cols)

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        # Log-odds of default. WoE is defined as ln(good share / bad share), so it enters negatively.
        return np.asarray(self.intercept - self._woe(X) @ np.asarray(self.coefficients), dtype=float)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Points below the best attainable for each characteristic: a scorecard's reason codes."""
        out = []
        for c in self.characteristics:
            pts = np.asarray(c.points, dtype=float)
            out.append(float(np.max(pts)) - pts[c.bin_of(X[c.feature].to_numpy(dtype=float))])
        return np.column_stack(out)

    def points_table(self) -> list[dict[str, Any]]:
        rows = []
        for c in self.characteristics:
            lows = [-math.inf, *c.edges]
            highs = [*c.edges, math.inf]
            for lo, hi, w, p in zip(lows, highs, c.woe, c.points, strict=True):
                rows.append({"feature": c.feature, "from": lo, "to": hi, "woe": round(w, 4), "points": round(p, 1)})
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "characteristics": [{"feature": c.feature, "edges": c.edges, "woe": c.woe, "points": c.points} for c in self.characteristics],
        }

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> Scorecard:
        chars = [Characteristic(c["feature"], c["edges"], c["woe"], c["points"]) for c in d["characteristics"]]
        return cls(name, chars, d["intercept"], d["coefficients"])

    def _save(self, directory: Path) -> None:
        (directory / "scorecard.json").write_text(json.dumps(self.to_dict(), indent=1))


def _monotone_bins(values: np.ndarray, target: np.ndarray, direction: int, max_bins: int = 6) -> tuple[list[float], list[float]]:
    """Quantile bins, merged until the bad rate moves in the declared direction (pool adjacent violators)."""
    edges = sorted(set(np.quantile(values, np.linspace(0, 1, max_bins + 1))[1:-1].tolist()))
    while True:
        idx = np.searchsorted(np.asarray(edges), values, side="left")
        counts = np.bincount(idx, minlength=len(edges) + 1)
        bads = np.bincount(idx, weights=target, minlength=len(edges) + 1)
        keep = counts > 0
        rates = bads[keep] / counts[keep]
        diffs = np.diff(rates)
        sign = direction if direction != 0 else (1 if rates[-1] >= rates[0] else -1)
        violations = np.where(diffs * sign < 0)[0] if len(edges) else np.array([], dtype=int)
        if len(violations) == 0 and keep.all():
            break
        if not keep.all():
            empty = int(np.where(~keep)[0][0])
            edges.pop(min(empty, len(edges) - 1))
            continue
        edges.pop(int(violations[0]))
    total_good, total_bad = (target == 0).sum(), (target == 1).sum()
    woe = []
    for b in range(len(edges) + 1):
        mask = idx == b
        good = ((target == 0) & mask).sum() + 0.5
        bad = ((target == 1) & mask).sum() + 0.5
        woe.append(float(math.log((good / total_good) / (bad / total_bad))))
    return [float(e) for e in edges], woe


def train_scorecard(X: pd.DataFrame, y: np.ndarray, name: str = "scorecard") -> Scorecard:
    chars = []
    for f in FEATURES:
        edges, bin_woe = _monotone_bins(X[f.name].to_numpy(dtype=float), y, f.direction)
        chars.append(Characteristic(f.name, edges, bin_woe))
    card = Scorecard(name, chars, 0.0, [0.0] * len(chars))
    woe = card._woe(X)
    # With correlated characteristics a coefficient can take the wrong sign, which would make
    # points move against the bin's weight of evidence. Drop those characteristics and refit
    # until every coefficient points the right way (the usual scorecard practice).
    active = list(range(len(chars)))
    while True:
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(woe[:, active], y)
        # sklearn models P(default) = sigmoid(b0 + b.w); our raw() is b0 - c.w, so c = -b.
        coefs = -lr.coef_[0]
        wrong = [active[i] for i, c in enumerate(coefs) if c < 0]
        if not wrong:
            break
        active.remove(wrong[0])
    card.intercept = float(lr.intercept_[0])
    card.coefficients = [0.0] * len(chars)
    for i, c in zip(active, coefs, strict=True):
        card.coefficients[i] = float(c)
    n = len(chars)
    for c, beta in zip(chars, card.coefficients, strict=True):
        # Points per bin: the share of the score this characteristic contributes (standard scaling).
        c.points = [float(FACTOR * (beta * w) + (OFFSET - FACTOR * card.intercept) / n) for w in c.woe]
    return card
