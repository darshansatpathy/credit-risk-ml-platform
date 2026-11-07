from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from creditrisk import models
from creditrisk.features import build
from creditrisk.pipeline import RunResult


def test_saved_models_reload_to_identical_predictions(run: RunResult, tmp_path: Path) -> None:
    X = build(run.split.test.head(500))
    for model in run.candidates:
        model.save(tmp_path / model.name)
        assert not list((tmp_path / model.name).glob("*.pkl"))
        again = models.load(tmp_path / model.name)
        assert np.allclose(again.predict_pd(X), model.predict_pd(X))
        assert np.allclose(again.contributions(X), model.contributions(X))


def test_load_refuses_mismatches(run: RunResult, tmp_path: Path) -> None:
    run.candidates[1].save(tmp_path)
    meta = json.loads((tmp_path / "model.json").read_text())
    (tmp_path / "model.json").write_text(json.dumps(meta | {"features": ["x"]}))
    with pytest.raises(ValueError, match="different features"):
        models.load(tmp_path)
    (tmp_path / "model.json").write_text(json.dumps(meta | {"kind": "neural"}))
    with pytest.raises(ValueError, match="unknown model kind"):
        models.load(tmp_path)


def test_score_scale_and_calibrator() -> None:
    assert models.score_from_pd(np.array([1 / 51]))[0] == 600
    assert models.score_from_pd(np.array([1 / 26]))[0] == 580  # odds halve, 20 points down
    assert np.array_equal(models.Calibrator()(np.array([0.5])), np.array([0.5]))


def test_scorecard_points_and_signs(run: RunResult) -> None:
    card = run.candidates[0]
    assert isinstance(card, models.Scorecard)
    assert all(c >= 0 for c in card.coefficients) and any(c == 0 for c in card.coefficients)
    table = card.points_table()
    assert {r["feature"] for r in table} >= {"recent_status", "limit_bal"}
    late = [r for r in table if r["feature"] == "recent_status"]
    assert late[0]["points"] > late[-1]["points"], "more recent delinquency earns fewer points"


def test_monotone_binning_merges_violations() -> None:
    rng = np.random.default_rng(1)
    x = rng.random(5000)
    y = (rng.random(5000) < 0.1 + 0.3 * x + 0.05 * np.sin(20 * x)).astype(int)
    edges, woe = models._monotone_bins(x, y, direction=1, max_bins=10)
    assert all(a >= b for a, b in itertools.pairwise(woe)), "woe falls as risk rises"
    assert len(edges) < 9
