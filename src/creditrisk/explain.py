"""Adverse-action reason codes from per-feature contributions."""

from __future__ import annotations

import numpy as np

from creditrisk.features import FEATURES


def reasons(contributions: np.ndarray, top: int = 4) -> list[list[str]]:
    """For each row, the reasons behind the features that raised its risk most, largest first."""
    out = []
    for row in contributions:
        order = np.argsort(-row, kind="stable")
        texts: list[str] = []
        for i in order:
            if row[i] <= 0 or len(texts) == top:
                break
            text = FEATURES[i].reason
            if text not in texts:
                texts.append(text)
        out.append(texts)
    return out


def global_importance(contributions: np.ndarray) -> list[tuple[str, float]]:
    """Mean absolute contribution per feature, largest first."""
    means = np.abs(contributions).mean(axis=0)
    return sorted(((f.name, float(m)) for f, m in zip(FEATURES, means, strict=True)), key=lambda t: -t[1])
