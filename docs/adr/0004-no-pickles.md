# ADR-0004: Models are stored as text and JSON, not pickles

**Status:** accepted

## Context

A pickle runs code when it is loaded, ties a model to library versions, and cannot be read by a
reviewer.

## Decision

Store LightGBM as its text model, XGBoost as JSON, the scorecard as bins, WoE, coefficients and
points, and calibration as isotonic knots. `model.json` records the feature list, checked on
load.

## Consequences

- A validator can read the scorecard and the calibration directly.
- Loading a model executes nothing; a model trained on different features is refused.
