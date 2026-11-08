# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-11-08

### Fixed

- `ScorecardModel.raw` returns a float array explicitly, so mypy passes with the NumPy stubs installed on Python 3.11.

## [0.1.0] - 2026-11-08

### Added

- Data validation and a stratified train, validation and test split of the UCI credit card
  default data.
- Twelve features with declared risk directions and adverse-action wording.
- Scorecard (monotone WoE bins, sign-constrained logistic regression, points), monotone
  LightGBM and XGBoost, isotonic calibration, plain-text persistence.
- Nine-check model-risk policy, champion selection on validation, MLflow runs and registry.
- Fairness by sex, age band, education and marital status; population stability and a stress
  scenario; TreeSHAP reason codes.
- Generated model validation report, scoring API and the `crisk` command.
