# Credit Risk ML Platform

[![CI](https://github.com/darshansatpathy/credit-risk-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/darshansatpathy/credit-risk-ml-platform/actions/workflows/ci.yml)
[![CodeQL](https://github.com/darshansatpathy/credit-risk-ml-platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/darshansatpathy/credit-risk-ml-platform/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Test AUC](https://img.shields.io/badge/test%20AUC-0.768-success)
![Policy checks](https://img.shields.io/badge/model--risk%20checks-9%2F9-success)
![License](https://img.shields.io/badge/license-MIT-green)

**Probability-of-default models built the way a bank's model-risk team would ask for them:
monotone in every input, calibrated, explained per decision, tested for disparate impact,
monitored for drift, and documented in a validation report generated from the evidence.**

```
data ─▶ validate ─▶ 12 features ──┬─▶ scorecard (binned WoE, points)  ─┐
         (reject,   (directions   ├─▶ LightGBM (monotone)             ├─▶ isotonic calibration ─▶ policy gate ─▶ champion
          not       declared)     └─▶ XGBoost  (monotone)             ─┘   (validation split)     (9 checks)     (MLflow alias)
          score)                                                                                        │
                                                       model validation report ◀────────────────────────┤
                                                       scoring API: PD, score, decision, reasons ◀───────┘
```

## Results

On the [UCI Default of Credit Card Clients](data/README.md) data (30,000 accounts, 22.1%
default), with a test split that is used for nothing but reporting
([full validation report](docs/model-validation-report.md)):

| Model | Test AUC (95% CI) | KS | Brier | Calibration error | Model-risk policy |
|---|---|---|---|---|---|
| Scorecard (benchmark) | 0.743 (0.728 to 0.758) | 0.385 | 0.1413 | 0.020 | fails the AUC minimum (0.75) |
| **LightGBM (champion)** | **0.768 (0.755 to 0.784)** | **0.413** | **0.1382** | **0.024** | **9 / 9** |
| XGBoost | 0.768 (0.755 to 0.783) | 0.415 | 0.1379 | 0.023 | 9 / 9 |

At the policy's 80% approval rate, 80.6% of test applicants are approved. 13.9% of those approved
defaulted, against 56.3% of those declined. The ten risk bands run from 4.7% to 67.3% observed
default, tracking predicted rates.

The interpretable scorecard falls 0.025 AUC short of the boosted models. That is the price of
interpretability on this data, and the report states it so the trade-off is a decision rather
than an accident.

## What the policy checks

Every candidate goes through the same nine checks from [`policy/model_risk.yaml`](policy/model_risk.yaml).
The champion is the eligible model with the best *validation* AUC; the test split never chooses.

| Check | What it guards against | Champion |
|---|---|---|
| Test AUC ≥ 0.75, KS ≥ 0.38 | A model too weak to justify its use | 0.768, 0.413 |
| Expected calibration error ≤ 0.03 | PDs that are wrong for pricing and provisioning | 0.024 |
| Train minus test AUC ≤ 0.05 | Overfitting | 0.033 |
| Test AUC ≤ 0.95 | Target leakage (no honest credit model scores that high) | pass |
| Score PSI train vs test ≤ 0.10 | An unrepresentative split | 0.005 |
| Zero monotonicity violations | Risk falling as delinquency rises, which no reviewer would accept | 0 of 300 applicants × 11 features |
| Adverse impact ratio ≥ 0.80 (sex, age) | Disparate impact | 0.965 (sex), 0.851 (age) |
| Every decline has a reason | Adverse-action notices without reasons | 100% |

The gate is tested against planted faults: an unconstrained model fails conceptual soundness,
inflated probabilities fail calibration, a leaked target fails the leakage check, skewed
approvals fail fairness, and a model with no explanations fails explainability.

## Design choices a validator would ask about

- **Protected attributes are not inputs.** Sex, age, marital status and education are kept
  beside the scores only to test outcomes. Age is the tightest: applicants under 25 are
  approved at 85% of the rate of the best-approved band, driven by lower limits and recent
  delinquency. The report flags it for review at every retraining.
- **Monotone by construction.** Each feature declares the direction a credit officer would
  expect. The boosted models enforce it with monotone constraints; the scorecard merges bins
  until default rates move the declared way and drops characteristics whose coefficient takes
  the wrong sign. The gate then checks it independently by sweeping each feature.
- **Decisions on the raw score, PDs from calibration.** Isotonic calibration is a step function
  with about 30 distinct values on this data, so a PD cutoff would approve whole steps at once:
  76% instead of 80%. The cutoff is set on the raw score, which has no ties.
- **Reason codes from TreeSHAP.** Per-feature contributions from LightGBM and XGBoost's built-in
  TreeSHAP, or points below the maximum for the scorecard, mapped to adverse-action wording.
- **No pickles.** Models are saved as LightGBM text, XGBoost JSON and scorecard JSON, with the
  calibration as its knots. A stored model can be read by a person and loaded without executing
  anything.
- **Stress tested.** A downturn scenario (balances +25%, payments −35%, more delinquency) moves
  score PSI to 0.267, raising alerts at the policy thresholds. Mean PD rises from 21.9% to 28.2%,
  and approvals fall from 80.6% to 71.7% at the same cutoff.

## Run it

```bash
pip install -e ".[dev]"
crisk train --report model-report.md --tracking-uri sqlite:///artifacts/mlflow.db --check
crisk score applications.csv          # PD, score, decision and reasons per line (JSON)
crisk monitor this_month.csv --check  # PSI of the score and every feature; exit 1 on an alert
crisk serve                           # POST /score
```

```bash
curl -s localhost:8000/score -H 'Content-Type: application/json' -d '{
  "limit_bal": 20000, "pay_status": [3, 2, 2, 2, 0, 0],
  "bill_amounts": [19500, 19500, 19500, 19500, 19500, 19500], "payments": [0, 0, 0, 0, 0, 0]}'
# {"model": "lightgbm", "pd": ..., "score": ..., "decision": "decline",
#  "reasons": ["Recent payment delinquency", ...]}
```

Training all three models and writing the report takes about ten seconds on a laptop. The
champion is registered in MLflow as `credit-risk-pd` with the alias `champion`, alongside a run
per candidate with its metrics, check results and model files.

## Engineering

- 22 tests, 99% line coverage (gate: 90%), including the planted-fault tests above, a scoring
  API test and an MLflow registry test.
- `ruff`, `mypy --strict`, Python 3.11 to 3.13.
- CI:
  - lint, types and tests on three Python versions;
  - the model-risk gate, with the validation report in the job summary;
  - a wheel install outside the repository that trains from the bundled data and policy;
  - CodeQL, pip-audit and Dependabot.
- Design notes: [architecture](docs/architecture.md) and [decision records](docs/adr).

## Layout

```
data/                   the dataset and its licence
policy/model_risk.yaml  decision rule, policy thresholds, fairness and monitoring settings
src/creditrisk/
  data.py               load, validate, stratified split
  features.py           12 features with declared risk directions and reason wording
  models.py             scorecard, LightGBM, XGBoost, isotonic calibration, plain-text persistence
  evaluation.py         the nine policy checks, deciles, monotonicity sweep
  fairness.py           approval rates, adverse impact, equal opportunity by group
  monitoring.py         population stability, stress scenario
  explain.py            reason codes, global importance
  pipeline.py           train, evaluate, choose, save
  tracking.py           MLflow runs and the model registry
  report.py             the model validation report
  api.py, cli.py        scoring API and the crisk command
```

## Known limits

- One public dataset from one lender in 2005. There are no dates in it, so validation is
  out-of-sample but not out-of-time.
- Booked accounts only: no reject inference.
- The adverse impact ratio is an outcome test at one cutoff. It does not establish that the
  model is fair in every sense, and the report shows the other measures.

## License

MIT for the code. See [LICENSE](LICENSE). The data is CC BY 4.0; see [data/README.md](data/README.md).
