# Model validation report: probability of default

**Champion: `lightgbm`.** Test AUC 0.768 (95% CI 0.755 to 0.784), KS 0.413, expected calibration error 0.024. All 9 policy checks pass.

| Model | Validation AUC | Test AUC (95% CI) | KS | Gini | Brier | ECE | Policy |
|---|---|---|---|---|---|---|---|
| scorecard | 0.776 | 0.743 (0.728 to 0.758) | 0.385 | 0.486 | 0.1413 | 0.0201 | fails discrimination |
| lightgbm (champion) | 0.802 | 0.768 (0.755 to 0.784) | 0.413 | 0.536 | 0.1382 | 0.0245 | pass |
| xgboost | 0.800 | 0.768 (0.755 to 0.783) | 0.415 | 0.536 | 0.1379 | 0.0230 | pass |

Validation AUC is higher than test AUC for every model because the validation split is used for early stopping and calibration. The test split is used for nothing but this report.

## 1. Purpose and use

Estimate the probability that a credit card account defaults on next month's payment, and decide approvals at a cutoff set so that 80.0% of the validation population is approved. Every decline carries adverse-action reasons.

## 2. Data

UCI Default of Credit Card Clients (Yeh, 2009; CC BY 4.0): 30,000 accounts, default rate 22.1%. Stratified split: 18,000 train, 6,000 validation, 6,000 test. Rows are validated on load (columns, nulls, code ranges, duplicate ids) and rejected rather than scored if they fail.

Sex, age, marriage, education are not model inputs. They are used only to test outcomes for fairness.

## 3. Methodology

| Feature | Meaning | Constraint |
|---|---|---|
| `limit_bal` | Credit limit | risk falls with it |
| `recent_status` | Repayment status last month (-2 no use, -1 paid in full, 0 revolving, n months late) | risk rises with it |
| `max_delinquency_6m` | Worst months-late status in the last six months | risk rises with it |
| `months_delinquent_6m` | Months with a late payment in the last six months | risk rises with it |
| `months_paid_full_6m` | Months paid in full in the last six months | risk falls with it |
| `utilization` | Last bill as a share of the limit | risk rises with it |
| `avg_utilization_6m` | Average bill as a share of the limit over six months | risk rises with it |
| `utilization_trend` | Change in utilization over six months | risk rises with it |
| `payment_ratio` | Last payment as a share of the previous bill | risk falls with it |
| `avg_payment_ratio_6m` | Average payment as a share of the previous bill | risk falls with it |
| `paid_to_limit_6m` | Total paid in six months as a share of the limit | risk falls with it |
| `bill_amt1` | Last bill amount | none |

- **Scorecard:** quantile bins merged until the default rate moves in the declared direction, weight of evidence, logistic regression with wrong-signed characteristics removed, points at 600 for 50:1 odds and 20 points to double the odds.
- **LightGBM and XGBoost:** gradient boosting with the monotone constraints above, early stopping on validation.
- **Calibration:** isotonic regression on the validation split, stored as its knots.
- **Selection:** the eligible model (all policy checks pass) with the best validation AUC.

## 4. Performance of the champion

| Risk band | Accounts | Predicted default | Observed default |
|---|---|---|---|
| 1 | 600 | 72.4% | 67.3% |
| 2 | 600 | 43.9% | 42.5% |
| 3 | 600 | 28.3% | 24.8% |
| 4 | 600 | 19.4% | 19.7% |
| 5 | 600 | 15.8% | 17.8% |
| 6 | 600 | 12.9% | 15.5% |
| 7 | 600 | 10.4% | 11.3% |
| 8 | 600 | 7.4% | 9.7% |
| 9 | 600 | 5.4% | 7.8% |
| 10 | 600 | 3.1% | 4.7% |

At the cutoff (raw score -0.684, PD about 0.315): 80.6% of test applicants are approved; 13.9% of approved and 56.3% of declined applicants defaulted.

## 5. Conceptual soundness

Each constrained feature was swept across its 1st to 99th percentile for 300 test applicants, holding everything else fixed. Applicants whose PD moved against the declared direction:

| Model | `limit_bal` | `recent_status` | `max_delinquency_6m` | `months_delinquent_6m` | `months_paid_full_6m` | `utilization` | `avg_utilization_6m` | `utilization_trend` | `payment_ratio` | `avg_payment_ratio_6m` | `paid_to_limit_6m` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scorecard | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lightgbm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| xgboost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Global importance of the champion (mean absolute contribution, log-odds):

1. `max_delinquency_6m` 0.344
2. `recent_status` 0.313
3. `bill_amt1` 0.290
4. `limit_bal` 0.187
5. `months_delinquent_6m` 0.132
6. `paid_to_limit_6m` 0.113
7. `utilization` 0.111
8. `avg_payment_ratio_6m` 0.061
9. `avg_utilization_6m` 0.051
10. `payment_ratio` 0.040
11. `utilization_trend` 0.037
12. `months_paid_full_6m` 0.014

## 6. Stability

Population stability index, train against test:

| Variable | PSI | Status |
|---|---|---|
| score (PD) | 0.0048 | stable |
| limit_bal | 0.0025 | stable |
| recent_status | 0.0003 | stable |
| max_delinquency_6m | 0.0002 | stable |
| months_delinquent_6m | 0.0006 | stable |
| months_paid_full_6m | 0.0005 | stable |
| utilization | 0.0031 | stable |
| avg_utilization_6m | 0.0029 | stable |
| utilization_trend | 0.0032 | stable |
| payment_ratio | 0.0023 | stable |
| avg_payment_ratio_6m | 0.0023 | stable |
| paid_to_limit_6m | 0.0025 | stable |
| bill_amt1 | 0.0036 | stable |

**Stress scenario** (balances +25%, payments -35%, one month more delinquency for a fifth of accounts): score PSI 0.267, 3 feature alerts and 5 warnings; mean PD 21.9% → 28.2%; approvals at the same cutoff 80.6% → 71.7%. Monitoring at the policy thresholds (warn 0.1, alert 0.25) flags this population.

## 7. Fairness

Outcomes at the cutoff, test split, groups of at least 200. Adverse impact ratio: lowest approval rate divided by highest. Equal opportunity gap: spread of approval rates among applicants who did not default. Gated attributes: sex, age_band (minimum 0.8).

**sex**: adverse impact ratio 0.965, equal opportunity gap 0.016

| Group | Accounts | Approved | Approved among non-defaulters | Mean PD | Default rate |
|---|---|---|---|---|---|
| female | 3,620 | 81.7% | 89.7% | 21.1% | 20.9% |
| male | 2,380 | 78.9% | 88.2% | 23.2% | 24.0% |

**age_band**: adverse impact ratio 0.851, equal opportunity gap 0.067

| Group | Accounts | Approved | Approved among non-defaulters | Mean PD | Default rate |
|---|---|---|---|---|---|
| 25-34 | 2,634 | 82.0% | 89.6% | 20.7% | 20.2% |
| 35-49 | 2,316 | 81.7% | 89.9% | 21.3% | 22.5% |
| 50 and over | 504 | 80.4% | 89.5% | 22.3% | 24.6% |
| under 25 | 546 | 69.8% | 83.1% | 29.9% | 27.3% |

**education**: adverse impact ratio 0.900, equal opportunity gap 0.066

| Group | Accounts | Approved | Approved among non-defaulters | Mean PD | Default rate |
|---|---|---|---|---|---|
| graduate school | 2,142 | 85.2% | 92.3% | 18.6% | 20.4% |
| high school | 1,022 | 76.6% | 85.8% | 25.4% | 25.1% |
| university | 2,746 | 78.1% | 87.5% | 23.4% | 22.8% |

**marriage**: adverse impact ratio 0.990, equal opportunity gap 0.005

| Group | Accounts | Approved | Approved among non-defaulters | Mean PD | Default rate |
|---|---|---|---|---|---|
| married | 2,676 | 80.2% | 88.9% | 21.8% | 23.4% |
| single | 3,246 | 81.0% | 89.4% | 21.9% | 21.2% |

## 8. Explainability

100.0% of declines carry at least one reason. Examples:

- Account 14796: PD 0.374, score 502: Low credit limit relative to other customers; Serious delinquency in the last six months; Large outstanding balance; Sustained high utilization
- Account 17711: PD 0.717, score 460: Recent payment delinquency; Serious delinquency in the last six months; High credit utilization; Persistently low payments
- Account 24200: PD 0.482, score 489: Recent payment delinquency; Serious delinquency in the last six months; Large outstanding balance; Little repaid relative to the limit

## 9. Policy checks

| Check | Threshold | scorecard | lightgbm | xgboost |
|---|---|---|---|---|
| discrimination: test AUC | >= 0.75 | 0.7429 **fail** | 0.7680 pass | 0.7678 pass |
| discrimination: test KS | >= 0.38 | 0.3849 pass | 0.4128 pass | 0.4150 pass |
| calibration: test ECE | <= 0.03 | 0.0201 pass | 0.0245 pass | 0.0230 pass |
| overfitting: train AUC - test AUC | <= 0.05 | 0.0247 pass | 0.0331 pass | 0.0317 pass |
| leakage: test AUC not implausibly high | <= 0.95 | 0.7429 pass | 0.7680 pass | 0.7678 pass |
| stability: score PSI train vs test | <= 0.1 | 0.0018 pass | 0.0048 pass | 0.0028 pass |
| conceptual soundness: monotonic in every constrained feature | == 0 | 0.0000 pass | 0.0000 pass | 0.0000 pass |
| fairness: lowest adverse impact ratio (sex, age_band) | >= 0.8 | 0.8328 pass | 0.8513 pass | 0.8422 pass |
| explainability: declines with a reason code | >= 1.0 | 1.0000 pass | 1.0000 pass | 1.0000 pass |

## 10. Limitations

- One public dataset from one lender and one period (2005); no out-of-time validation is possible with it.
- The adverse impact ratio for applicants under 25 is the closest to its threshold. Their lower approval rate follows lower credit limits and more recent delinquency, which are legitimate inputs, but it should be reviewed with every retraining.
- Reject inference is out of scope: the data contains only booked accounts.
- Isotonic calibration is piecewise constant, so applicants close together can share a PD. Decisions are made on the raw score, which has no ties; the PD is for pricing, provisioning and reporting.

## 11. Monitoring plan

- Monthly: score and feature PSI against the training population; warn at 0.1, alert at 0.25 (`crisk monitor`).
- Quarterly, once outcomes mature: AUC, KS and calibration by risk band; adverse impact by gated attribute.
- Retrain or recalibrate when score PSI alerts, AUC falls below the policy minimum, or calibration error exceeds it.
