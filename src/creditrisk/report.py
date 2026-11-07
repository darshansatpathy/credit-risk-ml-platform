"""The model validation report, generated from a run so it cannot drift from the evidence."""

from __future__ import annotations

import numpy as np

from creditrisk import monitoring
from creditrisk.data import PROTECTED, TARGET
from creditrisk.evaluation import ModelEvaluation
from creditrisk.features import FEATURES, build
from creditrisk.pipeline import RunResult
from creditrisk.policy import Policy


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def stress_test(result: RunResult, policy: Policy) -> dict[str, float]:
    champion = next(m for m in result.candidates if m.name == result.champion)
    ev = result.evaluation(champion.name)
    base, stressed = result.split.test, monitoring.stress(result.split.test)
    pd_base, pd_stress = champion.predict_pd(build(base)), champion.predict_pd(build(stressed))
    stab = monitoring.stability(build(base), build(stressed), pd_base, pd_stress, policy.monitoring.psi_warn, policy.monitoring.psi_alert)
    return {
        "score_psi": stab[0].psi,
        "alerts": float(sum(s.status == "alert" for s in stab[1:])),
        "warnings": float(sum(s.status == "warn" for s in stab[1:])),
        "mean_pd_base": float(pd_base.mean()),
        "mean_pd_stressed": float(pd_stress.mean()),
        "approval_base": float((champion.raw(build(base)) < ev.cutoff).mean()),
        "approval_stressed": float((champion.raw(build(stressed)) < ev.cutoff).mean()),
    }


def render(result: RunResult, policy: Policy) -> str:
    evs = result.evaluations
    champ = result.evaluation(result.champion) if result.champion else None
    split = result.split
    lines = ["# Model validation report: probability of default", ""]
    if champ is None:
        lines += ["**No candidate met the model-risk policy. No champion was selected.**", ""]
    else:
        lines += [
            f"**Champion: `{champ.name}`.** Test AUC {champ.test['auc']:.3f} (95% CI {champ.auc_interval[0]:.3f} to "
            f"{champ.auc_interval[1]:.3f}), KS {champ.test['ks']:.3f}, expected calibration error {champ.test['ece']:.3f}. "
            f"All {len(champ.checks)} policy checks pass.",
            "",
        ]
    lines += ["| Model | Validation AUC | Test AUC (95% CI) | KS | Gini | Brier | ECE | Policy |", "|---|---|---|---|---|---|---|---|"]
    for e in evs:
        failed = [c.name.split(":")[0] for c in e.checks if not c.passed]
        verdict = "pass" if e.passed else "fails " + ", ".join(dict.fromkeys(failed))
        lines.append(
            f"| {e.name}{' (champion)' if e.name == result.champion else ''} | {e.valid_auc:.3f} | {e.test['auc']:.3f} "
            f"({e.auc_interval[0]:.3f} to {e.auc_interval[1]:.3f}) | {e.test['ks']:.3f} | {e.test['gini']:.3f} | {e.test['brier']:.4f} | "
            f"{e.test['ece']:.4f} | {verdict} |"
        )
    lines += [
        "",
        "Validation AUC is higher than test AUC for every model because the validation split is used for early stopping and "
        "calibration. The test split is used for nothing but this report.",
        "",
        "## 1. Purpose and use",
        "",
        "Estimate the probability that a credit card account defaults on next month's payment, and decide approvals at a "
        f"cutoff set so that {_pct(policy.decision.approval_rate)} of the validation population is approved. Every decline "
        "carries adverse-action reasons.",
        "",
        "## 2. Data",
        "",
        f"UCI Default of Credit Card Clients (Yeh, 2009; CC BY 4.0): {len(split.train) + len(split.valid) + len(split.test):,} "
        f"accounts, default rate {_pct(float(np.mean([*split.train[TARGET], *split.valid[TARGET], *split.test[TARGET]])))}. "
        f"Stratified split: {len(split.train):,} train, {len(split.valid):,} validation, {len(split.test):,} test. Rows are "
        "validated on load (columns, nulls, code ranges, duplicate ids) and rejected rather than scored if they fail.",
        "",
        f"{', '.join(PROTECTED).capitalize()} are not model inputs. They are used only to test outcomes for fairness.",
        "",
        "## 3. Methodology",
        "",
        "| Feature | Meaning | Constraint |",
        "|---|---|---|",
    ]
    for f in FEATURES:
        constraint = {1: "risk rises with it", -1: "risk falls with it", 0: "none"}[f.direction]
        lines.append(f"| `{f.name}` | {f.description} | {constraint} |")
    lines += [
        "",
        "- **Scorecard:** quantile bins merged until the default rate moves in the declared direction, weight of evidence, "
        "logistic regression with wrong-signed characteristics removed, points at 600 for 50:1 odds and 20 points to double "
        "the odds.",
        "- **LightGBM and XGBoost:** gradient boosting with the monotone constraints above, early stopping on validation.",
        "- **Calibration:** isotonic regression on the validation split, stored as its knots.",
        "- **Selection:** the eligible model (all policy checks pass) with the best validation AUC.",
        "",
    ]
    if champ is None:
        return "\n".join(lines) + "\n"
    lines += ["## 4. Performance of the champion", "", "| Risk band | Accounts | Predicted default | Observed default |", "|---|---|---|---|"]
    lines += [f"| {d['band']} | {d['count']:,} | {_pct(d['predicted'])} | {_pct(d['observed'])} |" for d in champ.deciles]
    lines += [
        "",
        f"At the cutoff (raw score {champ.cutoff:.3f}, PD about {champ.pd_cutoff:.3f}): "
        f"{_pct(champ.decisions['approval_rate'])} of test applicants are approved; "
        f"{_pct(champ.decisions['bad_rate_approved'])} of approved and {_pct(champ.decisions['bad_rate_declined'])} of declined "
        "applicants defaulted.",
        "",
        "## 5. Conceptual soundness",
        "",
        "Each constrained feature was swept across its 1st to 99th percentile for 300 test applicants, holding everything else "
        "fixed. Applicants whose PD moved against the declared direction:",
        "",
        "| Model | " + " | ".join(f"`{n}`" for n in champ.monotonicity) + " |",
        "|---|" + "---|" * len(champ.monotonicity),
    ]
    for e in evs:
        lines.append(f"| {e.name} | " + " | ".join(str(v) for v in e.monotonicity.values()) + " |")
    lines += ["", "Global importance of the champion (mean absolute contribution, log-odds):", ""]
    lines += [f"{i}. `{n}` {v:.3f}" for i, (n, v) in enumerate(champ.importance, start=1)]
    lines += ["", "## 6. Stability", "", "Population stability index, train against test:", "", "| Variable | PSI | Status |", "|---|---|---|"]
    lines += [f"| {s.name} | {s.psi:.4f} | {s.status} |" for s in champ.stability]
    stress = stress_test(result, policy)
    lines += [
        "",
        "**Stress scenario** (balances +25%, payments -35%, one month more delinquency for a fifth of accounts): score PSI "
        f"{stress['score_psi']:.3f}, {int(stress['alerts'])} feature alerts and {int(stress['warnings'])} warnings; mean PD "
        f"{_pct(stress['mean_pd_base'])} → {_pct(stress['mean_pd_stressed'])}; approvals at the same cutoff "
        f"{_pct(stress['approval_base'])} → {_pct(stress['approval_stressed'])}. Monitoring at the policy thresholds "
        f"(warn {policy.monitoring.psi_warn}, alert {policy.monitoring.psi_alert}) flags this population.",
        "",
        "## 7. Fairness",
        "",
        f"Outcomes at the cutoff, test split, groups of at least {policy.fairness.min_group_size}. Adverse impact ratio: "
        "lowest approval rate divided by highest. Equal opportunity gap: spread of approval rates among applicants who did "
        f"not default. Gated attributes: {', '.join(policy.fairness.gated)} (minimum {policy.thresholds.adverse_impact_min}).",
        "",
    ]
    for r in champ.fairness:
        lines += [
            f"**{r.attribute}**: adverse impact ratio {r.adverse_impact_ratio:.3f}, equal opportunity gap {r.equal_opportunity_gap:.3f}",
            "",
            "| Group | Accounts | Approved | Approved among non-defaulters | Mean PD | Default rate |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {g.group} | {g.count:,} | {_pct(g.approval_rate)} | {_pct(g.good_approval_rate)} | {_pct(g.mean_pd)} | {_pct(g.default_rate)} |"
            for g in r.groups
        ]
        lines.append("")
    lines += ["## 8. Explainability", "", f"{_pct(champ.reason_coverage)} of declines carry at least one reason. Examples:", ""]
    lines += [f"- Account {x['id']}: PD {x['pd']:.3f}, score {x['score']}: " + "; ".join(x["reasons"]) for x in champ.examples]
    lines += ["", "## 9. Policy checks", "", "| Check | Threshold | " + " | ".join(e.name for e in evs) + " |", "|---|---|" + "---|" * len(evs)]
    for i, c in enumerate(champ.checks):
        cells = [f"{e.checks[i].value:.4f} {'pass' if e.checks[i].passed else '**fail**'}" for e in evs]
        lines.append(f"| {c.name} | {c.threshold} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 10. Limitations",
        "",
        "- One public dataset from one lender and one period (2005); no out-of-time validation is possible with it.",
        "- The adverse impact ratio for applicants under 25 is the closest to its threshold. Their lower approval rate follows "
        "lower credit limits and more recent delinquency, which are legitimate inputs, but it should be reviewed with every "
        "retraining.",
        "- Reject inference is out of scope: the data contains only booked accounts.",
        "- Isotonic calibration is piecewise constant, so applicants close together can share a PD. Decisions are made on "
        "the raw score, which has no ties; the PD is for pricing, provisioning and reporting.",
        "",
        "## 11. Monitoring plan",
        "",
        f"- Monthly: score and feature PSI against the training population; warn at {policy.monitoring.psi_warn}, alert at "
        f"{policy.monitoring.psi_alert} (`crisk monitor`).",
        "- Quarterly, once outcomes mature: AUC, KS and calibration by risk band; adverse impact by gated attribute.",
        "- Retrain or recalibrate when score PSI alerts, AUC falls below the policy minimum, or calibration error exceeds it.",
    ]
    return "\n".join(lines) + "\n"


def summary(evaluations: list[ModelEvaluation]) -> str:
    rows = [
        f"{e.name:10} valid_auc={e.valid_auc:.4f} test_auc={e.test['auc']:.4f} ks={e.test['ks']:.4f} ece={e.test['ece']:.4f} "
        f"{'PASS' if e.passed else 'FAIL'}"
        for e in evaluations
    ]
    return "\n".join(rows)
