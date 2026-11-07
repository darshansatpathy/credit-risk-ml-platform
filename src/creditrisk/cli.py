"""Command line: train, evaluate, score, monitor, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from creditrisk import __version__
from creditrisk import policy as policy_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crisk", description="Credit risk models with model-risk controls.")
    parser.add_argument("--version", action="version", version=f"crisk {__version__}")
    parser.add_argument("--policy", type=Path, help="model-risk policy (default: the bundled policy/model_risk.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train, evaluate, choose a champion, write the validation report")
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"), help="e.g. sqlite:///artifacts/mlflow.db")
    p.add_argument("--report", type=Path, help="write the model validation report here")
    p.add_argument("--check", action="store_true", help="exit 1 if no candidate meets the policy")

    p = sub.add_parser("score", help="score applications from a CSV in the source data's layout")
    p.add_argument("csv", type=Path)
    p.add_argument("--models", type=Path, default=Path("artifacts"))

    p = sub.add_parser("monitor", help="population stability of a new population against the training data")
    p.add_argument("csv", type=Path)
    p.add_argument("--models", type=Path, default=Path("artifacts"))
    p.add_argument("--check", action="store_true", help="exit 1 on any alert")

    p = sub.add_parser("serve", help="run the scoring API")
    p.add_argument("--models", type=Path, default=Path("artifacts"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    policy = policy_mod.load(args.policy)
    return int(COMMANDS[args.command](args, policy))


def _champion(models_dir: Path) -> tuple[str, float]:
    summary = json.loads((models_dir / "evaluation.json").read_text())
    if not summary["champion"]:
        raise SystemExit("no champion in this run: no candidate met the policy")
    ev = next(e for e in summary["evaluations"] if e["name"] == summary["champion"])
    return summary["champion"], float(ev["cutoff"])


def _train(args: argparse.Namespace, policy: policy_mod.Policy) -> int:
    from creditrisk import pipeline, report

    result = pipeline.run(policy, args.out, args.tracking_uri)
    print(report.summary(result.evaluations))
    print(f"champion: {result.champion or 'none'}")
    text = report.render(result, policy)
    if args.report:
        args.report.write_text(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(text)
    return 1 if args.check and result.champion is None else 0


def _score(args: argparse.Namespace, policy: policy_mod.Policy) -> int:
    import pandas as pd

    from creditrisk import data, explain, models
    from creditrisk.features import build

    name, cutoff = _champion(args.models)
    model = models.load(args.models / "models" / name)
    df = pd.read_csv(args.csv)
    if data.TARGET not in df:
        df[data.TARGET] = 0
    data.validate(df)
    X = build(df)
    pd_, raw = model.predict_pd(X), model.raw(X)
    reasons = explain.reasons(model.contributions(X))
    scores = models.score_from_pd(pd_)
    for i in range(len(df)):
        decision = "approve" if raw[i] < cutoff else "decline"
        out = {"id": int(df["id"].iloc[i]), "pd": round(float(pd_[i]), 4), "score": int(scores[i]), "decision": decision}
        if decision == "decline":
            out["reasons"] = reasons[i]
        print(json.dumps(out))
    return 0


def _monitor(args: argparse.Namespace, policy: policy_mod.Policy) -> int:
    import pandas as pd

    from creditrisk import data, models, monitoring
    from creditrisk.features import build

    name, _ = _champion(args.models)
    model = models.load(args.models / "models" / name)
    reference = data.split(data.load()).train
    current = pd.read_csv(args.csv)
    if data.TARGET not in current:
        current[data.TARGET] = 0
    data.validate(current)
    Xr, Xc = build(reference), build(current)
    rows = monitoring.stability(Xr, Xc, model.predict_pd(Xr), model.predict_pd(Xc), policy.monitoring.psi_warn, policy.monitoring.psi_alert)
    for r in rows:
        print(f"{r.name:24} psi={r.psi:.4f} {r.status}")
    alerts = [r for r in rows if r.status == "alert"]
    if alerts:
        print(f"{len(alerts)} alert(s): investigate before relying on the model for this population", file=sys.stderr)
    return 1 if args.check and alerts else 0


def _serve(args: argparse.Namespace, policy: policy_mod.Policy) -> int:  # pragma: no cover - runs a server
    import uvicorn

    from creditrisk.api import create_app

    name, cutoff = _champion(args.models)
    uvicorn.run(create_app(args.models / "models" / name, cutoff), host=args.host, port=args.port)
    return 0


COMMANDS = {"train": _train, "score": _score, "monitor": _monitor, "serve": _serve}
