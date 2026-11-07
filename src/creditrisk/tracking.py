"""MLflow: one run per candidate, the champion registered under the alias "champion"."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow import MlflowClient

from creditrisk.evaluation import ModelEvaluation
from creditrisk.models import RiskModel
from creditrisk.policy import Policy

EXPERIMENT = "credit-risk"
REGISTERED = "credit-risk-pd"


def log_run(
    tracking_uri: str, out: Path, candidates: list[RiskModel], evaluations: list[ModelEvaluation], champion: str | None, policy: Policy
) -> dict[str, str]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT, artifact_location=(out / "mlartifacts").resolve().as_uri())
    )
    run_ids = {}
    for model, ev in zip(candidates, evaluations, strict=True):
        with mlflow.start_run(experiment_id=experiment_id, run_name=model.name) as run:
            mlflow.set_tags({"model_kind": model.kind, "policy_passed": str(ev.passed), "champion": str(model.name == champion)})
            mlflow.log_params({"kind": model.kind, "approval_rate": policy.decision.approval_rate, "cutoff": round(ev.cutoff, 6)})
            mlflow.log_metrics({f"test_{k}": v for k, v in ev.test.items()} | {"valid_auc": ev.valid_auc, "train_auc": ev.train_auc})
            for check in ev.checks:
                mlflow.log_metric("check_" + "".join(ch if ch.isalnum() else "_" for ch in check.name)[:60], float(check.passed))
            mlflow.log_artifacts(str(out / "models" / model.name), artifact_path="model")
            mlflow.log_artifact(str(out / "evaluation.json"))
            run_ids[model.name] = run.info.run_id
    if champion:
        try:
            client.get_registered_model(REGISTERED)
        except mlflow.exceptions.MlflowException:
            client.create_registered_model(REGISTERED, description="Probability of default, 12 features, calibrated")
        version = client.create_model_version(REGISTERED, f"runs:/{run_ids[champion]}/model", run_ids[champion])
        client.set_registered_model_alias(REGISTERED, "champion", version.version)
    return run_ids
