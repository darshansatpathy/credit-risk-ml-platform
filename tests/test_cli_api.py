from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from creditrisk import monitoring
from creditrisk.api import create_app
from creditrisk.cli import main
from creditrisk.pipeline import RunResult


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli")
    assert main(["train", "--out", str(out), "--report", str(out / "report.md"), "--check"]) == 0
    return out


def test_train_writes_models_report_and_summary(trained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = json.loads((trained / "evaluation.json").read_text())
    assert summary["champion"] == "lightgbm" and len(summary["evaluations"]) == 3
    assert (trained / "models" / "lightgbm" / "booster.txt").exists()
    text = (trained / "report.md").read_text()
    assert "Champion: `lightgbm`" in text and "## 7. Fairness" in text and "Stress scenario" in text
    strict = tmp_path / "strict.yaml"
    strict.write_text((Path(__file__).parents[1] / "policy" / "model_risk.yaml").read_text().replace("auc_min: 0.75", "auc_min: 0.99"))
    gh = tmp_path / "gh.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh))
    assert main(["--policy", str(strict), "train", "--out", str(tmp_path / "x"), "--check"]) == 1
    assert "No candidate met the model-risk policy" in gh.read_text()


def test_score_and_monitor(trained: Path, run: RunResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sample = run.split.test.head(200).drop(columns=["default_next_month"])
    sample.to_csv(tmp_path / "apps.csv", index=False)
    assert main(["score", str(tmp_path / "apps.csv"), "--models", str(trained)]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 200 and {r["decision"] for r in rows} == {"approve", "decline"}
    assert all(r["reasons"] for r in rows if r["decision"] == "decline")
    run.split.test.to_csv(tmp_path / "same.csv", index=False)
    assert main(["monitor", str(tmp_path / "same.csv"), "--models", str(trained), "--check"]) == 0
    monitoring.stress(run.split.test).to_csv(tmp_path / "stressed.csv", index=False)
    assert main(["monitor", str(tmp_path / "stressed.csv"), "--models", str(trained), "--check"]) == 1
    assert "alert" in capsys.readouterr().out


def test_no_champion_is_reported(tmp_path: Path) -> None:
    (tmp_path / "evaluation.json").write_text(json.dumps({"champion": None, "evaluations": []}))
    with pytest.raises(SystemExit, match="no champion"):
        main(["score", "x.csv", "--models", str(tmp_path)])


def test_api(trained: Path) -> None:
    summary = json.loads((trained / "evaluation.json").read_text())
    cutoff = next(e["cutoff"] for e in summary["evaluations"] if e["name"] == "lightgbm")
    client = TestClient(create_app(trained / "models" / "lightgbm", cutoff))
    assert client.get("/healthz").json()["model"] == "lightgbm"
    good = {"limit_bal": 300000, "pay_status": [-1] * 6, "bill_amounts": [3000] * 6, "payments": [3000] * 6}
    r = client.post("/score", json=good).json()
    assert r["decision"] == "approve" and r["reasons"] == []
    bad = {"limit_bal": 20000, "pay_status": [3, 2, 2, 2, 0, 0], "bill_amounts": [19500] * 6, "payments": [0] * 6}
    declined = client.post("/score", json=bad).json()
    assert declined["decision"] == "decline" and "Recent payment delinquency" in declined["reasons"]
    assert declined["score"] < r["score"] - 50 and declined["pd"] > 4 * r["pd"]
    assert client.post("/score", json=good | {"pay_status": [12, 0, 0, 0, 0, 0]}).status_code == 422
    assert client.post("/score", json=good | {"sex": 2}).status_code == 422


def test_champion_registered_in_mlflow(run: RunResult) -> None:
    client = MlflowClient()
    version = client.get_model_version_by_alias("credit-risk-pd", "champion")
    assert version.run_id == run.run_ids["lightgbm"]
    assert client.get_run(version.run_id).data.tags["policy_passed"] == "True"
