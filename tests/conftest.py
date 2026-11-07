from __future__ import annotations

from pathlib import Path

import pytest

from creditrisk import data, pipeline, policy
from creditrisk.policy import Policy


@pytest.fixture(scope="session")
def pol() -> Policy:
    return policy.load()


@pytest.fixture(scope="session")
def split() -> data.Split:
    return data.split(data.load())


@pytest.fixture(scope="session")
def run(pol: Policy, tmp_path_factory: pytest.TempPathFactory) -> pipeline.RunResult:
    out: Path = tmp_path_factory.mktemp("run")
    return pipeline.run(pol, out, f"sqlite:///{out}/mlflow.db")
