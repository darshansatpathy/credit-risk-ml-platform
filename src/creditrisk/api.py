"""Scoring API: an application in, a PD, score, decision and reasons out."""

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from creditrisk import data, explain, models
from creditrisk.features import build


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit_bal: float = Field(gt=0)
    pay_status: list[int] = Field(min_length=6, max_length=6, description="Repayment status, most recent month first (-2..9)")
    bill_amounts: list[float] = Field(min_length=6, max_length=6, description="Statement balances, most recent first")
    payments: list[float] = Field(min_length=6, max_length=6, description="Payments made, most recent first")


def frame(app: Application) -> pd.DataFrame:
    row: dict[str, Any] = {"id": 0, "limit_bal": app.limit_bal, "sex": 1, "age": 30, "marriage": 1, "education": 1}
    row |= {c: v for c, v in zip(data.PAY_STATUS, app.pay_status, strict=True)}
    row |= {c: v for c, v in zip(data.BILLS, app.bill_amounts, strict=True)}
    row |= {c: v for c, v in zip(data.PAYMENTS, app.payments, strict=True)}
    df = pd.DataFrame([row])
    df[data.TARGET] = 0
    return df


def create_app(model_dir: Path, cutoff: float) -> FastAPI:
    model = models.load(model_dir)
    app = FastAPI(title="Credit risk scoring", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model": model.name}

    @app.post("/score")
    def score(application: Application) -> dict[str, Any]:
        df = frame(application)
        try:
            data.validate(df)
        except data.DataError as exc:
            raise HTTPException(422, str(exc)) from None
        X = build(df)
        pd_ = float(model.predict_pd(X)[0])
        approved = bool(model.raw(X)[0] < cutoff)
        return {
            "model": model.name,
            "pd": round(pd_, 4),
            "score": int(models.score_from_pd(model.predict_pd(X))[0]),
            "decision": "approve" if approved else "decline",
            "reasons": [] if approved else explain.reasons(model.contributions(X))[0],
        }

    return app
