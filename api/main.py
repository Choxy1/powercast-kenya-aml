import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
FEATURE_STORE_PATH = ROOT / "data" / "processed" / "ward_month_feature_store.csv"
LOG_PATH = Path(os.getenv("POWERCAST_LOG_PATH", str(ROOT / "logs" / "predictions.jsonl")))
LOG_LOCK = threading.Lock()


def _load_artifacts():
    missing = [p for p in (MODEL_PATH, METADATA_PATH, FEATURE_STORE_PATH) if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing required serving artifacts: {missing}")

    loaded_model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        loaded_metadata = json.load(f)

    store = pd.read_csv(FEATURE_STORE_PATH)
    store["period"] = pd.to_datetime(store["period"], errors="raise")

    required = {
        "ward_code", "ward_name", "county_name", "period", *loaded_metadata["features"]
    }
    missing_columns = sorted(required - set(store.columns))
    if missing_columns:
        raise RuntimeError(f"Feature store is missing required columns: {missing_columns}")

    return loaded_model, loaded_metadata, store


model, metadata, feature_store = _load_artifacts()
MODEL_VERSION = metadata.get("model_version", "powercast-logistic-v1")
LATEST_FEATURE_MONTH = feature_store["period"].max()

app = FastAPI(
    title="PowerCast Kenya API",
    version="1.2.0",
    description=(
        "Predicts the probability that a ward represented in the PawaSwitch feature store "
        "will have at least one mapped planned electricity interruption notice in the following month. "
        "It does not predict all physical or unplanned outages."
    ),
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ward_code: str = Field(..., min_length=3, examples=["ainabkoi--ainabkoi-olare"])
    as_of_month: str | None = Field(
        default=None,
        description="Feature month in YYYY-MM format. If omitted, the latest available month for the ward is used.",
        examples=["2026-08"],
    )

    @field_validator("ward_code")
    @classmethod
    def normalize_ward_code(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("as_of_month")
    @classmethod
    def validate_month(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            period = pd.Period(value, freq="M")
        except Exception as exc:
            raise ValueError("as_of_month must use YYYY-MM format") from exc
        normalized = str(period)
        if normalized != value:
            raise ValueError("as_of_month must use YYYY-MM format")
        return value


class PredictionResponse(BaseModel):
    ward_code: str
    ward_name: str
    county_name: str
    as_of_month: str
    prediction_month: str
    planned_interruption_probability: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    predicted_interruption: Literal[0, 1]
    interpretation: Literal["higher risk", "lower risk"]
    model_version: str
    scope_note: str


def _write_prediction_log(payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        **payload,
    }
    with LOG_LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/")
def root():
    return {
        "name": "PowerCast Kenya API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "scope": metadata.get("scope_note"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata["best_model"],
        "model_version": MODEL_VERSION,
        "model_stage": metadata.get("model_stage"),
        "decision_threshold": metadata.get("threshold"),
        "latest_feature_month": LATEST_FEATURE_MONTH.strftime("%Y-%m"),
        "target": metadata["target_definition"],
        "scope_note": metadata.get("scope_note"),
    }


@app.get("/model-info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "algorithm": metadata.get("best_model"),
        "selection_metric": metadata.get("selection_metric"),
        "validation_pr_auc": metadata.get("selection_validation_pr_auc"),
        "threshold": metadata.get("threshold"),
        "feature_count": len(metadata.get("features", [])),
        "development_target_period_end": metadata.get("development_target_period_end"),
        "test_period": metadata.get("test_period"),
        "test_metrics": metadata.get("test_metrics"),
        "scope_note": metadata.get("scope_note"),
    }


@app.get("/wards")
def wards(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    county: str | None = Query(default=None),
    search: str | None = Query(default=None),
):
    rows = feature_store[["ward_code", "ward_name", "county_name"]].drop_duplicates()

    if county:
        rows = rows[rows["county_name"].str.casefold() == county.strip().casefold()]
    if search:
        needle = search.strip()
        rows = rows[
            rows["ward_name"].str.contains(needle, case=False, na=False, regex=False)
            | rows["ward_code"].str.contains(needle, case=False, na=False, regex=False)
        ]

    rows = rows.sort_values(["county_name", "ward_name"]).iloc[offset : offset + limit]
    return rows.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    ward_rows = feature_store[feature_store["ward_code"] == request.ward_code].copy()
    if ward_rows.empty:
        raise HTTPException(
            status_code=404,
            detail="Ward code not found. Use GET /wards to list wards represented in the feature store.",
        )

    if request.as_of_month:
        as_of = pd.Period(request.as_of_month, freq="M").to_timestamp()
        row = ward_rows[ward_rows["period"] == as_of]
        if row.empty:
            available_min = ward_rows["period"].min().strftime("%Y-%m")
            available_max = ward_rows["period"].max().strftime("%Y-%m")
            raise HTTPException(
                status_code=404,
                detail=f"Requested month is not available for this ward. Available range: {available_min} to {available_max}.",
            )
    else:
        row = ward_rows.loc[[ward_rows["period"].idxmax()]]

    features = metadata["features"]
    probability = float(model.predict_proba(row[features])[:, 1][0])
    threshold = float(metadata["threshold"])
    predicted = int(probability >= threshold)
    as_of_date = pd.Timestamp(row.iloc[0]["period"])
    prediction_month = as_of_date + pd.offsets.MonthBegin(1)

    payload = {
        "ward_code": str(row.iloc[0]["ward_code"]),
        "ward_name": str(row.iloc[0]["ward_name"]),
        "county_name": str(row.iloc[0]["county_name"]),
        "as_of_month": as_of_date.strftime("%Y-%m"),
        "prediction_month": prediction_month.strftime("%Y-%m"),
        "planned_interruption_probability": round(probability, 4),
        "decision_threshold": round(threshold, 4),
        "predicted_interruption": predicted,
        "interpretation": "higher risk" if predicted else "lower risk",
        "model_version": MODEL_VERSION,
        "scope_note": "Predicts mapped planned interruption notices for represented wards, not all physical or unplanned outages.",
    }
    _write_prediction_log(payload)
    return payload
