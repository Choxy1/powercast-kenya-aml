import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROCESSED_DIR, MODEL_DIR, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END

NUMERIC_FEATURES = [
    "population", "events", "outage_hours",
    "events_lag1", "events_lag2", "events_lag3",
    "hours_lag1", "hours_lag2", "hours_lag3",
    "events_roll3", "events_roll6", "events_roll12",
    "hours_roll3", "hours_roll6", "hours_roll12",
    "flag_rate3", "flag_rate6", "flag_rate12",
    "hist_rate", "hist_events_mean", "months_since_outage",
    "month_sin", "month_cos",
]
CATEGORICAL_FEATURES = ["county_name"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target_next_month"
MODEL_VERSION = "powercast-logistic-2026-10-v1"


def build_model() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    classifier = LogisticRegression(
        C=0.02,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    pred = (probability >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = PROCESSED_DIR / "model_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "model_dataset.csv is missing. Run `python -m src.data_pipeline` first."
        )

    data = pd.read_csv(dataset_path, parse_dates=["period", "target_period"])

    train = data[data["target_period"] <= pd.Timestamp(TRAIN_END)].copy()
    val = data[
        (data["target_period"] >= pd.Timestamp(VAL_START))
        & (data["target_period"] <= pd.Timestamp(VAL_END))
    ].copy()
    test = data[
        (data["target_period"] >= pd.Timestamp(TEST_START))
        & (data["target_period"] <= pd.Timestamp(TEST_END))
    ].copy()

    # Development selection: tuned Logistic Regression selected on Jul-Dec 2025 PR AUC.
    selection_model = build_model()
    selection_model.fit(train[FEATURES], train[TARGET])
    val_probability = selection_model.predict_proba(val[FEATURES])[:, 1]
    threshold = choose_threshold(val[TARGET], val_probability)
    validation_metrics = evaluate(val[TARGET], val_probability, threshold)

    # Final refit uses all pre-2026 development data after choices are frozen.
    development = pd.concat([train, val], ignore_index=True)
    final_model = build_model()
    final_model.fit(development[FEATURES], development[TARGET])

    test_metrics = None
    if not test.empty:
        test_probability = final_model.predict_proba(test[FEATURES])[:, 1]
        test_metrics = evaluate(test[TARGET], test_probability, threshold)

    joblib.dump(final_model, MODEL_DIR / "best_model.joblib")
    metadata = {
        "best_model": "logistic_regression",
        "model_stage": "final_refit_on_pre_2026_development_data",
        "model_version": MODEL_VERSION,
        "serving_api_version": "1.2.0",
        "selection_metric": "July-Dec 2025 validation PR AUC",
        "selection_validation_pr_auc": validation_metrics["pr_auc"],
        "threshold": threshold,
        "threshold_source": "July-Dec 2025 validation F1 optimization",
        "features": FEATURES,
        "development_rows": int(len(development)),
        "test_rows": int(len(test)),
        "development_target_period_end": str(pd.Timestamp(VAL_END).date()),
        "test_period": [TEST_START, TEST_END],
        "target_definition": "At least one mapped PawaSwitch planned interruption in the ward during the following month.",
        "scope_note": "Predictions refer to mapped planned interruption notices for represented wards, not all physical or unplanned outages.",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
