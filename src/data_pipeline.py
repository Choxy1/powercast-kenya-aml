import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR, REPORT_DIR, HISTORY_START, MODEL_START, MODEL_END


def load_raw_data():
    events = pd.read_csv(RAW_DIR / "events.csv")
    event_wards = pd.read_csv(RAW_DIR / "event_wards.csv")
    localities = pd.read_csv(RAW_DIR / "localities_matched.csv")
    ward_stats = pd.read_csv(RAW_DIR / "ward_stats_monthly.csv")
    return events, event_wards, localities, ward_stats


def audit_data(events, event_wards, localities, ward_stats):
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    ward_stats = ward_stats.copy()
    ward_stats["period"] = pd.to_datetime(ward_stats["period"], errors="coerce")

    audit = {
        "events_rows": int(len(events)),
        "event_ward_rows": int(len(event_wards)),
        "locality_rows": int(len(localities)),
        "ward_stats_rows": int(len(ward_stats)),
        "event_date_min": str(events["event_date"].min().date()),
        "event_date_max": str(events["event_date"].max().date()),
        "missing_event_dates": int(events["event_date"].isna().sum()),
        "duplicate_event_ids": int(events["id"].duplicated().sum()),
        "missing_duration_hours": int(events["duration_hours"].isna().sum()),
        "unique_events": int(events["id"].nunique()),
        "mapped_events": int(event_wards["event_id"].nunique()),
        "mapped_event_share": float(event_wards["event_id"].nunique() / events["id"].nunique()),
        "unique_mapped_wards": int(event_wards["ward_code"].nunique()),
        "wards_in_monthly_stats": int(ward_stats["ward_code"].nunique()),
        "localities_with_ward": int(localities["ward_code"].notna().sum()),
        "locality_ward_share": float(localities["ward_code"].notna().mean()),
        "mean_match_confidence_when_available": float(localities["match_confidence"].mean()),
        "duplicate_event_ward_links": int(event_wards.duplicated(["event_id", "ward_code"]).sum()),
    }
    return audit


def _months_since_outage(series: pd.Series) -> pd.Series:
    result = []
    last_positive = None
    values = series.to_numpy()
    for i, value in enumerate(values):
        if value > 0:
            last_positive = i
            result.append(0)
        else:
            result.append(99 if last_positive is None else i - last_positive)
    return pd.Series(result, index=series.index)


def build_ward_month_panel(events, event_wards, ward_stats):
    events = events.copy()
    event_wards = event_wards.copy()
    ward_stats = ward_stats.copy()

    events["event_date"] = pd.to_datetime(events["event_date"], errors="raise")
    events["duration_hours"] = pd.to_numeric(events["duration_hours"], errors="coerce")
    ward_stats["period"] = pd.to_datetime(ward_stats["period"], errors="raise")

    # The monthly statistics file supplies a stable set of wards that are represented
    # in the PawaSwitch study period, plus static population/county metadata.
    ward_meta = (
        ward_stats.sort_values("period")
        .drop_duplicates("ward_code")
        [["ward_code", "ward_name", "county_code", "county_name", "population"]]
        .copy()
    )

    mapped = event_wards.merge(
        events[["id", "event_date", "duration_hours"]],
        left_on="event_id", right_on="id", how="left", validate="many_to_one"
    )
    mapped["period"] = mapped["event_date"].dt.to_period("M").dt.to_timestamp()
    mapped = mapped[
        (mapped["period"] >= pd.Timestamp(HISTORY_START)) &
        (mapped["period"] <= pd.Timestamp(MODEL_END)) &
        (mapped["ward_code"].isin(ward_meta["ward_code"]))
    ].copy()

    monthly = (
        mapped.groupby(["ward_code", "period"], as_index=False)
        .agg(events=("event_id", "nunique"), outage_hours=("duration_hours", "sum"))
    )

    months = pd.date_range(HISTORY_START, MODEL_END, freq="MS")
    panel = (
        ward_meta.assign(_key=1)
        .merge(pd.DataFrame({"period": months, "_key": 1}), on="_key")
        .drop(columns="_key")
        .merge(monthly, on=["ward_code", "period"], how="left")
        .sort_values(["ward_code", "period"])
        .reset_index(drop=True)
    )
    panel[["events", "outage_hours"]] = panel[["events", "outage_hours"]].fillna(0.0)
    panel["outage_flag"] = (panel["events"] > 0).astype(int)
    return panel


def add_features(panel):
    df = panel.copy().sort_values(["ward_code", "period"]).reset_index(drop=True)
    g = df.groupby("ward_code", group_keys=False)

    for lag in (1, 2, 3):
        df[f"events_lag{lag}"] = g["events"].shift(lag)
        df[f"hours_lag{lag}"] = g["outage_hours"].shift(lag)

    for window in (3, 6, 12):
        df[f"events_roll{window}"] = g["events"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )
        df[f"hours_roll{window}"] = g["outage_hours"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )
        df[f"flag_rate{window}"] = g["outage_flag"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    df["hist_rate"] = g["outage_flag"].transform(lambda s: s.expanding().mean())
    df["hist_events_mean"] = g["events"].transform(lambda s: s.expanding().mean())
    df["months_since_outage"] = (
        g["outage_flag"].apply(_months_since_outage).reset_index(level=0, drop=True)
    )

    df["month"] = df["period"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # At the end of month t, predict whether a mapped planned interruption is
    # announced for the same ward during month t+1.
    df["target_next_month"] = g["outage_flag"].shift(-1)
    df["target_period"] = df["period"] + pd.offsets.MonthBegin(1)
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    events, event_wards, localities, ward_stats = load_raw_data()
    audit = audit_data(events, event_wards, localities, ward_stats)
    panel = build_ward_month_panel(events, event_wards, ward_stats)
    feature_store = add_features(panel)
    model_data = feature_store[(feature_store["period"] >= pd.Timestamp(MODEL_START)) & feature_store["target_next_month"].notna()].copy()
    model_data["target_next_month"] = model_data["target_next_month"].astype(int)

    audit.update({
        "model_wards": int(panel["ward_code"].nunique()),
        "history_start": HISTORY_START,
        "panel_rows_with_warmup": int(len(panel)),
        "model_rows": int(len(model_data)),
        "model_positive_rate": float(model_data["target_next_month"].mean()),
        "model_period_start": str(model_data["period"].min().date()),
        "model_period_end": str(model_data["period"].max().date()),
        "target_period_start": str(model_data["target_period"].min().date()),
        "target_period_end": str(model_data["target_period"].max().date()),
    })

    feature_store.to_csv(PROCESSED_DIR / "ward_month_feature_store.csv", index=False)
    model_data.to_csv(PROCESSED_DIR / "model_dataset.csv", index=False)
    with open(REPORT_DIR / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
