from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_PATH = Path(
    "data/raw/pems_bay/PEMS-BAY.csv"
)

PROTOCOL_PATH = Path(
    "data/processed/pems_bay/"
    "pems_bay_evaluation_origins.csv"
)

OUT_DIR = Path(
    "results/pems_bay/baselines"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPLIT = "validation"

FORECAST_STEPS = 12
RESOLUTION_MINUTES = 5


# ---------------------------------------------------------------------
# Load protocol
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
        "target_start",
        "target_end",
    ],
)

protocol["sensor"] = (
    protocol["sensor"]
    .astype(str)
)

protocol = (
    protocol[
        protocol["split"] == SPLIT
    ]
    .copy()
    .sort_values(
        [
            "sensor",
            "eval_index",
        ]
    )
    .reset_index(drop=True)
)

sensors = sorted(
    protocol["sensor"].unique()
)


# ---------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------

header = pd.read_csv(
    RAW_PATH,
    nrows=0,
)

timestamp_col = header.columns[0]

raw = pd.read_csv(
    RAW_PATH,
    usecols=[
        timestamp_col,
        *sensors,
    ],
    low_memory=False,
)

raw[timestamp_col] = pd.to_datetime(
    raw[timestamp_col]
)

raw = (
    raw
    .set_index(timestamp_col)
    .sort_index()
)


print("=" * 78)
print("PersTSFM PEMS-BAY naive validation baselines")
print("=" * 78)

print()
print(f"Validation sensors: {len(sensors)}")
print(f"Evaluation cases:   {len(protocol)}")
print(f"Forecast steps:     {FORECAST_STEPS}")


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

rows = []


for _, case in protocol.iterrows():

    sensor = case["sensor"]

    eval_index = int(
        case["eval_index"]
    )

    origin = case["eval_origin"]

    target_index = pd.date_range(
        start=case["target_start"],
        end=case["target_end"],
        freq="5min",
    )

    if len(target_index) != FORECAST_STEPS:
        raise RuntimeError(
            f"{sensor} eval {eval_index}: "
            f"expected {FORECAST_STEPS} "
            f"target timestamps, "
            f"found {len(target_index)}."
        )

    truth = raw.loc[
        target_index,
        sensor,
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------
    # Persistence:
    # repeat last observed speed at the forecast origin
    # -------------------------------------------------------------

    last_value = float(
        raw.loc[
            origin,
            sensor,
        ]
    )

    persistence = np.full(
        FORECAST_STEPS,
        last_value,
        dtype=float,
    )

    # -------------------------------------------------------------
    # Daily seasonal naive:
    # same timestamps one day earlier
    # -------------------------------------------------------------

    daily_index = (
        target_index
        - pd.Timedelta(days=1)
    )

    daily = raw.loc[
        daily_index,
        sensor,
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------
    # Weekly seasonal naive:
    # same timestamps seven days earlier
    # -------------------------------------------------------------

    weekly_index = (
        target_index
        - pd.Timedelta(days=7)
    )

    weekly = raw.loc[
        weekly_index,
        sensor,
    ].to_numpy(
        dtype=float
    )

    methods = {
        "persistence": persistence,
        "daily_seasonal": daily,
        "weekly_seasonal": weekly,
    }

    for method, prediction in methods.items():

        error = (
            prediction
            - truth
        )

        mae = float(
            np.mean(
                np.abs(error)
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        rows.append(
            {
                "sensor": sensor,
                "eval_index": eval_index,
                "eval_origin": origin,
                "weekday": (
                    origin.day_name()
                ),
                "hour": int(
                    origin.hour
                ),
                "method": method,
                "mae": mae,
                "rmse": rmse,
            }
        )


case_metrics = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Entity-level aggregation
# ---------------------------------------------------------------------

entity_metrics = (
    case_metrics
    .groupby(
        [
            "sensor",
            "method",
        ],
        as_index=False,
    )
    .agg(
        mean_mae=(
            "mae",
            "mean",
        ),
        median_mae=(
            "mae",
            "median",
        ),
        mean_rmse=(
            "rmse",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------

summary = (
    case_metrics
    .groupby(
        "method",
        as_index=False,
    )
    .agg(
        mean_mae=(
            "mae",
            "mean",
        ),
        median_mae=(
            "mae",
            "median",
        ),
        mean_rmse=(
            "rmse",
            "mean",
        ),
    )
)


entity_summary = (
    entity_metrics
    .groupby(
        "method",
        as_index=False,
    )
    .agg(
        mean_entity_mae=(
            "mean_mae",
            "mean",
        ),
        median_entity_mae=(
            "mean_mae",
            "median",
        ),
    )
)

summary = summary.merge(
    entity_summary,
    on="method",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------------------
# Daypart diagnostic
# ---------------------------------------------------------------------

daypart = (
    case_metrics
    .groupby(
        [
            "method",
            "hour",
        ],
        as_index=False,
    )
    .agg(
        mean_mae=(
            "mae",
            "mean",
        ),
        median_mae=(
            "mae",
            "median",
        ),
    )
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

case_path = (
    OUT_DIR
    / "pems_bay_naive_case_metrics.csv"
)

entity_path = (
    OUT_DIR
    / "pems_bay_naive_entity_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "pems_bay_naive_summary.csv"
)

daypart_path = (
    OUT_DIR
    / "pems_bay_naive_by_hour.csv"
)

case_metrics.to_csv(
    case_path,
    index=False,
)

entity_metrics.to_csv(
    entity_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

daypart.to_csv(
    daypart_path,
    index=False,
)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("Naive baseline summary")
print("=" * 78)

display = summary.copy()

for col in [
    "mean_mae",
    "median_mae",
    "mean_rmse",
    "mean_entity_mae",
    "median_entity_mae",
]:
    display[col] = (
        display[col]
        .round(4)
    )

print()
print(
    display.to_string(
        index=False
    )
)

print()
print("=" * 78)
print("Mean MAE by forecast-origin hour")
print("=" * 78)

pivot = (
    daypart
    .pivot(
        index="hour",
        columns="method",
        values="mean_mae",
    )
    .round(4)
)

print()
print(
    pivot.to_string()
)

print()
print("Saved:")
print(f"  {case_path}")
print(f"  {entity_path}")
print(f"  {summary_path}")
print(f"  {daypart_path}")
