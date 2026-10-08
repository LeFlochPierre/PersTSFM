from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path(
    "data/raw/electricity/LD2011_2014.txt"
)

PROTOCOL_PATH = Path(
    "data/processed/electricity/"
    "electricity_evaluation_origins.csv"
)

OUT_DIR = Path(
    "results/baselines"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPLIT = "validation"

FORECAST_STEPS = 96
DAILY_LAG_STEPS = 96
WEEKLY_LAG_STEPS = 7 * 96


# ---------------------------------------------------------------------
# Load evaluation protocol
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
        "context_start",
        "context_end",
        "target_start",
        "target_end",
    ],
)

protocol = protocol[
    protocol["split"] == SPLIT
].copy()

entities = sorted(
    protocol["entity"].unique()
)

print("=" * 72)
print("PersTSFM naive validation baselines")
print("=" * 72)

print()
print(f"Validation entities: {len(entities)}")
print(f"Evaluation cases:    {len(protocol)}")


# ---------------------------------------------------------------------
# Load only the validation entity columns
# ---------------------------------------------------------------------

header = pd.read_csv(
    RAW_PATH,
    sep=";",
    decimal=",",
    nrows=0,
)

timestamp_col = header.columns[0]

usecols = [
    timestamp_col,
    *entities,
]

print()
print("Loading validation time series...")

raw = pd.read_csv(
    RAW_PATH,
    sep=";",
    decimal=",",
    usecols=usecols,
    low_memory=False,
)

raw[timestamp_col] = pd.to_datetime(
    raw[timestamp_col]
)

raw = raw.set_index(
    timestamp_col
).sort_index()

print(
    f"Loaded {len(raw):,} timestamps "
    f"for {len(entities)} entities."
)


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

rows = []

for case_idx, row in protocol.iterrows():

    entity = row["entity"]
    origin = row["eval_origin"]

    target_start = row["target_start"]
    target_end = row["target_end"]

    target = raw.loc[
        target_start:target_end,
        entity,
    ].to_numpy(
        dtype=float
    )

    if len(target) != FORECAST_STEPS:
        raise RuntimeError(
            f"{entity} at {origin}: "
            f"expected {FORECAST_STEPS} target points, "
            f"got {len(target)}."
        )

    # -------------------------------------------------------------
    # Persistence
    # Same last observed value for all 96 future steps.
    # -------------------------------------------------------------

    last_value = float(
        raw.loc[
            origin,
            entity,
        ]
    )

    persistence = np.full(
        FORECAST_STEPS,
        last_value,
        dtype=float,
    )

    # -------------------------------------------------------------
    # Daily seasonal naive
    #
    # Forecast tomorrow using the exact corresponding values
    # from the previous day.
    # -------------------------------------------------------------

    daily_start = (
        target_start
        - pd.Timedelta(days=1)
    )

    daily_end = (
        target_end
        - pd.Timedelta(days=1)
    )

    seasonal_daily = raw.loc[
        daily_start:daily_end,
        entity,
    ].to_numpy(
        dtype=float
    )

    if len(seasonal_daily) != FORECAST_STEPS:
        raise RuntimeError(
            f"{entity}: daily seasonal baseline "
            f"has {len(seasonal_daily)} points."
        )

    # -------------------------------------------------------------
    # Weekly seasonal naive
    #
    # Forecast using the same quarter-hour one week earlier.
    # -------------------------------------------------------------

    weekly_start = (
        target_start
        - pd.Timedelta(days=7)
    )

    weekly_end = (
        target_end
        - pd.Timedelta(days=7)
    )

    seasonal_weekly = raw.loc[
        weekly_start:weekly_end,
        entity,
    ].to_numpy(
        dtype=float
    )

    if len(seasonal_weekly) != FORECAST_STEPS:
        raise RuntimeError(
            f"{entity}: weekly seasonal baseline "
            f"has {len(seasonal_weekly)} points."
        )

    predictions = {
        "persistence": persistence,
        "seasonal_daily": seasonal_daily,
        "seasonal_weekly": seasonal_weekly,
    }

    for model_name, pred in predictions.items():

        error = pred - target

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
                "entity": entity,
                "eval_index": row["eval_index"],
                "eval_origin": origin,
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
                "target_mean": float(
                    np.mean(target)
                ),
                "target_std": float(
                    np.std(target)
                ),
            }
        )


results = pd.DataFrame(
    rows
)

out_path = (
    OUT_DIR
    / "electricity_validation_naive.csv"
)

results.to_csv(
    out_path,
    index=False,
)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

summary = (
    results
    .groupby("model")
    .agg(
        n_cases=("mae", "size"),
        median_mae=("mae", "median"),
        mean_mae=("mae", "mean"),
        median_rmse=("rmse", "median"),
        mean_rmse=("rmse", "mean"),
    )
    .sort_values("median_mae")
)

print()
print("=" * 72)
print("Validation results")
print("=" * 72)

print(
    summary.round(4).to_string()
)

print()
print(f"Saved: {out_path}")
