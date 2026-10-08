from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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

MODEL_ID = "amazon/chronos-2"

SPLIT = "validation"

FREQUENCY_MINUTES = 15
CONTEXT_STEPS = 384
FORECAST_STEPS = 96

MAX_PERSONALIZATION_DAYS = 64

BATCH_SIZE = 64


# ---------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------

freq = pd.Timedelta(
    minutes=FREQUENCY_MINUTES
)

context_span = (
    CONTEXT_STEPS - 1
) * freq

forecast_span = (
    FORECAST_STEPS
) * freq


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
        "first_active_timestamp",
        "last_active_timestamp",
    ],
)

protocol = (
    protocol[
        protocol["split"] == SPLIT
    ]
    .copy()
    .sort_values(
        [
            "entity",
            "eval_origin",
        ]
    )
    .reset_index(drop=True)
)

entities = sorted(
    protocol["entity"].unique()
)


# ---------------------------------------------------------------------
# Build union of required historical daily forecast origins
#
# For an evaluation origin t, personalization with n days may use
# completed forecasts from:
#
# t - 1 day
# t - 2 days
# ...
# t - 64 days
#
# We generate the union once so overlapping requirements are not
# forecast repeatedly.
# ---------------------------------------------------------------------

required = set()

for _, row in protocol.iterrows():

    entity = row["entity"]
    eval_origin = row["eval_origin"]

    for lag_days in range(
        1,
        MAX_PERSONALIZATION_DAYS + 1,
    ):

        personal_origin = (
            eval_origin
            - pd.Timedelta(
                days=lag_days
            )
        )

        required.add(
            (
                entity,
                personal_origin,
            )
        )


origin_df = pd.DataFrame(
    sorted(required),
    columns=[
        "entity",
        "personal_origin",
    ],
)

origin_df["personal_origin"] = pd.to_datetime(
    origin_df["personal_origin"]
)


print("=" * 72)
print("PersTSFM Chronos-2 validation residual bank")
print("=" * 72)

print()
print(f"Model:                    {MODEL_ID}")
print(f"Validation entities:      {len(entities)}")
print(f"Evaluation origins:       {len(protocol)}")
print(
    f"Unique historical origins:{len(origin_df):8d}"
)
print(
    f"Maximum personal history: "
    f"{MAX_PERSONALIZATION_DAYS} days"
)
print(f"Context steps:            {CONTEXT_STEPS}")
print(f"Forecast steps:           {FORECAST_STEPS}")
print(f"Batch size:               {BATCH_SIZE}")


# ---------------------------------------------------------------------
# Check how many historical origins exist per entity
# ---------------------------------------------------------------------

origin_counts = (
    origin_df
    .groupby("entity")
    .size()
)

print()
print("Historical origins per entity:")
print(
    origin_counts
    .describe()
    .round(2)
    .to_string()
)


# ---------------------------------------------------------------------
# Load validation series only
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

raw = (
    raw
    .set_index(timestamp_col)
    .sort_index()
)

print(
    f"Loaded {len(raw):,} timestamps "
    f"for {len(entities)} entities."
)


# ---------------------------------------------------------------------
# Entity active bounds
# ---------------------------------------------------------------------

bounds = (
    protocol[
        [
            "entity",
            "first_active_timestamp",
            "last_active_timestamp",
        ]
    ]
    .drop_duplicates(
        subset=["entity"]
    )
    .set_index("entity")
)


# ---------------------------------------------------------------------
# Validate all required origins before running Chronos
# ---------------------------------------------------------------------

for _, row in origin_df.iterrows():

    entity = row["entity"]
    origin = row["personal_origin"]

    first_active = bounds.loc[
        entity,
        "first_active_timestamp",
    ]

    last_active = bounds.loc[
        entity,
        "last_active_timestamp",
    ]

    context_start = (
        origin
        - context_span
    )

    target_end = (
        origin
        + forecast_span
    )

    if context_start < first_active:
        raise RuntimeError(
            f"{entity} origin {origin}: "
            f"context begins before active history."
        )

    if target_end > last_active:
        raise RuntimeError(
            f"{entity} origin {origin}: "
            f"target extends after active history."
        )


print()
print("Historical-origin safety checks: PASSED")


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(f"Loading {MODEL_ID}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        "GPU: "
        + torch.cuda.get_device_name(0)
    )

pipeline = Chronos2Pipeline.from_pretrained(
    MODEL_ID,
    device_map=device,
)


# ---------------------------------------------------------------------
# Batch constructor
# ---------------------------------------------------------------------

def build_batch(batch_origins):

    context_frames = []
    truth_frames = []

    for _, row in batch_origins.iterrows():

        entity = row["entity"]
        origin = row["personal_origin"]

        case_id = (
            f"{entity}__"
            f"{origin.strftime('%Y%m%d')}"
        )

        context_start = (
            origin
            - context_span
        )

        target_start = (
            origin
            + freq
        )

        target_end = (
            origin
            + forecast_span
        )

        context = raw.loc[
            context_start:origin,
            entity,
        ]

        truth = raw.loc[
            target_start:target_end,
            entity,
        ]

        if len(context) != CONTEXT_STEPS:
            raise RuntimeError(
                f"{case_id}: expected "
                f"{CONTEXT_STEPS} context points, "
                f"got {len(context)}."
            )

        if len(truth) != FORECAST_STEPS:
            raise RuntimeError(
                f"{case_id}: expected "
                f"{FORECAST_STEPS} target points, "
                f"got {len(truth)}."
            )

        context_frames.append(
            pd.DataFrame(
                {
                    "id": case_id,
                    "timestamp": context.index,
                    "target": context.to_numpy(
                        dtype=float
                    ),
                }
            )
        )

        truth_frames.append(
            pd.DataFrame(
                {
                    "id": case_id,
                    "entity": entity,
                    "personal_origin": origin,
                    "timestamp": truth.index,
                    "truth": truth.to_numpy(
                        dtype=float
                    ),
                    "forecast_step": np.arange(
                        1,
                        FORECAST_STEPS + 1,
                    ),
                }
            )
        )

    return (
        pd.concat(
            context_frames,
            ignore_index=True,
        ),
        pd.concat(
            truth_frames,
            ignore_index=True,
        ),
    )


# ---------------------------------------------------------------------
# Forecast all required historical origins
# ---------------------------------------------------------------------

all_predictions = []

n_cases = len(origin_df)

n_batches = int(
    np.ceil(
        n_cases / BATCH_SIZE
    )
)

start_time = time.time()

for batch_number, start in enumerate(
    range(
        0,
        n_cases,
        BATCH_SIZE,
    ),
    start=1,
):

    end = min(
        start + BATCH_SIZE,
        n_cases,
    )

    batch_origins = origin_df.iloc[
        start:end
    ]

    context_df, truth_df = build_batch(
        batch_origins
    )

    batch_start = time.time()

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=FORECAST_STEPS,
        quantile_levels=[
            0.1,
            0.5,
            0.9,
        ],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        context_length=CONTEXT_STEPS,
        batch_size=BATCH_SIZE,
        freq="15min",
    )

    expected_rows = (
        len(batch_origins)
        * FORECAST_STEPS
    )

    if len(pred_df) != expected_rows:
        raise RuntimeError(
            f"Batch {batch_number}: expected "
            f"{expected_rows} prediction rows, "
            f"got {len(pred_df)}."
        )

    merged = pred_df.merge(
        truth_df,
        on=[
            "id",
            "timestamp",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != expected_rows:
        raise RuntimeError(
            f"Batch {batch_number}: merge "
            f"produced {len(merged)} rows, "
            f"expected {expected_rows}."
        )

    all_predictions.append(
        merged
    )

    batch_elapsed = (
        time.time()
        - batch_start
    )

    total_elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Batch {batch_number:03d}/"
        f"{n_batches:03d} | "
        f"cases {start + 1:04d}-"
        f"{end:04d} | "
        f"{batch_elapsed:.2f}s | "
        f"{total_elapsed:.1f}s total"
    )


# ---------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------

bank = pd.concat(
    all_predictions,
    ignore_index=True,
)

bank = bank.rename(
    columns={
        "predictions": "global_pred",
        "0.1": "q10",
        "0.5": "q50",
        "0.9": "q90",
    }
)


# ---------------------------------------------------------------------
# Residual convention
#
# Positive residual:
# Chronos under-predicted.
#
# Negative residual:
# Chronos over-predicted.
#
# The personalized correction will therefore be ADDED to global_pred.
# ---------------------------------------------------------------------

bank["residual"] = (
    bank["truth"]
    - bank["global_pred"]
)

bank["abs_error"] = (
    bank["residual"].abs()
)

bank["squared_error"] = (
    bank["residual"] ** 2
)


# ---------------------------------------------------------------------
# Time-of-day position
# ---------------------------------------------------------------------

bank["quarter_hour"] = (
    (
        bank["timestamp"].dt.hour
        * 4
    )
    +
    (
        bank["timestamp"].dt.minute
        // 15
    )
)

assert (
    bank["quarter_hour"]
    .between(
        0,
        95,
    )
    .all()
)


# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------

expected_total_rows = (
    len(origin_df)
    * FORECAST_STEPS
)

assert len(bank) == expected_total_rows

assert not bank.duplicated(
    [
        "entity",
        "personal_origin",
        "timestamp",
    ]
).any()

assert bank[
    [
        "truth",
        "global_pred",
        "residual",
    ]
].notna().all().all()


# ---------------------------------------------------------------------
# Daily residual summaries
# ---------------------------------------------------------------------

daily = (
    bank
    .groupby(
        [
            "entity",
            "personal_origin",
        ],
        as_index=False,
    )
    .agg(
        residual_mean=(
            "residual",
            "mean",
        ),
        residual_median=(
            "residual",
            "median",
        ),
        residual_std=(
            "residual",
            "std",
        ),
        mae=(
            "abs_error",
            "mean",
        ),
        mse=(
            "squared_error",
            "mean",
        ),
        truth_mean=(
            "truth",
            "mean",
        ),
        global_pred_mean=(
            "global_pred",
            "mean",
        ),
    )
)

daily["rmse"] = np.sqrt(
    daily["mse"]
)

daily = daily.drop(
    columns="mse"
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

bank_path = (
    OUT_DIR
    / "chronos2_validation_residual_bank.parquet"
)

daily_path = (
    OUT_DIR
    / "chronos2_validation_daily_residuals.csv"
)

bank[
    [
        "entity",
        "personal_origin",
        "timestamp",
        "forecast_step",
        "quarter_hour",
        "truth",
        "global_pred",
        "q10",
        "q50",
        "q90",
        "residual",
    ]
].to_parquet(
    bank_path,
    index=False,
)

daily.to_csv(
    daily_path,
    index=False,
)


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

runtime = (
    time.time()
    - start_time
)

print()
print("=" * 72)
print("Residual-bank summary")
print("=" * 72)

print()
print(
    f"Unique historical forecast days: "
    f"{len(daily):,}"
)

print(
    f"Per-timestep residual rows:       "
    f"{len(bank):,}"
)

print(
    f"Entities:                         "
    f"{bank['entity'].nunique()}"
)

print(
    f"Mean residual:                    "
    f"{bank['residual'].mean():.4f}"
)

print(
    f"Median residual:                  "
    f"{bank['residual'].median():.4f}"
)

print(
    f"Residual SD:                      "
    f"{bank['residual'].std():.4f}"
)

print(
    f"Mean absolute error:              "
    f"{bank['abs_error'].mean():.4f}"
)

print(
    f"Runtime seconds:                  "
    f"{runtime:.2f}"
)

print()
print("Saved:")
print(f"  {bank_path}")
print(f"  {daily_path}")
