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

CONTEXT_STEPS = 384
FORECAST_STEPS = 96

QUANTILES = [
    0.1,
    0.5,
    0.9,
]

BATCH_SIZE = 32


# ---------------------------------------------------------------------
# Load protocol
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

protocol = (
    protocol[
        protocol["split"] == SPLIT
    ]
    .copy()
    .sort_values(
        [
            "entity",
            "eval_index",
        ]
    )
    .reset_index(drop=True)
)

entities = sorted(
    protocol["entity"].unique()
)

print("=" * 72)
print("PersTSFM Chronos-2 validation cache")
print("=" * 72)

print()
print(f"Model:               {MODEL_ID}")
print(f"Split:               {SPLIT}")
print(f"Entities:            {len(entities)}")
print(f"Evaluation cases:    {len(protocol)}")
print(f"Context steps:       {CONTEXT_STEPS}")
print(f"Forecast steps:      {FORECAST_STEPS}")
print(f"Batch size:          {BATCH_SIZE}")


# ---------------------------------------------------------------------
# Load validation time series
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
# Load Chronos-2
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
# Helper: construct one batch
# ---------------------------------------------------------------------

def build_batch(batch_protocol):

    context_frames = []
    truth_frames = []

    for _, row in batch_protocol.iterrows():

        entity = row["entity"]
        eval_index = int(
            row["eval_index"]
        )

        case_id = (
            f"{entity}__"
            f"{eval_index:02d}"
        )

        context = raw.loc[
            row["context_start"]:
            row["context_end"],
            entity,
        ]

        truth = raw.loc[
            row["target_start"]:
            row["target_end"],
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

        context_frame = pd.DataFrame(
            {
                "id": case_id,
                "timestamp": context.index,
                "target": context.to_numpy(
                    dtype=float
                ),
            }
        )

        truth_frame = pd.DataFrame(
            {
                "id": case_id,
                "entity": entity,
                "eval_index": eval_index,
                "eval_origin": row[
                    "eval_origin"
                ],
                "timestamp": truth.index,
                "truth": truth.to_numpy(
                    dtype=float
                ),
            }
        )

        context_frames.append(
            context_frame
        )

        truth_frames.append(
            truth_frame
        )

    context_df = pd.concat(
        context_frames,
        ignore_index=True,
    )

    truth_df = pd.concat(
        truth_frames,
        ignore_index=True,
    )

    return context_df, truth_df


# ---------------------------------------------------------------------
# Forecast in batches
# ---------------------------------------------------------------------

all_predictions = []

n_cases = len(protocol)
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

    batch_protocol = protocol.iloc[
        start:end
    ]

    context_df, truth_df = build_batch(
        batch_protocol
    )

    batch_start = time.time()

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=FORECAST_STEPS,
        quantile_levels=QUANTILES,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        context_length=CONTEXT_STEPS,
        batch_size=BATCH_SIZE,
        freq="15min",
    )

    expected_rows = (
        len(batch_protocol)
        * FORECAST_STEPS
    )

    if len(pred_df) != expected_rows:
        raise RuntimeError(
            f"Batch {batch_number}: expected "
            f"{expected_rows} predictions, "
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
            f"Batch {batch_number}: timestamp "
            f"merge produced {len(merged)} rows, "
            f"expected {expected_rows}."
        )

    if merged["truth"].isna().any():
        raise RuntimeError(
            f"Batch {batch_number}: missing "
            "ground-truth values."
        )

    all_predictions.append(
        merged
    )

    elapsed_batch = (
        time.time()
        - batch_start
    )

    elapsed_total = (
        time.time()
        - start_time
    )

    print(
        f"Batch {batch_number:02d}/"
        f"{n_batches:02d} | "
        f"cases {start + 1:03d}-"
        f"{end:03d} | "
        f"{elapsed_batch:.1f}s batch | "
        f"{elapsed_total:.1f}s total"
    )


# ---------------------------------------------------------------------
# Combine predictions
# ---------------------------------------------------------------------

predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

predictions = predictions.rename(
    columns={
        "predictions": "global_pred",
        "0.1": "q10",
        "0.5": "q50",
        "0.9": "q90",
    }
)

keep_cols = [
    "entity",
    "eval_index",
    "eval_origin",
    "timestamp",
    "truth",
    "global_pred",
    "q10",
    "q50",
    "q90",
]

predictions = predictions[
    keep_cols
].copy()

predictions = predictions.sort_values(
    [
        "entity",
        "eval_index",
        "timestamp",
    ]
).reset_index(drop=True)


# ---------------------------------------------------------------------
# Final safety checks
# ---------------------------------------------------------------------

expected_total_rows = (
    len(protocol)
    * FORECAST_STEPS
)

assert (
    len(predictions)
    == expected_total_rows
)

assert not predictions.duplicated(
    [
        "entity",
        "eval_index",
        "timestamp",
    ]
).any()

assert predictions[
    [
        "truth",
        "global_pred",
        "q10",
        "q50",
        "q90",
    ]
].notna().all().all()

# ---------------------------------------------------------------------
# Quantile-crossing diagnostics
#
# Quantile forecasts from neural models are not guaranteed to be
# perfectly ordered at every timestamp. Record crossing rather than
# crashing the experiment.
# ---------------------------------------------------------------------

predictions["cross_q10_q50"] = (
    predictions["q10"]
    > predictions["q50"]
)

predictions["cross_q50_q90"] = (
    predictions["q50"]
    > predictions["q90"]
)

predictions["cross_q10_q90"] = (
    predictions["q10"]
    > predictions["q90"]
)

predictions["quantile_crossing"] = (
    predictions[
        [
            "cross_q10_q50",
            "cross_q50_q90",
            "cross_q10_q90",
        ]
    ]
    .any(axis=1)
)

n_crossing = int(
    predictions["quantile_crossing"].sum()
)

crossing_fraction = (
    n_crossing
    / len(predictions)
)

print()
print(
    f"Quantile-crossing rows: "
    f"{n_crossing:,} / "
    f"{len(predictions):,} "
    f"({100 * crossing_fraction:.4f}%)"
)


# ---------------------------------------------------------------------
# Monotone rearrangement for interval evaluation only
#
# Preserve the raw Chronos quantiles above, but sort the three
# predicted quantiles before calculating interval diagnostics.
# The point forecast remains completely unchanged.
# ---------------------------------------------------------------------

quantile_values = predictions[
    [
        "q10",
        "q50",
        "q90",
    ]
].to_numpy()

quantile_values_sorted = np.sort(
    quantile_values,
    axis=1,
)

predictions["q10_eval"] = (
    quantile_values_sorted[:, 0]
)

predictions["q50_eval"] = (
    quantile_values_sorted[:, 1]
)

predictions["q90_eval"] = (
    quantile_values_sorted[:, 2]
)


# ---------------------------------------------------------------------
# Per-case metrics
# ---------------------------------------------------------------------

predictions["error"] = (
    predictions["global_pred"]
    - predictions["truth"]
)

predictions["abs_error"] = (
    predictions["error"].abs()
)

predictions["squared_error"] = (
    predictions["error"] ** 2
)

predictions["covered_80"] = (
    (
        predictions["truth"]
        >= predictions["q10_eval"]
    )
    &
    (
        predictions["truth"]
        <= predictions["q90_eval"]
    )
)

predictions["interval_width_80"] = (
    predictions["q90_eval"]
    - predictions["q10_eval"]
)


case_metrics = (
    predictions
    .groupby(
        [
            "entity",
            "eval_index",
            "eval_origin",
        ],
        as_index=False,
    )
    .agg(
        mae=(
            "abs_error",
            "mean",
        ),
        mse=(
            "squared_error",
            "mean",
        ),
        coverage_80=(
            "covered_80",
            "mean",
        ),
        interval_width_80=(
            "interval_width_80",
            "mean",
        ),
        mean_truth=(
            "truth",
            "mean",
        ),
        mean_prediction=(
            "global_pred",
            "mean",
        ),
    )
)

case_metrics["rmse"] = np.sqrt(
    case_metrics["mse"]
)

case_metrics = case_metrics.drop(
    columns=[
        "mse",
    ]
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

prediction_path = (
    OUT_DIR
    / "chronos2_validation_predictions.parquet"
)

metric_path = (
    OUT_DIR
    / "chronos2_validation_case_metrics.csv"
)

predictions[
    keep_cols
].to_parquet(
    prediction_path,
    index=False,
)

case_metrics.to_csv(
    metric_path,
    index=False,
)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

total_runtime = (
    time.time()
    - start_time
)

negative_fraction = float(
    (
        predictions["global_pred"]
        < 0
    ).mean()
)

summary = {
    "n_cases": len(case_metrics),
    "median_mae": (
        case_metrics["mae"].median()
    ),
    "mean_mae": (
        case_metrics["mae"].mean()
    ),
    "median_rmse": (
        case_metrics["rmse"].median()
    ),
    "mean_rmse": (
        case_metrics["rmse"].mean()
    ),
    "mean_coverage_80": (
        case_metrics[
            "coverage_80"
        ].mean()
    ),
    "mean_interval_width_80": (
        case_metrics[
            "interval_width_80"
        ].mean()
    ),
    "negative_prediction_fraction": (
        negative_fraction
    ),
    "runtime_seconds": (
        total_runtime
    ),
}

print()
print("=" * 72)
print("Chronos-2 validation summary")
print("=" * 72)

for key, value in summary.items():

    if isinstance(value, float):
        print(
            f"{key:32s}: "
            f"{value:.4f}"
        )

    else:
        print(
            f"{key:32s}: "
            f"{value}"
        )

print()
print("Saved:")
print(f"  {prediction_path}")
print(f"  {metric_path}")
