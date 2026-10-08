from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "results/context_threshold"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_ID = "amazon/chronos-2"

SPLIT = "validation"

FREQUENCY_MINUTES = 15
FORECAST_STEPS = 96

CONTEXT_DAYS = [
    1,
    2,
    4,
    8,
    16,
    21,
]

BATCH_SIZE = 64


# ---------------------------------------------------------------------
# Load protocol
# ---------------------------------------------------------------------

protocol = pd.read_csv(
    PROTOCOL_PATH,
    parse_dates=[
        "eval_origin",
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
            "eval_index",
        ]
    )
    .reset_index(drop=True)
)

entities = sorted(
    protocol["entity"].unique()
)


print("=" * 78)
print("PersTSFM Chronos-2 in-context personalization curve")
print("=" * 78)

print()
print(f"Model:              {MODEL_ID}")
print(f"Entities:           {len(entities)}")
print(f"Evaluation cases:   {len(protocol)}")
print(f"Forecast steps:     {FORECAST_STEPS}")
print(
    "Context days:       "
    + ", ".join(
        str(x)
        for x in CONTEXT_DAYS
    )
)


# ---------------------------------------------------------------------
# Load validation series
# ---------------------------------------------------------------------

header = pd.read_csv(
    RAW_PATH,
    sep=";",
    decimal=",",
    nrows=0,
)

timestamp_col = header.columns[0]

raw = pd.read_csv(
    RAW_PATH,
    sep=";",
    decimal=",",
    usecols=[
        timestamp_col,
        *entities,
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

print()
print(
    f"Loaded {len(raw):,} timestamps "
    f"for {len(entities)} entities."
)


# ---------------------------------------------------------------------
# Load Chronos
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
# Forecast for each context length
# ---------------------------------------------------------------------

all_predictions = []

total_start = time.time()


for context_days in CONTEXT_DAYS:

    context_steps = (
        context_days
        * 96
    )

    context_span = (
        context_steps - 1
    ) * pd.Timedelta(
        minutes=FREQUENCY_MINUTES
    )

    print()
    print("-" * 78)
    print(
        f"Context: {context_days} days "
        f"({context_steps} steps)"
    )
    print("-" * 78)

    n_cases = len(protocol)

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

        context_frames = []
        truth_frames = []

        for _, row in batch_protocol.iterrows():

            entity = row["entity"]

            eval_index = int(
                row["eval_index"]
            )

            origin = row[
                "eval_origin"
            ]

            case_id = (
                f"{entity}__"
                f"{eval_index:02d}"
            )

            context_start = (
                origin
                - context_span
            )

            if (
                context_start
                < row[
                    "first_active_timestamp"
                ]
            ):
                raise RuntimeError(
                    f"{entity} eval {eval_index}: "
                    f"{context_days}-day context "
                    "starts before active history."
                )

            context = raw.loc[
                context_start:origin,
                entity,
            ]

            truth = raw.loc[
                row["target_start"]:
                row["target_end"],
                entity,
            ]

            if len(context) != context_steps:
                raise RuntimeError(
                    f"{case_id}: expected "
                    f"{context_steps} context rows, "
                    f"got {len(context)}."
                )

            if len(truth) != FORECAST_STEPS:
                raise RuntimeError(
                    f"{case_id}: expected "
                    f"{FORECAST_STEPS} target rows, "
                    f"got {len(truth)}."
                )

            context_frames.append(
                pd.DataFrame(
                    {
                        "id": case_id,
                        "timestamp": (
                            context.index
                        ),
                        "target": (
                            context
                            .to_numpy(
                                dtype=float
                            )
                        ),
                    }
                )
            )

            truth_frames.append(
                pd.DataFrame(
                    {
                        "id": case_id,
                        "entity": entity,
                        "eval_index": (
                            eval_index
                        ),
                        "eval_origin": (
                            origin
                        ),
                        "timestamp": (
                            truth.index
                        ),
                        "truth": (
                            truth.to_numpy(
                                dtype=float
                            )
                        ),
                    }
                )
            )

        context_df = pd.concat(
            context_frames,
            ignore_index=True,
        )

        truth_df = pd.concat(
            truth_frames,
            ignore_index=True,
        )

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
            context_length=context_steps,
            batch_size=BATCH_SIZE,
            freq="15min",
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

        expected_rows = (
            len(batch_protocol)
            * FORECAST_STEPS
        )

        if len(merged) != expected_rows:
            raise RuntimeError(
                f"Context {context_days}, "
                f"batch {batch_number}: "
                f"expected {expected_rows} rows, "
                f"got {len(merged)}."
            )

        merged[
            "context_days"
        ] = context_days

        merged[
            "context_steps"
        ] = context_steps

        all_predictions.append(
            merged
        )

        print(
            f"  batch "
            f"{batch_number:02d} | "
            f"cases {start + 1:03d}-"
            f"{end:03d}"
        )


# ---------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------

pred = pd.concat(
    all_predictions,
    ignore_index=True,
)

pred = pred.rename(
    columns={
        "predictions": "prediction",
        "0.1": "q10",
        "0.5": "q50",
        "0.9": "q90",
    }
)


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

pred["error"] = (
    pred["prediction"]
    - pred["truth"]
)

pred["abs_error"] = (
    pred["error"].abs()
)

pred["squared_error"] = (
    pred["error"] ** 2
)


# ---------------------------------------------------------------------
# Case metrics
# ---------------------------------------------------------------------

case_metrics = (
    pred
    .groupby(
        [
            "context_days",
            "context_steps",
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
    )
)

case_metrics["rmse"] = np.sqrt(
    case_metrics["mse"]
)

case_metrics = case_metrics.drop(
    columns="mse"
)


# ---------------------------------------------------------------------
# Use 4-day context as our current reference
# ---------------------------------------------------------------------

reference = (
    case_metrics[
        case_metrics[
            "context_days"
        ] == 4
    ]
    [
        [
            "entity",
            "eval_index",
            "mae",
        ]
    ]
    .rename(
        columns={
            "mae": "mae_4day",
        }
    )
)

case_metrics = case_metrics.merge(
    reference,
    on=[
        "entity",
        "eval_index",
    ],
    how="left",
    validate="many_to_one",
)

case_metrics[
    "delta_mae_vs_4day"
] = (
    case_metrics["mae"]
    - case_metrics["mae_4day"]
)


# ---------------------------------------------------------------------
# Entity metrics
# ---------------------------------------------------------------------

entity_metrics = (
    case_metrics
    .groupby(
        [
            "context_days",
            "entity",
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
        mean_delta_mae_vs_4day=(
            "delta_mae_vs_4day",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------

summary_rows = []

for context_days in CONTEXT_DAYS:

    cases = case_metrics[
        case_metrics[
            "context_days"
        ] == context_days
    ]

    entities_part = entity_metrics[
        entity_metrics[
            "context_days"
        ] == context_days
    ]

    summary_rows.append(
        {
            "context_days": (
                context_days
            ),

            "context_steps": (
                context_days * 96
            ),

            "mean_mae": (
                cases["mae"].mean()
            ),

            "median_mae": (
                cases["mae"].median()
            ),

            "mean_rmse": (
                cases["rmse"].mean()
            ),

            "mean_delta_mae_vs_4day": (
                cases[
                    "delta_mae_vs_4day"
                ].mean()
            ),

            "case_better_than_4day_fraction": (
                (
                    cases[
                        "delta_mae_vs_4day"
                    ]
                    < 0
                ).mean()
            ),

            "entity_mean_mae": (
                entities_part[
                    "mean_mae"
                ].mean()
            ),

            "entity_better_than_4day_fraction": (
                (
                    entities_part[
                        "mean_delta_mae_vs_4day"
                    ]
                    < 0
                ).mean()
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

pred_path = (
    OUT_DIR
    / "chronos2_context_threshold_predictions.parquet"
)

case_path = (
    OUT_DIR
    / "chronos2_context_threshold_case_metrics.csv"
)

entity_path = (
    OUT_DIR
    / "chronos2_context_threshold_entity_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "chronos2_context_threshold_summary.csv"
)

pred[
    [
        "context_days",
        "context_steps",
        "entity",
        "eval_index",
        "eval_origin",
        "timestamp",
        "truth",
        "prediction",
        "q10",
        "q50",
        "q90",
    ]
].to_parquet(
    pred_path,
    index=False,
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


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    summary["context_days"],
    summary["mean_mae"],
    marker="o",
    label="Mean MAE",
)

ax.plot(
    summary["context_days"],
    summary["median_mae"],
    marker="o",
    label="Median MAE",
)

ax.set_xlabel(
    "Same-entity context available to Chronos (days)"
)

ax.set_ylabel(
    "MAE"
)

ax.set_title(
    "In-context personalization curve"
)

ax.set_xscale(
    "log",
    base=2,
)

ax.set_xticks(
    CONTEXT_DAYS
)

ax.set_xticklabels(
    CONTEXT_DAYS
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "chronos2_context_threshold_curve.png"
)

fig.savefig(
    figure_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

runtime = (
    time.time()
    - total_start
)

print()
print("=" * 78)
print("In-context personalization summary")
print("=" * 78)

display = summary.copy()

for col in [
    "mean_mae",
    "median_mae",
    "mean_rmse",
    "mean_delta_mae_vs_4day",
    "entity_mean_mae",
]:
    display[col] = (
        display[col]
        .round(4)
    )

for col in [
    "case_better_than_4day_fraction",
    "entity_better_than_4day_fraction",
]:
    display[col] = (
        display[col]
        * 100
    ).round(1)

print()
print(
    display.to_string(
        index=False
    )
)

print()
print(
    f"Runtime seconds: {runtime:.2f}"
)

print()
print("Saved:")
print(f"  {pred_path}")
print(f"  {case_path}")
print(f"  {entity_path}")
print(f"  {summary_path}")
print(f"  {figure_path}")
