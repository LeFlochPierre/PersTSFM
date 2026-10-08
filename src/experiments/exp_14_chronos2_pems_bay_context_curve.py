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
    "data/raw/pems_bay/PEMS-BAY.csv"
)

PROTOCOL_PATH = Path(
    "data/processed/pems_bay/"
    "pems_bay_evaluation_origins.csv"
)

NAIVE_PATH = Path(
    "results/pems_bay/baselines/"
    "pems_bay_naive_summary.csv"
)

OUT_DIR = Path(
    "results/pems_bay/context"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_ID = "amazon/chronos-2"

SPLIT = "validation"

RESOLUTION_MINUTES = 5
FORECAST_STEPS = 12

# 31 days is NOT used because Chronos-2 caps context at 8192 steps.
#
# 28 days = 8064 five-minute observations.
CONTEXT_HOURS = [
    3,
    6,
    12,
    24,
    48,
    72,
    120,
    168,
    240,
    336,
    504,
    672,
]

BATCH_SIZE = 32


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def context_label(hours):

    if hours < 24:
        return f"{hours}h"

    days = hours // 24

    return f"{days}d"


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
    protocol[
        "sensor"
    ].unique()
)


print("=" * 78)
print("PersTSFM Chronos-2 PEMS-BAY context curve")
print("=" * 78)

print()
print(f"Model:             {MODEL_ID}")
print(f"Validation sensors:{len(sensors):5d}")
print(f"Evaluation cases:  {len(protocol):5d}")
print(f"Forecast steps:    {FORECAST_STEPS:5d}")

print()
print(
    "Context grid: "
    + ", ".join(
        context_label(x)
        for x in CONTEXT_HOURS
    )
)


# ---------------------------------------------------------------------
# Load raw validation data
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

print()
print(
    f"Loaded {len(raw):,} timestamps "
    f"for {len(sensors)} sensors."
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
# Run context sweep
# ---------------------------------------------------------------------

all_predictions = []

total_start = time.time()


for context_hours in CONTEXT_HOURS:

    context_steps = int(
        context_hours
        * 60
        // RESOLUTION_MINUTES
    )

    if context_steps > 8192:
        raise RuntimeError(
            f"Requested context "
            f"{context_steps} exceeds "
            "Chronos-2 maximum 8192."
        )

    context_span = (
        context_steps - 1
    ) * pd.Timedelta(
        minutes=RESOLUTION_MINUTES
    )

    print()
    print("-" * 78)
    print(
        f"Context: "
        f"{context_label(context_hours)} "
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

        for _, row in (
            batch_protocol.iterrows()
        ):

            sensor = row["sensor"]

            eval_index = int(
                row["eval_index"]
            )

            origin = row[
                "eval_origin"
            ]

            case_id = (
                f"{sensor}__"
                f"{eval_index:02d}"
            )

            context_start = (
                origin
                - context_span
            )

            context = raw.loc[
                context_start:origin,
                sensor,
            ]

            truth = raw.loc[
                row["target_start"]:
                row["target_end"],
                sensor,
            ]

            if (
                len(context)
                != context_steps
            ):
                raise RuntimeError(
                    f"{case_id}: expected "
                    f"{context_steps} context "
                    f"rows, got {len(context)}."
                )

            if (
                len(truth)
                != FORECAST_STEPS
            ):
                raise RuntimeError(
                    f"{case_id}: expected "
                    f"{FORECAST_STEPS} target "
                    f"rows, got {len(truth)}."
                )

            context_frames.append(
                pd.DataFrame(
                    {
                        "id": case_id,
                        "timestamp": (
                            context.index
                        ),
                        "target": (
                            context.to_numpy(
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
                        "sensor": sensor,
                        "eval_index": (
                            eval_index
                        ),
                        "eval_origin": (
                            origin
                        ),
                        "hour": int(
                            origin.hour
                        ),
                        "weekday": (
                            origin.day_name()
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
            freq="5min",
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
                f"Context "
                f"{context_label(context_hours)}, "
                f"batch {batch_number}: expected "
                f"{expected_rows} rows, "
                f"got {len(merged)}."
            )

        merged[
            "context_hours"
        ] = context_hours

        merged[
            "context_steps"
        ] = context_steps

        all_predictions.append(
            merged
        )

        print(
            f"  batch "
            f"{batch_number:02d} | "
            f"cases "
            f"{start + 1:03d}-"
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
# Quantile crossing diagnostic
# ---------------------------------------------------------------------

crossing = (
    (
        pred["q10"]
        > pred["q50"]
    )
    |
    (
        pred["q50"]
        > pred["q90"]
    )
)

crossing_fraction = float(
    crossing.mean()
)

print()
print(
    f"Quantile-crossing rows: "
    f"{int(crossing.sum()):,} / "
    f"{len(pred):,} "
    f"({100 * crossing_fraction:.4f}%)"
)


# ---------------------------------------------------------------------
# Evaluation quantiles
#
# Preserve raw quantiles, but sort only for interval coverage.
# ---------------------------------------------------------------------

q_values = pred[
    [
        "q10",
        "q50",
        "q90",
    ]
].to_numpy(
    dtype=float
)

q_sorted = np.sort(
    q_values,
    axis=1,
)

pred[
    "q10_eval"
] = q_sorted[:, 0]

pred[
    "q50_eval"
] = q_sorted[:, 1]

pred[
    "q90_eval"
] = q_sorted[:, 2]


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

pred["covered_80"] = (
    (
        pred["truth"]
        >= pred["q10_eval"]
    )
    &
    (
        pred["truth"]
        <= pred["q90_eval"]
    )
)

pred["interval_width_80"] = (
    pred["q90_eval"]
    - pred["q10_eval"]
)

pred["negative_prediction"] = (
    pred["prediction"]
    < 0
)


# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------

expected_rows = (
    len(CONTEXT_HOURS)
    * len(protocol)
    * FORECAST_STEPS
)

if len(pred) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows:,} "
        f"prediction rows, "
        f"got {len(pred):,}."
    )

duplicates = pred.duplicated(
    [
        "context_hours",
        "sensor",
        "eval_index",
        "timestamp",
    ]
).sum()

if duplicates != 0:
    raise RuntimeError(
        f"Found {duplicates} "
        "duplicate prediction rows."
    )


# ---------------------------------------------------------------------
# Case metrics
# ---------------------------------------------------------------------

case_metrics = (
    pred
    .groupby(
        [
            "context_hours",
            "context_steps",
            "sensor",
            "eval_index",
            "eval_origin",
            "hour",
            "weekday",
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
        negative_fraction=(
            "negative_prediction",
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
# 24-hour context reference
# ---------------------------------------------------------------------

reference = (
    case_metrics[
        case_metrics[
            "context_hours"
        ] == 24
    ]
    [
        [
            "sensor",
            "eval_index",
            "mae",
        ]
    ]
    .rename(
        columns={
            "mae": "mae_24h",
        }
    )
)

case_metrics = case_metrics.merge(
    reference,
    on=[
        "sensor",
        "eval_index",
    ],
    how="left",
    validate="many_to_one",
)

case_metrics[
    "delta_mae_vs_24h"
] = (
    case_metrics["mae"]
    - case_metrics["mae_24h"]
)


# ---------------------------------------------------------------------
# Entity-level aggregation
# ---------------------------------------------------------------------

entity_metrics = (
    case_metrics
    .groupby(
        [
            "context_hours",
            "sensor",
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
        mean_delta_mae_vs_24h=(
            "delta_mae_vs_24h",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Overall context summary
# ---------------------------------------------------------------------

summary_rows = []

for context_hours in CONTEXT_HOURS:

    cases = case_metrics[
        case_metrics[
            "context_hours"
        ] == context_hours
    ]

    entities_part = entity_metrics[
        entity_metrics[
            "context_hours"
        ] == context_hours
    ]

    summary_rows.append(
        {
            "context_hours": (
                context_hours
            ),

            "context_label": (
                context_label(
                    context_hours
                )
            ),

            "context_steps": int(
                context_hours
                * 60
                // RESOLUTION_MINUTES
            ),

            "mean_mae": float(
                cases[
                    "mae"
                ].mean()
            ),

            "median_mae": float(
                cases[
                    "mae"
                ].median()
            ),

            "mean_rmse": float(
                cases[
                    "rmse"
                ].mean()
            ),

            "mean_coverage_80": float(
                cases[
                    "coverage_80"
                ].mean()
            ),

            "mean_interval_width_80": float(
                cases[
                    "interval_width_80"
                ].mean()
            ),

            "mean_delta_mae_vs_24h": float(
                cases[
                    "delta_mae_vs_24h"
                ].mean()
            ),

            "case_better_than_24h_fraction": float(
                (
                    cases[
                        "delta_mae_vs_24h"
                    ]
                    < 0
                ).mean()
            ),

            "entity_mean_mae": float(
                entities_part[
                    "mean_mae"
                ].mean()
            ),

            "entity_better_than_24h_fraction": float(
                (
                    entities_part[
                        "mean_delta_mae_vs_24h"
                    ]
                    < 0
                ).mean()
            ),

            "negative_prediction_fraction": float(
                cases[
                    "negative_fraction"
                ].mean()
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Hour-specific performance
# ---------------------------------------------------------------------

hour_summary = (
    case_metrics
    .groupby(
        [
            "context_hours",
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

pred_path = (
    OUT_DIR
    / "chronos2_pems_bay_context_predictions.parquet"
)

case_path = (
    OUT_DIR
    / "chronos2_pems_bay_context_case_metrics.csv"
)

entity_path = (
    OUT_DIR
    / "chronos2_pems_bay_context_entity_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "chronos2_pems_bay_context_summary.csv"
)

hour_path = (
    OUT_DIR
    / "chronos2_pems_bay_context_by_hour.csv"
)

pred.to_parquet(
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

hour_summary.to_csv(
    hour_path,
    index=False,
)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5)
)

x = np.arange(
    len(summary)
)

ax.plot(
    x,
    summary["mean_mae"],
    marker="o",
    label="Chronos-2",
)

if NAIVE_PATH.exists():

    naive = pd.read_csv(
        NAIVE_PATH
    )

    for method in [
        "persistence",
        "daily_seasonal",
        "weekly_seasonal",
    ]:

        value = float(
            naive.loc[
                naive["method"] == method,
                "mean_mae",
            ].iloc[0]
        )

        ax.axhline(
            value,
            linestyle="--",
            label=method,
        )

ax.set_xticks(
    x
)

ax.set_xticklabels(
    summary[
        "context_label"
    ]
)

ax.set_xlabel(
    "Same-sensor context"
)

ax.set_ylabel(
    "Mean MAE"
)

ax.set_title(
    "PEMS-BAY: Chronos-2 context curve"
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "chronos2_pems_bay_context_curve.png"
)

fig.savefig(
    figure_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Best context
# ---------------------------------------------------------------------

best = summary.sort_values(
    "mean_mae"
).iloc[0]

mae_24h = float(
    summary.loc[
        summary[
            "context_hours"
        ] == 24,
        "mean_mae",
    ].iloc[0]
)

best_improvement = (
    100
    * (
        mae_24h
        - float(
            best["mean_mae"]
        )
    )
    / mae_24h
)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

runtime = (
    time.time()
    - total_start
)

print()
print("=" * 78)
print("Chronos-2 PEMS-BAY context summary")
print("=" * 78)

display = summary.copy()

for col in [
    "mean_mae",
    "median_mae",
    "mean_rmse",
    "mean_coverage_80",
    "mean_interval_width_80",
    "mean_delta_mae_vs_24h",
    "entity_mean_mae",
]:
    display[col] = (
        display[col]
        .round(4)
    )

for col in [
    "case_better_than_24h_fraction",
    "entity_better_than_24h_fraction",
    "negative_prediction_fraction",
]:
    display[col] = (
        100
        * display[col]
    ).round(1)

print()
print(
    display.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("Best Chronos-2 PEMS-BAY context")
print("=" * 78)

print()
print(
    f"Best context: "
    f"{best['context_label']}"
)

print(
    f"Mean MAE:     "
    f"{best['mean_mae']:.4f}"
)

print(
    f"Improvement vs 24h: "
    f"{best_improvement:.2f}%"
)


# ---------------------------------------------------------------------
# Compare against naive baselines
# ---------------------------------------------------------------------

if NAIVE_PATH.exists():

    naive = pd.read_csv(
        NAIVE_PATH
    )

    print()
    print("=" * 78)
    print("Naive comparison")
    print("=" * 78)

    naive_display = naive[
        [
            "method",
            "mean_mae",
            "median_mae",
            "mean_rmse",
        ]
    ].copy()

    for col in [
        "mean_mae",
        "median_mae",
        "mean_rmse",
    ]:
        naive_display[col] = (
            naive_display[col]
            .round(4)
        )

    print()
    print(
        naive_display.to_string(
            index=False
        )
    )

    print()
    print(
        f"Best Chronos MAE: "
        f"{best['mean_mae']:.4f}"
    )


print()
print(
    f"Quantile crossing: "
    f"{100 * crossing_fraction:.4f}%"
)

print(
    f"Runtime seconds: "
    f"{runtime:.2f}"
)

print()
print("Saved:")
print(f"  {pred_path}")
print(f"  {case_path}")
print(f"  {entity_path}")
print(f"  {summary_path}")
print(f"  {hour_path}")
print(f"  {figure_path}")
