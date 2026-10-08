from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from timesfm3 import (
    ModelConfig,
    TimesFM3Evaluator,
)


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

CHRONOS_PATH = Path(
    "results/pems_bay/context/"
    "chronos2_pems_bay_context_summary.csv"
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

MODEL_ID = "google/timesfm-3.0-pytorch"

SPLIT = "validation"

RESOLUTION_MINUTES = 5
FORECAST_STEPS = 12

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

BATCH_SIZE = 16


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def context_label(hours):

    if hours < 24:
        return f"{hours}h"

    days = hours // 24

    return f"{days}d"


# ---------------------------------------------------------------------
# Protocol
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


print("=" * 78)
print("PersTSFM TimesFM 3 PEMS-BAY context curve")
print("=" * 78)

print()
print(f"Model:              {MODEL_ID}")
print(f"Validation sensors: {len(sensors)}")
print(f"Evaluation cases:   {len(protocol)}")
print(f"Forecast steps:     {FORECAST_STEPS}")

print()
print(
    "Context grid: "
    + ", ".join(
        context_label(x)
        for x in CONTEXT_HOURS
    )
)


# ---------------------------------------------------------------------
# Raw data
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
# TimesFM
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

config = ModelConfig(
    checkpoint_path=MODEL_ID,
    per_core_batch_size=BATCH_SIZE,
    device=device,
)

print()
print(f"Loading {MODEL_ID}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        "GPU: "
        + torch.cuda.get_device_name(0)
    )

forecaster = TimesFM3Evaluator(
    config
)


# ---------------------------------------------------------------------
# Context sweep
# ---------------------------------------------------------------------

prediction_rows = []

total_start = time.time()


for context_hours in CONTEXT_HOURS:

    context_steps = int(
        context_hours
        * 60
        // RESOLUTION_MINUTES
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

        contexts = []
        records = []

        for _, row in batch_protocol.iterrows():

            sensor = row["sensor"]

            eval_index = int(
                row["eval_index"]
            )

            origin = row[
                "eval_origin"
            ]

            context_start = (
                origin
                - context_span
            )

            context = raw.loc[
                context_start:origin,
                sensor,
            ].to_numpy(
                dtype=np.float32
            )

            truth_series = raw.loc[
                row["target_start"]:
                row["target_end"],
                sensor,
            ]

            truth = truth_series.to_numpy(
                dtype=np.float32
            )

            if len(context) != context_steps:
                raise RuntimeError(
                    f"{sensor} eval {eval_index}: "
                    f"expected {context_steps} "
                    f"context points, "
                    f"got {len(context)}."
                )

            if len(truth) != FORECAST_STEPS:
                raise RuntimeError(
                    f"{sensor} eval {eval_index}: "
                    f"expected {FORECAST_STEPS} "
                    f"target points, "
                    f"got {len(truth)}."
                )

            contexts.append(
                context
            )

            records.append(
                {
                    "sensor": sensor,
                    "eval_index": eval_index,
                    "eval_origin": origin,
                    "hour": int(
                        origin.hour
                    ),
                    "weekday": (
                        origin.day_name()
                    ),
                    "timestamps": (
                        truth_series.index
                    ),
                    "truth": truth,
                }
            )


        outputs = list(
            forecaster.predict_batch(
                contexts=contexts,
                horizon=FORECAST_STEPS,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )


        if len(outputs) != len(records):
            raise RuntimeError(
                f"Expected {len(records)} "
                f"outputs, received "
                f"{len(outputs)}."
            )


        for output, record in zip(
            outputs,
            records,
        ):

            point = np.asarray(
                output.forecast,
                dtype=float,
            )

            quantiles = np.asarray(
                output.quantiles,
                dtype=float,
            )

            if point.shape != (
                FORECAST_STEPS,
            ):
                raise RuntimeError(
                    "Unexpected point shape: "
                    f"{point.shape}"
                )

            if quantiles.shape != (
                FORECAST_STEPS,
                9,
            ):
                raise RuntimeError(
                    "Unexpected quantile shape: "
                    f"{quantiles.shape}"
                )

            q10 = quantiles[:, 0]
            q50 = quantiles[:, 4]
            q90 = quantiles[:, 8]

            for step in range(
                FORECAST_STEPS
            ):

                prediction_rows.append(
                    {
                        "context_hours": (
                            context_hours
                        ),
                        "context_label": (
                            context_label(
                                context_hours
                            )
                        ),
                        "context_steps": (
                            context_steps
                        ),
                        "sensor": (
                            record["sensor"]
                        ),
                        "eval_index": (
                            record[
                                "eval_index"
                            ]
                        ),
                        "eval_origin": (
                            record[
                                "eval_origin"
                            ]
                        ),
                        "hour": (
                            record["hour"]
                        ),
                        "weekday": (
                            record["weekday"]
                        ),
                        "timestamp": (
                            record[
                                "timestamps"
                            ][step]
                        ),
                        "truth": float(
                            record[
                                "truth"
                            ][step]
                        ),
                        "prediction": float(
                            point[step]
                        ),
                        "q10": float(
                            q10[step]
                        ),
                        "q50": float(
                            q50[step]
                        ),
                        "q90": float(
                            q90[step]
                        ),
                    }
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

pred = pd.DataFrame(
    prediction_rows
)

expected_rows = (
    len(CONTEXT_HOURS)
    * len(protocol)
    * FORECAST_STEPS
)

if len(pred) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows:,} rows, "
        f"found {len(pred):,}."
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

pred["covered_80"] = (
    (
        pred["truth"]
        >= pred["q10"]
    )
    &
    (
        pred["truth"]
        <= pred["q90"]
    )
)

pred["interval_width_80"] = (
    pred["q90"]
    - pred["q10"]
)

pred["negative_prediction"] = (
    pred["prediction"]
    < 0
)


# ---------------------------------------------------------------------
# Case metrics
# ---------------------------------------------------------------------

case_metrics = (
    pred
    .groupby(
        [
            "context_hours",
            "context_label",
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
# 24h baseline
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
# Entity metrics
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
        mean_delta_mae_vs_24h=(
            "delta_mae_vs_24h",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Summary
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
                cases["mae"].mean()
            ),
            "median_mae": float(
                cases["mae"].median()
            ),
            "mean_rmse": float(
                cases["rmse"].mean()
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
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Hour summary
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
# Save TimesFM results
# ---------------------------------------------------------------------

pred_path = (
    OUT_DIR
    / "timesfm3_pems_bay_context_predictions.parquet"
)

case_path = (
    OUT_DIR
    / "timesfm3_pems_bay_context_case_metrics.csv"
)

entity_path = (
    OUT_DIR
    / "timesfm3_pems_bay_context_entity_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "timesfm3_pems_bay_context_summary.csv"
)

hour_path = (
    OUT_DIR
    / "timesfm3_pems_bay_context_by_hour.csv"
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
# Cross-model comparison
# ---------------------------------------------------------------------

chronos = pd.read_csv(
    CHRONOS_PATH
)

comparison = summary[
    [
        "context_hours",
        "context_label",
        "mean_mae",
        "median_mae",
    ]
].rename(
    columns={
        "mean_mae": (
            "timesfm_mean_mae"
        ),
        "median_mae": (
            "timesfm_median_mae"
        ),
    }
)

comparison = comparison.merge(
    chronos[
        [
            "context_hours",
            "mean_mae",
            "median_mae",
        ]
    ].rename(
        columns={
            "mean_mae": (
                "chronos_mean_mae"
            ),
            "median_mae": (
                "chronos_median_mae"
            ),
        }
    ),
    on="context_hours",
    how="inner",
    validate="one_to_one",
)

comparison[
    "timesfm_minus_chronos_mae"
] = (
    comparison[
        "timesfm_mean_mae"
    ]
    - comparison[
        "chronos_mean_mae"
    ]
)

comparison_path = (
    OUT_DIR
    / "chronos2_vs_timesfm3_pems_bay_context.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5)
)

x = np.arange(
    len(comparison)
)

ax.plot(
    x,
    comparison[
        "chronos_mean_mae"
    ],
    marker="o",
    label="Chronos-2",
)

ax.plot(
    x,
    comparison[
        "timesfm_mean_mae"
    ],
    marker="o",
    label="TimesFM 3",
)

if NAIVE_PATH.exists():

    naive = pd.read_csv(
        NAIVE_PATH
    )

    weekly = float(
        naive.loc[
            naive["method"]
            == "weekly_seasonal",
            "mean_mae",
        ].iloc[0]
    )

    ax.axhline(
        weekly,
        linestyle="--",
        label="Weekly seasonal naive",
    )

ax.set_xticks(
    x
)

ax.set_xticklabels(
    comparison[
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
    "PEMS-BAY context scaling across TSFMs"
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "chronos2_vs_timesfm3_pems_bay_context.png"
)

fig.savefig(
    figure_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Best and saturation contexts
# ---------------------------------------------------------------------

best = summary.sort_values(
    "mean_mae"
).iloc[0]

best_mae = float(
    best["mean_mae"]
)

# Earliest context within 1% of best MAE.
threshold = (
    best_mae * 1.01
)

within_one_percent = (
    summary[
        summary[
            "mean_mae"
        ] <= threshold
    ]
    .sort_values(
        "context_hours"
    )
)

saturation = (
    within_one_percent.iloc[0]
)


mae_24h = float(
    summary.loc[
        summary[
            "context_hours"
        ] == 24,
        "mean_mae",
    ].iloc[0]
)

improvement = (
    100
    * (
        mae_24h
        - best_mae
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
print("TimesFM 3 PEMS-BAY context summary")
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
print("=" * 78)
print("Best TimesFM 3 PEMS-BAY context")
print("=" * 78)

print()
print(
    f"Best context: "
    f"{best['context_label']}"
)

print(
    f"Mean MAE:     "
    f"{best_mae:.4f}"
)

print(
    f"Improvement vs 24h: "
    f"{improvement:.2f}%"
)

print(
    f"Earliest context within 1% "
    f"of best: "
    f"{saturation['context_label']}"
)


print()
print("=" * 78)
print("Chronos-2 vs TimesFM 3")
print("=" * 78)

comparison_display = (
    comparison.copy()
)

for col in [
    "timesfm_mean_mae",
    "timesfm_median_mae",
    "chronos_mean_mae",
    "chronos_median_mae",
    "timesfm_minus_chronos_mae",
]:
    comparison_display[col] = (
        comparison_display[col]
        .round(4)
    )

print()
print(
    comparison_display.to_string(
        index=False
    )
)

print()
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
print(f"  {comparison_path}")
print(f"  {figure_path}")
