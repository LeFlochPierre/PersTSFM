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

MODEL_ID = "google/timesfm-3.0-pytorch"

SPLIT = "validation"

FREQUENCY_MINUTES = 15
FORECAST_STEPS = 96

CONTEXT_DAYS = list(
    range(3, 32)
)

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
print("PersTSFM TimesFM 3 in-context personalization curve")
print("=" * 78)

print()
print(f"Model:              {MODEL_ID}")
print(f"Entities:           {len(entities)}")
print(f"Evaluation cases:   {len(protocol)}")
print(f"Forecast steps:     {FORECAST_STEPS}")
print(
    f"Context days:       "
    f"{min(CONTEXT_DAYS)}-"
    f"{max(CONTEXT_DAYS)}"
)


# ---------------------------------------------------------------------
# Load validation data
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
# Load TimesFM
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
# Forecast
# ---------------------------------------------------------------------

prediction_rows = []

total_start = time.time()


for context_days in CONTEXT_DAYS:

    context_steps = (
        context_days * 96
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

        contexts = []
        truth_records = []

        for _, row in (
            batch_protocol.iterrows()
        ):

            entity = row["entity"]

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

            if (
                context_start
                < row[
                    "first_active_timestamp"
                ]
            ):
                raise RuntimeError(
                    f"{entity} eval {eval_index}: "
                    f"{context_days}-day context "
                    "begins before active history."
                )

            context = raw.loc[
                context_start:origin,
                entity,
            ].to_numpy(
                dtype=np.float32
            )

            truth_series = raw.loc[
                row["target_start"]:
                row["target_end"],
                entity,
            ]

            truth = truth_series.to_numpy(
                dtype=np.float32
            )

            if (
                len(context)
                != context_steps
            ):
                raise RuntimeError(
                    f"{entity} eval {eval_index}: "
                    f"expected {context_steps} "
                    f"context points, "
                    f"got {len(context)}."
                )

            if (
                len(truth)
                != FORECAST_STEPS
            ):
                raise RuntimeError(
                    f"{entity} eval {eval_index}: "
                    f"expected {FORECAST_STEPS} "
                    f"target points, "
                    f"got {len(truth)}."
                )

            contexts.append(
                context
            )

            truth_records.append(
                {
                    "entity": entity,
                    "eval_index": (
                        eval_index
                    ),
                    "eval_origin": (
                        origin
                    ),
                    "timestamps": (
                        truth_series.index
                    ),
                    "truth": truth,
                }
            )


        # -------------------------------------------------------------
        # TimesFM batch prediction
        # -------------------------------------------------------------

        outputs = list(
            forecaster.predict_batch(
                contexts=contexts,
                horizon=FORECAST_STEPS,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )

        if (
            len(outputs)
            != len(contexts)
        ):
            raise RuntimeError(
                f"Expected {len(contexts)} "
                f"outputs but received "
                f"{len(outputs)}."
            )


        # -------------------------------------------------------------
        # Save batch predictions
        # -------------------------------------------------------------

        for output, truth_record in zip(
            outputs,
            truth_records,
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
                    "Unexpected point "
                    f"forecast shape: "
                    f"{point.shape}"
                )

            if quantiles.shape != (
                FORECAST_STEPS,
                9,
            ):
                raise RuntimeError(
                    "Unexpected quantile "
                    f"shape: "
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
                        "context_days": (
                            context_days
                        ),
                        "context_steps": (
                            context_steps
                        ),
                        "entity": (
                            truth_record[
                                "entity"
                            ]
                        ),
                        "eval_index": (
                            truth_record[
                                "eval_index"
                            ]
                        ),
                        "eval_origin": (
                            truth_record[
                                "eval_origin"
                            ]
                        ),
                        "timestamp": (
                            truth_record[
                                "timestamps"
                            ][step]
                        ),
                        "truth": float(
                            truth_record[
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


# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------

expected_rows = (
    len(CONTEXT_DAYS)
    * len(protocol)
    * FORECAST_STEPS
)

assert len(pred) == expected_rows

assert not pred.duplicated(
    [
        "context_days",
        "entity",
        "eval_index",
        "timestamp",
    ]
).any()

assert pred[
    [
        "truth",
        "prediction",
        "q10",
        "q50",
        "q90",
    ]
].notna().all().all()


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
        coverage_80=(
            "covered_80",
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
# Four-day reference
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

    entity_part = entity_metrics[
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
                cases[
                    "mae"
                ].mean()
            ),

            "median_mae": (
                cases[
                    "mae"
                ].median()
            ),

            "mean_rmse": (
                cases[
                    "rmse"
                ].mean()
            ),

            "mean_coverage_80": (
                cases[
                    "coverage_80"
                ].mean()
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
                entity_part[
                    "mean_mae"
                ].mean()
            ),

            "entity_better_than_4day_fraction": (
                (
                    entity_part[
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
    / "timesfm3_context_predictions.parquet"
)

case_path = (
    OUT_DIR
    / "timesfm3_context_case_metrics.csv"
)

entity_path = (
    OUT_DIR
    / "timesfm3_context_entity_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "timesfm3_context_summary.csv"
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


# ---------------------------------------------------------------------
# Compare against Chronos-2
# ---------------------------------------------------------------------

chronos_candidates = [
    Path(
        "results/context_threshold/"
        "chronos2_context_summary.csv"
    ),
    Path(
        "results/context_threshold/"
        "chronos2_context_threshold_summary.csv"
    ),
]

chronos_path = None

for candidate in chronos_candidates:
    if candidate.exists():
        chronos_path = candidate
        break

if chronos_path is None:
    raise FileNotFoundError(
        "Could not find Chronos-2 "
        "context summary."
    )

chronos = pd.read_csv(
    chronos_path
)

comparison = summary[
    [
        "context_days",
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
            "context_days",
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
    on="context_days",
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
    / "chronos2_vs_timesfm3_context.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ---------------------------------------------------------------------
# Cross-model figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    comparison[
        "context_days"
    ],
    comparison[
        "chronos_mean_mae"
    ],
    marker="o",
    label="Chronos-2",
)

ax.plot(
    comparison[
        "context_days"
    ],
    comparison[
        "timesfm_mean_mae"
    ],
    marker="o",
    label="TimesFM 3",
)

ax.set_xlabel(
    "Same-entity context (days)"
)

ax.set_ylabel(
    "Mean MAE"
)

ax.set_title(
    "In-context personalization across foundation models"
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "chronos2_vs_timesfm3_context.png"
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
print("TimesFM 3 context summary")
print("=" * 78)

display = summary.copy()

for col in [
    "mean_mae",
    "median_mae",
    "mean_rmse",
    "mean_coverage_80",
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
        100
        * display[col]
    ).round(1)

print()
print(
    display.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Best context
# ---------------------------------------------------------------------

best = summary.sort_values(
    "mean_mae"
).iloc[0]

print()
print("=" * 78)
print("Best TimesFM 3 context")
print("=" * 78)

print()
print(
    f"Best context: "
    f"{int(best['context_days'])} days"
)

print(
    f"Mean MAE:     "
    f"{best['mean_mae']:.4f}"
)

mae_4 = float(
    summary.loc[
        summary[
            "context_days"
        ] == 4,
        "mean_mae",
    ].iloc[0]
)

relative_improvement = (
    100
    * (
        mae_4
        - best["mean_mae"]
    )
    / mae_4
)

print(
    f"Improvement vs 4 days: "
    f"{relative_improvement:.2f}%"
)


print()
print("=" * 78)
print("Chronos-2 vs TimesFM 3")
print("=" * 78)

comparison_display = (
    comparison.copy()
)

for col in comparison_display.columns:
    if col != "context_days":
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
print(f"  {comparison_path}")
print(f"  {figure_path}")
