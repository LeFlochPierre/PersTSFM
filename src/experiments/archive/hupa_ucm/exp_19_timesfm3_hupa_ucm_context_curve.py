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

RAW_DIR = Path(
    "data/raw/hupa_ucm/"
    "HUPA-UCM-extracted-glucose-files"
)

PROTOCOL_PATH = Path(
    "data/processed/hupa_ucm/"
    "hupa_ucm_evaluation_origins.csv"
)

CHRONOS_PATH = Path(
    "results/hupa_ucm/context/"
    "chronos2_hupa_context_summary.csv"
)

NAIVE_PATH = Path(
    "results/hupa_ucm/baselines/"
    "hupa_ucm_naive_summary.csv"
)

OUT_DIR = Path(
    "results/hupa_ucm/context"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_ID = "google/timesfm-3.0-pytorch"

SPLIT = "development"

RESOLUTION_MINUTES = 5
FORECAST_STEPS = 12

CONTEXT_HOURS = [
    3,
    6,
    12,
    18,
    24,
    36,
    48,
    72,
    96,
    120,
    168,
]

BATCH_SIZE = 16


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def context_label(hours):

    if hours < 24:
        return f"{hours}h"

    if hours % 24 == 0:
        return f"{hours // 24}d"

    return f"{hours}h"


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

protocol = (
    protocol[
        protocol["split"] == SPLIT
    ]
    .copy()
    .sort_values(
        [
            "person_id",
            "eval_index",
        ]
    )
    .reset_index(drop=True)
)

people = sorted(
    protocol[
        "person_id"
    ].unique()
)


print("=" * 78)
print("PersTSFM TimesFM 3 HUPA-UCM context curve")
print("=" * 78)

print()
print(f"Model:                    {MODEL_ID}")
print(f"Development participants: {len(people)}")
print(f"Evaluation cases:         {len(protocol)}")
print(f"Forecast steps:           {FORECAST_STEPS}")

print()
print(
    "Context grid: "
    + ", ".join(
        context_label(x)
        for x in CONTEXT_HOURS
    )
)


# ---------------------------------------------------------------------
# Load participant series
# ---------------------------------------------------------------------

series_by_person = {}

for person_id in people:

    path = (
        RAW_DIR
        / f"{person_id}.csv"
    )

    df = pd.read_csv(
        path,
        usecols=[
            "timestamp",
            "glucose_value_mg_dl",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df[
        "glucose_value_mg_dl"
    ] = pd.to_numeric(
        df[
            "glucose_value_mg_dl"
        ]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .set_index("timestamp")
    )

    series_by_person[
        person_id
    ] = df[
        "glucose_value_mg_dl"
    ].astype(float)


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
# Run context sweep
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

    for batch_number, start in enumerate(
        range(
            0,
            len(protocol),
            BATCH_SIZE,
        ),
        start=1,
    ):

        end = min(
            start + BATCH_SIZE,
            len(protocol),
        )

        batch_protocol = protocol.iloc[
            start:end
        ]

        contexts = []
        records = []

        for _, row in batch_protocol.iterrows():

            person_id = row[
                "person_id"
            ]

            eval_index = int(
                row["eval_index"]
            )

            origin = row[
                "eval_origin"
            ]

            series = series_by_person[
                person_id
            ]

            context_start = (
                origin
                - context_span
            )

            context = series.loc[
                context_start:origin
            ].to_numpy(
                dtype=np.float32
            )

            truth_series = series.loc[
                row["target_start"]:
                row["target_end"]
            ]

            truth = truth_series.to_numpy(
                dtype=np.float32
            )

            if len(context) != context_steps:
                raise RuntimeError(
                    f"{person_id} eval {eval_index}: "
                    f"expected {context_steps} "
                    f"context points, "
                    f"found {len(context)}."
                )

            if len(truth) != FORECAST_STEPS:
                raise RuntimeError(
                    f"{person_id} eval {eval_index}: "
                    f"expected {FORECAST_STEPS} "
                    f"target points, "
                    f"found {len(truth)}."
                )

            contexts.append(
                context
            )

            records.append(
                {
                    "person_id": person_id,
                    "eval_index": eval_index,
                    "eval_origin": origin,
                    "hour": int(
                        origin.hour
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
                f"outputs, got "
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
                    f"Unexpected forecast shape: "
                    f"{point.shape}"
                )

            if quantiles.shape != (
                FORECAST_STEPS,
                9,
            ):
                raise RuntimeError(
                    f"Unexpected quantile shape: "
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
                        "person_id": (
                            record[
                                "person_id"
                            ]
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
# Combine predictions
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
# Metrics
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


# ---------------------------------------------------------------------
# Case-level metrics
# ---------------------------------------------------------------------

case_metrics = (
    pred
    .groupby(
        [
            "context_hours",
            "context_label",
            "context_steps",
            "person_id",
            "eval_index",
            "eval_origin",
            "hour",
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
    )
)

case_metrics["rmse"] = np.sqrt(
    case_metrics["mse"]
)

case_metrics = case_metrics.drop(
    columns="mse"
)


# ---------------------------------------------------------------------
# 24h reference
# ---------------------------------------------------------------------

reference = (
    case_metrics[
        case_metrics[
            "context_hours"
        ] == 24
    ]
    [
        [
            "person_id",
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
        "person_id",
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
# Participant-level metrics
# ---------------------------------------------------------------------

participant_metrics = (
    case_metrics
    .groupby(
        [
            "context_hours",
            "context_label",
            "person_id",
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
        mean_coverage_80=(
            "coverage_80",
            "mean",
        ),
        mean_delta_mae_vs_24h=(
            "delta_mae_vs_24h",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------

summary_rows = []

for context_hours in CONTEXT_HOURS:

    part = participant_metrics[
        participant_metrics[
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
            "mean_participant_mae": float(
                part[
                    "mean_mae"
                ].mean()
            ),
            "median_participant_mae": float(
                part[
                    "mean_mae"
                ].median()
            ),
            "mean_participant_rmse": float(
                part[
                    "mean_rmse"
                ].mean()
            ),
            "mean_coverage_80": float(
                part[
                    "mean_coverage_80"
                ].mean()
            ),
            "mean_delta_mae_vs_24h": float(
                part[
                    "mean_delta_mae_vs_24h"
                ].mean()
            ),
            "participant_better_than_24h_fraction": float(
                (
                    part[
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
# Best / saturation
# ---------------------------------------------------------------------

best = summary.sort_values(
    "mean_participant_mae"
).iloc[0]

best_mae = float(
    best[
        "mean_participant_mae"
    ]
)

mae_24h = float(
    summary.loc[
        summary[
            "context_hours"
        ] == 24,
        "mean_participant_mae",
    ].iloc[0]
)

improvement_vs_24h = (
    100
    * (
        mae_24h
        - best_mae
    )
    / mae_24h
)

within_one_percent = (
    summary[
        summary[
            "mean_participant_mae"
        ]
        <= best_mae * 1.01
    ]
    .sort_values(
        "context_hours"
    )
)

earliest_near_best = (
    within_one_percent.iloc[0]
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

pred_path = (
    OUT_DIR
    / "timesfm3_hupa_context_predictions.parquet"
)

case_path = (
    OUT_DIR
    / "timesfm3_hupa_context_case_metrics.csv"
)

participant_path = (
    OUT_DIR
    / "timesfm3_hupa_context_participant_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "timesfm3_hupa_context_summary.csv"
)

pred.to_parquet(
    pred_path,
    index=False,
)

case_metrics.to_csv(
    case_path,
    index=False,
)

participant_metrics.to_csv(
    participant_path,
    index=False,
)

summary.to_csv(
    summary_path,
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
        "mean_participant_mae",
    ]
].rename(
    columns={
        "mean_participant_mae":
        "timesfm_mean_mae",
    }
)

comparison = comparison.merge(
    chronos[
        [
            "context_hours",
            "mean_participant_mae",
        ]
    ].rename(
        columns={
            "mean_participant_mae":
            "chronos_mean_mae",
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
    / "chronos2_vs_timesfm3_hupa_context.csv"
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

    persistence = naive[
        naive["method"]
        == "persistence"
    ]

    if len(persistence) > 0:

        value = float(
            persistence[
                "mean_participant_mae"
            ].iloc[0]
        )

        ax.axhline(
            value,
            linestyle="--",
            label="Persistence",
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
    "Same-person CGM context"
)

ax.set_ylabel(
    "Participant-averaged MAE (mg/dL)"
)

ax.set_title(
    "HUPA-UCM context scaling across TSFMs"
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "chronos2_vs_timesfm3_hupa_context.png"
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
print("TimesFM 3 HUPA-UCM context summary")
print("=" * 78)

display = summary.copy()

for col in [
    "mean_participant_mae",
    "median_participant_mae",
    "mean_participant_rmse",
    "mean_coverage_80",
    "mean_delta_mae_vs_24h",
]:
    display[col] = (
        display[col]
        .round(4)
    )

display[
    "participant_better_than_24h_fraction"
] = (
    100
    * display[
        "participant_better_than_24h_fraction"
    ]
).round(1)

print()
print(
    display.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("Best TimesFM 3 HUPA-UCM context")
print("=" * 78)

print()
print(
    f"Best context: "
    f"{best['context_label']}"
)

print(
    f"Mean participant MAE: "
    f"{best_mae:.4f} mg/dL"
)

print(
    f"Improvement vs 24h: "
    f"{improvement_vs_24h:.2f}%"
)

print(
    f"Earliest context within 1% "
    f"of best: "
    f"{earliest_near_best['context_label']}"
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
    "chronos_mean_mae",
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
print(f"  {participant_path}")
print(f"  {summary_path}")
print(f"  {comparison_path}")
print(f"  {figure_path}")
