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

RAW_DIR = Path(
    "data/raw/hupa_ucm/"
    "HUPA-UCM-extracted-glucose-files"
)

PROTOCOL_PATH = Path(
    "data/processed/hupa_ucm/"
    "hupa_ucm_evaluation_origins.csv"
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

MODEL_ID = "amazon/chronos-2"

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

BATCH_SIZE = 32


# ---------------------------------------------------------------------
# Context label
# ---------------------------------------------------------------------

def context_label(hours):

    if hours < 24:
        return f"{hours}h"

    if hours % 24 == 0:
        days = hours // 24
        return f"{days}d"

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
print("PersTSFM Chronos-2 HUPA-UCM context curve")
print("=" * 78)

print()
print(
    f"Model:                    "
    f"{MODEL_ID}"
)

print(
    f"Development participants: "
    f"{len(people)}"
)

print(
    f"Evaluation cases:         "
    f"{len(protocol)}"
)

print(
    f"Forecast steps:           "
    f"{FORECAST_STEPS}"
)

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

pipeline = (
    Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
    )
)


# ---------------------------------------------------------------------
# Forecast all contexts
# ---------------------------------------------------------------------

all_predictions = []

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

        context_frames = []
        truth_frames = []

        for _, row in (
            batch_protocol.iterrows()
        ):

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
            ]

            truth = series.loc[
                row["target_start"]:
                row["target_end"]
            ]

            if len(context) != context_steps:

                raise RuntimeError(
                    f"{person_id} eval "
                    f"{eval_index}: expected "
                    f"{context_steps} context "
                    f"points, found "
                    f"{len(context)}."
                )

            if len(truth) != FORECAST_STEPS:

                raise RuntimeError(
                    f"{person_id} eval "
                    f"{eval_index}: expected "
                    f"{FORECAST_STEPS} target "
                    f"points, found "
                    f"{len(truth)}."
                )

            case_id = (
                f"{person_id}__"
                f"{eval_index:02d}"
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
                        "person_id": (
                            person_id
                        ),
                        "eval_index": (
                            eval_index
                        ),
                        "eval_origin": (
                            origin
                        ),
                        "hour": int(
                            origin.hour
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
                f"{context_label(context_hours)} "
                f"batch {batch_number}: "
                f"expected {expected_rows} rows, "
                f"found {len(merged)}."
            )

        merged[
            "context_hours"
        ] = context_hours

        merged[
            "context_label"
        ] = context_label(
            context_hours
        )

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
        "predictions": (
            "prediction"
        ),
        "0.1": "q10",
        "0.5": "q50",
        "0.9": "q90",
    }
)


# ---------------------------------------------------------------------
# Quantile crossing
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


# ---------------------------------------------------------------------
# Evaluation quantiles
# ---------------------------------------------------------------------

quantile_values = pred[
    [
        "q10",
        "q50",
        "q90",
    ]
].to_numpy(
    dtype=float
)

quantile_sorted = np.sort(
    quantile_values,
    axis=1,
)

pred["q10_eval"] = (
    quantile_sorted[:, 0]
)

pred["q50_eval"] = (
    quantile_sorted[:, 1]
)

pred["q90_eval"] = (
    quantile_sorted[:, 2]
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

case_metrics = (
    case_metrics.drop(
        columns="mse"
    )
)


# ---------------------------------------------------------------------
# 24-hour reference
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
#
# IMPORTANT:
# participant first, cohort second.
# ---------------------------------------------------------------------

summary_rows = []

for context_hours in CONTEXT_HOURS:

    people_part = (
        participant_metrics[
            participant_metrics[
                "context_hours"
            ] == context_hours
        ]
    )

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
                people_part[
                    "mean_mae"
                ].mean()
            ),

            "median_participant_mae": float(
                people_part[
                    "mean_mae"
                ].median()
            ),

            "mean_participant_rmse": float(
                people_part[
                    "mean_rmse"
                ].mean()
            ),

            "mean_coverage_80": float(
                people_part[
                    "mean_coverage_80"
                ].mean()
            ),

            "mean_delta_mae_vs_24h": float(
                people_part[
                    "mean_delta_mae_vs_24h"
                ].mean()
            ),

            "participant_better_than_24h_fraction": float(
                (
                    people_part[
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
# Best context
# ---------------------------------------------------------------------

best = (
    summary
    .sort_values(
        "mean_participant_mae"
    )
    .iloc[0]
)

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
    100.0
    * (
        mae_24h
        - best_mae
    )
    / mae_24h
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

pred_path = (
    OUT_DIR
    / "chronos2_hupa_context_predictions.parquet"
)

case_path = (
    OUT_DIR
    / "chronos2_hupa_context_case_metrics.csv"
)

participant_path = (
    OUT_DIR
    / "chronos2_hupa_context_participant_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "chronos2_hupa_context_summary.csv"
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
    summary[
        "mean_participant_mae"
    ],
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

        match = naive[
            naive[
                "method"
            ] == method
        ]

        if len(match) == 0:
            continue

        value = float(
            match[
                "mean_participant_mae"
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
    "Same-person CGM context"
)

ax.set_ylabel(
    "Participant-averaged MAE (mg/dL)"
)

ax.set_title(
    "HUPA-UCM: Chronos-2 context curve"
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "chronos2_hupa_context_curve.png"
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
print("Chronos-2 HUPA-UCM context summary")
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
    100.0
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
print("Best Chronos-2 HUPA-UCM context")
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
print(f"  {participant_path}")
print(f"  {summary_path}")
print(f"  {figure_path}")
