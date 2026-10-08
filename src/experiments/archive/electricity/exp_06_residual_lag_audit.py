from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

GLOBAL_PATH = Path(
    "results/baselines/"
    "chronos2_validation_predictions.parquet"
)

RESIDUAL_PATH = Path(
    "results/baselines/"
    "chronos2_validation_residual_bank.parquet"
)

OUT_DIR = Path(
    "results/personalization"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_LAG_DAYS = 28
FORECAST_STEPS = 96


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

global_pred = pd.read_parquet(
    GLOBAL_PATH
)

bank = pd.read_parquet(
    RESIDUAL_PATH
)

global_pred["eval_origin"] = pd.to_datetime(
    global_pred["eval_origin"]
)

global_pred["timestamp"] = pd.to_datetime(
    global_pred["timestamp"]
)

bank["personal_origin"] = pd.to_datetime(
    bank["personal_origin"]
)


print("=" * 76)
print("PersTSFM residual-memory lag audit")
print("=" * 76)

print()
print(
    f"Evaluation cases: "
    f"{global_pred[['entity', 'eval_index']].drop_duplicates().shape[0]}"
)

print(
    f"Entities:         "
    f"{global_pred['entity'].nunique()}"
)

print(
    f"Lags tested:      "
    f"1-{MAX_LAG_DAYS} days"
)


# ---------------------------------------------------------------------
# Build lookup:
#
# (entity, historical forecast origin)
#        -> 96-dimensional residual vector
# ---------------------------------------------------------------------

residual_lookup = {}

for (
    entity,
    personal_origin,
), frame in bank.groupby(
    [
        "entity",
        "personal_origin",
    ],
    sort=False,
):

    frame = frame.sort_values(
        "forecast_step"
    )

    if len(frame) != FORECAST_STEPS:
        raise RuntimeError(
            f"{entity} {personal_origin}: "
            f"expected {FORECAST_STEPS} rows, "
            f"found {len(frame)}."
        )

    residual_lookup[
        (
            entity,
            personal_origin,
        )
    ] = frame["residual"].to_numpy(
        dtype=float
    )


# ---------------------------------------------------------------------
# Evaluate each lag separately
# ---------------------------------------------------------------------

rows = []

case_groups = global_pred.groupby(
    [
        "entity",
        "eval_index",
        "eval_origin",
    ],
    sort=False,
)


for (
    entity,
    eval_index,
    eval_origin,
), target_df in case_groups:

    target_df = (
        target_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    truth = target_df[
        "truth"
    ].to_numpy(
        dtype=float
    )

    global_values = target_df[
        "global_pred"
    ].to_numpy(
        dtype=float
    )

    global_error = (
        global_values
        - truth
    )

    global_mae = float(
        np.mean(
            np.abs(global_error)
        )
    )

    for lag_days in range(
        1,
        MAX_LAG_DAYS + 1,
    ):

        historical_origin = (
            eval_origin
            - pd.Timedelta(
                days=lag_days
            )
        )

        key = (
            entity,
            historical_origin,
        )

        if key not in residual_lookup:
            raise RuntimeError(
                f"Missing historical residual: "
                f"{entity}, {historical_origin}"
            )

        residual_profile = residual_lookup[
            key
        ]

        # -------------------------------------------------------------
        # Method A:
        # single scalar residual from that historical day
        # -------------------------------------------------------------

        scalar_correction = float(
            np.mean(
                residual_profile
            )
        )

        scalar_pred = (
            global_values
            + scalar_correction
        )

        scalar_mae = float(
            np.mean(
                np.abs(
                    scalar_pred
                    - truth
                )
            )
        )

        # -------------------------------------------------------------
        # Method B:
        # reuse the complete 96-step residual profile
        # -------------------------------------------------------------

        profile_pred = (
            global_values
            + residual_profile
        )

        profile_mae = float(
            np.mean(
                np.abs(
                    profile_pred
                    - truth
                )
            )
        )

        rows.append(
            {
                "entity": entity,
                "eval_index": int(
                    eval_index
                ),
                "eval_origin": eval_origin,
                "lag_days": lag_days,
                "weekday_match": (
                    lag_days % 7 == 0
                ),
                "global_mae": global_mae,
                "scalar_mae": scalar_mae,
                "profile_mae": profile_mae,
                "scalar_delta_mae": (
                    scalar_mae
                    - global_mae
                ),
                "profile_delta_mae": (
                    profile_mae
                    - global_mae
                ),
                "scalar_improved": (
                    scalar_mae
                    < global_mae
                ),
                "profile_improved": (
                    profile_mae
                    < global_mae
                ),
            }
        )


results = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------

summary = (
    results
    .groupby(
        "lag_days",
        as_index=False,
    )
    .agg(
        mean_scalar_delta_mae=(
            "scalar_delta_mae",
            "mean",
        ),
        median_scalar_delta_mae=(
            "scalar_delta_mae",
            "median",
        ),
        scalar_improvement_fraction=(
            "scalar_improved",
            "mean",
        ),
        mean_profile_delta_mae=(
            "profile_delta_mae",
            "mean",
        ),
        median_profile_delta_mae=(
            "profile_delta_mae",
            "median",
        ),
        profile_improvement_fraction=(
            "profile_improved",
            "mean",
        ),
    )
)

summary["weekday_match"] = (
    summary["lag_days"]
    % 7
    == 0
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

case_path = (
    OUT_DIR
    / "residual_lag_audit_cases.csv"
)

summary_path = (
    OUT_DIR
    / "residual_lag_audit_summary.csv"
)

results.to_csv(
    case_path,
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

ax.axhline(
    0.0,
    linewidth=1,
)

ax.plot(
    summary["lag_days"],
    summary["mean_scalar_delta_mae"],
    marker="o",
    label="Scalar correction",
)

ax.plot(
    summary["lag_days"],
    summary["mean_profile_delta_mae"],
    marker="o",
    label="96-step profile correction",
)

ax.set_xlabel(
    "Lag of personal residual used (days)"
)

ax.set_ylabel(
    "MAE difference vs frozen Chronos"
)

ax.set_title(
    "How persistent are entity-specific Chronos residuals?"
)

ax.set_xticks(
    range(
        1,
        MAX_LAG_DAYS + 1,
    )
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "residual_lag_audit.png"
)

fig.savefig(
    figure_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print()
print("=" * 76)
print("Best scalar residual lags")
print("=" * 76)

scalar_best = (
    summary
    .sort_values(
        "mean_scalar_delta_mae"
    )
    .head(10)
    [
        [
            "lag_days",
            "weekday_match",
            "mean_scalar_delta_mae",
            "scalar_improvement_fraction",
        ]
    ]
    .copy()
)

scalar_best[
    "mean_scalar_delta_mae"
] = (
    scalar_best[
        "mean_scalar_delta_mae"
    ]
    .round(4)
)

scalar_best[
    "scalar_improvement_fraction"
] = (
    100
    * scalar_best[
        "scalar_improvement_fraction"
    ]
).round(1)

print()
print(
    scalar_best.to_string(
        index=False
    )
)


print()
print("=" * 76)
print("Best profile residual lags")
print("=" * 76)

profile_best = (
    summary
    .sort_values(
        "mean_profile_delta_mae"
    )
    .head(10)
    [
        [
            "lag_days",
            "weekday_match",
            "mean_profile_delta_mae",
            "profile_improvement_fraction",
        ]
    ]
    .copy()
)

profile_best[
    "mean_profile_delta_mae"
] = (
    profile_best[
        "mean_profile_delta_mae"
    ]
    .round(4)
)

profile_best[
    "profile_improvement_fraction"
] = (
    100
    * profile_best[
        "profile_improvement_fraction"
    ]
).round(1)

print()
print(
    profile_best.to_string(
        index=False
    )
)


print()
print("=" * 76)
print("Weekly lags only")
print("=" * 76)

weekly = summary[
    summary[
        "weekday_match"
    ]
].copy()

for col in [
    "mean_scalar_delta_mae",
    "mean_profile_delta_mae",
]:
    weekly[col] = (
        weekly[col]
        .round(4)
    )

for col in [
    "scalar_improvement_fraction",
    "profile_improvement_fraction",
]:
    weekly[col] = (
        weekly[col]
        * 100
    ).round(1)

print()
print(
    weekly[
        [
            "lag_days",
            "mean_scalar_delta_mae",
            "scalar_improvement_fraction",
            "mean_profile_delta_mae",
            "profile_improvement_fraction",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Saved:")
print(f"  {case_path}")
print(f"  {summary_path}")
print(f"  {figure_path}")
