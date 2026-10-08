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

PERSONALIZATION_DAYS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]


# ---------------------------------------------------------------------
# Load data
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

bank["timestamp"] = pd.to_datetime(
    bank["timestamp"]
)


print("=" * 76)
print("PersTSFM personal-bias adaptation curve")
print("=" * 76)

print()
print(
    f"Evaluation rows:      "
    f"{len(global_pred):,}"
)

print(
    f"Residual-bank rows:   "
    f"{len(bank):,}"
)

print(
    f"Evaluation entities:  "
    f"{global_pred['entity'].nunique()}"
)

print(
    f"Evaluation cases:     "
    f"{global_pred[['entity', 'eval_index']].drop_duplicates().shape[0]}"
)


# ---------------------------------------------------------------------
# Build fast entity lookup for residual bank
# ---------------------------------------------------------------------

bank_by_entity = {
    entity: frame.copy()
    for entity, frame in bank.groupby(
        "entity",
        sort=False,
    )
}


# ---------------------------------------------------------------------
# Evaluate each fixed target under increasing personal evidence
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

    target_df = target_df.sort_values(
        "timestamp"
    )

    truth = target_df[
        "truth"
    ].to_numpy(
        dtype=float
    )

    pred_global = target_df[
        "global_pred"
    ].to_numpy(
        dtype=float
    )

    if len(truth) != 96:
        raise RuntimeError(
            f"{entity} eval {eval_index}: "
            f"expected 96 target points, "
            f"got {len(truth)}."
        )

    entity_bank = bank_by_entity[
        entity
    ]

    for n_days in PERSONALIZATION_DAYS:

        # -------------------------------------------------------------
        # No personalization
        # -------------------------------------------------------------

        if n_days == 0:

            personal_bias = 0.0

            n_residual_points = 0
            n_residual_days = 0

            residual_std = np.nan

        # -------------------------------------------------------------
        # Personalization from previous n completed daily forecasts
        # -------------------------------------------------------------

        else:

            start_origin = (
                eval_origin
                - pd.Timedelta(
                    days=n_days
                )
            )

            history = entity_bank[
                (
                    entity_bank[
                        "personal_origin"
                    ]
                    >= start_origin
                )
                &
                (
                    entity_bank[
                        "personal_origin"
                    ]
                    < eval_origin
                )
            ].copy()

            unique_days = (
                history[
                    "personal_origin"
                ]
                .drop_duplicates()
                .sort_values()
            )

            n_residual_days = len(
                unique_days
            )

            if n_residual_days != n_days:
                raise RuntimeError(
                    f"{entity} eval {eval_index}, "
                    f"n={n_days}: expected "
                    f"{n_days} historical days, "
                    f"found {n_residual_days}."
                )

            expected_points = (
                n_days * 96
            )

            if len(history) != expected_points:
                raise RuntimeError(
                    f"{entity} eval {eval_index}, "
                    f"n={n_days}: expected "
                    f"{expected_points} residual points, "
                    f"found {len(history)}."
                )

            personal_bias = float(
                history[
                    "residual"
                ].mean()
            )

            residual_std = float(
                history[
                    "residual"
                ].std()
            )

            n_residual_points = len(
                history
            )

        # -------------------------------------------------------------
        # Personalized prediction
        # -------------------------------------------------------------

        pred_personal = (
            pred_global
            + personal_bias
        )

        error = (
            pred_personal
            - truth
        )

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

        mean_error = float(
            np.mean(error)
        )

        negative_fraction = float(
            np.mean(
                pred_personal < 0
            )
        )

        rows.append(
            {
                "entity": entity,
                "eval_index": int(
                    eval_index
                ),
                "eval_origin": eval_origin,
                "n_personal_days": n_days,
                "personal_bias": personal_bias,
                "residual_std_history": (
                    residual_std
                ),
                "n_residual_days": (
                    n_residual_days
                ),
                "n_residual_points": (
                    n_residual_points
                ),
                "mae": mae,
                "rmse": rmse,
                "mean_error": mean_error,
                "negative_fraction": (
                    negative_fraction
                ),
            }
        )


case_metrics = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Attach each case's n=0 global baseline
# ---------------------------------------------------------------------

global_case = (
    case_metrics[
        case_metrics[
            "n_personal_days"
        ] == 0
    ]
    [
        [
            "entity",
            "eval_index",
            "mae",
            "rmse",
        ]
    ]
    .rename(
        columns={
            "mae": "global_mae",
            "rmse": "global_rmse",
        }
    )
)

case_metrics = case_metrics.merge(
    global_case,
    on=[
        "entity",
        "eval_index",
    ],
    how="left",
    validate="many_to_one",
)

case_metrics["delta_mae"] = (
    case_metrics["mae"]
    - case_metrics["global_mae"]
)

case_metrics["delta_rmse"] = (
    case_metrics["rmse"]
    - case_metrics["global_rmse"]
)

case_metrics["mae_skill_vs_global"] = (
    1.0
    - (
        case_metrics["mae"]
        / case_metrics["global_mae"]
    )
)

case_metrics["improved_vs_global"] = (
    case_metrics["mae"]
    < case_metrics["global_mae"]
)


# ---------------------------------------------------------------------
# Aggregate by entity first
#
# This prevents entities with repeated weekly evaluation days from
# being treated as independent subjects in entity-level summaries.
# ---------------------------------------------------------------------

entity_metrics = (
    case_metrics
    .groupby(
        [
            "entity",
            "n_personal_days",
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
        mean_delta_mae=(
            "delta_mae",
            "mean",
        ),
        mean_skill_vs_global=(
            "mae_skill_vs_global",
            "mean",
        ),
        mean_personal_bias=(
            "personal_bias",
            "mean",
        ),
    )
)


# ---------------------------------------------------------------------
# Entity-level global baseline
# ---------------------------------------------------------------------

entity_global = (
    entity_metrics[
        entity_metrics[
            "n_personal_days"
        ] == 0
    ]
    [
        [
            "entity",
            "mean_mae",
        ]
    ]
    .rename(
        columns={
            "mean_mae": (
                "entity_global_mae"
            ),
        }
    )
)

entity_metrics = entity_metrics.merge(
    entity_global,
    on="entity",
    how="left",
    validate="many_to_one",
)

entity_metrics[
    "entity_improved_vs_global"
] = (
    entity_metrics["mean_mae"]
    < entity_metrics[
        "entity_global_mae"
    ]
)


# ---------------------------------------------------------------------
# Overall adaptation summary
# ---------------------------------------------------------------------

summary_rows = []

for n_days in PERSONALIZATION_DAYS:

    cases_n = case_metrics[
        case_metrics[
            "n_personal_days"
        ] == n_days
    ]

    entities_n = entity_metrics[
        entity_metrics[
            "n_personal_days"
        ] == n_days
    ]

    summary_rows.append(
        {
            "n_personal_days": n_days,

            "n_cases": len(
                cases_n
            ),

            "n_entities": len(
                entities_n
            ),

            "mean_mae": (
                cases_n[
                    "mae"
                ].mean()
            ),

            "median_mae": (
                cases_n[
                    "mae"
                ].median()
            ),

            "mean_rmse": (
                cases_n[
                    "rmse"
                ].mean()
            ),

            "median_rmse": (
                cases_n[
                    "rmse"
                ].median()
            ),

            "mean_delta_mae": (
                cases_n[
                    "delta_mae"
                ].mean()
            ),

            "median_delta_mae": (
                cases_n[
                    "delta_mae"
                ].median()
            ),

            "mean_skill_vs_global": (
                cases_n[
                    "mae_skill_vs_global"
                ].mean()
            ),

            "case_improvement_fraction": (
                cases_n[
                    "improved_vs_global"
                ].mean()
            ),

            "entity_mean_mae": (
                entities_n[
                    "mean_mae"
                ].mean()
            ),

            "entity_median_mae": (
                entities_n[
                    "mean_mae"
                ].median()
            ),

            "entity_improvement_fraction": (
                entities_n[
                    "entity_improved_vs_global"
                ].mean()
            ),

            "mean_abs_personal_bias": (
                cases_n[
                    "personal_bias"
                ].abs().mean()
            ),

            "negative_prediction_fraction": (
                cases_n[
                    "negative_fraction"
                ].mean()
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Estimate simple per-entity crossover
#
# Definition for this exploratory analysis:
# smallest n > 0 for which the entity's average MAE is below its
# n=0 global average MAE.
#
# This is NOT yet our final theoretical definition of crossover.
# ---------------------------------------------------------------------

crossover_rows = []

for entity, group in entity_metrics.groupby(
    "entity"
):

    group = group.sort_values(
        "n_personal_days"
    )

    global_mae = float(
        group.loc[
            group[
                "n_personal_days"
            ] == 0,
            "mean_mae",
        ].iloc[0]
    )

    candidates = group[
        (
            group[
                "n_personal_days"
            ] > 0
        )
        &
        (
            group[
                "mean_mae"
            ] < global_mae
        )
    ]

    if len(candidates) == 0:
        crossover = np.nan

    else:
        crossover = int(
            candidates[
                "n_personal_days"
            ].iloc[0]
        )

    crossover_rows.append(
        {
            "entity": entity,
            "global_mean_mae": (
                global_mae
            ),
            "first_improvement_days": (
                crossover
            ),
        }
    )


crossovers = pd.DataFrame(
    crossover_rows
)


# ---------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------

case_path = (
    OUT_DIR
    / "personal_bias_case_metrics.csv"
)

entity_path = (
    OUT_DIR
    / "personal_bias_entity_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "personal_bias_summary.csv"
)

crossover_path = (
    OUT_DIR
    / "personal_bias_crossovers.csv"
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

crossovers.to_csv(
    crossover_path,
    index=False,
)


# ---------------------------------------------------------------------
# Figure 1: adaptation curve
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

x = np.arange(
    len(PERSONALIZATION_DAYS)
)

ax.plot(
    x,
    summary["mean_mae"],
    marker="o",
    label="Mean case MAE",
)

ax.plot(
    x,
    summary["median_mae"],
    marker="o",
    label="Median case MAE",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    PERSONALIZATION_DAYS
)

ax.set_xlabel(
    "Previous same-entity days used for personalization"
)

ax.set_ylabel(
    "MAE"
)

ax.set_title(
    "Chronos-2 + scalar personal bias"
)

ax.legend()

fig.tight_layout()

curve_path = (
    OUT_DIR
    / "personal_bias_adaptation_curve.png"
)

fig.savefig(
    curve_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: improvement relative to frozen global Chronos
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.axhline(
    0.0,
    linewidth=1,
)

ax.plot(
    x,
    summary["mean_delta_mae"],
    marker="o",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    PERSONALIZATION_DAYS
)

ax.set_xlabel(
    "Previous same-entity days used for personalization"
)

ax.set_ylabel(
    "MAE difference vs frozen Chronos"
)

ax.set_title(
    "Negative values mean personalization improves forecasting"
)

fig.tight_layout()

delta_path = (
    OUT_DIR
    / "personal_bias_delta_vs_global.png"
)

fig.savefig(
    delta_path,
    dpi=200,
)

plt.close(fig)


# ---------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------

print()
print("=" * 76)
print("Personalization adaptation summary")
print("=" * 76)

display_cols = [
    "n_personal_days",
    "mean_mae",
    "median_mae",
    "mean_delta_mae",
    "case_improvement_fraction",
    "entity_improvement_fraction",
    "mean_abs_personal_bias",
]

display = summary[
    display_cols
].copy()

for col in [
    "mean_mae",
    "median_mae",
    "mean_delta_mae",
    "mean_abs_personal_bias",
]:
    display[col] = (
        display[col]
        .round(4)
    )

for col in [
    "case_improvement_fraction",
    "entity_improvement_fraction",
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
print("=" * 76)
print("Simple entity crossover summary")
print("=" * 76)

n_never = int(
    crossovers[
        "first_improvement_days"
    ].isna().sum()
)

print()
print(
    f"Entities improving at some n: "
    f"{len(crossovers) - n_never} / "
    f"{len(crossovers)}"
)

print(
    f"Entities never improving:     "
    f"{n_never} / "
    f"{len(crossovers)}"
)

print()

print(
    "First-improvement distribution:"
)

print(
    crossovers[
        "first_improvement_days"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .to_string()
)

print()
print("Saved:")
print(f"  {case_path}")
print(f"  {entity_path}")
print(f"  {summary_path}")
print(f"  {crossover_path}")
print(f"  {curve_path}")
print(f"  {delta_path}")
