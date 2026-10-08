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

SCALAR_SUMMARY_PATH = Path(
    "results/personalization/"
    "personal_bias_summary.csv"
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

bank["timestamp"] = pd.to_datetime(
    bank["timestamp"]
)


print("=" * 76)
print("PersTSFM personal residual-profile adaptation curve")
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
    f"Residual rows:    "
    f"{len(bank):,}"
)


# ---------------------------------------------------------------------
# Cache entity residual banks
# ---------------------------------------------------------------------

bank_by_entity = {
    entity: frame.copy()
    for entity, frame in bank.groupby(
        "entity",
        sort=False,
    )
}


# ---------------------------------------------------------------------
# Evaluate
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

    if len(target_df) != FORECAST_STEPS:
        raise RuntimeError(
            f"{entity} eval {eval_index}: "
            f"expected {FORECAST_STEPS} target rows, "
            f"found {len(target_df)}."
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

    entity_bank = bank_by_entity[
        entity
    ]


    for n_days in PERSONALIZATION_DAYS:

        # -------------------------------------------------------------
        # Frozen global model
        # -------------------------------------------------------------

        if n_days == 0:

            correction = np.zeros(
                FORECAST_STEPS,
                dtype=float,
            )

            history_residual_sd = np.nan


        # -------------------------------------------------------------
        # Horizon-specific personal residual profile
        # -------------------------------------------------------------

        else:

            history_start = (
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
                    >= history_start
                )
                &
                (
                    entity_bank[
                        "personal_origin"
                    ]
                    < eval_origin
                )
            ].copy()

            n_found_days = (
                history[
                    "personal_origin"
                ]
                .nunique()
            )

            if n_found_days != n_days:
                raise RuntimeError(
                    f"{entity} eval {eval_index}, "
                    f"n={n_days}: expected "
                    f"{n_days} history days but found "
                    f"{n_found_days}."
                )

            expected_rows = (
                n_days
                * FORECAST_STEPS
            )

            if len(history) != expected_rows:
                raise RuntimeError(
                    f"{entity} eval {eval_index}, "
                    f"n={n_days}: expected "
                    f"{expected_rows} history rows, "
                    f"found {len(history)}."
                )

            # Each forecast step gets its own personal correction.
            #
            # Example:
            #
            # step 1  <- mean residual at step 1
            # step 2  <- mean residual at step 2
            # ...
            # step 96 <- mean residual at step 96
            #
            profile = (
                history
                .groupby(
                    "forecast_step"
                )["residual"]
                .mean()
                .reindex(
                    range(
                        1,
                        FORECAST_STEPS + 1
                    )
                )
            )

            if profile.isna().any():
                raise RuntimeError(
                    f"{entity} eval {eval_index}, "
                    f"n={n_days}: incomplete "
                    "forecast-step profile."
                )

            correction = profile.to_numpy(
                dtype=float
            )

            history_residual_sd = float(
                history["residual"].std()
            )


        # -------------------------------------------------------------
        # Personalized forecast
        # -------------------------------------------------------------

        personal_pred = (
            global_values
            + correction
        )

        error = (
            personal_pred
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

        rows.append(
            {
                "entity": entity,
                "eval_index": int(
                    eval_index
                ),
                "eval_origin": eval_origin,
                "n_personal_days": (
                    n_days
                ),
                "mae": mae,
                "rmse": rmse,
                "mean_correction": float(
                    np.mean(correction)
                ),
                "mean_abs_correction": float(
                    np.mean(
                        np.abs(correction)
                    )
                ),
                "correction_sd": float(
                    np.std(correction)
                ),
                "history_residual_sd": (
                    history_residual_sd
                ),
                "negative_fraction": float(
                    np.mean(
                        personal_pred < 0
                    )
                ),
            }
        )


case_metrics = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Global baseline for each case
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

case_metrics["skill_vs_global"] = (
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
# Entity-level summaries
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
            "skill_vs_global",
            "mean",
        ),
        mean_abs_correction=(
            "mean_abs_correction",
            "mean",
        ),
    )
)


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
            "mean_mae": "global_entity_mae",
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
    "improved_vs_global"
] = (
    entity_metrics["mean_mae"]
    < entity_metrics[
        "global_entity_mae"
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
            "n_personal_days": (
                n_days
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
                    "skill_vs_global"
                ].mean()
            ),

            "case_improvement_fraction": (
                cases_n[
                    "improved_vs_global"
                ].mean()
            ),

            "entity_improvement_fraction": (
                entities_n[
                    "improved_vs_global"
                ].mean()
            ),

            "mean_abs_correction": (
                cases_n[
                    "mean_abs_correction"
                ].mean()
            ),

            "mean_correction_sd": (
                cases_n[
                    "correction_sd"
                ].mean()
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
# First entity-level improvement
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

    if candidates.empty:
        first_improvement = np.nan

    else:
        first_improvement = int(
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
                first_improvement
            ),
        }
    )


crossovers = pd.DataFrame(
    crossover_rows
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

case_path = (
    OUT_DIR
    / "personal_profile_case_metrics.csv"
)

entity_path = (
    OUT_DIR
    / "personal_profile_entity_metrics.csv"
)

summary_path = (
    OUT_DIR
    / "personal_profile_summary.csv"
)

crossover_path = (
    OUT_DIR
    / "personal_profile_crossovers.csv"
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
# Compare profile correction against scalar correction
# ---------------------------------------------------------------------

scalar_summary = pd.read_csv(
    SCALAR_SUMMARY_PATH
)

comparison = summary[
    [
        "n_personal_days",
        "mean_mae",
        "mean_delta_mae",
    ]
].rename(
    columns={
        "mean_mae": "profile_mean_mae",
        "mean_delta_mae": (
            "profile_delta_mae"
        ),
    }
)

comparison = comparison.merge(
    scalar_summary[
        [
            "n_personal_days",
            "mean_mae",
            "mean_delta_mae",
        ]
    ].rename(
        columns={
            "mean_mae": (
                "scalar_mean_mae"
            ),
            "mean_delta_mae": (
                "scalar_delta_mae"
            ),
        }
    ),
    on="n_personal_days",
    how="left",
    validate="one_to_one",
)

comparison_path = (
    OUT_DIR
    / "scalar_vs_profile_summary.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)

x = np.arange(
    len(PERSONALIZATION_DAYS)
)

ax.axhline(
    0.0,
    linewidth=1,
)

ax.plot(
    x,
    comparison[
        "scalar_delta_mae"
    ],
    marker="o",
    label="Scalar bias",
)

ax.plot(
    x,
    comparison[
        "profile_delta_mae"
    ],
    marker="o",
    label="96-step residual profile",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    PERSONALIZATION_DAYS
)

ax.set_xlabel(
    "Previous same-entity days"
)

ax.set_ylabel(
    "MAE difference vs frozen Chronos"
)

ax.set_title(
    "Does richer personalization reveal persistent entity structure?"
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "scalar_vs_profile_delta_mae.png"
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
print("Personal residual-profile adaptation summary")
print("=" * 76)

display = summary[
    [
        "n_personal_days",
        "mean_mae",
        "median_mae",
        "mean_delta_mae",
        "case_improvement_fraction",
        "entity_improvement_fraction",
        "mean_abs_correction",
    ]
].copy()

for col in [
    "mean_mae",
    "median_mae",
    "mean_delta_mae",
    "mean_abs_correction",
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
print("Scalar vs profile correction")
print("=" * 76)

print()

comparison_display = (
    comparison.copy()
)

for col in comparison_display.columns:
    if col != "n_personal_days":
        comparison_display[col] = (
            comparison_display[col]
            .round(4)
        )

print(
    comparison_display.to_string(
        index=False
    )
)


print()
print("=" * 76)
print("Entity first-improvement summary")
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
print(f"  {comparison_path}")
print(f"  {figure_path}")
