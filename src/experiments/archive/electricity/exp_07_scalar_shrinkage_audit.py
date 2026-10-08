from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

GLOBAL_PATH = Path(
    "results/baselines/"
    "chronos2_validation_predictions.parquet"
)

BIAS_PATH = Path(
    "results/personalization/"
    "personal_bias_case_metrics.csv"
)

OUT_DIR = Path(
    "results/personalization"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PERSONALIZATION_DAYS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]

ALPHAS = np.round(
    np.linspace(
        0.0,
        1.0,
        21,
    ),
    2,
)

N_FOLDS = 5
SEED = 42


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

pred = pd.read_parquet(
    GLOBAL_PATH
)

bias = pd.read_csv(
    BIAS_PATH,
    parse_dates=[
        "eval_origin",
    ],
)

pred["eval_origin"] = pd.to_datetime(
    pred["eval_origin"]
)

pred["timestamp"] = pd.to_datetime(
    pred["timestamp"]
)


print("=" * 78)
print("PersTSFM scalar shrinkage audit")
print("=" * 78)

print()
print(
    f"Entities: "
    f"{pred['entity'].nunique()}"
)

print(
    f"Cases:    "
    f"{pred[['entity', 'eval_index']].drop_duplicates().shape[0]}"
)

print(
    "Alpha grid: "
    + ", ".join(
        f"{x:.2f}"
        for x in ALPHAS
    )
)


# ---------------------------------------------------------------------
# Build case lookup
# ---------------------------------------------------------------------

pred_groups = {}

for (
    entity,
    eval_index,
), frame in pred.groupby(
    [
        "entity",
        "eval_index",
    ],
    sort=False,
):

    frame = frame.sort_values(
        "timestamp"
    )

    pred_groups[
        (
            entity,
            int(eval_index),
        )
    ] = (
        frame["truth"].to_numpy(
            dtype=float
        ),
        frame["global_pred"].to_numpy(
            dtype=float
        ),
    )


# ---------------------------------------------------------------------
# Evaluate all alpha values
# ---------------------------------------------------------------------

rows = []

for _, row in bias.iterrows():

    n_days = int(
        row["n_personal_days"]
    )

    if n_days == 0:
        continue

    entity = row["entity"]
    eval_index = int(
        row["eval_index"]
    )

    personal_bias = float(
        row["personal_bias"]
    )

    truth, global_values = pred_groups[
        (
            entity,
            eval_index,
        )
    ]

    global_mae = float(
        np.mean(
            np.abs(
                global_values
                - truth
            )
        )
    )

    for alpha in ALPHAS:

        adjusted = (
            global_values
            + alpha
            * personal_bias
        )

        mae = float(
            np.mean(
                np.abs(
                    adjusted
                    - truth
                )
            )
        )

        rows.append(
            {
                "entity": entity,
                "eval_index": eval_index,
                "n_personal_days": n_days,
                "alpha": float(alpha),
                "personal_bias": (
                    personal_bias
                ),
                "global_mae": (
                    global_mae
                ),
                "mae": mae,
                "delta_mae": (
                    mae
                    - global_mae
                ),
            }
        )


grid = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Overall grid summary
#
# Aggregate by entity first so each entity has equal weight.
# ---------------------------------------------------------------------

entity_grid = (
    grid
    .groupby(
        [
            "entity",
            "n_personal_days",
            "alpha",
        ],
        as_index=False,
    )
    .agg(
        mean_mae=(
            "mae",
            "mean",
        ),
        mean_delta_mae=(
            "delta_mae",
            "mean",
        ),
    )
)


summary = (
    entity_grid
    .groupby(
        [
            "n_personal_days",
            "alpha",
        ],
        as_index=False,
    )
    .agg(
        mean_entity_mae=(
            "mean_mae",
            "mean",
        ),
        mean_delta_mae=(
            "mean_delta_mae",
            "mean",
        ),
        median_delta_mae=(
            "mean_delta_mae",
            "median",
        ),
        entity_improvement_fraction=(
            "mean_delta_mae",
            lambda x: float(
                np.mean(
                    x < 0
                )
            ),
        ),
    )
)


# ---------------------------------------------------------------------
# Best alpha if selected on ALL validation entities
#
# Useful diagnostic only.
# ---------------------------------------------------------------------

best_global_rows = []

for n_days in PERSONALIZATION_DAYS:

    part = summary[
        summary[
            "n_personal_days"
        ] == n_days
    ].copy()

    best = part.sort_values(
        [
            "mean_entity_mae",
            "alpha",
        ]
    ).iloc[0]

    best_global_rows.append(
        {
            "n_personal_days": n_days,
            "best_alpha": float(
                best["alpha"]
            ),
            "mean_entity_mae": float(
                best[
                    "mean_entity_mae"
                ]
            ),
            "mean_delta_mae": float(
                best[
                    "mean_delta_mae"
                ]
            ),
            (
                "entity_improvement_"
                "fraction"
            ): float(
                best[
                    "entity_improvement_fraction"
                ]
            ),
        }
    )


best_global = pd.DataFrame(
    best_global_rows
)


# ---------------------------------------------------------------------
# 5-fold ENTITY-LEVEL cross-fitting
#
# Within validation:
#
# 4 folds:
# choose alpha
#
# 1 fold:
# evaluate chosen alpha
#
# This tells us whether shrinkage strength transfers to unseen entities.
# ---------------------------------------------------------------------

entities = np.array(
    sorted(
        entity_grid[
            "entity"
        ].unique()
    )
)

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

cv_rows = []


for fold, (
    train_idx,
    test_idx,
) in enumerate(
    kf.split(entities),
    start=1,
):

    train_entities = set(
        entities[
            train_idx
        ]
    )

    test_entities = set(
        entities[
            test_idx
        ]
    )

    for n_days in PERSONALIZATION_DAYS:

        train_part = entity_grid[
            (
                entity_grid[
                    "entity"
                ].isin(
                    train_entities
                )
            )
            &
            (
                entity_grid[
                    "n_personal_days"
                ]
                == n_days
            )
        ]

        alpha_performance = (
            train_part
            .groupby(
                "alpha",
                as_index=False,
            )
            .agg(
                mean_mae=(
                    "mean_mae",
                    "mean",
                )
            )
        )

        best_alpha = float(
            alpha_performance
            .sort_values(
                [
                    "mean_mae",
                    "alpha",
                ]
            )
            .iloc[0][
                "alpha"
            ]
        )

        test_part = entity_grid[
            (
                entity_grid[
                    "entity"
                ].isin(
                    test_entities
                )
            )
            &
            (
                entity_grid[
                    "n_personal_days"
                ]
                == n_days
            )
            &
            (
                entity_grid[
                    "alpha"
                ]
                == best_alpha
            )
        ]

        for _, test_row in (
            test_part.iterrows()
        ):

            cv_rows.append(
                {
                    "fold": fold,
                    "entity": (
                        test_row[
                            "entity"
                        ]
                    ),
                    "n_personal_days": (
                        n_days
                    ),
                    "selected_alpha": (
                        best_alpha
                    ),
                    "mean_mae": float(
                        test_row[
                            "mean_mae"
                        ]
                    ),
                    "mean_delta_mae": float(
                        test_row[
                            "mean_delta_mae"
                        ]
                    ),
                }
            )


cv = pd.DataFrame(
    cv_rows
)


cv_summary = (
    cv
    .groupby(
        "n_personal_days",
        as_index=False,
    )
    .agg(
        mean_selected_alpha=(
            "selected_alpha",
            "mean",
        ),
        min_selected_alpha=(
            "selected_alpha",
            "min",
        ),
        max_selected_alpha=(
            "selected_alpha",
            "max",
        ),
        mean_entity_mae=(
            "mean_mae",
            "mean",
        ),
        mean_delta_mae=(
            "mean_delta_mae",
            "mean",
        ),
        median_delta_mae=(
            "mean_delta_mae",
            "median",
        ),
        entity_improvement_fraction=(
            "mean_delta_mae",
            lambda x: float(
                np.mean(
                    x < 0
                )
            ),
        ),
    )
)


# ---------------------------------------------------------------------
# Oracle per-entity alpha
#
# NOT a deployable result.
#
# This asks:
# if we somehow knew the best shrinkage strength for each entity,
# how much potential personalization signal exists?
# ---------------------------------------------------------------------

oracle_rows = []

for (
    entity,
    n_days,
), group in entity_grid.groupby(
    [
        "entity",
        "n_personal_days",
    ]
):

    best = group.sort_values(
        [
            "mean_mae",
            "alpha",
        ]
    ).iloc[0]

    oracle_rows.append(
        {
            "entity": entity,
            "n_personal_days": (
                n_days
            ),
            "oracle_alpha": float(
                best["alpha"]
            ),
            "oracle_mae": float(
                best[
                    "mean_mae"
                ]
            ),
            "oracle_delta_mae": float(
                best[
                    "mean_delta_mae"
                ]
            ),
        }
    )


oracle = pd.DataFrame(
    oracle_rows
)


oracle_summary = (
    oracle
    .groupby(
        "n_personal_days",
        as_index=False,
    )
    .agg(
        mean_oracle_alpha=(
            "oracle_alpha",
            "mean",
        ),
        median_oracle_alpha=(
            "oracle_alpha",
            "median",
        ),
        mean_oracle_delta_mae=(
            "oracle_delta_mae",
            "mean",
        ),
        oracle_improvement_fraction=(
            "oracle_delta_mae",
            lambda x: float(
                np.mean(
                    x < 0
                )
            ),
        ),
    )
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

grid_path = (
    OUT_DIR
    / "scalar_shrinkage_grid.csv"
)

best_path = (
    OUT_DIR
    / "scalar_shrinkage_best_global.csv"
)

cv_path = (
    OUT_DIR
    / "scalar_shrinkage_crossfit.csv"
)

cv_summary_path = (
    OUT_DIR
    / "scalar_shrinkage_crossfit_summary.csv"
)

oracle_path = (
    OUT_DIR
    / "scalar_shrinkage_oracle.csv"
)

oracle_summary_path = (
    OUT_DIR
    / "scalar_shrinkage_oracle_summary.csv"
)


summary.to_csv(
    grid_path,
    index=False,
)

best_global.to_csv(
    best_path,
    index=False,
)

cv.to_csv(
    cv_path,
    index=False,
)

cv_summary.to_csv(
    cv_summary_path,
    index=False,
)

oracle.to_csv(
    oracle_path,
    index=False,
)

oracle_summary.to_csv(
    oracle_summary_path,
    index=False,
)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5)
)

for n_days in [
    1,
    4,
    16,
    64,
]:

    part = summary[
        summary[
            "n_personal_days"
        ] == n_days
    ]

    ax.plot(
        part["alpha"],
        part[
            "mean_delta_mae"
        ],
        marker="o",
        label=f"{n_days} days",
    )


ax.axhline(
    0.0,
    linewidth=1,
)

ax.set_xlabel(
    "Personalization weight alpha"
)

ax.set_ylabel(
    "Entity-averaged MAE difference vs frozen Chronos"
)

ax.set_title(
    "How strongly should personal residual information be trusted?"
)

ax.legend()

fig.tight_layout()

figure_path = (
    OUT_DIR
    / "scalar_shrinkage_curves.png"
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
print("=" * 78)
print("Best alpha using all validation entities")
print("=" * 78)

display = best_global.copy()

display[
    "mean_entity_mae"
] = (
    display[
        "mean_entity_mae"
    ].round(4)
)

display[
    "mean_delta_mae"
] = (
    display[
        "mean_delta_mae"
    ].round(4)
)

display[
    "entity_improvement_fraction"
] = (
    100
    * display[
        "entity_improvement_fraction"
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
print("5-fold entity-level cross-fitted shrinkage")
print("=" * 78)

cv_display = (
    cv_summary.copy()
)

for col in [
    "mean_selected_alpha",
    "min_selected_alpha",
    "max_selected_alpha",
    "mean_entity_mae",
    "mean_delta_mae",
    "median_delta_mae",
]:
    cv_display[col] = (
        cv_display[col]
        .round(4)
    )

cv_display[
    "entity_improvement_fraction"
] = (
    100
    * cv_display[
        "entity_improvement_fraction"
    ]
).round(1)

print()
print(
    cv_display.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("Oracle per-entity shrinkage headroom")
print("=" * 78)

oracle_display = (
    oracle_summary.copy()
)

for col in [
    "mean_oracle_alpha",
    "median_oracle_alpha",
    "mean_oracle_delta_mae",
]:
    oracle_display[col] = (
        oracle_display[col]
        .round(4)
    )

oracle_display[
    "oracle_improvement_fraction"
] = (
    100
    * oracle_display[
        "oracle_improvement_fraction"
    ]
).round(1)

print()
print(
    oracle_display.to_string(
        index=False
    )
)

print()
print("Saved:")
print(f"  {grid_path}")
print(f"  {best_path}")
print(f"  {cv_path}")
print(f"  {cv_summary_path}")
print(f"  {oracle_path}")
print(f"  {oracle_summary_path}")
print(f"  {figure_path}")
