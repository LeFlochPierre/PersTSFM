from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    pearsonr,
    spearmanr,
)


ROOT = Path(
    "results/synthetic_phase_diagram_final"
)

OUT_DIR = (
    ROOT
    / "synthesis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = [
    "chronos2",
    "timesfm3",
]

SEEDS = [
    11,
    22,
    33,
    44,
    55,
]


# ---------------------------------------------------------------------
# Load all synthetic worlds
# ---------------------------------------------------------------------

frames = []


for model in MODELS:

    for seed in SEEDS:

        path = (
            ROOT
            / model
            / f"seed_{seed}"
            / "summary.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        df = pd.read_csv(
            path
        )

        if len(df) != 49:
            raise RuntimeError(
                f"{path}: expected 49 "
                f"conditions, found {len(df)}."
            )

        if "seed" not in df.columns:
            df["seed"] = seed

        df["seed"] = seed
        df["model"] = model

        frames.append(
            df
        )


combined = pd.concat(
    frames,
    ignore_index=True,
)


combined[
    "signal_noise_ratio"
] = (
    combined[
        "heterogeneity"
    ]
    ** 2
    / (
        combined[
            "noise_sd"
        ]
        ** 2
    )
)


combined[
    "log1p_signal_noise_ratio"
] = np.log1p(
    combined[
        "signal_noise_ratio"
    ]
)


combined_path = (
    OUT_DIR
    / "combined_summary.csv"
)

combined.to_csv(
    combined_path,
    index=False,
)


# ---------------------------------------------------------------------
# Average phase surface across synthetic worlds
# ---------------------------------------------------------------------

cell_summary = (
    combined
    .groupby(
        [
            "model",
            "heterogeneity",
            "noise_sd",
        ],
        as_index=False,
    )
    .agg(
        n_seeds=(
            "seed",
            "nunique",
        ),
        entity_gain_mean=(
            "entity_gain",
            "mean",
        ),
        entity_gain_sd=(
            "entity_gain",
            "std",
        ),
        generic_gain_mean=(
            "generic_gain",
            "mean",
        ),
        total_gain_mean=(
            "total_gain",
            "mean",
        ),
        donor_distance_mean=(
            "donor_distance",
            "mean",
        ),
        entity_ratio_mean=(
            "entity_contribution_ratio",
            "mean",
        ),
    )
)


cell_summary[
    "entity_gain_sem"
] = (
    cell_summary[
        "entity_gain_sd"
    ]
    / np.sqrt(
        cell_summary[
            "n_seeds"
        ]
    )
)


cell_path = (
    OUT_DIR
    / "phase_surface_summary.csv"
)

cell_summary.to_csv(
    cell_path,
    index=False,
)


# ---------------------------------------------------------------------
# Zero-heterogeneity sanity check
# ---------------------------------------------------------------------

print("=" * 108)
print(
    "SYNTHETIC PHASE-DIAGRAM SYNTHESIS"
)
print("=" * 108)


print()
print("=" * 108)
print(
    "ZERO-HETEROGENEITY SANITY CHECK"
)
print("=" * 108)


zero = (
    combined[
        combined[
            "heterogeneity"
        ]
        == 0
    ]
    .groupby(
        [
            "model",
            "noise_sd",
        ],
        as_index=False,
    )
    .agg(
        mean_entity_gain=(
            "entity_gain",
            "mean",
        ),
        sd_entity_gain=(
            "entity_gain",
            "std",
        ),
    )
)


print(
    zero
    .round(4)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Print mean phase surfaces
# ---------------------------------------------------------------------

for model in MODELS:

    print()
    print("=" * 108)
    print(
        f"{model.upper()} | "
        "MEAN ENTITY-GAIN SURFACE"
    )
    print("=" * 108)

    subset = cell_summary[
        cell_summary[
            "model"
        ]
        == model
    ]

    pivot = subset.pivot(
        index="heterogeneity",
        columns="noise_sd",
        values="entity_gain_mean",
    )

    print()
    print(
        pivot
        .round(3)
        .to_string()
    )


# ---------------------------------------------------------------------
# Monotonicity tests on the mean phase surface
# ---------------------------------------------------------------------

monotonic_rows = []


print()
print("=" * 108)
print(
    "MONOTONICITY OF THE MEAN SURFACE"
)
print("=" * 108)


for model in MODELS:

    subset = cell_summary[
        cell_summary[
            "model"
        ]
        == model
    ].copy()

    print()
    print(model)
    print("-" * 108)

    print()
    print(
        "Heterogeneity effect "
        "at fixed noise"
    )

    for noise_sd, group in (
        subset.groupby(
            "noise_sd"
        )
    ):
        group = group.sort_values(
            "heterogeneity"
        )

        rho, p_value = spearmanr(
            group[
                "heterogeneity"
            ],
            group[
                "entity_gain_mean"
            ],
        )

        monotonic_rows.append(
            {
                "model": model,
                "direction": (
                    "heterogeneity"
                ),
                "fixed_value": (
                    noise_sd
                ),
                "spearman_rho": (
                    float(rho)
                ),
                "p_value": (
                    float(p_value)
                ),
            }
        )

        print(
            f"  noise={noise_sd:>4g} | "
            f"rho={rho:>6.3f} | "
            f"p={p_value:.4g}"
        )

    print()
    print(
        "Noise effect at fixed "
        "non-zero heterogeneity"
    )

    nonzero = subset[
        subset[
            "heterogeneity"
        ]
        > 0
    ]

    for heterogeneity, group in (
        nonzero.groupby(
            "heterogeneity"
        )
    ):
        group = group.sort_values(
            "noise_sd"
        )

        rho, p_value = spearmanr(
            group[
                "noise_sd"
            ],
            group[
                "entity_gain_mean"
            ],
        )

        monotonic_rows.append(
            {
                "model": model,
                "direction": (
                    "noise"
                ),
                "fixed_value": (
                    heterogeneity
                ),
                "spearman_rho": (
                    float(rho)
                ),
                "p_value": (
                    float(p_value)
                ),
            }
        )

        print(
            f"  heterogeneity="
            f"{heterogeneity:>4g} | "
            f"rho={rho:>6.3f} | "
            f"p={p_value:.4g}"
        )


monotonic = pd.DataFrame(
    monotonic_rows
)

monotonic_path = (
    OUT_DIR
    / "monotonicity_tests.csv"
)

monotonic.to_csv(
    monotonic_path,
    index=False,
)


# ---------------------------------------------------------------------
# Repeat monotonicity inside each independent synthetic world.
#
# This checks that the mean surface is not being created by one seed.
# ---------------------------------------------------------------------

seed_rows = []


for model in MODELS:

    model_df = combined[
        combined[
            "model"
        ]
        == model
    ]

    for seed in SEEDS:

        seed_df = model_df[
            model_df[
                "seed"
            ]
            == seed
        ]

        for noise_sd, group in (
            seed_df.groupby(
                "noise_sd"
            )
        ):
            rho = spearmanr(
                group[
                    "heterogeneity"
                ],
                group[
                    "entity_gain"
                ],
            ).statistic

            seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "effect": (
                        "heterogeneity"
                    ),
                    "fixed_value": (
                        noise_sd
                    ),
                    "rho": float(
                        rho
                    ),
                }
            )

        for heterogeneity, group in (
            seed_df[
                seed_df[
                    "heterogeneity"
                ]
                > 0
            ]
            .groupby(
                "heterogeneity"
            )
        ):
            rho = spearmanr(
                group[
                    "noise_sd"
                ],
                group[
                    "entity_gain"
                ],
            ).statistic

            seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "effect": (
                        "noise"
                    ),
                    "fixed_value": (
                        heterogeneity
                    ),
                    "rho": float(
                        rho
                    ),
                }
            )


seed_monotonic = pd.DataFrame(
    seed_rows
)


seed_monotonic_path = (
    OUT_DIR
    / "seed_level_monotonicity.csv"
)

seed_monotonic.to_csv(
    seed_monotonic_path,
    index=False,
)


print()
print("=" * 108)
print(
    "ACROSS-SEED MONOTONICITY"
)
print("=" * 108)


for model in MODELS:

    print()
    print(model)
    print("-" * 108)

    model_df = seed_monotonic[
        seed_monotonic[
            "model"
        ]
        == model
    ]

    h_rho = model_df[
        model_df[
            "effect"
        ]
        == "heterogeneity"
    ][
        "rho"
    ]

    n_rho = model_df[
        model_df[
            "effect"
        ]
        == "noise"
    ][
        "rho"
    ]

    print(
        "Heterogeneity correlations:"
    )

    print(
        "  median rho:       "
        f"{h_rho.median():.3f}"
    )

    print(
        "  range:            "
        f"[{h_rho.min():.3f}, "
        f"{h_rho.max():.3f}]"
    )

    print(
        "  fraction positive:"
        f" {(h_rho > 0).mean():.3f}"
    )

    print()
    print(
        "Noise correlations:"
    )

    print(
        "  median rho:       "
        f"{n_rho.median():.3f}"
    )

    print(
        "  range:            "
        f"[{n_rho.min():.3f}, "
        f"{n_rho.max():.3f}]"
    )

    print(
        "  fraction negative:"
        f" {(n_rho < 0).mean():.3f}"
    )


# ---------------------------------------------------------------------
# Does the theoretical h^2 / sigma^2 ratio organize the surface?
# ---------------------------------------------------------------------

ratio_rows = []


print()
print("=" * 108)
print(
    "SIGNAL-TO-NOISE ORGANIZATION"
)
print("=" * 108)


for model in MODELS:

    model_df = combined[
        combined[
            "model"
        ]
        == model
    ]

    seed_rhos = []

    for seed in SEEDS:

        subset = model_df[
            model_df[
                "seed"
            ]
            == seed
        ]

        rho, p_value = spearmanr(
            subset[
                "log1p_signal_noise_ratio"
            ],
            subset[
                "entity_gain"
            ],
        )

        seed_rhos.append(
            float(rho)
        )

        ratio_rows.append(
            {
                "model": model,
                "seed": seed,
                "spearman_rho": (
                    float(rho)
                ),
                "p_value": (
                    float(p_value)
                ),
            }
        )

    mean_surface = cell_summary[
        cell_summary[
            "model"
        ]
        == model
    ].copy()

    mean_surface[
        "signal_noise_ratio"
    ] = (
        mean_surface[
            "heterogeneity"
        ]
        ** 2
        / (
            mean_surface[
                "noise_sd"
            ]
            ** 2
        )
    )

    mean_surface[
        "log1p_signal_noise_ratio"
    ] = np.log1p(
        mean_surface[
            "signal_noise_ratio"
        ]
    )

    rho_surface, p_surface = (
        spearmanr(
            mean_surface[
                "log1p_signal_noise_ratio"
            ],
            mean_surface[
                "entity_gain_mean"
            ],
        )
    )

    print()
    print(model)
    print("-" * 108)

    print(
        "Mean-surface Spearman rho: "
        f"{rho_surface:.3f}"
    )

    print(
        "Mean-surface p-value:      "
        f"{p_surface:.6g}"
    )

    print(
        "Per-seed median rho:       "
        f"{np.median(seed_rhos):.3f}"
    )

    print(
        "Per-seed range:            "
        f"[{np.min(seed_rhos):.3f}, "
        f"{np.max(seed_rhos):.3f}]"
    )


ratio_stats = pd.DataFrame(
    ratio_rows
)

ratio_path = (
    OUT_DIR
    / "signal_noise_ratio_associations.csv"
)

ratio_stats.to_csv(
    ratio_path,
    index=False,
)


# ---------------------------------------------------------------------
# Cross-model reproducibility of the entire 7x7 phase surface
# ---------------------------------------------------------------------

chronos = (
    cell_summary[
        cell_summary[
            "model"
        ]
        == "chronos2"
    ][
        [
            "heterogeneity",
            "noise_sd",
            "entity_gain_mean",
        ]
    ]
    .rename(
        columns={
            "entity_gain_mean":
            "chronos_entity_gain"
        }
    )
)


timesfm = (
    cell_summary[
        cell_summary[
            "model"
        ]
        == "timesfm3"
    ][
        [
            "heterogeneity",
            "noise_sd",
            "entity_gain_mean",
        ]
    ]
    .rename(
        columns={
            "entity_gain_mean":
            "timesfm_entity_gain"
        }
    )
)


comparison = chronos.merge(
    timesfm,
    on=[
        "heterogeneity",
        "noise_sd",
    ],
    validate="one_to_one",
)


spearman_rho, spearman_p = (
    spearmanr(
        comparison[
            "chronos_entity_gain"
        ],
        comparison[
            "timesfm_entity_gain"
        ],
    )
)


pearson_r, pearson_p = (
    pearsonr(
        comparison[
            "chronos_entity_gain"
        ],
        comparison[
            "timesfm_entity_gain"
        ],
    )
)


comparison_path = (
    OUT_DIR
    / "cross_model_phase_surface.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


print()
print("=" * 108)
print(
    "CROSS-MODEL PHASE-SURFACE REPRODUCIBILITY"
)
print("=" * 108)

print()
print(
    "Spearman rho: "
    f"{spearman_rho:.4f}"
)

print(
    "Spearman p:   "
    f"{spearman_p:.6g}"
)

print(
    "Pearson r:    "
    f"{pearson_r:.4f}"
)

print(
    "Pearson p:    "
    f"{pearson_p:.6g}"
)


# ---------------------------------------------------------------------
# Corner contrasts
# ---------------------------------------------------------------------

print()
print("=" * 108)
print(
    "CORNER CONTRASTS"
)
print("=" * 108)


for model in MODELS:

    subset = combined[
        combined[
            "model"
        ]
        == model
    ]

    print()
    print(model)
    print("-" * 108)

    for noise_sd in [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ]:

        low = subset[
            (
                subset[
                    "heterogeneity"
                ]
                == 0
            )
            & (
                subset[
                    "noise_sd"
                ]
                == noise_sd
            )
        ].set_index(
            "seed"
        )[
            "entity_gain"
        ]

        high = subset[
            (
                subset[
                    "heterogeneity"
                ]
                == 8
            )
            & (
                subset[
                    "noise_sd"
                ]
                == noise_sd
            )
        ].set_index(
            "seed"
        )[
            "entity_gain"
        ]

        delta = (
            high
            - low
        )

        print(
            f"  noise={noise_sd:>4g}: "
            f"h=8 minus h=0 = "
            f"{delta.mean():.3f} "
            f"(seed range "
            f"{delta.min():.3f} to "
            f"{delta.max():.3f})"
        )


print()
print("=" * 108)
print("Saved")
print("=" * 108)

print(combined_path)
print(cell_path)
print(monotonic_path)
print(seed_monotonic_path)
print(ratio_path)
print(comparison_path)
