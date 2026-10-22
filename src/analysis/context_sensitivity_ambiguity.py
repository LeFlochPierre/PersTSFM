from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path("results")

OUT = (
    ROOT
    / "context_regime_comparison"
    / "context_sensitivity_ambiguity"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


GAINS_PATH = (
    ROOT
    / "context_regime_comparison"
    / "context_sensitivity_phenotype"
    / "entity_gains_scale_normalized.csv"
)

PHENOTYPE_PATH = (
    ROOT
    / "context_regime_comparison"
    / "context_sensitivity_phenotype"
    / "scale_normalized_consensus_phenotype.csv"
)


CONFIGS = {
    "pems_bay": {
        "label": "PEMS-BAY",
        "recent": 24,
        "long": {
            "chronos2": 240,
            "timesfm3": 504,
        },
    },
    "electricity": {
        "label": "Electricity",
        "recent": 96,
        "long": {
            "chronos2": 192,
            "timesfm3": 240,
        },
    },
    "hupa_ucm": {
        "label": "HUPA CGM",
        "recent": 24,
        "long": {
            "chronos2": 96,
            "timesfm3": 168,
        },
    },
    "vitaldb": {
        "label": "VitalDB MAP",
        "recent": 0.5,
        "long": {
            "chronos2": 4,
            "timesfm3": 4,
        },
    },
    "solar": {
        "label": "Solar PV",
        "recent": 24,
        "long": {
            "chronos2": 96,
            "timesfm3": 96,
        },
    },
    "physionet2019": {
        "label": "PhysioNet HR",
        "recent": 6,
        "long": {
            "chronos2": 48,
            "timesfm3": 48,
        },
    },
}


MODELS = [
    "chronos2",
    "timesfm3",
]


MODEL_LABELS = {
    "chronos2": "Chronos-2",
    "timesfm3": "TimesFM3",
}


EPS = 1e-12


def folder(
    dataset,
    model,
):
    cfg = CONFIGS[
        dataset
    ]

    recent = cfg[
        "recent"
    ]

    long = cfg[
        "long"
    ][
        model
    ]

    return (
        ROOT
        / dataset
        / "interventions"
        / model
        / (
            f"recent_{recent:g}h"
            f"_long_{long:g}h"
        )
    )


def zscore(x):
    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    sd = x.std(
        ddof=1
    )

    if (
        not np.isfinite(sd)
        or sd < EPS
    ):
        return pd.Series(
            np.zeros(
                len(x)
            ),
            index=x.index,
        )

    return (
        x - x.mean()
    ) / sd


def rho_p(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    rho, p = spearmanr(
        x[good],
        y[good],
    )

    return (
        float(rho),
        float(p),
        int(
            good.sum()
        ),
    )


def rank01(x):
    return (
        pd.Series(x)
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=float
        )
    )


def residualize(
    y,
    x,
):
    """
    Residualize ranked y against ranked x.

    Used only as a robustness / partial-rank analysis.
    """
    yr = rank01(
        y
    )

    xr = rank01(
        x
    )

    X = np.column_stack(
        [
            np.ones(
                len(xr)
            ),
            xr,
        ]
    )

    beta = np.linalg.lstsq(
        X,
        yr,
        rcond=None,
    )[0]

    return (
        yr
        - X @ beta
    )


# ============================================================
# Load normalized effects
# ============================================================

gains = pd.read_csv(
    GAINS_PATH,
    dtype={
        "entity_id": str,
    },
)


# ============================================================
# 1. Difficulty-adjusted cross-model agreement
# ============================================================

difficulty_rows = []


print("=" * 104)
print(
    "DIFFICULTY-ADJUSTED CROSS-MODEL AGREEMENT"
)
print(
    "Rank residuals after removing each model's "
    "own short-context MAE"
)
print("=" * 104)


for dataset, cfg in (
    CONFIGS.items()
):

    part = gains[
        gains[
            "dataset"
        ] == dataset
    ]

    c = (
        part[
            part[
                "model"
            ] == "chronos2"
        ]
        .set_index(
            "entity_id"
        )
    )

    t = (
        part[
            part[
                "model"
            ] == "timesfm3"
        ]
        .set_index(
            "entity_id"
        )
    )

    joined = c.join(
        t,
        lsuffix="_c",
        rsuffix="_t",
        how="inner",
    )

    c_resid = residualize(
        joined[
            "entity_gain_pct_c"
        ],
        joined[
            "short_mae_c"
        ],
    )

    t_resid = residualize(
        joined[
            "entity_gain_pct_t"
        ],
        joined[
            "short_mae_t"
        ],
    )

    rho, p, n = rho_p(
        c_resid,
        t_resid,
    )

    # Also show how similar difficulty itself is.
    rho_diff, p_diff, _ = rho_p(
        joined[
            "short_mae_c"
        ],
        joined[
            "short_mae_t"
        ],
    )

    difficulty_rows.append(
        {
            "dataset": dataset,
            "dataset_label": (
                cfg[
                    "label"
                ]
            ),
            "n": n,
            "partial_rank_rho": rho,
            "partial_rank_p": p,
            "short_mae_agreement_rho": (
                rho_diff
            ),
            "short_mae_agreement_p": (
                p_diff
            ),
        }
    )

    print()
    print(
        cfg[
            "label"
        ]
    )

    print(
        f"  adjusted entity-gain agreement "
        f"rho={rho:7.3f} "
        f"p={p:.5g} "
        f"N={n}"
    )

    print(
        f"  short-MAE agreement            "
        f"rho={rho_diff:7.3f} "
        f"p={p_diff:.5g}"
    )


difficulty = pd.DataFrame(
    difficulty_rows
)

difficulty.to_csv(
    OUT
    / "difficulty_adjusted_cross_model_agreement.csv",
    index=False,
)


# ============================================================
# 2. Summarize actual donor behaviour per entity/model
# ============================================================

ambiguity_rows = []


for dataset, cfg in (
    CONFIGS.items()
):

    for model in MODELS:

        manifest = pd.read_csv(
            folder(
                dataset,
                model,
            )
            / "donor_manifest.csv",
            dtype={
                "entity_id": str,
                "donor_id": str,
            },
        )

        for entity_id, part in (
            manifest.groupby(
                "entity_id"
            )
        ):

            n = len(
                part
            )

            counts = (
                part[
                    "donor_id"
                ]
                .value_counts()
            )

            probs = (
                counts.to_numpy(
                    dtype=float
                )
                / n
            )

            entropy = float(
                -np.sum(
                    probs
                    * np.log(
                        probs
                    )
                )
            )

            if n > 1:
                entropy_norm = (
                    entropy
                    / np.log(
                        n
                    )
                )
            else:
                entropy_norm = 0.0

            unique_fraction = (
                counts.size
                / n
            )

            max_reuse_fraction = float(
                counts.max()
                / n
            )

            distances = (
                part[
                    "donor_distance"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            ambiguity_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": (
                        cfg[
                            "label"
                        ]
                    ),
                    "model": model,
                    "model_label": (
                        MODEL_LABELS[
                            model
                        ]
                    ),
                    "entity_id": str(
                        entity_id
                    ),
                    "n_origins": n,
                    "mean_donor_distance": (
                        float(
                            np.mean(
                                distances
                            )
                        )
                    ),
                    "median_donor_distance": (
                        float(
                            np.median(
                                distances
                            )
                        )
                    ),
                    "sd_donor_distance": (
                        float(
                            np.std(
                                distances,
                                ddof=1,
                            )
                        )
                        if n > 1
                        else 0.0
                    ),
                    "unique_donors": (
                        int(
                            counts.size
                        )
                    ),
                    "unique_donor_fraction": (
                        float(
                            unique_fraction
                        )
                    ),
                    "donor_entropy": (
                        float(
                            entropy_norm
                        )
                    ),
                    "max_reuse_fraction": (
                        max_reuse_fraction
                    ),
                }
            )


ambiguity = pd.DataFrame(
    ambiguity_rows
)


ambiguity.to_csv(
    OUT
    / "entity_recent_state_ambiguity.csv",
    index=False,
)


# ============================================================
# Merge normalized outcome
# ============================================================

merged = gains.merge(
    ambiguity,
    on=[
        "dataset",
        "model",
        "entity_id",
    ],
    how="inner",
    validate="one_to_one",
)


if len(
    merged
) != len(
    gains
):
    raise RuntimeError(
        "Ambiguity merge lost entity/model rows."
    )


# Higher value = a closer alternative entity exists.
merged[
    "recent_neighbor_closeness"
] = (
    -merged.groupby(
        [
            "dataset",
            "model",
        ]
    )[
        "mean_donor_distance"
    ]
    .transform(
        zscore
    )
)


# Higher entropy = the entity is matched to many
# different plausible donors across origins.
merged[
    "donor_entropy_z"
] = (
    merged.groupby(
        [
            "dataset",
            "model",
        ]
    )[
        "donor_entropy"
    ]
    .transform(
        zscore
    )
)


# Higher = repeatedly matched to the same donor.
merged[
    "donor_reuse_z"
] = (
    merged.groupby(
        [
            "dataset",
            "model",
        ]
    )[
        "max_reuse_fraction"
    ]
    .transform(
        zscore
    )
)


merged.to_csv(
    OUT
    / "entity_gain_with_recent_ambiguity.csv",
    index=False,
)


# ============================================================
# 3. Model-specific ambiguity associations
# ============================================================

FEATURES = [
    (
        "recent_neighbor_closeness",
        "Recent neighbour closeness",
    ),
    (
        "donor_entropy",
        "Donor identity entropy",
    ),
    (
        "unique_donor_fraction",
        "Unique donor fraction",
    ),
    (
        "max_reuse_fraction",
        "Maximum donor reuse",
    ),
]


association_rows = []


print()
print("=" * 104)
print(
    "RECENT-STATE AMBIGUITY vs NORMALIZED ENTITY GAIN"
)
print(
    "Positive closeness rho would support: "
    "more locally confusable entities benefit more from own history."
)
print("=" * 104)


for dataset, cfg in (
    CONFIGS.items()
):

    print()
    print(
        cfg[
            "label"
        ]
    )

    for model in MODELS:

        part = merged[
            (
                merged[
                    "dataset"
                ]
                == dataset
            )
            &
            (
                merged[
                    "model"
                ]
                == model
            )
        ]

        print(
            f"  {MODEL_LABELS[model]}"
        )

        for feature, label in (
            FEATURES
        ):

            rho, p, n = rho_p(
                part[
                    feature
                ],
                part[
                    "entity_gain_pct"
                ],
            )

            association_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": (
                        cfg[
                            "label"
                        ]
                    ),
                    "model": model,
                    "model_label": (
                        MODEL_LABELS[
                            model
                        ]
                    ),
                    "feature": feature,
                    "feature_label": label,
                    "n": n,
                    "rho": rho,
                    "p": p,
                }
            )

            print(
                f"    {label:<25} "
                f"rho={rho:7.3f} "
                f"p={p:.5g}"
            )


associations = pd.DataFrame(
    association_rows
)

associations.to_csv(
    OUT
    / "ambiguity_gain_correlations.csv",
    index=False,
)


# ============================================================
# 4. Consensus ambiguity phenotype
# ============================================================

phenotype = pd.read_csv(
    PHENOTYPE_PATH,
    dtype={
        "entity_id": str,
    },
)


consensus_parts = []


for dataset, cfg in (
    CONFIGS.items()
):

    part = merged[
        merged[
            "dataset"
        ] == dataset
    ].copy()

    cols = [
        "recent_neighbor_closeness",
        "donor_entropy_z",
        "donor_reuse_z",
    ]

    wide = part.pivot(
        index="entity_id",
        columns="model",
        values=cols,
    )

    out = pd.DataFrame(
        index=wide.index
    )

    for feature in cols:

        out[
            f"{feature}_consensus"
        ] = (
            (
                wide[
                    (
                        feature,
                        "chronos2",
                    )
                ]
                + wide[
                    (
                        feature,
                        "timesfm3",
                    )
                ]
            )
            / 2.0
        )

    out = (
        out
        .reset_index()
    )

    out[
        "dataset"
    ] = dataset

    consensus_parts.append(
        out
    )


consensus_ambiguity = pd.concat(
    consensus_parts,
    ignore_index=True,
)


consensus = phenotype.merge(
    consensus_ambiguity,
    on=[
        "dataset",
        "entity_id",
    ],
    how="inner",
    validate="one_to_one",
)


consensus.to_csv(
    OUT
    / "consensus_sensitivity_and_ambiguity.csv",
    index=False,
)


print()
print("=" * 104)
print(
    "CONSENSUS AMBIGUITY vs CONSENSUS CONTEXT SENSITIVITY"
)
print("=" * 104)


consensus_rows = []


CONSENSUS_FEATURES = [
    (
        "recent_neighbor_closeness_consensus",
        "Recent neighbour closeness",
    ),
    (
        "donor_entropy_z_consensus",
        "Donor identity entropy",
    ),
    (
        "donor_reuse_z_consensus",
        "Repeated donor reuse",
    ),
]


for dataset, cfg in (
    CONFIGS.items()
):

    part = consensus[
        consensus[
            "dataset"
        ] == dataset
    ]

    print()
    print(
        cfg[
            "label"
        ]
    )

    for feature, label in (
        CONSENSUS_FEATURES
    ):

        rho, p, n = rho_p(
            part[
                feature
            ],
            part[
                "normalized_consensus_sensitivity"
            ],
        )

        consensus_rows.append(
            {
                "dataset": dataset,
                "dataset_label": (
                    cfg[
                        "label"
                    ]
                ),
                "feature": feature,
                "feature_label": label,
                "n": n,
                "rho": rho,
                "p": p,
            }
        )

        print(
            f"  {label:<25} "
            f"rho={rho:7.3f} "
            f"p={p:.5g}"
        )


consensus_corr = pd.DataFrame(
    consensus_rows
)

consensus_corr.to_csv(
    OUT
    / "consensus_ambiguity_correlations.csv",
    index=False,
)


# ============================================================
# 5. Pooled within-domain standardized consensus
# ============================================================

pooled = consensus.copy()


for feature, _ in (
    CONSENSUS_FEATURES
):

    pooled[
        f"{feature}_within_z"
    ] = (
        pooled.groupby(
            "dataset"
        )[
            feature
        ]
        .transform(
            zscore
        )
    )


print()
print("=" * 104)
print(
    "POOLED WITHIN-DOMAIN CONSENSUS ANALYSIS"
)
print("=" * 104)


pooled_rows = []


for feature, label in (
    CONSENSUS_FEATURES
):

    rho, p, n = rho_p(
        pooled[
            f"{feature}_within_z"
        ],
        pooled[
            "normalized_consensus_sensitivity"
        ],
    )

    pooled_rows.append(
        {
            "feature": feature,
            "feature_label": label,
            "n": n,
            "rho": rho,
            "p": p,
        }
    )


pooled_df = pd.DataFrame(
    pooled_rows
)

print(
    pooled_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:8.3f}"
        ),
    )
)


pooled_df.to_csv(
    OUT
    / "pooled_consensus_ambiguity.csv",
    index=False,
)


# ============================================================
# Figures
# ============================================================

def save(
    fig,
    filename,
):
    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT
        / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)

axes = axes.ravel()


for ax, (
    dataset,
    cfg,
) in zip(
    axes,
    CONFIGS.items(),
):

    part = consensus[
        consensus[
            "dataset"
        ] == dataset
    ]

    x = part[
        "recent_neighbor_closeness_consensus"
    ]

    y = part[
        "normalized_consensus_sensitivity"
    ]

    rho, p, _ = rho_p(
        x,
        y,
    )

    ax.scatter(
        x,
        y,
        s=40,
        alpha=0.7,
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        (
            f"{cfg['label']}\n"
            f"ρ={rho:.3f}, "
            f"p={p:.3g}"
        )
    )

    ax.set_xlabel(
        "Recent-neighbour closeness"
    )

    ax.set_ylabel(
        "Normalized consensus\n"
        "context sensitivity"
    )


save(
    fig,
    "01_recent_ambiguity_vs_context_sensitivity",
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)

axes = axes.ravel()


for ax, (
    dataset,
    cfg,
) in zip(
    axes,
    CONFIGS.items(),
):

    part = consensus[
        consensus[
            "dataset"
        ] == dataset
    ]

    x = part[
        "donor_entropy_z_consensus"
    ]

    y = part[
        "normalized_consensus_sensitivity"
    ]

    rho, p, _ = rho_p(
        x,
        y,
    )

    ax.scatter(
        x,
        y,
        s=40,
        alpha=0.7,
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        (
            f"{cfg['label']}\n"
            f"ρ={rho:.3f}, "
            f"p={p:.3g}"
        )
    )

    ax.set_xlabel(
        "Donor identity entropy"
    )

    ax.set_ylabel(
        "Normalized consensus\n"
        "context sensitivity"
    )


save(
    fig,
    "02_donor_entropy_vs_context_sensitivity",
)


print()
print("=" * 104)
print("SAVED")
print("=" * 104)
print(OUT)
