from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path("results/context_regime_comparison")

SOURCE = (
    ROOT
    / "matched_history_divergence"
    / "all_entity_level_divergence.csv"
)

SIGNATURES = (
    ROOT
    / "domain_signatures"
    / "entity_statistical_signatures.csv"
)

OUT = (
    ROOT
    / "context_sensitivity_phenotype"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DATASET_ORDER = [
    "pems_bay",
    "electricity",
    "hupa_ucm",
    "vitaldb",
    "solar",
    "physionet2019",
]


DATASET_LABELS = {
    "pems_bay": "PEMS-BAY",
    "electricity": "Electricity",
    "hupa_ucm": "HUPA CGM",
    "vitaldb": "VitalDB MAP",
    "solar": "Solar PV",
    "physionet2019": "PhysioNet HR",
}


MODEL_LABELS = {
    "chronos2": "Chronos-2",
    "timesfm3": "TimesFM3",
}


EPS = 1e-12


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


def spearman(
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

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    rho, p = spearmanr(
        x[valid],
        y[valid],
    )

    return (
        float(rho),
        float(p),
        int(valid.sum()),
    )


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

    plt.close(fig)

    print(
        "Saved:",
        OUT
        / f"{filename}.png"
    )


# ============================================================
# Load model-specific entity gains
# ============================================================

df = pd.read_csv(
    SOURCE,
    dtype={
        "entity_id": str,
    },
)


required = {
    "dataset",
    "entity_id",
    "model",
    "entity_gain",
    "generic_gain",
    "total_gain",
}

missing = (
    required
    - set(
        df.columns
    )
)

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# ============================================================
# Standardize each mechanism within dataset + model.
#
# This prevents differences in MAE units and model-specific
# selected context lengths from determining the consensus.
# ============================================================

for effect in [
    "entity_gain",
    "generic_gain",
    "total_gain",
]:
    df[
        f"{effect}_z"
    ] = (
        df.groupby(
            [
                "dataset",
                "model",
            ]
        )[
            effect
        ]
        .transform(
            zscore
        )
    )


# ============================================================
# Put Chronos and TimesFM side-by-side
# ============================================================

keep = [
    "dataset",
    "entity_id",
    "model",
    "entity_gain",
    "generic_gain",
    "total_gain",
    "entity_gain_z",
    "generic_gain_z",
    "total_gain_z",
]


wide = (
    df[
        keep
    ]
    .pivot(
        index=[
            "dataset",
            "entity_id",
        ],
        columns="model",
    )
)


wide.columns = [
    f"{variable}_{model}"
    for variable, model
    in wide.columns
]


wide = (
    wide
    .reset_index()
)


required_models = [
    "chronos2",
    "timesfm3",
]


for effect in [
    "entity_gain",
    "generic_gain",
    "total_gain",
]:

    for model in (
        required_models
    ):
        col = (
            f"{effect}_{model}"
        )

        if col not in (
            wide.columns
        ):
            raise RuntimeError(
                f"Missing {col}"
            )


# ============================================================
# Consensus phenotypes
# ============================================================

wide[
    "entity_context_sensitivity"
] = (
    (
        wide[
            "entity_gain_z_chronos2"
        ]
        + wide[
            "entity_gain_z_timesfm3"
        ]
    )
    / 2.0
)


wide[
    "generic_context_sensitivity"
] = (
    (
        wide[
            "generic_gain_z_chronos2"
        ]
        + wide[
            "generic_gain_z_timesfm3"
        ]
    )
    / 2.0
)


wide[
    "total_context_sensitivity"
] = (
    (
        wide[
            "total_gain_z_chronos2"
        ]
        + wide[
            "total_gain_z_timesfm3"
        ]
    )
    / 2.0
)


wide[
    "entity_model_disagreement"
] = abs(
    wide[
        "entity_gain_z_chronos2"
    ]
    - wide[
        "entity_gain_z_timesfm3"
    ]
)


wide[
    "entity_gain_mean_raw"
] = (
    (
        wide[
            "entity_gain_chronos2"
        ]
        + wide[
            "entity_gain_timesfm3"
        ]
    )
    / 2.0
)


wide[
    "dataset_label"
] = wide[
    "dataset"
].map(
    DATASET_LABELS
)


# ============================================================
# Cross-model reproducibility
# ============================================================

agreement_rows = []


print("=" * 100)
print(
    "CONTEXT-SENSITIVITY PHENOTYPE"
)
print("=" * 100)


for dataset in (
    DATASET_ORDER
):

    part = wide[
        wide[
            "dataset"
        ] == dataset
    ]

    print()
    print(
        "-" * 100
    )

    print(
        DATASET_LABELS[
            dataset
        ]
    )

    print(
        "-" * 100
    )

    for effect in [
        "entity_gain",
        "generic_gain",
        "total_gain",
    ]:

        rho, p, n = (
            spearman(
                part[
                    f"{effect}_chronos2"
                ],
                part[
                    f"{effect}_timesfm3"
                ],
            )
        )

        agreement_rows.append(
            {
                "dataset": (
                    dataset
                ),
                "dataset_label": (
                    DATASET_LABELS[
                        dataset
                    ]
                ),
                "effect": (
                    effect
                ),
                "n": (
                    n
                ),
                "rho": (
                    rho
                ),
                "p": (
                    p
                ),
            }
        )

        print(
            f"{effect:<18} "
            f"rho={rho:7.3f} "
            f"p={p:.4g} "
            f"N={n}"
        )


agreement = pd.DataFrame(
    agreement_rows
)

agreement.to_csv(
    OUT
    / "cross_model_reproducibility.csv",
    index=False,
)


# ============================================================
# Main figure:
# Chronos vs TimesFM entity-specific gain
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)

axes = axes.ravel()


for ax, dataset in zip(
    axes,
    DATASET_ORDER,
):

    part = wide[
        wide[
            "dataset"
        ] == dataset
    ]

    x = (
        part[
            "entity_gain_chronos2"
        ]
        .to_numpy()
    )

    y = (
        part[
            "entity_gain_timesfm3"
        ]
        .to_numpy()
    )

    rho, p, n = spearman(
        x,
        y,
    )

    ax.scatter(
        x,
        y,
        s=42,
        alpha=0.7,
    )

    lower = min(
        np.nanmin(x),
        np.nanmin(y),
    )

    upper = max(
        np.nanmax(x),
        np.nanmax(y),
    )

    padding = (
        0.05
        * (
            upper
            - lower
            + EPS
        )
    )

    lower -= padding
    upper += padding

    ax.plot(
        [
            lower,
            upper,
        ],
        [
            lower,
            upper,
        ],
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_xlim(
        lower,
        upper,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    ax.set_title(
        (
            f"{DATASET_LABELS[dataset]}\n"
            f"ρ={rho:.3f}, "
            f"N={n}"
        )
    )

    ax.set_xlabel(
        "Chronos-2 entity / episode gain"
    )

    ax.set_ylabel(
        "TimesFM3 entity / episode gain"
    )


save(
    fig,
    "01_cross_model_entity_gain_reproducibility",
)


# ============================================================
# Ranked consensus phenotype
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)

axes = axes.ravel()


for ax, dataset in zip(
    axes,
    DATASET_ORDER,
):

    part = (
        wide[
            wide[
                "dataset"
            ] == dataset
        ]
        .sort_values(
            "entity_context_sensitivity"
        )
        .reset_index(
            drop=True
        )
    )

    rank = np.arange(
        1,
        len(part) + 1,
    )

    ax.scatter(
        rank,
        part[
            "entity_gain_z_chronos2"
        ],
        s=24,
        alpha=0.65,
        label="Chronos-2",
    )

    ax.scatter(
        rank,
        part[
            "entity_gain_z_timesfm3"
        ],
        s=24,
        alpha=0.65,
        label="TimesFM3",
    )

    ax.plot(
        rank,
        part[
            "entity_context_sensitivity"
        ],
        linewidth=1.5,
        label="Consensus",
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        DATASET_LABELS[
            dataset
        ]
    )

    ax.set_xlabel(
        "Entities ranked by consensus sensitivity"
    )

    ax.set_ylabel(
        "Within-model standardized gain"
    )

    ax.legend(
        fontsize=8
    )


save(
    fig,
    "02_ranked_entity_context_sensitivity",
)


# ============================================================
# Highest and lowest sensitivity entities
# ============================================================

rank_rows = []


print()
print("=" * 100)
print(
    "MOST / LEAST CONTEXT-SENSITIVE ENTITIES"
)
print("=" * 100)


for dataset in (
    DATASET_ORDER
):

    part = (
        wide[
            wide[
                "dataset"
            ] == dataset
        ]
        .sort_values(
            "entity_context_sensitivity",
            ascending=False,
        )
    )

    print()
    print(
        DATASET_LABELS[
            dataset
        ]
    )

    print(
        "  Highest:"
    )

    print(
        part[
            [
                "entity_id",
                "entity_context_sensitivity",
                "entity_gain_chronos2",
                "entity_gain_timesfm3",
            ]
        ]
        .head(5)
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:8.3f}"
            ),
        )
    )

    print(
        "  Lowest:"
    )

    print(
        part[
            [
                "entity_id",
                "entity_context_sensitivity",
                "entity_gain_chronos2",
                "entity_gain_timesfm3",
            ]
        ]
        .tail(5)
        .sort_values(
            "entity_context_sensitivity"
        )
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:8.3f}"
            ),
        )
    )

    high = (
        part.head(5)
        .copy()
    )

    high[
        "group"
    ] = "highest"

    low = (
        part.tail(5)
        .copy()
    )

    low[
        "group"
    ] = "lowest"

    rank_rows.extend(
        [
            high,
            low,
        ]
    )


extremes = pd.concat(
    rank_rows,
    ignore_index=True,
)

extremes.to_csv(
    OUT
    / "context_sensitivity_extremes.csv",
    index=False,
)


# ============================================================
# Merge simple entity-level statistical signatures
# ============================================================

sig = pd.read_csv(
    SIGNATURES,
    dtype={
        "entity_id": str,
    },
)


phenotype = wide.merge(
    sig,
    on=[
        "dataset",
        "entity_id",
    ],
    how="left",
    validate="one_to_one",
    suffixes=(
        "",
        "_signature",
    ),
)


# Interpretable scale-free entity descriptors.
phenotype[
    "baseline_prominence"
] = (
    abs(
        phenotype[
            "mean_level"
        ]
    )
    / (
        phenotype[
            "mean_scale"
        ].abs()
        + EPS
    )
)


phenotype[
    "level_nonstationarity"
] = (
    phenotype[
        "within_level_sd"
    ]
    / (
        phenotype[
            "mean_scale"
        ].abs()
        + EPS
    )
)


phenotype.to_csv(
    OUT
    / "entity_context_sensitivity_consensus.csv",
    index=False,
)


# ============================================================
# Do simple entity properties predict the consensus phenotype?
# ============================================================

FEATURES = {
    "baseline_prominence": (
        "Baseline prominence"
    ),
    "mean_scale": (
        "Signal scale"
    ),
    "level_nonstationarity": (
        "Level nonstationarity"
    ),
    "median_lag1": (
        "Lag-1 persistence"
    ),
    "median_innovation_ratio": (
        "Innovation ratio"
    ),
}


feature_rows = []


print()
print("=" * 100)
print(
    "ENTITY FEATURES vs CONSENSUS PERSONALIZATION"
)
print("=" * 100)


for dataset in (
    DATASET_ORDER
):

    part = phenotype[
        phenotype[
            "dataset"
        ] == dataset
    ]

    print()
    print(
        DATASET_LABELS[
            dataset
        ]
    )

    for feature, label in (
        FEATURES.items()
    ):

        rho, p, n = (
            spearman(
                part[
                    feature
                ],
                part[
                    "entity_context_sensitivity"
                ],
            )
        )

        feature_rows.append(
            {
                "dataset": (
                    dataset
                ),
                "dataset_label": (
                    DATASET_LABELS[
                        dataset
                    ]
                ),
                "feature": (
                    feature
                ),
                "feature_label": (
                    label
                ),
                "n": (
                    n
                ),
                "rho": (
                    rho
                ),
                "p": (
                    p
                ),
            }
        )

        print(
            f"  {label:<24} "
            f"rho={rho:7.3f} "
            f"p={p:.4g}"
        )


feature_corr = pd.DataFrame(
    feature_rows
)

feature_corr.to_csv(
    OUT
    / "entity_feature_correlations.csv",
    index=False,
)


# ============================================================
# Pooled analysis after within-domain standardization
# ============================================================

pooled = (
    phenotype.copy()
)


for feature in (
    FEATURES
):

    pooled[
        f"{feature}_z"
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


pooled_rows = []


print()
print("=" * 100)
print(
    "POOLED WITHIN-DOMAIN STANDARDIZED FEATURE ANALYSIS"
)
print("=" * 100)


for feature, label in (
    FEATURES.items()
):

    rho, p, n = (
        spearman(
            pooled[
                f"{feature}_z"
            ],
            pooled[
                "entity_context_sensitivity"
            ],
        )
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


pooled_result = (
    pd.DataFrame(
        pooled_rows
    )
    .sort_values(
        "rho",
        key=lambda x: abs(x),
        ascending=False,
    )
)


print(
    pooled_result.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:8.3f}"
        ),
    )
)


pooled_result.to_csv(
    OUT
    / "pooled_entity_feature_correlations.csv",
    index=False,
)


# ============================================================
# Heatmap: feature → context-sensitivity correlation
# ============================================================

heat = (
    feature_corr.pivot(
        index="feature_label",
        columns="dataset",
        values="rho",
    )
)


heat = heat.loc[
    list(
        FEATURES.values()
    ),
    DATASET_ORDER,
]


fig, ax = plt.subplots(
    figsize=(10, 6)
)

im = ax.imshow(
    heat.to_numpy(),
    aspect="auto",
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
)


ax.set_xticks(
    np.arange(
        len(
            DATASET_ORDER
        )
    )
)

ax.set_xticklabels(
    [
        DATASET_LABELS[
            dataset
        ]
        for dataset
        in DATASET_ORDER
    ],
    rotation=30,
    ha="right",
)


ax.set_yticks(
    np.arange(
        len(
            FEATURES
        )
    )
)

ax.set_yticklabels(
    list(
        FEATURES.values()
    )
)


for i in range(
    heat.shape[0]
):
    for j in range(
        heat.shape[1]
    ):
        ax.text(
            j,
            i,
            f"{heat.iloc[i, j]:.2f}",
            ha="center",
            va="center",
        )


fig.colorbar(
    im,
    ax=ax,
    label=(
        "Spearman ρ with "
        "consensus context sensitivity"
    ),
)


ax.set_title(
    "What predicts entity-level personalization susceptibility?"
)


save(
    fig,
    "03_entity_feature_correlation_heatmap",
)


# ============================================================
# Model disagreement
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9),
)

axes = axes.ravel()


for ax, dataset in zip(
    axes,
    DATASET_ORDER,
):

    part = phenotype[
        phenotype[
            "dataset"
        ] == dataset
    ]

    ax.hist(
        part[
            "entity_model_disagreement"
        ],
        bins=15,
    )

    ax.set_title(
        DATASET_LABELS[
            dataset
        ]
    )

    ax.set_xlabel(
        "|Chronos z − TimesFM z|"
    )

    ax.set_ylabel(
        "Entities"
    )


save(
    fig,
    "04_entity_model_disagreement",
)


print()
print("=" * 100)
print(
    "SAVED"
)
print("=" * 100)

print(
    OUT
)
