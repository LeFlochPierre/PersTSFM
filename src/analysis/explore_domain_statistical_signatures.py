from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.data.registry import (
    load_dataset,
    extract_case,
)


OUT = Path(
    "results/context_regime_comparison/domain_signatures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MECHANISM_PATH = Path(
    "results/context_regime_comparison/"
    "exploratory_figures/"
    "exploratory_mechanism_summary.csv"
)


# Use the recent context that defines the intervention
# in each domain. This is identical across Chronos/TimesFM.
CONFIGS = {
    "pems_bay": {
        "label": "PEMS-BAY",
        "recent_hours": 24,
    },
    "electricity": {
        "label": "Electricity",
        "recent_hours": 96,
    },
    "hupa_ucm": {
        "label": "HUPA CGM",
        "recent_hours": 24,
    },
    "vitaldb": {
        "label": "VitalDB MAP",
        "recent_hours": 0.5,
    },
    "solar": {
        "label": "Solar PV",
        "recent_hours": 24,
    },
    "physionet2019": {
        "label": "PhysioNet HR",
        "recent_hours": 6,
    },
}


EPS = 1e-8


def safe_std(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    if len(x) < 2:
        return np.nan

    return float(
        np.std(
            x,
            ddof=1,
        )
    )


def standardize(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    mean = float(
        np.mean(x)
    )

    std = float(
        np.std(x)
    )

    if (
        not np.isfinite(std)
        or std < EPS
    ):
        return np.zeros_like(
            x,
            dtype=float,
        )

    return (
        x - mean
    ) / std


def lag1_corr(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    if len(x) < 3:
        return np.nan

    a = x[:-1]
    b = x[1:]

    if (
        np.std(a) < EPS
        or np.std(b) < EPS
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def innovation_ratio(x):
    """
    Standard deviation of first differences
    relative to signal standard deviation.

    Smaller values -> smoother / more persistent.
    Larger values -> more locally unpredictable.
    """
    x = np.asarray(
        x,
        dtype=float,
    )

    std = np.std(x)

    if (
        len(x) < 2
        or std < EPS
    ):
        return np.nan

    return float(
        np.std(
            np.diff(x)
        )
        / std
    )


def rmse(a, b):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(a)
                    - np.asarray(b)
                )
                ** 2
            )
        )
    )


def compute_identifiability(
    samples,
):
    """
    Leave-one-case-out entity identification.

    Every context is independently z-normalized, so this
    measures persistent trajectory SHAPE rather than mean
    or scale.

    For each case:
      distance to own entity's leave-one-out centroid
      vs nearest centroid belonging to another entity.

    Positive margin means the correct entity is closer.
    """
    by_entity = {}

    for sample in samples:
        by_entity.setdefault(
            sample[
                "entity_id"
            ],
            [],
        ).append(
            sample[
                "z"
            ]
        )

    centroids = {
        entity: np.mean(
            np.stack(
                windows
            ),
            axis=0,
        )
        for entity, windows
        in by_entity.items()
    }

    margins = []
    ratios = []
    correct = []
    own_distances = []
    other_distances = []

    for sample in samples:

        entity = sample[
            "entity_id"
        ]

        z = sample[
            "z"
        ]

        own_windows = (
            by_entity[
                entity
            ]
        )

        if len(
            own_windows
        ) < 2:
            continue

        # Leave this exact case out of the
        # entity centroid.
        own_stack = np.stack(
            own_windows
        )

        matches = []

        for i in range(
            len(
                own_windows
            )
        ):
            if np.array_equal(
                own_windows[i],
                z,
            ):
                matches.append(i)

        # Duplicate windows are possible. Rather
        # than depending on object identity, remove
        # one best-matching occurrence.
        distances_to_windows = np.asarray(
            [
                rmse(
                    z,
                    w,
                )
                for w
                in own_windows
            ]
        )

        remove_idx = int(
            np.argmin(
                distances_to_windows
            )
        )

        own_remaining = np.delete(
            own_stack,
            remove_idx,
            axis=0,
        )

        own_centroid = np.mean(
            own_remaining,
            axis=0,
        )

        own_distance = rmse(
            z,
            own_centroid,
        )

        other = []

        for other_entity, centroid in (
            centroids.items()
        ):
            if (
                other_entity
                == entity
            ):
                continue

            other.append(
                rmse(
                    z,
                    centroid,
                )
            )

        nearest_other = float(
            np.min(
                other
            )
        )

        margin = (
            nearest_other
            - own_distance
        )

        normalized_margin = (
            margin
            / (
                nearest_other
                + EPS
            )
        )

        ratio = (
            nearest_other
            / (
                own_distance
                + EPS
            )
        )

        margins.append(
            normalized_margin
        )

        ratios.append(
            ratio
        )

        correct.append(
            own_distance
            < nearest_other
        )

        own_distances.append(
            own_distance
        )

        other_distances.append(
            nearest_other
        )

    return {
        "identification_accuracy": (
            float(
                np.mean(
                    correct
                )
            )
        ),
        "identification_margin": (
            float(
                np.median(
                    margins
                )
            )
        ),
        "identification_ratio": (
            float(
                np.median(
                    ratios
                )
            )
        ),
        "median_own_distance": (
            float(
                np.median(
                    own_distances
                )
            )
        ),
        "median_nearest_other_distance": (
            float(
                np.median(
                    other_distances
                )
            )
        ),
    }


domain_rows = []
entity_rows = []


print(
    "=" * 100
)

print(
    "DATA-ONLY DOMAIN STATISTICAL SIGNATURES"
)

print(
    "=" * 100
)


for dataset_key, config in (
    CONFIGS.items()
):

    print()
    print(
        "-" * 100
    )

    print(
        config[
            "label"
        ]
    )

    print(
        "-" * 100
    )

    dataset = load_dataset(
        dataset_key
    )

    recent_hours = float(
        config[
            "recent_hours"
        ]
    )

    samples = []

    for _, row in (
        dataset.protocol.iterrows()
    ):

        case = extract_case(
            dataset,
            row,
            recent_hours,
        )

        x = (
            case[
                "context"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.all(
            np.isfinite(
                x
            )
        ):
            raise RuntimeError(
                f"{dataset_key}: "
                "non-finite context"
            )

        sample = {
            "entity_id": (
                case[
                    "entity_id"
                ]
            ),
            "eval_index": (
                case[
                    "eval_index"
                ]
            ),
            "mean": float(
                np.mean(x)
            ),
            "std": float(
                np.std(
                    x,
                    ddof=1,
                )
            ),
            "lag1": (
                lag1_corr(x)
            ),
            "innovation_ratio": (
                innovation_ratio(
                    x
                )
            ),
            "z": standardize(
                x
            ),
        }

        samples.append(
            sample
        )

    sample_df = pd.DataFrame(
        [
            {
                key: value
                for key, value
                in sample.items()
                if key != "z"
            }
            for sample
            in samples
        ]
    )

    # --------------------------------------------------------
    # Per-entity summaries
    # --------------------------------------------------------

    for entity_id, part in (
        sample_df.groupby(
            "entity_id"
        )
    ):

        entity_rows.append(
            {
                "dataset": (
                    dataset_key
                ),
                "dataset_label": (
                    config[
                        "label"
                    ]
                ),
                "entity_id": (
                    entity_id
                ),
                "mean_level": float(
                    part[
                        "mean"
                    ].mean()
                ),
                "mean_scale": float(
                    part[
                        "std"
                    ].mean()
                ),
                "within_level_sd": (
                    safe_std(
                        part[
                            "mean"
                        ]
                    )
                ),
                "median_lag1": float(
                    np.nanmedian(
                        part[
                            "lag1"
                        ]
                    )
                ),
                "median_innovation_ratio": float(
                    np.nanmedian(
                        part[
                            "innovation_ratio"
                        ]
                    )
                ),
            }
        )

    entity_df = pd.DataFrame(
        [
            row
            for row in entity_rows
            if row[
                "dataset"
            ] == dataset_key
        ]
    )

    # Between-entity typical level variation,
    # normalized by typical within-context scale.
    between_level_sd = float(
        np.std(
            entity_df[
                "mean_level"
            ],
            ddof=1,
        )
    )

    typical_signal_scale = float(
        np.nanmedian(
            entity_df[
                "mean_scale"
            ]
        )
    )

    level_heterogeneity_ratio = (
        between_level_sd
        / (
            typical_signal_scale
            + EPS
        )
    )

    # Between-entity level differences relative
    # to how much an entity's own level changes
    # between evaluation cases.
    typical_within_level_sd = float(
        np.nanmedian(
            entity_df[
                "within_level_sd"
            ]
        )
    )

    level_stability_ratio = (
        between_level_sd
        / (
            typical_within_level_sd
            + EPS
        )
    )

    valid_scales = (
        entity_df[
            "mean_scale"
        ]
        .to_numpy(
            dtype=float
        )
    )

    valid_scales = (
        valid_scales[
            np.isfinite(
                valid_scales
            )
            & (
                valid_scales
                > EPS
            )
        ]
    )

    scale_heterogeneity = float(
        np.std(
            np.log(
                valid_scales
            ),
            ddof=1,
        )
    )

    identity = (
        compute_identifiability(
            samples
        )
    )

    domain_row = {
        "dataset": (
            dataset_key
        ),
        "dataset_label": (
            config[
                "label"
            ]
        ),
        "recent_hours": (
            recent_hours
        ),
        "n_entities": int(
            sample_df[
                "entity_id"
            ].nunique()
        ),
        "n_cases": int(
            len(
                sample_df
            )
        ),
        "level_heterogeneity_ratio": (
            level_heterogeneity_ratio
        ),
        "level_stability_ratio": (
            level_stability_ratio
        ),
        "scale_heterogeneity": (
            scale_heterogeneity
        ),
        "median_lag1": float(
            np.nanmedian(
                entity_df[
                    "median_lag1"
                ]
            )
        ),
        "median_innovation_ratio": float(
            np.nanmedian(
                entity_df[
                    "median_innovation_ratio"
                ]
            )
        ),
        **identity,
    }

    domain_rows.append(
        domain_row
    )

    print(
        f"Entities:                  "
        f"{domain_row['n_entities']}"
    )

    print(
        f"Cases:                     "
        f"{domain_row['n_cases']}"
    )

    print(
        f"Level heterogeneity ratio: "
        f"{level_heterogeneity_ratio:.4f}"
    )

    print(
        f"Level stability ratio:     "
        f"{level_stability_ratio:.4f}"
    )

    print(
        f"Scale heterogeneity:       "
        f"{scale_heterogeneity:.4f}"
    )

    print(
        f"Median lag-1 correlation:  "
        f"{domain_row['median_lag1']:.4f}"
    )

    print(
        f"Innovation ratio:          "
        f"{domain_row['median_innovation_ratio']:.4f}"
    )

    print(
        f"Identification accuracy:   "
        f"{identity['identification_accuracy']:.4f}"
    )

    print(
        f"Identification margin:     "
        f"{identity['identification_margin']:.4f}"
    )

    print(
        f"Identification ratio:      "
        f"{identity['identification_ratio']:.4f}"
    )


domains = pd.DataFrame(
    domain_rows
)

entities = pd.DataFrame(
    entity_rows
)


domains.to_csv(
    OUT
    / "domain_statistical_signatures.csv",
    index=False,
)

entities.to_csv(
    OUT
    / "entity_statistical_signatures.csv",
    index=False,
)


# ============================================================
# Merge with intervention mechanisms
# ============================================================

mechanisms = pd.read_csv(
    MECHANISM_PATH
)

domain_mechanisms = (
    mechanisms.groupby(
        [
            "dataset",
            "dataset_label",
        ],
        as_index=False,
    )
    [
        [
            "total_pct",
            "generic_pct",
            "entity_pct",
            "order_pct",
        ]
    ]
    .mean()
)

merged = domains.merge(
    domain_mechanisms,
    on=[
        "dataset",
        "dataset_label",
    ],
    how="inner",
)

merged.to_csv(
    OUT
    / "domain_signatures_with_mechanisms.csv",
    index=False,
)


print()
print(
    "=" * 100
)

print(
    "DOMAIN SIGNATURE + MECHANISM TABLE"
)

print(
    "=" * 100
)

cols = [
    "dataset_label",
    "level_heterogeneity_ratio",
    "level_stability_ratio",
    "scale_heterogeneity",
    "median_lag1",
    "median_innovation_ratio",
    "identification_accuracy",
    "identification_margin",
    "generic_pct",
    "entity_pct",
]

print(
    merged[
        cols
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:8.3f}"
        ),
    )
)


# ============================================================
# Exploratory cross-domain correlations
# N=6 — descriptive only.
# ============================================================

signature_cols = [
    "level_heterogeneity_ratio",
    "level_stability_ratio",
    "scale_heterogeneity",
    "median_lag1",
    "median_innovation_ratio",
    "identification_accuracy",
    "identification_margin",
]

effect_cols = [
    "total_pct",
    "generic_pct",
    "entity_pct",
    "order_pct",
]


corr_rows = []


print()
print(
    "=" * 100
)

print(
    "EXPLORATORY SPEARMAN CORRELATIONS — N=6 DOMAINS"
)

print(
    "Interpret descriptively; this is not confirmatory inference."
)

print(
    "=" * 100
)


for signature in signature_cols:

    for effect in effect_cols:

        rho, p = spearmanr(
            merged[
                signature
            ],
            merged[
                effect
            ],
        )

        corr_rows.append(
            {
                "signature": (
                    signature
                ),
                "effect": (
                    effect
                ),
                "rho": (
                    rho
                ),
                "p": (
                    p
                ),
            }
        )


corr = pd.DataFrame(
    corr_rows
)

corr.to_csv(
    OUT
    / "domain_signature_correlations.csv",
    index=False,
)


for effect in effect_cols:

    print()
    print(
        effect
    )

    part = (
        corr[
            corr[
                "effect"
            ] == effect
        ]
        .copy()
        .sort_values(
            "rho",
            key=lambda x: (
                np.abs(x)
            ),
            ascending=False,
        )
    )

    print(
        part[
            [
                "signature",
                "rho",
                "p",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:8.3f}"
            ),
        )
    )


# ============================================================
# Plot helpers
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


def scatter_with_labels(
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    filename,
):
    rho, p = spearmanr(
        merged[
            x_col
        ],
        merged[
            y_col
        ],
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.scatter(
        merged[
            x_col
        ],
        merged[
            y_col
        ],
        s=90,
    )

    for _, row in (
        merged.iterrows()
    ):
        ax.annotate(
            row[
                "dataset_label"
            ],
            (
                row[
                    x_col
                ],
                row[
                    y_col
                ],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        y_label
    )

    ax.set_title(
        (
            f"{title}\n"
            f"Spearman ρ={rho:.3f}, "
            f"p={p:.3f}; N=6"
        )
    )

    save(
        fig,
        filename,
    )


# ============================================================
# Key explanatory plots
# ============================================================

scatter_with_labels(
    "identification_margin",
    "entity_pct",
    "Dynamic identifiability margin",
    "Entity / episode gain (% short MAE)",
    "Does identifiable entity structure predict personalization?",
    "01_identifiability_vs_entity_gain",
)

scatter_with_labels(
    "identification_accuracy",
    "entity_pct",
    "Entity identification accuracy",
    "Entity / episode gain (% short MAE)",
    "Entity recognition vs entity-specific context value",
    "02_identification_accuracy_vs_entity_gain",
)

scatter_with_labels(
    "level_heterogeneity_ratio",
    "entity_pct",
    "Between-entity level heterogeneity / within scale",
    "Entity / episode gain (% short MAE)",
    "Do persistent level differences predict personalization?",
    "03_level_heterogeneity_vs_entity_gain",
)

scatter_with_labels(
    "level_stability_ratio",
    "entity_pct",
    "Between-entity / within-entity level variation",
    "Entity / episode gain (% short MAE)",
    "Stable entity differences vs personalization",
    "04_level_stability_vs_entity_gain",
)

scatter_with_labels(
    "median_innovation_ratio",
    "entity_pct",
    "Within-entity innovation ratio",
    "Entity / episode gain (% short MAE)",
    "Does local unpredictability suppress personalization?",
    "05_innovation_vs_entity_gain",
)

scatter_with_labels(
    "median_lag1",
    "generic_pct",
    "Median lag-1 autocorrelation",
    "Generic-context gain (% short MAE)",
    "Persistence vs transferable context",
    "06_persistence_vs_generic_gain",
)

scatter_with_labels(
    "identification_margin",
    "generic_pct",
    "Dynamic identifiability margin",
    "Generic-context gain (% short MAE)",
    "Identifiability vs transferable context",
    "07_identifiability_vs_generic_gain",
)

scatter_with_labels(
    "median_lag1",
    "order_pct",
    "Median lag-1 autocorrelation",
    "Temporal-order gain (% short MAE)",
    "Persistence vs value of temporal ordering",
    "08_persistence_vs_order_gain",
)


# ============================================================
# Correlation heatmap
# ============================================================

heat = (
    corr.pivot(
        index="signature",
        columns="effect",
        values="rho",
    )
    .loc[
        signature_cols,
        effect_cols,
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 7)
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
            effect_cols
        )
    )
)

ax.set_xticklabels(
    [
        "Total",
        "Generic",
        "Entity",
        "Order",
    ]
)

ax.set_yticks(
    np.arange(
        len(
            signature_cols
        )
    )
)

ax.set_yticklabels(
    [
        "Level heterogeneity",
        "Level stability",
        "Scale heterogeneity",
        "Lag-1 persistence",
        "Innovation ratio",
        "Identification accuracy",
        "Identification margin",
    ]
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
    label="Spearman ρ",
)

ax.set_title(
    "Which data properties align with context mechanisms?"
)

save(
    fig,
    "09_signature_mechanism_correlation_heatmap",
)


print()
print(
    "=" * 100
)

print(
    "SAVED"
)

print(
    "=" * 100
)

print(
    OUT
)
