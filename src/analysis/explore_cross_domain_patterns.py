from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


RESULTS = Path("results")

OUT = (
    RESULTS
    / "context_regime_comparison"
    / "exploratory_figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42
N_BOOTSTRAP = 10000


CONFIGS = [
    ("pems_bay", "chronos2", 24, 240),
    ("pems_bay", "timesfm3", 24, 504),

    ("electricity", "chronos2", 96, 192),
    ("electricity", "timesfm3", 96, 240),

    ("hupa_ucm", "chronos2", 24, 96),
    ("hupa_ucm", "timesfm3", 24, 168),

    ("vitaldb", "chronos2", 0.5, 4),
    ("vitaldb", "timesfm3", 0.5, 4),

    ("solar", "chronos2", 24, 96),
    ("solar", "timesfm3", 24, 96),

    ("physionet2019", "chronos2", 6, 48),
    ("physionet2019", "timesfm3", 6, 48),
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


MODEL_SHORT = {
    "chronos2": "C",
    "timesfm3": "T",
}


DATASET_ORDER = [
    "pems_bay",
    "electricity",
    "hupa_ucm",
    "vitaldb",
    "solar",
    "physionet2019",
]


def result_folder(
    dataset,
    model,
    recent_hours,
    long_hours,
):
    return (
        RESULTS
        / dataset
        / "interventions"
        / model
        / (
            f"recent_{recent_hours:g}h"
            f"_long_{long_hours:g}h"
        )
    )


def bootstrap_normalized_ci(
    effect,
    short,
    seed,
):
    effect = np.asarray(
        effect,
        dtype=float,
    )

    short = np.asarray(
        short,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(effect)

    idx = rng.integers(
        0,
        n,
        size=(
            N_BOOTSTRAP,
            n,
        ),
    )

    effect_b = (
        effect[idx]
        .mean(axis=1)
    )

    short_b = (
        short[idx]
        .mean(axis=1)
    )

    values = (
        100.0
        * effect_b
        / short_b
    )

    return np.quantile(
        values,
        [
            0.025,
            0.975,
        ],
    )


def save_figure(
    fig,
    name,
):
    fig.tight_layout()

    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        png,
    )


# ============================================================
# Load intervention results
# ============================================================

rows = []


for config_index, (
    dataset,
    model,
    recent_hours,
    long_hours,
) in enumerate(CONFIGS):

    folder = result_folder(
        dataset,
        model,
        recent_hours,
        long_hours,
    )

    path = (
        folder
        / "entity_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        dtype={
            "entity_id": str,
        },
    )

    wide = df.pivot(
        index="entity_id",
        columns="intervention",
        values="mean_mae",
    )

    if (
        "same_entity_day_shuffle"
        in wide.columns
    ):
        temporal_control = (
            "same_entity_day_shuffle"
        )

    elif (
        "same_entity_block_shuffle"
        in wide.columns
    ):
        temporal_control = (
            "same_entity_block_shuffle"
        )

    else:
        temporal_control = None

    required = [
        "short",
        "same_entity",
        "matched_other_affine",
    ]

    for condition in required:
        if condition not in wide.columns:
            raise RuntimeError(
                f"{dataset}/{model}: "
                f"missing {condition}"
            )

    short = (
        wide["short"]
        .to_numpy(dtype=float)
    )

    same = (
        wide["same_entity"]
        .to_numpy(dtype=float)
    )

    other = (
        wide[
            "matched_other_affine"
        ]
        .to_numpy(dtype=float)
    )

    total = (
        short
        - same
    )

    generic = (
        short
        - other
    )

    entity = (
        other
        - same
    )

    if temporal_control is None:
        order = np.full(
            len(short),
            np.nan,
        )
    else:
        shuffled = (
            wide[
                temporal_control
            ]
            .to_numpy(dtype=float)
        )

        order = (
            shuffled
            - same
        )

    if not np.allclose(
        total,
        generic + entity,
    ):
        raise RuntimeError(
            f"{dataset}/{model}: "
            "decomposition failed"
        )

    short_mean = float(
        np.mean(short)
    )

    effects = {
        "total": total,
        "generic": generic,
        "entity": entity,
        "order": order,
    }

    result = {
        "dataset": dataset,
        "dataset_label": (
            DATASET_LABELS[
                dataset
            ]
        ),
        "model": model,
        "model_label": (
            MODEL_LABELS[
                model
            ]
        ),
        "n_entities": len(short),
        "recent_hours": (
            recent_hours
        ),
        "long_hours": (
            long_hours
        ),
        "short_mae": (
            short_mean
        ),
    }

    for effect_name, values in (
        effects.items()
    ):

        mean_effect = float(
            np.nanmean(values)
        )

        normalized = (
            100.0
            * mean_effect
            / short_mean
        )

        result[
            f"{effect_name}_gain"
        ] = mean_effect

        result[
            f"{effect_name}_pct"
        ] = normalized

        if not np.isnan(
            values
        ).all():

            low, high = (
                bootstrap_normalized_ci(
                    values,
                    short,
                    seed=(
                        SEED
                        + config_index * 10
                        + len(
                            effect_name
                        )
                    ),
                )
            )

            result[
                f"{effect_name}_pct_low"
            ] = low

            result[
                f"{effect_name}_pct_high"
            ] = high

        else:
            result[
                f"{effect_name}_pct_low"
            ] = np.nan

            result[
                f"{effect_name}_pct_high"
            ] = np.nan

    denom = (
        abs(
            result[
                "entity_pct"
            ]
        )
        + abs(
            result[
                "generic_pct"
            ]
        )
    )

    if denom > 0:
        result[
            "dominance_index"
        ] = (
            result[
                "entity_pct"
            ]
            - result[
                "generic_pct"
            ]
        ) / denom
    else:
        result[
            "dominance_index"
        ] = np.nan

    rows.append(
        result
    )


summary = pd.DataFrame(
    rows
)


summary_path = (
    OUT
    / "exploratory_mechanism_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print()
print("=" * 100)
print("NORMALIZED MECHANISM TABLE")
print("=" * 100)

display_cols = [
    "dataset_label",
    "model_label",
    "short_mae",
    "total_pct",
    "generic_pct",
    "entity_pct",
    "order_pct",
    "dominance_index",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:8.3f}"
        ),
    )
)


# ============================================================
# Figure 1
# Mechanism map
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

for dataset in DATASET_ORDER:

    part = (
        summary[
            summary[
                "dataset"
            ] == dataset
        ]
        .set_index(
            "model"
        )
    )

    if {
        "chronos2",
        "timesfm3",
    }.issubset(
        part.index
    ):
        ax.plot(
            [
                part.loc[
                    "chronos2",
                    "generic_pct",
                ],
                part.loc[
                    "timesfm3",
                    "generic_pct",
                ],
            ],
            [
                part.loc[
                    "chronos2",
                    "entity_pct",
                ],
                part.loc[
                    "timesfm3",
                    "entity_pct",
                ],
            ],
            linewidth=1.5,
            alpha=0.55,
        )


for model, marker in [
    ("chronos2", "o"),
    ("timesfm3", "s"),
]:

    part = summary[
        summary[
            "model"
        ] == model
    ]

    ax.scatter(
        part[
            "generic_pct"
        ],
        part[
            "entity_pct"
        ],
        marker=marker,
        s=80,
        label=(
            MODEL_LABELS[
                model
            ]
        ),
        zorder=3,
    )


for _, row in (
    summary.iterrows()
):
    ax.annotate(
        (
            f"{row['dataset_label']} "
            f"{MODEL_SHORT[row['model']]}"
        ),
        (
            row[
                "generic_pct"
            ],
            row[
                "entity_pct"
            ],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )


ax.axvline(
    0,
    linewidth=1,
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Generic-context gain (% of short-context MAE)"
)

ax.set_ylabel(
    "Entity / episode-specific gain (% of short-context MAE)"
)

ax.set_title(
    "Cross-domain context mechanism map"
)

ax.legend()

save_figure(
    fig,
    "01_mechanism_map_normalized",
)


# ============================================================
# Figure 2
# Generic + entity forest plot
# ============================================================

plot_df = (
    summary.copy()
    .reset_index(drop=True)
)

plot_df[
    "row_label"
] = (
    plot_df[
        "dataset_label"
    ]
    + " — "
    + plot_df[
        "model_label"
    ]
)

y = np.arange(
    len(plot_df)
)

fig, ax = plt.subplots(
    figsize=(10, 9)
)


for effect_name, offset, marker in [
    ("generic", -0.12, "o"),
    ("entity", 0.12, "s"),
]:

    x = (
        plot_df[
            f"{effect_name}_pct"
        ].to_numpy()
    )

    low = (
        plot_df[
            f"{effect_name}_pct_low"
        ].to_numpy()
    )

    high = (
        plot_df[
            f"{effect_name}_pct_high"
        ].to_numpy()
    )

    xerr = np.vstack(
        [
            x - low,
            high - x,
        ]
    )

    ax.errorbar(
        x,
        y + offset,
        xerr=xerr,
        fmt=marker,
        capsize=3,
        label=(
            "Generic"
            if effect_name
            == "generic"
            else "Entity / episode"
        ),
    )


ax.axvline(
    0,
    linewidth=1,
)

ax.set_yticks(
    y
)

ax.set_yticklabels(
    plot_df[
        "row_label"
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    "MAE improvement (% of short-context MAE)"
)

ax.set_title(
    "Decomposition of long-context benefit"
)

ax.legend()

save_figure(
    fig,
    "02_decomposition_forest_normalized",
)


# ============================================================
# Figure 3
# Total context gain
# ============================================================

total_df = (
    summary.sort_values(
        "total_pct"
    )
    .copy()
)

labels = (
    total_df[
        "dataset_label"
    ]
    + " — "
    + total_df[
        "model_label"
    ]
)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

ax.barh(
    labels,
    total_df[
        "total_pct"
    ],
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Long-context MAE improvement (% of short-context MAE)"
)

ax.set_title(
    "How much does longer context help?"
)

save_figure(
    fig,
    "03_total_context_gain_normalized",
)


# ============================================================
# Figure 4
# Generic ↔ entity dominance
# ============================================================

dominance_df = (
    summary.sort_values(
        "dominance_index"
    )
    .copy()
)

labels = (
    dominance_df[
        "dataset_label"
    ]
    + " — "
    + dominance_df[
        "model_label"
    ]
)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

ax.barh(
    labels,
    dominance_df[
        "dominance_index"
    ],
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlim(
    -1.05,
    1.05,
)

ax.set_xlabel(
    "Mechanism dominance (−1 generic, +1 entity / episode)"
)

ax.set_title(
    "What type of information drives the context benefit?"
)

save_figure(
    fig,
    "04_mechanism_dominance_index",
)


# ============================================================
# Figure 5
# Architecture shift: TimesFM - Chronos
# ============================================================

architecture_rows = []


for dataset in DATASET_ORDER:

    part = (
        summary[
            summary[
                "dataset"
            ] == dataset
        ]
        .set_index(
            "model"
        )
    )

    c = part.loc[
        "chronos2"
    ]

    t = part.loc[
        "timesfm3"
    ]

    architecture_rows.append(
        {
            "dataset": (
                DATASET_LABELS[
                    dataset
                ]
            ),
            "delta_generic": (
                t[
                    "generic_pct"
                ]
                - c[
                    "generic_pct"
                ]
            ),
            "delta_entity": (
                t[
                    "entity_pct"
                ]
                - c[
                    "entity_pct"
                ]
            ),
            "delta_total": (
                t[
                    "total_pct"
                ]
                - c[
                    "total_pct"
                ]
            ),
            "delta_order": (
                t[
                    "order_pct"
                ]
                - c[
                    "order_pct"
                ]
            ),
        }
    )


architecture = pd.DataFrame(
    architecture_rows
)

architecture.to_csv(
    OUT
    / "architecture_shift_summary.csv",
    index=False,
)


y = np.arange(
    len(architecture)
)

height = 0.35


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    y - height / 2,
    architecture[
        "delta_generic"
    ],
    height=height,
    label="Δ generic",
)

ax.barh(
    y + height / 2,
    architecture[
        "delta_entity"
    ],
    height=height,
    label="Δ entity / episode",
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_yticks(
    y
)

ax.set_yticklabels(
    architecture[
        "dataset"
    ]
)

ax.set_xlabel(
    "TimesFM3 − Chronos-2 normalized gain (percentage points)"
)

ax.set_title(
    "Where do the architectures differ?"
)

ax.legend()

save_figure(
    fig,
    "05_architecture_shift_generic_vs_entity",
)


# ============================================================
# Figures 6–7
# Cross-model agreement
# ============================================================

def cross_model_scatter(
    metric,
    title,
    filename,
):

    points = []

    for dataset in DATASET_ORDER:

        part = (
            summary[
                summary[
                    "dataset"
                ] == dataset
            ]
            .set_index(
                "model"
            )
        )

        points.append(
            {
                "dataset": (
                    DATASET_LABELS[
                        dataset
                    ]
                ),
                "chronos": (
                    part.loc[
                        "chronos2",
                        metric,
                    ]
                ),
                "timesfm": (
                    part.loc[
                        "timesfm3",
                        metric,
                    ]
                ),
            }
        )

    points = pd.DataFrame(
        points
    )

    rho, p = spearmanr(
        points[
            "chronos"
        ],
        points[
            "timesfm"
        ],
    )

    values = np.concatenate(
        [
            points[
                "chronos"
            ].to_numpy(),
            points[
                "timesfm"
            ].to_numpy(),
        ]
    )

    margin = max(
        1.0,
        0.08
        * (
            values.max()
            - values.min()
        ),
    )

    lower = (
        values.min()
        - margin
    )

    upper = (
        values.max()
        + margin
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        points[
            "chronos"
        ],
        points[
            "timesfm"
        ],
        s=90,
    )

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

    for _, row in (
        points.iterrows()
    ):
        ax.annotate(
            row[
                "dataset"
            ],
            (
                row[
                    "chronos"
                ],
                row[
                    "timesfm"
                ],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(
        lower,
        upper,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    ax.set_xlabel(
        "Chronos-2 normalized gain (%)"
    )

    ax.set_ylabel(
        "TimesFM3 normalized gain (%)"
    )

    ax.set_title(
        (
            f"{title}\n"
            f"Cross-domain Spearman "
            f"ρ={rho:.3f}, p={p:.3g}"
        )
    )

    save_figure(
        fig,
        filename,
    )

    return rho, p


rho_generic, p_generic = (
    cross_model_scatter(
        "generic_pct",
        "Cross-model agreement: generic context",
        "06_cross_model_generic_agreement",
    )
)

rho_entity, p_entity = (
    cross_model_scatter(
        "entity_pct",
        "Cross-model agreement: entity / episode context",
        "07_cross_model_entity_agreement",
    )
)


# ============================================================
# Figure 8
# Temporal order vs entity identity
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

for model, marker in [
    ("chronos2", "o"),
    ("timesfm3", "s"),
]:

    part = summary[
        summary[
            "model"
        ] == model
    ]

    ax.scatter(
        part[
            "entity_pct"
        ],
        part[
            "order_pct"
        ],
        marker=marker,
        s=80,
        label=(
            MODEL_LABELS[
                model
            ]
        ),
    )


for _, row in (
    summary.iterrows()
):
    ax.annotate(
        (
            f"{row['dataset_label']} "
            f"{MODEL_SHORT[row['model']]}"
        ),
        (
            row[
                "entity_pct"
            ],
            row[
                "order_pct"
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

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Entity / episode-specific gain (% of short MAE)"
)

ax.set_ylabel(
    "Temporal-order gain (% of short MAE)"
)

ax.set_title(
    "Is identity more important than temporal ordering?"
)

ax.legend()

save_figure(
    fig,
    "08_order_vs_entity_gain",
)


# ============================================================
# Figure 9
# Mechanism heatmap
# ============================================================

heat_rows = []

heat_labels = []


for dataset in DATASET_ORDER:

    part = (
        summary[
            summary[
                "dataset"
            ] == dataset
        ]
        .set_index(
            "model"
        )
    )

    heat_rows.append(
        [
            part.loc[
                "chronos2",
                "generic_pct",
            ],
            part.loc[
                "chronos2",
                "entity_pct",
            ],
            part.loc[
                "timesfm3",
                "generic_pct",
            ],
            part.loc[
                "timesfm3",
                "entity_pct",
            ],
        ]
    )

    heat_labels.append(
        DATASET_LABELS[
            dataset
        ]
    )


heat = np.asarray(
    heat_rows
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

max_abs = float(
    np.nanmax(
        np.abs(
            heat
        )
    )
)

im = ax.imshow(
    heat,
    aspect="auto",
    vmin=-max_abs,
    vmax=max_abs,
    cmap="coolwarm",
)

ax.set_xticks(
    np.arange(4)
)

ax.set_xticklabels(
    [
        "Chronos\nGeneric",
        "Chronos\nEntity",
        "TimesFM\nGeneric",
        "TimesFM\nEntity",
    ]
)

ax.set_yticks(
    np.arange(
        len(
            heat_labels
        )
    )
)

ax.set_yticklabels(
    heat_labels
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
            f"{heat[i, j]:.1f}",
            ha="center",
            va="center",
        )

fig.colorbar(
    im,
    ax=ax,
    label=(
        "MAE improvement "
        "(% of short-context MAE)"
    ),
)

ax.set_title(
    "Cross-domain mechanism fingerprint"
)

save_figure(
    fig,
    "09_mechanism_heatmap",
)


# ============================================================
# Figures 10+
# Normalized context acquisition curves
# ============================================================

context_data = {}


for dataset in DATASET_ORDER:

    context_data[
        dataset
    ] = {}

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for model, marker in [
        ("chronos2", "o"),
        ("timesfm3", "s"),
    ]:

        path = (
            RESULTS
            / dataset
            / "context"
            / model
            / "summary.csv"
        )

        if not path.exists():
            print(
                "WARNING: missing context curve:",
                path,
            )
            continue

        curve = pd.read_csv(
            path
        )

        curve = (
            curve.sort_values(
                "context_hours"
            )
            .reset_index(
                drop=True
            )
        )

        reference_mae = float(
            curve.iloc[0][
                "mean_entity_mae"
            ]
        )

        curve[
            "improvement_pct"
        ] = (
            100.0
            * (
                reference_mae
                - curve[
                    "mean_entity_mae"
                ]
            )
            / reference_mae
        )

        context_data[
            dataset
        ][model] = (
            curve
        )

        ax.plot(
            curve[
                "context_hours"
            ],
            curve[
                "improvement_pct"
            ],
            marker=marker,
            label=(
                MODEL_LABELS[
                    model
                ]
            ),
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xlabel(
        "Historical context (hours, log2 scale)"
    )

    ax.set_ylabel(
        "MAE improvement vs shortest context (%)"
    )

    ax.set_title(
        (
            f"{DATASET_LABELS[dataset]}: "
            "context acquisition"
        )
    )

    ax.legend()

    save_figure(
        fig,
        (
            "10_context_curve_"
            f"{dataset}"
        ),
    )


# ============================================================
# Figure 16
# All context curves together
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9),
)

axes = (
    axes
    .ravel()
)


for ax, dataset in zip(
    axes,
    DATASET_ORDER,
):

    for model, marker in [
        ("chronos2", "o"),
        ("timesfm3", "s"),
    ]:

        if (
            model
            not in context_data[
                dataset
            ]
        ):
            continue

        curve = (
            context_data[
                dataset
            ][model]
        )

        ax.plot(
            curve[
                "context_hours"
            ],
            curve[
                "improvement_pct"
            ],
            marker=marker,
            label=(
                MODEL_LABELS[
                    model
                ]
            ),
        )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_title(
        DATASET_LABELS[
            dataset
        ]
    )

    ax.set_xlabel(
        "Context (h)"
    )

    ax.set_ylabel(
        "MAE improvement (%)"
    )

    ax.legend(
        fontsize=8
    )


save_figure(
    fig,
    "16_all_context_acquisition_curves",
)


# ============================================================
# Architecture agreement statistics
# ============================================================

print()
print("=" * 100)
print("CROSS-MODEL AGREEMENT ACROSS DOMAINS")
print("=" * 100)


for metric, label in [
    (
        "total_pct",
        "Total context gain",
    ),
    (
        "generic_pct",
        "Generic-context gain",
    ),
    (
        "entity_pct",
        "Entity/episode gain",
    ),
    (
        "order_pct",
        "Temporal-order gain",
    ),
]:

    c = (
        summary[
            summary[
                "model"
            ] == "chronos2"
        ]
        .set_index(
            "dataset"
        )
        .loc[
            DATASET_ORDER,
            metric,
        ]
        .to_numpy()
    )

    t = (
        summary[
            summary[
                "model"
            ] == "timesfm3"
        ]
        .set_index(
            "dataset"
        )
        .loc[
            DATASET_ORDER,
            metric,
        ]
        .to_numpy()
    )

    rho, p = spearmanr(
        c,
        t,
    )

    print(
        f"{label:<28} "
        f"rho={rho:7.3f}  "
        f"p={p:.5f}"
    )


print()
print("=" * 100)
print("ARCHITECTURE SHIFT")
print("=" * 100)

print(
    architecture.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:8.3f}"
        ),
    )
)


print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    f"Figures: {OUT}"
)

print(
    f"Summary: {summary_path}"
)
