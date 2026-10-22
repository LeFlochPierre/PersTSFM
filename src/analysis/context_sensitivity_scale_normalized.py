from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("results")

OUT = Path(
    "results/context_regime_comparison/"
    "context_sensitivity_phenotype"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


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


LABELS = {
    "pems_bay": "PEMS-BAY",
    "electricity": "Electricity",
    "hupa_ucm": "HUPA CGM",
    "vitaldb": "VitalDB MAP",
    "solar": "Solar PV",
    "physionet2019": "PhysioNet HR",
}


EPS = 1e-12


def folder(
    dataset,
    model,
    recent,
    long,
):
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


rows = []


for (
    dataset,
    model,
    recent,
    long,
) in CONFIGS:

    df = pd.read_csv(
        folder(
            dataset,
            model,
            recent,
            long,
        )
        / "entity_metrics.csv",
        dtype={
            "entity_id": str,
        },
    )

    wide = df.pivot(
        index="entity_id",
        columns="intervention",
        values="mean_mae",
    )

    required = [
        "short",
        "same_entity",
        "matched_other_affine",
    ]

    missing = [
        x
        for x in required
        if x not in wide.columns
    ]

    if missing:
        raise RuntimeError(
            f"{dataset}/{model}: "
            f"missing {missing}"
        )

    short = (
        wide["short"]
        .astype(float)
    )

    same = (
        wide["same_entity"]
        .astype(float)
    )

    other = (
        wide[
            "matched_other_affine"
        ]
        .astype(float)
    )

    denominator = (
        short.abs()
        + EPS
    )

    out = pd.DataFrame(
        {
            "dataset": dataset,
            "entity_id": (
                wide.index.astype(str)
            ),
            "model": model,
            "short_mae": short.values,
            "entity_gain": (
                other - same
            ).values,
            "generic_gain": (
                short - other
            ).values,
            "total_gain": (
                short - same
            ).values,
            "entity_gain_pct": (
                100
                * (
                    other - same
                )
                / denominator
            ).values,
            "generic_gain_pct": (
                100
                * (
                    short - other
                )
                / denominator
            ).values,
            "total_gain_pct": (
                100
                * (
                    short - same
                )
                / denominator
            ).values,
        }
    )

    rows.append(
        out
    )


all_df = pd.concat(
    rows,
    ignore_index=True,
)


all_df.to_csv(
    OUT
    / "entity_gains_scale_normalized.csv",
    index=False,
)


print("=" * 100)
print(
    "CROSS-MODEL ENTITY-LEVEL AGREEMENT"
)
print(
    "Entity-normalized effects: "
    "% of each entity's short-context MAE"
)
print("=" * 100)


agreement_rows = []


for dataset in LABELS:

    part = all_df[
        all_df[
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

    print()
    print(
        LABELS[
            dataset
        ]
    )

    for effect in [
        "entity_gain_pct",
        "generic_gain_pct",
        "total_gain_pct",
    ]:

        rho, p = spearmanr(
            joined[
                f"{effect}_c"
            ],
            joined[
                f"{effect}_t"
            ],
        )

        agreement_rows.append(
            {
                "dataset": dataset,
                "dataset_label": (
                    LABELS[
                        dataset
                    ]
                ),
                "effect": effect,
                "n": len(joined),
                "rho": rho,
                "p": p,
            }
        )

        print(
            f"  {effect:<20} "
            f"rho={rho:7.3f} "
            f"p={p:.5g} "
            f"N={len(joined)}"
        )


agreement = pd.DataFrame(
    agreement_rows
)

agreement.to_csv(
    OUT
    / "scale_normalized_cross_model_agreement.csv",
    index=False,
)


# ============================================================
# Consensus normalized phenotype
# ============================================================

def zscore(x):
    sd = x.std(
        ddof=1
    )

    return (
        x - x.mean()
    ) / (
        sd + EPS
    )


all_df[
    "entity_gain_pct_z"
] = (
    all_df.groupby(
        [
            "dataset",
            "model",
        ]
    )[
        "entity_gain_pct"
    ]
    .transform(
        zscore
    )
)


wide = (
    all_df.pivot(
        index=[
            "dataset",
            "entity_id",
        ],
        columns="model",
        values="entity_gain_pct_z",
    )
    .reset_index()
)


wide[
    "normalized_consensus_sensitivity"
] = (
    (
        wide[
            "chronos2"
        ]
        + wide[
            "timesfm3"
        ]
    )
    / 2
)


wide.to_csv(
    OUT
    / "scale_normalized_consensus_phenotype.csv",
    index=False,
)


# ============================================================
# Re-test signal scale
# ============================================================

sig = pd.read_csv(
    (
        "results/context_regime_comparison/"
        "domain_signatures/"
        "entity_statistical_signatures.csv"
    ),
    dtype={
        "entity_id": str,
    },
)


merged = wide.merge(
    sig[
        [
            "dataset",
            "entity_id",
            "mean_scale",
            "within_level_sd",
            "median_lag1",
            "median_innovation_ratio",
        ]
    ],
    on=[
        "dataset",
        "entity_id",
    ],
    how="left",
    validate="one_to_one",
)


print()
print("=" * 100)
print(
    "SIGNAL SCALE vs NORMALIZED CONSENSUS PHENOTYPE"
)
print("=" * 100)


scale_rows = []


for dataset in LABELS:

    part = merged[
        merged[
            "dataset"
        ] == dataset
    ]

    rho, p = spearmanr(
        part[
            "mean_scale"
        ],
        part[
            "normalized_consensus_sensitivity"
        ],
        nan_policy="omit",
    )

    scale_rows.append(
        {
            "dataset": dataset,
            "dataset_label": (
                LABELS[
                    dataset
                ]
            ),
            "n": len(part),
            "rho": rho,
            "p": p,
        }
    )

    print(
        f"{LABELS[dataset]:<18} "
        f"rho={rho:7.3f} "
        f"p={p:.5g}"
    )


scale_df = pd.DataFrame(
    scale_rows
)

scale_df.to_csv(
    OUT
    / "scale_vs_normalized_phenotype.csv",
    index=False,
)


print()
print("Saved to:", OUT)
