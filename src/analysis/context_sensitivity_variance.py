from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("results")

OUT = (
    ROOT
    / "context_regime_comparison"
    / "context_sensitivity_variance"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
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
    cfg = CONFIGS[dataset]

    return (
        ROOT
        / dataset
        / "interventions"
        / model
        / (
            f"recent_{cfg['recent']:g}h"
            f"_long_{cfg['long'][model]:g}h"
        )
    )


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
        int(good.sum()),
    )


def icc_oneway(
    df,
    entity_col,
    value_col,
):
    """
    One-way random-effects ICC(1).

    Works with balanced or approximately balanced groups.
    Uses the standard unequal-n effective group-size
    correction.
    """

    work = (
        df[
            [
                entity_col,
                value_col,
            ]
        ]
        .dropna()
        .copy()
    )

    grouped = work.groupby(
        entity_col
    )[value_col]

    sizes = grouped.size()

    means = grouped.mean()

    n_groups = len(
        sizes
    )

    n_total = len(
        work
    )

    grand = work[
        value_col
    ].mean()

    ss_between = float(
        np.sum(
            sizes
            * (
                means - grand
            )
            ** 2
        )
    )

    ss_within = 0.0

    for entity, part in (
        work.groupby(
            entity_col
        )
    ):
        m = part[
            value_col
        ].mean()

        ss_within += float(
            np.sum(
                (
                    part[
                        value_col
                    ]
                    - m
                )
                ** 2
            )
        )

    df_between = (
        n_groups - 1
    )

    df_within = (
        n_total - n_groups
    )

    ms_between = (
        ss_between
        / df_between
    )

    ms_within = (
        ss_within
        / df_within
    )

    k_effective = (
        (
            n_total
            - (
                np.sum(
                    sizes.to_numpy() ** 2
                )
                / n_total
            )
        )
        / df_between
    )

    sigma_entity = max(
        (
            ms_between
            - ms_within
        )
        / (
            k_effective
            + EPS
        ),
        0.0,
    )

    sigma_within = max(
        ms_within,
        0.0,
    )

    icc = (
        sigma_entity
        / (
            sigma_entity
            + sigma_within
            + EPS
        )
    )

    return {
        "n_entities": (
            n_groups
        ),
        "n_cases": (
            n_total
        ),
        "origins_mean": float(
            sizes.mean()
        ),
        "sigma_entity": (
            sigma_entity
        ),
        "sigma_within": (
            sigma_within
        ),
        "icc": (
            icc
        ),
    }


def load_cases(
    dataset,
    model,
):
    df = pd.read_csv(
        folder(
            dataset,
            model,
        )
        / "case_metrics.csv",
        dtype={
            "entity_id": str,
            "donor_id": str,
        },
    )

    df = (
        df[
            df[
                "intervention"
            ]
            == "matched_other_affine"
        ]
        .copy()
    )

    df[
        "entity_gain"
    ] = (
        df[
            "mae"
        ]
        - df[
            "same_mae"
        ]
    )

    df[
        "entity_gain_pct"
    ] = (
        100
        * df[
            "entity_gain"
        ]
        / (
            df[
                "short_mae"
            ].abs()
            + EPS
        )
    )

    df[
        "generic_gain_pct"
    ] = (
        100
        * (
            df[
                "short_mae"
            ]
            - df[
                "mae"
            ]
        )
        / (
            df[
                "short_mae"
            ].abs()
            + EPS
        )
    )

    df[
        "total_gain_pct"
    ] = (
        100
        * (
            df[
                "short_mae"
            ]
            - df[
                "same_mae"
            ]
        )
        / (
            df[
                "short_mae"
            ].abs()
            + EPS
        )
    )

    return df


all_cases = []
icc_rows = []
case_tables = {}


print("=" * 104)
print(
    "ENTITY vs STATE DECOMPOSITION OF CONTEXT SENSITIVITY"
)
print("=" * 104)


for dataset, cfg in (
    CONFIGS.items()
):

    print()
    print("=" * 104)
    print(
        cfg[
            "label"
        ]
    )
    print("=" * 104)

    for model in MODELS:

        df = load_cases(
            dataset,
            model,
        )

        df[
            "dataset"
        ] = dataset

        df[
            "model"
        ] = model

        all_cases.append(
            df
        )

        case_tables[
            (
                dataset,
                model,
            )
        ] = df

        result = icc_oneway(
            df,
            "entity_id",
            "entity_gain_pct",
        )

        result.update(
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
            }
        )

        icc_rows.append(
            result
        )

        print(
            f"{MODEL_LABELS[model]:<10} "
            f"ICC={result['icc']:.3f}  "
            f"between_var={result['sigma_entity']:.3f}  "
            f"within_var={result['sigma_within']:.3f}  "
            f"N={result['n_entities']} entities / "
            f"{result['n_cases']} origins"
        )


icc_df = pd.DataFrame(
    icc_rows
)

icc_df.to_csv(
    OUT
    / "context_sensitivity_icc.csv",
    index=False,
)


cases = pd.concat(
    all_cases,
    ignore_index=True,
)

cases.to_csv(
    OUT
    / "case_level_normalized_gains.csv",
    index=False,
)


# ============================================================
# Cross-model agreement at exact evaluation origins
# ============================================================

agreement_rows = []


print()
print("=" * 104)
print(
    "CROSS-MODEL AGREEMENT AT EXACT EVALUATION ORIGINS"
)
print("=" * 104)


for dataset, cfg in (
    CONFIGS.items()
):

    c = (
        case_tables[
            (
                dataset,
                "chronos2",
            )
        ][
            [
                "entity_id",
                "eval_index",
                "entity_gain_pct",
                "short_mae",
            ]
        ]
        .copy()
    )

    t = (
        case_tables[
            (
                dataset,
                "timesfm3",
            )
        ][
            [
                "entity_id",
                "eval_index",
                "entity_gain_pct",
                "short_mae",
            ]
        ]
        .copy()
    )

    joined = c.merge(
        t,
        on=[
            "entity_id",
            "eval_index",
        ],
        suffixes=(
            "_chronos",
            "_timesfm",
        ),
        validate="one_to_one",
    )

    rho_case, p_case, n = spearman(
        joined[
            "entity_gain_pct_chronos"
        ],
        joined[
            "entity_gain_pct_timesfm"
        ],
    )

    # Remove stable entity mean separately in each model.
    joined[
        "chronos_within"
    ] = (
        joined[
            "entity_gain_pct_chronos"
        ]
        - joined.groupby(
            "entity_id"
        )[
            "entity_gain_pct_chronos"
        ]
        .transform(
            "mean"
        )
    )

    joined[
        "timesfm_within"
    ] = (
        joined[
            "entity_gain_pct_timesfm"
        ]
        - joined.groupby(
            "entity_id"
        )[
            "entity_gain_pct_timesfm"
        ]
        .transform(
            "mean"
        )
    )

    rho_within, p_within, _ = spearman(
        joined[
            "chronos_within"
        ],
        joined[
            "timesfm_within"
        ],
    )

    # Entity means only.
    entity = (
        joined.groupby(
            "entity_id",
            as_index=False,
        )[
            [
                "entity_gain_pct_chronos",
                "entity_gain_pct_timesfm",
            ]
        ]
        .mean()
    )

    rho_between, p_between, n_entities = spearman(
        entity[
            "entity_gain_pct_chronos"
        ],
        entity[
            "entity_gain_pct_timesfm"
        ],
    )

    agreement_rows.append(
        {
            "dataset": dataset,
            "dataset_label": (
                cfg[
                    "label"
                ]
            ),
            "n_cases": n,
            "n_entities": (
                n_entities
            ),
            "rho_case": (
                rho_case
            ),
            "p_case": (
                p_case
            ),
            "rho_between_entity": (
                rho_between
            ),
            "p_between_entity": (
                p_between
            ),
            "rho_within_entity_state": (
                rho_within
            ),
            "p_within_entity_state": (
                p_within
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
        f"  Exact-origin agreement:     "
        f"rho={rho_case:7.3f} "
        f"p={p_case:.4g} "
        f"N={n}"
    )

    print(
        f"  Between-entity agreement:   "
        f"rho={rho_between:7.3f} "
        f"p={p_between:.4g} "
        f"N={n_entities}"
    )

    print(
        f"  Within-entity state agreement: "
        f"rho={rho_within:7.3f} "
        f"p={p_within:.4g}"
    )


agreement = pd.DataFrame(
    agreement_rows
)

agreement.to_csv(
    OUT
    / "cross_model_entity_state_agreement.csv",
    index=False,
)


print()
print("=" * 104)
print(
    "INTERPRETATION GUIDE"
)
print("=" * 104)

print(
    "High ICC              -> sensitivity is stable across "
    "origins within an entity."
)

print(
    "High between rho      -> models agree on which entities "
    "benefit from history."
)

print(
    "High within-state rho -> models also agree on WHEN, within "
    "the same entity, history is useful."
)

print()
print(
    "Saved:",
    OUT
)
