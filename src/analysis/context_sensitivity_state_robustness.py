from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


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


N_PERM = 10000
SEED = 42
EPS = 1e-12


def folder(dataset, model):

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


def load_cases(dataset, model):

    df = pd.read_csv(
        folder(dataset, model)
        / "case_metrics.csv",
        dtype={
            "entity_id": str,
            "donor_id": str,
        },
    )

    df = (
        df[
            df["intervention"]
            == "matched_other_affine"
        ]
        .copy()
    )

    # Raw per-origin entity / episode gain.
    df["entity_gain_raw"] = (
        df["mae"]
        - df["same_mae"]
    )

    # FIXED denominator within entity/model.
    # This is crucial: it does not change across origins.
    df["entity_short_mean"] = (
        df.groupby("entity_id")[
            "short_mae"
        ]
        .transform("mean")
    )

    df["entity_gain_fixed_pct"] = (
        100.0
        * df["entity_gain_raw"]
        / (
            df["entity_short_mean"]
            + EPS
        )
    )

    # Relative short-context difficulty at this origin.
    df["short_difficulty_relative"] = (
        df["short_mae"]
        / (
            df["entity_short_mean"]
            + EPS
        )
    )

    # Remove stable entity mean.
    df["gain_within_entity"] = (
        df["entity_gain_fixed_pct"]
        - df.groupby("entity_id")[
            "entity_gain_fixed_pct"
        ]
        .transform("mean")
    )

    df["difficulty_within_entity"] = (
        df["short_difficulty_relative"]
        - df.groupby("entity_id")[
            "short_difficulty_relative"
        ]
        .transform("mean")
    )

    return df


def icc1(df, value_col):

    work = (
        df[
            [
                "entity_id",
                value_col,
            ]
        ]
        .dropna()
        .copy()
    )

    groups = work.groupby(
        "entity_id"
    )[value_col]

    sizes = groups.size()
    means = groups.mean()

    k = len(sizes)
    n = len(work)

    grand = work[
        value_col
    ].mean()

    ss_between = float(
        np.sum(
            sizes
            * (
                means - grand
            ) ** 2
        )
    )

    ss_within = 0.0

    for entity_id, part in (
        work.groupby("entity_id")
    ):

        m = part[
            value_col
        ].mean()

        ss_within += float(
            np.sum(
                (
                    part[value_col]
                    - m
                ) ** 2
            )
        )

    ms_between = (
        ss_between
        / (k - 1)
    )

    ms_within = (
        ss_within
        / (n - k)
    )

    k_eff = (
        (
            n
            - (
                np.sum(
                    sizes.to_numpy() ** 2
                )
                / n
            )
        )
        / (k - 1)
    )

    icc = (
        (
            ms_between
            - ms_within
        )
        / (
            ms_between
            + (
                k_eff - 1
            )
            * ms_within
            + EPS
        )
    )

    return {
        "n_entities": k,
        "n_cases": n,
        "origins_mean": float(
            sizes.mean()
        ),
        "ms_between": (
            ms_between
        ),
        "ms_within": (
            ms_within
        ),
        "icc": float(
            icc
        ),
    }


def rank_residual(y, x):
    """
    Partial-rank residual:
    rank(y) residualized against rank(x).
    """

    y = np.asarray(
        y,
        dtype=float,
    )

    x = np.asarray(
        x,
        dtype=float,
    )

    yr = rankdata(
        y,
        method="average",
    )

    xr = rankdata(
        x,
        method="average",
    )

    X = np.column_stack(
        [
            np.ones(len(xr)),
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


def corr(a, b):

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def within_entity_permutation_test(
    df,
    c_col,
    t_col,
    n_perm=N_PERM,
    seed=SEED,
):

    rng = np.random.default_rng(
        seed
    )

    c = df[c_col].to_numpy(
        dtype=float
    )

    t = df[t_col].to_numpy(
        dtype=float
    )

    observed = corr(
        c,
        t,
    )

    groups = [
        idx.to_numpy()
        for _, idx
        in (
            df.groupby(
                "entity_id"
            ).groups.items()
        )
    ]

    null = np.empty(
        n_perm,
        dtype=float,
    )

    for p in range(
        n_perm
    ):

        t_perm = t.copy()

        for idx in groups:
            t_perm[idx] = (
                t[
                    rng.permutation(
                        idx
                    )
                ]
            )

        null[p] = corr(
            c,
            t_perm,
        )

    p_two = (
        1
        + np.sum(
            np.abs(null)
            >= abs(observed)
        )
    ) / (
        n_perm + 1
    )

    return (
        observed,
        float(p_two),
        null,
    )


tables = {}

icc_rows = []
agreement_rows = []


print("=" * 108)
print(
    "ROBUST ENTITY / STATE CONTEXT-SENSITIVITY ANALYSIS"
)
print(
    "Per-origin gains normalized by each entity's "
    "fixed mean short-context MAE"
)
print("=" * 108)


# ============================================================
# Load + robust ICC
# ============================================================

for dataset, cfg in (
    CONFIGS.items()
):

    print()
    print("=" * 108)
    print(
        cfg["label"]
    )
    print("=" * 108)

    for model in MODELS:

        df = load_cases(
            dataset,
            model,
        )

        tables[
            (
                dataset,
                model,
            )
        ] = df

        result = icc1(
            df,
            "entity_gain_fixed_pct",
        )

        result.update(
            {
                "dataset": dataset,
                "dataset_label": (
                    cfg["label"]
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
            f"ICC={result['icc']:7.3f}  "
            f"MS_between={result['ms_between']:10.3f}  "
            f"MS_within={result['ms_within']:10.3f}  "
            f"N={result['n_entities']} / "
            f"{result['n_cases']}"
        )


icc_df = pd.DataFrame(
    icc_rows
)

icc_df.to_csv(
    OUT
    / "robust_fixed_denominator_icc.csv",
    index=False,
)


# ============================================================
# Cross-model exact-state analysis
# ============================================================

print()
print("=" * 108)
print(
    "CROSS-MODEL STATE AGREEMENT"
)
print(
    "Includes within-entity difficulty adjustment "
    "and within-entity permutation inference"
)
print("=" * 108)


for dataset, cfg in (
    CONFIGS.items()
):

    c = (
        tables[
            (
                dataset,
                "chronos2",
            )
        ][
            [
                "entity_id",
                "eval_index",
                "entity_gain_fixed_pct",
                "gain_within_entity",
                "difficulty_within_entity",
            ]
        ]
        .copy()
    )

    t = (
        tables[
            (
                dataset,
                "timesfm3",
            )
        ][
            [
                "entity_id",
                "eval_index",
                "entity_gain_fixed_pct",
                "gain_within_entity",
                "difficulty_within_entity",
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

    # ---------------------------------------------
    # Between-entity agreement
    # ---------------------------------------------

    entity = (
        joined.groupby(
            "entity_id",
            as_index=False,
        )[
            [
                "entity_gain_fixed_pct_chronos",
                "entity_gain_fixed_pct_timesfm",
            ]
        ]
        .mean()
    )

    rho_between, p_between = (
        spearmanr(
            entity[
                "entity_gain_fixed_pct_chronos"
            ],
            entity[
                "entity_gain_fixed_pct_timesfm"
            ],
        )
    )

    # ---------------------------------------------
    # Within-entity state agreement before
    # difficulty adjustment.
    # ---------------------------------------------

    rho_within, _ = spearmanr(
        joined[
            "gain_within_entity_chronos"
        ],
        joined[
            "gain_within_entity_timesfm"
        ],
    )

    # ---------------------------------------------
    # Remove each model's own origin-specific
    # short-context difficulty.
    # ---------------------------------------------

    joined[
        "chronos_state_resid"
    ] = rank_residual(
        joined[
            "gain_within_entity_chronos"
        ],
        joined[
            "difficulty_within_entity_chronos"
        ],
    )

    joined[
        "timesfm_state_resid"
    ] = rank_residual(
        joined[
            "gain_within_entity_timesfm"
        ],
        joined[
            "difficulty_within_entity_timesfm"
        ],
    )

    (
        rho_adjusted,
        p_perm,
        null,
    ) = within_entity_permutation_test(
        joined,
        "chronos_state_resid",
        "timesfm_state_resid",
    )

    agreement_rows.append(
        {
            "dataset": dataset,
            "dataset_label": (
                cfg["label"]
            ),
            "n_entities": (
                len(entity)
            ),
            "n_cases": (
                len(joined)
            ),
            "between_entity_rho": (
                float(
                    rho_between
                )
            ),
            "between_entity_p": (
                float(
                    p_between
                )
            ),
            "within_state_rho": (
                float(
                    rho_within
                )
            ),
            "difficulty_adjusted_state_rho": (
                rho_adjusted
            ),
            "within_entity_permutation_p": (
                p_perm
            ),
            "null_rho_mean": (
                float(
                    np.mean(
                        null
                    )
                )
            ),
            "null_rho_sd": (
                float(
                    np.std(
                        null,
                        ddof=1,
                    )
                )
            ),
        }
    )

    print()
    print(
        cfg["label"]
    )

    print(
        f"  Between entities:                  "
        f"rho={rho_between:7.3f} "
        f"p={p_between:.4g}"
    )

    print(
        f"  Within entity, exact state:        "
        f"rho={rho_within:7.3f}"
    )

    print(
        f"  After state-difficulty adjustment: "
        f"rho={rho_adjusted:7.3f}"
    )

    print(
        f"  Within-entity permutation p:       "
        f"{p_perm:.5g}"
    )

    print(
        f"  Permutation null:                  "
        f"{np.mean(null):7.3f} "
        f"± {np.std(null, ddof=1):.3f}"
    )


agreement = pd.DataFrame(
    agreement_rows
)

agreement.to_csv(
    OUT
    / "robust_cross_model_state_agreement.csv",
    index=False,
)


print()
print("=" * 108)
print("SAVED")
print("=" * 108)

print(
    OUT
)
