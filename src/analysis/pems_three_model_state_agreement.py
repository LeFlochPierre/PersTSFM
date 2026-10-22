from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    rankdata,
    spearmanr,
)


ROOT = Path("results")

OUT = (
    ROOT
    / "context_regime_comparison"
    / "pems_three_model_state_agreement"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = [
    "chronos2",
    "timesfm3",
    "tirex2",
]


LABELS = {
    "chronos2": "Chronos-2",
    "timesfm3": "TimesFM3",
    "tirex2": "TiRex-2",
}


PATHS = {
    model: (
        ROOT
        / "pems_bay"
        / "interventions"
        / model
        / "recent_24h_long_168h"
        / "case_metrics.csv"
    )
    for model in MODELS
}


N_PERM = 10000
SEED = 42
EPS = 1e-12


def load_model_cases(
    model,
):
    df = pd.read_csv(
        PATHS[model],
        dtype={
            "entity_id": str,
            "case_id": str,
        },
    )

    df = (
        df[
            df["intervention"]
            == "matched_other_affine"
        ]
        .copy()
        .reset_index(drop=True)
    )

    df["entity_gain_raw"] = (
        df["mae"]
        - df["same_mae"]
    )

    # Fixed denominator within entity.
    df["entity_short_mean"] = (
        df.groupby("entity_id")[
            "short_mae"
        ]
        .transform("mean")
    )

    df["entity_gain_pct"] = (
        100.0
        * df["entity_gain_raw"]
        / (
            df["entity_short_mean"]
            + EPS
        )
    )

    # Remove stable entity mean.
    df["gain_within_entity"] = (
        df["entity_gain_pct"]
        - df.groupby("entity_id")[
            "entity_gain_pct"
        ]
        .transform("mean")
    )

    # Relative origin-specific difficulty.
    df["difficulty_relative"] = (
        df["short_mae"]
        / (
            df["entity_short_mean"]
            + EPS
        )
    )

    df["difficulty_within_entity"] = (
        df["difficulty_relative"]
        - df.groupby("entity_id")[
            "difficulty_relative"
        ]
        .transform("mean")
    )

    return df


def residualize_rank(
    y,
    x,
):
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


def standardize(
    x,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    sd = np.std(
        x,
        ddof=1,
    )

    if sd < EPS:
        return np.zeros_like(
            x
        )

    return (
        x - np.mean(x)
    ) / sd


def within_entity_permutation(
    df,
    col_a,
    col_b,
    n_perm=N_PERM,
    seed=SEED,
):
    rng = np.random.default_rng(
        seed
    )

    a = df[col_a].to_numpy(
        dtype=float
    )

    b = df[col_b].to_numpy(
        dtype=float
    )

    observed = float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )

    groups = [
        np.asarray(
            idx,
            dtype=int,
        )
        for idx
        in df.groupby(
            "entity_id"
        ).indices.values()
    ]

    null = np.empty(
        n_perm,
        dtype=float,
    )

    for p in range(
        n_perm
    ):
        b_perm = b.copy()

        for idx in groups:
            b_perm[idx] = (
                b[
                    rng.permutation(
                        idx
                    )
                ]
            )

        null[p] = float(
            np.corrcoef(
                a,
                b_perm,
            )[0, 1]
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
        float(np.mean(null)),
        float(
            np.std(
                null,
                ddof=1,
            )
        ),
    )


tables = {
    model: load_model_cases(
        model
    )
    for model in MODELS
}


# ============================================================
# Merge exact evaluation origins
# ============================================================

key_cols = [
    "entity_id",
    "eval_index",
]

merged = None


for model in MODELS:

    df = tables[model][
        key_cols
        + [
            "entity_gain_pct",
            "gain_within_entity",
            "difficulty_within_entity",
            "short_mae",
        ]
    ].copy()

    rename = {
        col: f"{col}_{model}"
        for col in df.columns
        if col not in key_cols
    }

    df = df.rename(
        columns=rename
    )

    if merged is None:
        merged = df
    else:
        merged = merged.merge(
            df,
            on=key_cols,
            how="inner",
            validate="one_to_one",
        )


print("=" * 112)
print("PEMS-BAY THREE-MODEL COMMON-CONTEXT STATE AGREEMENT")
print("Common intervention regime: recent 24 h -> long 168 h")
print("=" * 112)

print(
    "Exact shared origins:",
    len(merged),
)

print(
    "Shared entities:",
    merged[
        "entity_id"
    ].nunique(),
)


# ============================================================
# Difficulty-adjusted state scores
# ============================================================

for model in MODELS:

    resid = residualize_rank(
        merged[
            f"gain_within_entity_{model}"
        ],
        merged[
            f"difficulty_within_entity_{model}"
        ],
    )

    merged[
        f"state_resid_{model}"
    ] = resid

    # Important:
    # remove any residual entity structure again
    # after rank residualization.
    merged[
        f"state_resid_{model}"
    ] = (
        merged[
            f"state_resid_{model}"
        ]
        - merged.groupby(
            "entity_id"
        )[
            f"state_resid_{model}"
        ]
        .transform("mean")
    )

    merged[
        f"state_z_{model}"
    ] = standardize(
        merged[
            f"state_resid_{model}"
        ]
    )


# ============================================================
# Pairwise agreement
# ============================================================

pairs = [
    (
        "chronos2",
        "timesfm3",
    ),
    (
        "chronos2",
        "tirex2",
    ),
    (
        "timesfm3",
        "tirex2",
    ),
]


pair_rows = []


print()
print("=" * 112)
print("PAIRWISE CROSS-MODEL AGREEMENT")
print("=" * 112)


for a, b in pairs:

    # Exact-origin raw normalized gain.
    rho_case, p_case = (
        spearmanr(
            merged[
                f"entity_gain_pct_{a}"
            ],
            merged[
                f"entity_gain_pct_{b}"
            ],
        )
    )

    # Between-entity means.
    entity = (
        merged.groupby(
            "entity_id",
            as_index=False,
        )[
            [
                f"entity_gain_pct_{a}",
                f"entity_gain_pct_{b}",
            ]
        ]
        .mean()
    )

    rho_between, p_between = (
        spearmanr(
            entity[
                f"entity_gain_pct_{a}"
            ],
            entity[
                f"entity_gain_pct_{b}"
            ],
        )
    )

    # Within-entity exact-state agreement.
    rho_within, p_within = (
        spearmanr(
            merged[
                f"gain_within_entity_{a}"
            ],
            merged[
                f"gain_within_entity_{b}"
            ],
        )
    )

    (
        rho_adjusted,
        p_perm,
        null_mean,
        null_sd,
    ) = within_entity_permutation(
        merged,
        f"state_resid_{a}",
        f"state_resid_{b}",
    )

    pair_rows.append(
        {
            "model_a": a,
            "model_b": b,
            "n_cases": len(merged),
            "n_entities": (
                merged[
                    "entity_id"
                ].nunique()
            ),
            "raw_exact_origin_rho": (
                float(rho_case)
            ),
            "raw_exact_origin_p": (
                float(p_case)
            ),
            "between_entity_rho": (
                float(rho_between)
            ),
            "between_entity_p": (
                float(p_between)
            ),
            "within_entity_state_rho": (
                float(rho_within)
            ),
            "within_entity_state_p_naive": (
                float(p_within)
            ),
            "difficulty_adjusted_state_rho": (
                rho_adjusted
            ),
            "permutation_p": (
                p_perm
            ),
            "permutation_null_mean": (
                null_mean
            ),
            "permutation_null_sd": (
                null_sd
            ),
        }
    )

    print()
    print(
        f"{LABELS[a]} vs {LABELS[b]}"
    )

    print(
        f"  Exact-origin normalized gain:      "
        f"rho={rho_case:7.3f} "
        f"p={p_case:.3g}"
    )

    print(
        f"  Between-entity agreement:          "
        f"rho={rho_between:7.3f} "
        f"p={p_between:.3g}"
    )

    print(
        f"  Within-entity state agreement:     "
        f"rho={rho_within:7.3f}"
    )

    print(
        f"  Difficulty-adjusted state:         "
        f"rho={rho_adjusted:7.3f}"
    )

    if (
        p_perm
        <= 1.0
        / N_PERM
    ):
        p_text = (
            f"< {1.0 / N_PERM:.1e}"
        )
    else:
        p_text = (
            f"{p_perm:.5g}"
        )

    print(
        f"  Within-entity permutation p:       "
        f"{p_text}"
    )

    print(
        f"  Permutation null:                  "
        f"{null_mean:7.3f} "
        f"± {null_sd:.3f}"
    )


pair_df = pd.DataFrame(
    pair_rows
)

pair_df.to_csv(
    OUT
    / "pairwise_three_model_state_agreement.csv",
    index=False,
)


# ============================================================
# Three-model consensus
# ============================================================

merged[
    "three_model_consensus_state"
] = (
    merged[
        [
            "state_z_chronos2",
            "state_z_timesfm3",
            "state_z_tirex2",
        ]
    ]
    .mean(
        axis=1
    )
)


print()
print("=" * 112)
print("LEAVE-ONE-MODEL-OUT CONSENSUS")
print("=" * 112)


loo_rows = []


for model in MODELS:

    others = [
        x
        for x in MODELS
        if x != model
    ]

    other_consensus = (
        merged[
            [
                f"state_z_{others[0]}",
                f"state_z_{others[1]}",
            ]
        ]
        .mean(
            axis=1
        )
    )

    rho, p = spearmanr(
        merged[
            f"state_z_{model}"
        ],
        other_consensus,
    )

    loo_rows.append(
        {
            "model": model,
            "rho_vs_other_two_consensus": (
                float(rho)
            ),
            "p_naive": (
                float(p)
            ),
        }
    )

    print(
        f"{LABELS[model]:<10} vs other-two consensus: "
        f"rho={rho:7.3f} "
        f"p={p:.3g}"
    )


pd.DataFrame(
    loo_rows
).to_csv(
    OUT
    / "leave_one_model_out_consensus.csv",
    index=False,
)


# ============================================================
# Rank the most / least context-sensitive states
# ============================================================

ranked = (
    merged[
        [
            "entity_id",
            "eval_index",
            "three_model_consensus_state",
            "state_z_chronos2",
            "state_z_timesfm3",
            "state_z_tirex2",
            "entity_gain_pct_chronos2",
            "entity_gain_pct_timesfm3",
            "entity_gain_pct_tirex2",
        ]
    ]
    .sort_values(
        "three_model_consensus_state",
        ascending=False,
    )
    .reset_index(drop=True)
)


ranked.to_csv(
    OUT
    / "three_model_consensus_states.csv",
    index=False,
)


print()
print("=" * 112)
print("TOP 10 CONSENSUS CONTEXT-SENSITIVE STATES")
print("=" * 112)

print(
    ranked.head(10).to_string(
        index=False
    )
)


print()
print("=" * 112)
print("BOTTOM 10 CONSENSUS CONTEXT-SENSITIVE STATES")
print("=" * 112)

print(
    ranked.tail(10).to_string(
        index=False
    )
)


print()
print("=" * 112)
print("SAVED")
print("=" * 112)

print(OUT)
