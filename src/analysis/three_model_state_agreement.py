import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    rankdata,
    spearmanr,
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

EPS = 1e-12
N_PERM = 10000
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--recent-hours",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--long-hours",
        required=True,
        type=float,
    )

    return parser.parse_args()


def fmt(x):
    return f"{x:g}"


def load_cases(
    dataset,
    model,
    recent,
    long,
):
    path = (
        Path("results")
        / dataset
        / "interventions"
        / model
        / (
            f"recent_{fmt(recent)}h"
            f"_long_{fmt(long)}h"
        )
        / "case_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
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

    df["gain_within_entity"] = (
        df["entity_gain_pct"]
        - df.groupby("entity_id")[
            "entity_gain_pct"
        ]
        .transform("mean")
    )

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


def standardize(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    sd = np.std(
        x,
        ddof=1,
    )

    if sd < EPS:
        return np.zeros_like(x)

    return (
        x - np.mean(x)
    ) / sd


def permutation_test(
    df,
    col_a,
    col_b,
):
    rng = np.random.default_rng(
        SEED
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
        for idx in df.groupby(
            "entity_id"
        ).indices.values()
    ]

    null = np.empty(
        N_PERM,
        dtype=float,
    )

    for p in range(N_PERM):

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
        N_PERM + 1
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


def main():

    args = parse_args()

    out = (
        Path("results")
        / "context_regime_comparison"
        / "three_model_state_agreement"
        / args.dataset
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables = {
        model: load_cases(
            args.dataset,
            model,
            args.recent_hours,
            args.long_hours,
        )
        for model in MODELS
    }

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
            ]
        ].copy()

        df = df.rename(
            columns={
                col: f"{col}_{model}"
                for col in df.columns
                if col not in key_cols
            }
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

    print("=" * 100)
    print(
        f"THREE-MODEL STATE AGREEMENT: "
        f"{args.dataset}"
    )
    print(
        f"Common regime: "
        f"{fmt(args.recent_hours)} h "
        f"-> {fmt(args.long_hours)} h"
    )
    print("=" * 100)

    print(
        "Shared origins:",
        len(merged),
    )

    print(
        "Shared entities:",
        merged[
            "entity_id"
        ].nunique(),
    )

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

        # Remove residual entity structure again.
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

    pairs = [
        ("chronos2", "timesfm3"),
        ("chronos2", "tirex2"),
        ("timesfm3", "tirex2"),
    ]

    rows = []

    print()
    print("=" * 100)
    print("PAIRWISE AGREEMENT")
    print("=" * 100)

    for a, b in pairs:

        rho_raw, p_raw = spearmanr(
            merged[
                f"entity_gain_pct_{a}"
            ],
            merged[
                f"entity_gain_pct_{b}"
            ],
        )

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

        rho_within, _ = spearmanr(
            merged[
                f"gain_within_entity_{a}"
            ],
            merged[
                f"gain_within_entity_{b}"
            ],
        )

        (
            rho_adjusted,
            p_perm,
            null_mean,
            null_sd,
        ) = permutation_test(
            merged,
            f"state_resid_{a}",
            f"state_resid_{b}",
        )

        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "raw_rho": rho_raw,
                "raw_p": p_raw,
                "between_entity_rho": (
                    rho_between
                ),
                "between_entity_p": (
                    p_between
                ),
                "within_state_rho": (
                    rho_within
                ),
                "adjusted_state_rho": (
                    rho_adjusted
                ),
                "permutation_p": (
                    p_perm
                ),
                "null_mean": (
                    null_mean
                ),
                "null_sd": (
                    null_sd
                ),
            }
        )

        print()
        print(
            f"{LABELS[a]} vs "
            f"{LABELS[b]}"
        )

        print(
            f"  Exact-origin gain:        "
            f"rho={rho_raw:7.3f}"
        )

        print(
            f"  Between entities:         "
            f"rho={rho_between:7.3f}"
        )

        print(
            f"  Within-entity state:      "
            f"rho={rho_within:7.3f}"
        )

        print(
            f"  Difficulty-adjusted:      "
            f"rho={rho_adjusted:7.3f}"
        )

        print(
            f"  Permutation p:            "
            f"{p_perm:.5g}"
        )

        print(
            f"  Null:                     "
            f"{null_mean:7.3f} "
            f"± {null_sd:.3f}"
        )

    pd.DataFrame(
        rows
    ).to_csv(
        out
        / "pairwise_agreement.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("LEAVE-ONE-MODEL-OUT CONSENSUS")
    print("=" * 100)

    loo_rows = []

    for model in MODELS:

        others = [
            x
            for x in MODELS
            if x != model
        ]

        consensus = (
            merged[
                [
                    f"state_z_{others[0]}",
                    f"state_z_{others[1]}",
                ]
            ]
            .mean(axis=1)
        )

        rho, p = spearmanr(
            merged[
                f"state_z_{model}"
            ],
            consensus,
        )

        loo_rows.append(
            {
                "model": model,
                "rho": rho,
                "p": p,
            }
        )

        print(
            f"{LABELS[model]:<10} "
            f"vs other-two consensus: "
            f"rho={rho:7.3f} "
            f"p={p:.3g}"
        )

    pd.DataFrame(
        loo_rows
    ).to_csv(
        out
        / "leave_one_out_consensus.csv",
        index=False,
    )

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
        .mean(axis=1)
    )

    merged.to_csv(
        out
        / "state_scores.csv",
        index=False,
    )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
