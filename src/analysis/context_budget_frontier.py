from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results")

SCORE_PATH = (
    ROOT
    / "context_regime_comparison"
    / "personalization_router_v1"
    / "cross_domain"
    / "router_v1_oof_scores.csv"
)

OUT = (
    ROOT
    / "context_regime_comparison"
    / "context_budget_frontier"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


REGIMES = {
    "pems_bay": (24, 168),
    "electricity": (96, 192),
    "hupa_ucm": (24, 96),
    "vitaldb": (0.5, 4),
    "solar": (24, 96),
    "physionet2019": (6, 48),
}

DOMAIN_LABELS = {
    "pems_bay": "PEMS-BAY",
    "electricity": "Electricity",
    "hupa_ucm": "HUPA-UCM",
    "vitaldb": "VitalDB",
    "solar": "Solar",
    "physionet2019": "PhysioNet 2019",
}

MODELS = [
    "chronos2",
    "timesfm3",
    "tirex2",
]

MODEL_LABELS = {
    "chronos2": "Chronos-2",
    "timesfm3": "TimesFM3",
    "tirex2": "TiRex-2",
}


# 0%, then every 5% to 100%.
BUDGETS = np.arange(
    0.0,
    1.0001,
    0.05,
)


def fmt(x):
    return f"{x:g}"


def intervention_root(
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
            f"recent_{fmt(recent)}h"
            f"_long_{fmt(long)}h"
        )
    )


def percentile_score(x):
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


def evaluate_selection(
    short,
    same,
    other,
    selected,
):
    """
    Deployment action:
      unselected -> short context
      selected   -> long same-target context
    """

    routed = short.copy()

    if len(selected) > 0:
        routed[selected] = (
            same[selected]
        )

    short_mean = float(
        np.mean(short)
    )

    routed_mean = float(
        np.mean(routed)
    )

    gain_pct = (
        100.0
        * (
            short_mean
            - routed_mean
        )
        / short_mean
    )

    if len(selected) > 0:

        long_precision = float(
            np.mean(
                same[selected]
                < short[selected]
            )
        )

        personal_precision = float(
            np.mean(
                same[selected]
                < other[selected]
            )
        )

        both_precision = float(
            np.mean(
                (
                    same[selected]
                    < short[selected]
                )
                &
                (
                    same[selected]
                    < other[selected]
                )
            )
        )

    else:
        long_precision = np.nan
        personal_precision = np.nan
        both_precision = np.nan

    return {
        "mae": routed_mean,
        "gain_pct_vs_short": (
            gain_pct
        ),
        "precision_long_helps": (
            long_precision
        ),
        "precision_personal_helps": (
            personal_precision
        ),
        "precision_both": (
            both_precision
        ),
    }


def top_k_exact(
    score,
    k,
):
    """
    Spend exactly the full budget.
    """

    if k <= 0:
        return np.array(
            [],
            dtype=int,
        )

    return np.argsort(
        -score
    )[:k]


def top_k_at_most(
    score,
    k,
    positive_mask,
):
    """
    Spend at most k selections.

    Never select cases that the policy itself predicts
    should not receive long context.
    """

    if k <= 0:
        return np.array(
            [],
            dtype=int,
        )

    order = np.argsort(
        -score
    )

    eligible = order[
        positive_mask[
            order
        ]
    ]

    return eligible[:k]


scores = pd.read_csv(
    SCORE_PATH
)

scores[
    "entity_id"
] = (
    scores[
        "entity_id"
    ].astype(str)
)


rows = []
setting_rows = []


for dataset, (
    recent,
    long,
) in REGIMES.items():

    for model in MODELS:

        print()
        print(
            "=" * 120
        )

        print(
            f"{DOMAIN_LABELS[dataset]} "
            f"| {MODEL_LABELS[model]}"
        )

        print(
            "=" * 120
        )

        metrics = pd.read_csv(
            intervention_root(
                dataset,
                model,
                recent,
                long,
            )
            / "case_metrics.csv"
        )

        metrics[
            "entity_id"
        ] = (
            metrics[
                "entity_id"
            ].astype(str)
        )

        metrics = (
            metrics[
                metrics[
                    "intervention"
                ]
                == "matched_other_affine"
            ]
            .copy()
            .reset_index(drop=True)
        )

        setting_scores = (
            scores[
                (
                    scores[
                        "dataset"
                    ]
                    == dataset
                )
                &
                (
                    scores[
                        "model"
                    ]
                    == model
                )
            ]
            .copy()
        )

        data = metrics.merge(
            setting_scores[
                [
                    "entity_id",
                    "eval_index",
                    "predicted_long_value",
                    "predicted_personal_value",
                ]
            ],
            on=[
                "entity_id",
                "eval_index",
            ],
            how="inner",
            validate="one_to_one",
        )

        if len(data) != len(
            metrics
        ):
            raise RuntimeError(
                f"{dataset}/{model}: "
                f"{len(data)}/"
                f"{len(metrics)} aligned."
            )

        short = data[
            "short_mae"
        ].to_numpy(
            dtype=float
        )

        same = data[
            "same_mae"
        ].to_numpy(
            dtype=float
        )

        other = data[
            "mae"
        ].to_numpy(
            dtype=float
        )

        r1 = data[
            "predicted_long_value"
        ].to_numpy(
            dtype=float
        )

        r2 = data[
            "predicted_personal_value"
        ].to_numpy(
            dtype=float
        )

        actual_long = (
            short - same
        )

        actual_personal = (
            other - same
        )

        n = len(data)

        short_mean = float(
            np.mean(short)
        )

        always_long_mean = float(
            np.mean(same)
        )

        always_long_gain = (
            100.0
            * (
                short_mean
                - always_long_mean
            )
            / short_mean
        )

        # --------------------------------------------
        # Existing hard-cascade score.
        # Used only as one comparator.
        # --------------------------------------------

        r1_rank = percentile_score(
            r1
        )

        r2_rank = percentile_score(
            r2
        )

        cascade_score = np.minimum(
            r1_rank,
            r2_rank,
        )

        policies = {
            "router1": {
                "score": r1,
                "positive": (
                    r1 > 0
                ),
            },

            "router2": {
                "score": r2,
                "positive": (
                    r2 > 0
                ),
            },

            "cascade": {
                "score": (
                    cascade_score
                ),
                "positive": (
                    (r1 > 0)
                    &
                    (r2 > 0)
                ),
            },

            "oracle": {
                "score": (
                    actual_long
                ),
                "positive": (
                    actual_long > 0
                ),
            },
        }

        setting_rows.append(
            {
                "dataset": dataset,
                "domain": (
                    DOMAIN_LABELS[
                        dataset
                    ]
                ),
                "model": model,
                "model_label": (
                    MODEL_LABELS[
                        model
                    ]
                ),
                "n_cases": n,
                "fraction_long_helps": (
                    np.mean(
                        actual_long > 0
                    )
                ),
                "fraction_personal_helps": (
                    np.mean(
                        actual_personal > 0
                    )
                ),
                "short_mae": (
                    short_mean
                ),
                "always_long_mae": (
                    always_long_mean
                ),
                "always_long_gain_pct": (
                    always_long_gain
                ),
            }
        )

        for budget in BUDGETS:

            k = int(
                round(
                    budget * n
                )
            )

            # ----------------------------------------
            # Random baseline
            #
            # With positive average long-context gain,
            # random has no reason to use less than
            # the allowed budget.
            # ----------------------------------------

            random_gain = (
                budget
                * always_long_gain
            )

            random_mae = (
                short_mean
                * (
                    1.0
                    - random_gain
                    / 100.0
                )
            )

            for mode in [
                "exact",
                "at_most",
            ]:

                rows.append(
                    {
                        "dataset": dataset,
                        "domain": (
                            DOMAIN_LABELS[
                                dataset
                            ]
                        ),
                        "model": model,
                        "model_label": (
                            MODEL_LABELS[
                                model
                            ]
                        ),
                        "budget_fraction": (
                            budget
                        ),
                        "mode": mode,
                        "policy": "random",
                        "n_selected": (
                            k
                        ),
                        "actual_fraction_used": (
                            k / n
                        ),
                        "mae": (
                            random_mae
                        ),
                        "gain_pct_vs_short": (
                            random_gain
                        ),
                        "precision_long_helps": (
                            np.mean(
                                actual_long
                                > 0
                            )
                        ),
                        "precision_personal_helps": (
                            np.mean(
                                actual_personal
                                > 0
                            )
                        ),
                        "precision_both": (
                            np.mean(
                                (
                                    actual_long
                                    > 0
                                )
                                &
                                (
                                    actual_personal
                                    > 0
                                )
                            )
                        ),
                    }
                )

            # ----------------------------------------
            # Learned policies + oracle
            # ----------------------------------------

            for policy, cfg in (
                policies.items()
            ):

                score = cfg[
                    "score"
                ]

                positive = cfg[
                    "positive"
                ]

                exact_idx = (
                    top_k_exact(
                        score,
                        k,
                    )
                )

                atmost_idx = (
                    top_k_at_most(
                        score,
                        k,
                        positive,
                    )
                )

                for mode, idx in [
                    (
                        "exact",
                        exact_idx,
                    ),
                    (
                        "at_most",
                        atmost_idx,
                    ),
                ]:

                    result = (
                        evaluate_selection(
                            short,
                            same,
                            other,
                            idx,
                        )
                    )

                    rows.append(
                        {
                            "dataset": (
                                dataset
                            ),
                            "domain": (
                                DOMAIN_LABELS[
                                    dataset
                                ]
                            ),
                            "model": (
                                model
                            ),
                            "model_label": (
                                MODEL_LABELS[
                                    model
                                ]
                            ),
                            "budget_fraction": (
                                budget
                            ),
                            "mode": mode,
                            "policy": (
                                policy
                            ),
                            "n_selected": (
                                len(idx)
                            ),
                            "actual_fraction_used": (
                                len(idx)
                                / n
                            ),
                            **result,
                        }
                    )


frontier = pd.DataFrame(
    rows
)

settings = pd.DataFrame(
    setting_rows
)


# ============================================================
# Oracle-normalized quantities
# ============================================================

oracle_lookup = (
    frontier[
        frontier[
            "policy"
        ]
        == "oracle"
    ][
        [
            "dataset",
            "model",
            "budget_fraction",
            "mode",
            "gain_pct_vs_short",
        ]
    ]
    .rename(
        columns={
            "gain_pct_vs_short":
                "oracle_gain_pct"
        }
    )
)

frontier = frontier.merge(
    oracle_lookup,
    on=[
        "dataset",
        "model",
        "budget_fraction",
        "mode",
    ],
    how="left",
    validate="many_to_one",
)

frontier[
    "oracle_regret_pct"
] = (
    frontier[
        "oracle_gain_pct"
    ]
    - frontier[
        "gain_pct_vs_short"
    ]
)

frontier[
    "oracle_capture"
] = np.where(
    frontier[
        "oracle_gain_pct"
    ] > 1e-12,
    frontier[
        "gain_pct_vs_short"
    ]
    / frontier[
        "oracle_gain_pct"
    ],
    np.nan,
)


# ============================================================
# Save raw curves
# ============================================================

frontier.to_csv(
    OUT
    / "context_budget_frontiers.csv",
    index=False,
)

settings.to_csv(
    OUT
    / "setting_baselines.csv",
    index=False,
)


# ============================================================
# Aggregate curve across 18 settings
# ============================================================

aggregate = (
    frontier.groupby(
        [
            "mode",
            "policy",
            "budget_fraction",
        ],
        as_index=False,
    )
    .agg(
        mean_gain_pct=(
            "gain_pct_vs_short",
            "mean",
        ),
        median_gain_pct=(
            "gain_pct_vs_short",
            "median",
        ),
        mean_oracle_capture=(
            "oracle_capture",
            "mean",
        ),
        median_oracle_capture=(
            "oracle_capture",
            "median",
        ),
        mean_oracle_regret_pct=(
            "oracle_regret_pct",
            "mean",
        ),
        mean_fraction_used=(
            "actual_fraction_used",
            "mean",
        ),
        mean_long_precision=(
            "precision_long_helps",
            "mean",
        ),
        mean_personal_precision=(
            "precision_personal_helps",
            "mean",
        ),
    )
)

aggregate.to_csv(
    OUT
    / "aggregate_budget_frontiers.csv",
    index=False,
)


# ============================================================
# Area under gain-budget curve
# ============================================================

auc_rows = []

for (
    dataset,
    model,
    mode,
    policy,
), group in frontier.groupby(
    [
        "dataset",
        "model",
        "mode",
        "policy",
    ]
):

    group = group.sort_values(
        "budget_fraction"
    )

    x = group[
        "budget_fraction"
    ].to_numpy(
        dtype=float
    )

    y = group[
        "gain_pct_vs_short"
    ].to_numpy(
        dtype=float
    )

    area = float(
        np.trapezoid(
            y,
            x,
        )
    )

    regret_area = float(
        np.trapezoid(
            group[
                "oracle_regret_pct"
            ].to_numpy(
                dtype=float
            ),
            x,
        )
    )

    auc_rows.append(
        {
            "dataset": dataset,
            "domain": (
                DOMAIN_LABELS[
                    dataset
                ]
            ),
            "model": model,
            "model_label": (
                MODEL_LABELS[
                    model
                ]
            ),
            "mode": mode,
            "policy": policy,
            "gain_budget_auc": (
                area
            ),
            "oracle_regret_auc": (
                regret_area
            ),
        }
    )


auc = pd.DataFrame(
    auc_rows
)


oracle_auc = (
    auc[
        auc[
            "policy"
        ]
        == "oracle"
    ][
        [
            "dataset",
            "model",
            "mode",
            "gain_budget_auc",
        ]
    ]
    .rename(
        columns={
            "gain_budget_auc":
                "oracle_gain_budget_auc"
        }
    )
)


auc = auc.merge(
    oracle_auc,
    on=[
        "dataset",
        "model",
        "mode",
    ],
    how="left",
    validate="many_to_one",
)

auc[
    "auc_oracle_capture"
] = np.where(
    auc[
        "oracle_gain_budget_auc"
    ] > 1e-12,
    auc[
        "gain_budget_auc"
    ]
    / auc[
        "oracle_gain_budget_auc"
    ],
    np.nan,
)

auc.to_csv(
    OUT
    / "setting_frontier_auc.csv",
    index=False,
)


global_auc = (
    auc.groupby(
        [
            "mode",
            "policy",
        ],
        as_index=False,
    )
    .agg(
        mean_gain_budget_auc=(
            "gain_budget_auc",
            "mean",
        ),
        median_gain_budget_auc=(
            "gain_budget_auc",
            "median",
        ),
        mean_auc_oracle_capture=(
            "auc_oracle_capture",
            "mean",
        ),
        median_auc_oracle_capture=(
            "auc_oracle_capture",
            "median",
        ),
        mean_oracle_regret_auc=(
            "oracle_regret_auc",
            "mean",
        ),
    )
)

global_auc.to_csv(
    OUT
    / "global_frontier_auc.csv",
    index=False,
)


# ============================================================
# Best diagnostic budget for each policy/setting.
#
# IMPORTANT:
# Post-hoc diagnostic only, not a valid tuned deployment budget.
# ============================================================

best_rows = []

for (
    dataset,
    model,
    mode,
    policy,
), group in frontier[
    frontier[
        "policy"
    ].isin(
        [
            "router1",
            "router2",
            "cascade",
            "oracle",
        ]
    )
].groupby(
    [
        "dataset",
        "model",
        "mode",
        "policy",
    ]
):

    best = group.loc[
        group[
            "gain_pct_vs_short"
        ].idxmax()
    ]

    best_rows.append(
        {
            "dataset": dataset,
            "domain": (
                DOMAIN_LABELS[
                    dataset
                ]
            ),
            "model": model,
            "model_label": (
                MODEL_LABELS[
                    model
                ]
            ),
            "mode": mode,
            "policy": policy,
            "best_allowed_budget": (
                best[
                    "budget_fraction"
                ]
            ),
            "fraction_actually_used": (
                best[
                    "actual_fraction_used"
                ]
            ),
            "best_gain_pct": (
                best[
                    "gain_pct_vs_short"
                ]
            ),
            "oracle_gain_at_budget_pct": (
                best[
                    "oracle_gain_pct"
                ]
            ),
            "oracle_capture": (
                best[
                    "oracle_capture"
                ]
            ),
        }
    )


best = pd.DataFrame(
    best_rows
)

best.to_csv(
    OUT
    / "diagnostic_best_budgets.csv",
    index=False,
)


# ============================================================
# Does at-most-budget improve over forced exact-budget?
# ============================================================

exact = (
    frontier[
        frontier[
            "mode"
        ]
        == "exact"
    ][
        [
            "dataset",
            "model",
            "policy",
            "budget_fraction",
            "gain_pct_vs_short",
        ]
    ]
    .rename(
        columns={
            "gain_pct_vs_short":
                "exact_gain"
        }
    )
)

atmost = (
    frontier[
        frontier[
            "mode"
        ]
        == "at_most"
    ][
        [
            "dataset",
            "model",
            "policy",
            "budget_fraction",
            "gain_pct_vs_short",
            "actual_fraction_used",
        ]
    ]
    .rename(
        columns={
            "gain_pct_vs_short":
                "atmost_gain"
        }
    )
)

comparison = exact.merge(
    atmost,
    on=[
        "dataset",
        "model",
        "policy",
        "budget_fraction",
    ],
    validate="one_to_one",
)

comparison[
    "atmost_minus_exact_gain"
] = (
    comparison[
        "atmost_gain"
    ]
    - comparison[
        "exact_gain"
    ]
)

comparison.to_csv(
    OUT
    / "atmost_vs_exact.csv",
    index=False,
)


# ============================================================
# Selected operating points for terminal output
# ============================================================

print()
print()
print("=" * 145)
print(
    "MEAN GAIN ACROSS 18 SETTINGS"
)
print("=" * 145)

for budget in [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]:

    print()
    print(
        f"ALLOWED BUDGET = "
        f"{100*budget:.0f}%"
    )

    part = aggregate[
        np.isclose(
            aggregate[
                "budget_fraction"
            ],
            budget,
        )
        &
        (
            aggregate[
                "mode"
            ]
            == "at_most"
        )
    ][
        [
            "policy",
            "mean_gain_pct",
            "mean_oracle_capture",
            "mean_fraction_used",
        ]
    ].copy()

    part[
        "mean_gain_pct"
    ] = part[
        "mean_gain_pct"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    part[
        "mean_oracle_capture"
    ] = part[
        "mean_oracle_capture"
    ].map(
        lambda x: (
            "nan"
            if pd.isna(x)
            else f"{100*x:.1f}%"
        )
    )

    part[
        "mean_fraction_used"
    ] = part[
        "mean_fraction_used"
    ].map(
        lambda x: f"{100*x:.1f}%"
    )

    print(
        part.to_string(
            index=False
        )
    )


print()
print("=" * 145)
print(
    "GLOBAL AREA UNDER GAIN-BUDGET FRONTIER"
)
print("=" * 145)

pretty_auc = (
    global_auc[
        global_auc[
            "mode"
        ]
        == "at_most"
    ]
    .copy()
)

pretty_auc[
    "mean_gain_budget_auc"
] = pretty_auc[
    "mean_gain_budget_auc"
].map(
    lambda x: f"{x:.3f}"
)

pretty_auc[
    "mean_auc_oracle_capture"
] = pretty_auc[
    "mean_auc_oracle_capture"
].map(
    lambda x: (
        "nan"
        if pd.isna(x)
        else f"{100*x:.1f}%"
    )
)

pretty_auc[
    "mean_oracle_regret_auc"
] = pretty_auc[
    "mean_oracle_regret_auc"
].map(
    lambda x: f"{x:.3f}"
)

print(
    pretty_auc[
        [
            "policy",
            "mean_gain_budget_auc",
            "mean_auc_oracle_capture",
            "mean_oracle_regret_auc",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Router 1: at-most vs exact
# ============================================================

r1cmp = comparison[
    comparison[
        "policy"
    ]
    == "router1"
].copy()

print()
print("=" * 145)
print(
    "ROUTER 1: AT-MOST VS EXACT"
)
print("=" * 145)

for budget in [
    0.25,
    0.50,
    0.75,
    1.00,
]:

    p = r1cmp[
        np.isclose(
            r1cmp[
                "budget_fraction"
            ],
            budget,
        )
    ]

    print(
        f"Budget {100*budget:3.0f}% | "
        f"mean delta gain = "
        f"{p['atmost_minus_exact_gain'].mean():+.2f} pp | "
        f"at-most better in "
        f"{int((p['atmost_minus_exact_gain'] > 1e-12).sum())}/18 | "
        f"mean actually used = "
        f"{100*p['actual_fraction_used'].mean():.1f}%"
    )


# ============================================================
# How often at-most Router 1 beats always-long?
# ============================================================

r1 = frontier[
    (
        frontier[
            "policy"
        ]
        == "router1"
    )
    &
    (
        frontier[
            "mode"
        ]
        == "at_most"
    )
].merge(
    settings[
        [
            "dataset",
            "model",
            "always_long_gain_pct",
        ]
    ],
    on=[
        "dataset",
        "model",
    ],
    how="left",
    validate="many_to_one",
)

print()
print("=" * 145)
print(
    "AT-MOST ROUTER 1 VS ALWAYS-LONG"
)
print("=" * 145)

for budget in [
    0.25,
    0.50,
    0.75,
    1.00,
]:

    p = r1[
        np.isclose(
            r1[
                "budget_fraction"
            ],
            budget,
        )
    ]

    wins = (
        p[
            "gain_pct_vs_short"
        ]
        >
        p[
            "always_long_gain_pct"
        ]
    )

    print(
        f"Allowed budget "
        f"{100*budget:3.0f}% | "
        f"beats always-long: "
        f"{int(wins.sum())}/18 | "
        f"mean actual usage: "
        f"{100*p['actual_fraction_used'].mean():.1f}%"
    )


print()
print(
    "NOTE: diagnostic best budgets are post-hoc "
    "and must not be reported as tuned test performance."
)

print(
    "Saved to:",
    OUT,
)
