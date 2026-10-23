from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results")

OUT = (
    ROOT
    / "context_regime_comparison"
    / "oracle_personalization_routing"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Frozen common regimes
# ------------------------------------------------------------

REGIMES = {
    "pems_bay": (24, 168),
    "electricity": (96, 192),
    "hupa_ucm": (24, 96),
    "vitaldb": (0.5, 4),
    "solar": (24, 96),
    "physionet2019": (6, 48),
}

LABELS = {
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


def fmt(x):
    return f"{x:g}"


def load_cases(
    dataset,
    model,
    recent,
    long,
):
    path = (
        ROOT
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

    df = pd.read_csv(path)

    # One row per forecasting origin containing:
    # - mae       = matched-other-affine MAE
    # - same_mae  = same-target long-context MAE
    # - short_mae = short-context MAE
    df = (
        df[
            df["intervention"]
            == "matched_other_affine"
        ]
        .copy()
        .reset_index(drop=True)
    )

    required = {
        "mae",
        "same_mae",
        "short_mae",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing columns "
            f"{sorted(missing)}"
        )

    return df


summary_rows = []
budget_rows = []


for dataset, (
    recent,
    long,
) in REGIMES.items():

    for model in MODELS:

        df = load_cases(
            dataset,
            model,
            recent,
            long,
        )

        short = (
            df["short_mae"]
            .to_numpy(
                dtype=float
            )
        )

        same = (
            df["same_mae"]
            .to_numpy(
                dtype=float
            )
        )

        other = (
            df["mae"]
            .to_numpy(
                dtype=float
            )
        )

        # ====================================================
        # DEPLOYMENT-RELEVANT QUESTION
        #
        # Should we retrieve/use the target's long history,
        # or is short context enough?
        # ====================================================

        long_value = (
            short - same
        )

        use_long = (
            same < short
        )

        oracle = np.minimum(
            short,
            same,
        )

        short_mean = float(
            np.mean(short)
        )

        same_mean = float(
            np.mean(same)
        )

        oracle_mean = float(
            np.mean(oracle)
        )

        frac_long = float(
            np.mean(use_long)
        )

        always_long_gain = (
            short_mean
            - same_mean
        )

        oracle_gain = (
            short_mean
            - oracle_mean
        )

        oracle_advantage_vs_long = (
            same_mean
            - oracle_mean
        )

        # ====================================================
        # DIAGNOSTIC PERSONALIZATION QUESTION
        #
        # Given long history, how often is the correct target
        # history better than matched history from another
        # entity?
        #
        # matched-other is a counterfactual diagnostic, not
        # intended as a deployment action.
        # ====================================================

        entity_value = (
            other - same
        )

        frac_target_specific = float(
            np.mean(
                same < other
            )
        )

        mean_entity_value = float(
            np.mean(
                entity_value
            )
        )

        summary_rows.append(
            {
                "dataset": dataset,
                "domain": (
                    LABELS[dataset]
                ),
                "model": model,
                "model_label": (
                    MODEL_LABELS[model]
                ),
                "n_cases": len(df),
                "short_mae": (
                    short_mean
                ),
                "always_long_mae": (
                    same_mean
                ),
                "oracle_mae": (
                    oracle_mean
                ),
                "always_long_gain": (
                    always_long_gain
                ),
                "oracle_gain": (
                    oracle_gain
                ),
                "always_long_gain_pct": (
                    100.0
                    * always_long_gain
                    / short_mean
                ),
                "oracle_gain_pct": (
                    100.0
                    * oracle_gain
                    / short_mean
                ),
                "oracle_advantage_vs_always_long": (
                    oracle_advantage_vs_long
                ),
                "oracle_advantage_vs_always_long_pct": (
                    100.0
                    * oracle_advantage_vs_long
                    / short_mean
                ),
                "oracle_fraction_long": (
                    frac_long
                ),
                "oracle_fraction_short": (
                    1.0 - frac_long
                ),
                "mean_target_specific_gain": (
                    mean_entity_value
                ),
                "fraction_correct_target_history_better": (
                    frac_target_specific
                ),
            }
        )

        # ====================================================
        # ORACLE BUDGET CURVE
        #
        # Suppose only fraction q of forecasting states may
        # retrieve long target-specific history.
        #
        # The oracle assigns that budget to the states with
        # largest realized long-context benefit.
        # ====================================================

        order = np.argsort(
            -long_value
        )

        n = len(df)

        budgets = np.linspace(
            0.0,
            1.0,
            21,
        )

        for budget in budgets:

            k = int(
                round(
                    budget * n
                )
            )

            routed = short.copy()

            if k > 0:
                chosen = order[:k]
                routed[chosen] = (
                    same[chosen]
                )

            routed_mean = float(
                np.mean(routed)
            )

            budget_rows.append(
                {
                    "dataset": (
                        dataset
                    ),
                    "domain": (
                        LABELS[dataset]
                    ),
                    "model": model,
                    "model_label": (
                        MODEL_LABELS[
                            model
                        ]
                    ),
                    "budget_fraction": (
                        float(
                            budget
                        )
                    ),
                    "n_long_context": (
                        k
                    ),
                    "n_cases": (
                        n
                    ),
                    "routed_mae": (
                        routed_mean
                    ),
                    "relative_mae_vs_short": (
                        routed_mean
                        / short_mean
                    ),
                    "gain_pct_vs_short": (
                        100.0
                        * (
                            short_mean
                            - routed_mean
                        )
                        / short_mean
                    ),
                }
            )


summary = pd.DataFrame(
    summary_rows
)

budget = pd.DataFrame(
    budget_rows
)


summary.to_csv(
    OUT / "oracle_summary.csv",
    index=False,
)

budget.to_csv(
    OUT / "oracle_budget_curve.csv",
    index=False,
)


# ------------------------------------------------------------
# Cross-setting budget summary
# Normalize first because domains have different units.
# ------------------------------------------------------------

aggregate_budget = (
    budget
    .groupby(
        "budget_fraction",
        as_index=False,
    )
    .agg(
        mean_relative_mae=(
            "relative_mae_vs_short",
            "mean",
        ),
        median_relative_mae=(
            "relative_mae_vs_short",
            "median",
        ),
        mean_gain_pct=(
            "gain_pct_vs_short",
            "mean",
        ),
    )
)

aggregate_budget.to_csv(
    OUT
    / "aggregate_budget_curve.csv",
    index=False,
)


# ------------------------------------------------------------
# Terminal report
# ------------------------------------------------------------

print("=" * 120)
print("ORACLE STATE-ADAPTIVE PERSONALIZATION")
print("=" * 120)

display_cols = [
    "domain",
    "model_label",
    "short_mae",
    "always_long_mae",
    "oracle_mae",
    "always_long_gain_pct",
    "oracle_gain_pct",
    "oracle_advantage_vs_always_long_pct",
    "oracle_fraction_long",
]

pretty = (
    summary[
        display_cols
    ]
    .copy()
)

for col in [
    "short_mae",
    "always_long_mae",
    "oracle_mae",
]:
    pretty[col] = (
        pretty[col]
        .map(
            lambda x: f"{x:.4f}"
        )
    )

for col in [
    "always_long_gain_pct",
    "oracle_gain_pct",
    "oracle_advantage_vs_always_long_pct",
]:
    pretty[col] = (
        pretty[col]
        .map(
            lambda x: f"{x:.2f}%"
        )
    )

pretty[
    "oracle_fraction_long"
] = (
    pretty[
        "oracle_fraction_long"
    ]
    .map(
        lambda x: f"{100*x:.1f}%"
    )
)

print()
print(
    pretty.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("CROSS-SETTING SUMMARY")
print("=" * 120)

print(
    "Mean always-long gain vs short: "
    f"{summary['always_long_gain_pct'].mean():.2f}%"
)

print(
    "Mean oracle gain vs short:      "
    f"{summary['oracle_gain_pct'].mean():.2f}%"
)

print(
    "Mean oracle advantage over "
    "always-long:                   "
    f"{summary['oracle_advantage_vs_always_long_pct'].mean():.2f}%"
)

print(
    "Mean fraction of states using "
    "long target history:           "
    f"{100*summary['oracle_fraction_long'].mean():.1f}%"
)

print(
    "Median fraction using long "
    "target history:                "
    f"{100*summary['oracle_fraction_long'].median():.1f}%"
)


print()
print("=" * 120)
print("TARGET-SPECIFIC HISTORY DIAGNOSTIC")
print("=" * 120)

diagnostic = summary[
    [
        "domain",
        "model_label",
        "mean_target_specific_gain",
        "fraction_correct_target_history_better",
    ]
].copy()

diagnostic[
    "fraction_correct_target_history_better"
] = (
    diagnostic[
        "fraction_correct_target_history_better"
    ]
    .map(
        lambda x: f"{100*x:.1f}%"
    )
)

print(
    diagnostic.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("ORACLE CONTEXT-BUDGET FRONTIER")
print("=" * 120)

selected_budgets = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

for q in selected_budgets:

    part = aggregate_budget[
        np.isclose(
            aggregate_budget[
                "budget_fraction"
            ],
            q,
        )
    ]

    # Budget grid is 0.05, so 0.25 etc exist.
    row = part.iloc[0]

    print(
        f"Long-context budget "
        f"{100*q:5.0f}% -> "
        f"mean gain vs short "
        f"{row['mean_gain_pct']:7.2f}%"
    )


print()
print("Saved to:", OUT)
