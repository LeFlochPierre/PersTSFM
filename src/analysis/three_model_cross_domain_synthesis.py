from pathlib import Path

import numpy as np
import pandas as pd


ROOT = (
    Path("results")
    / "context_regime_comparison"
    / "three_model_state_agreement"
)

OUT = (
    Path("results")
    / "context_regime_comparison"
    / "three_model_cross_domain_synthesis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DOMAINS = [
    ("pems_bay", "PEMS-BAY"),
    ("electricity", "Electricity"),
    ("hupa_ucm", "HUPA-UCM"),
    ("vitaldb", "VitalDB"),
    ("solar", "Solar"),
    ("physionet2019", "PhysioNet 2019"),
]


PAIR_LABELS = {
    ("chronos2", "timesfm3"):
        "Chronos-2 ↔ TimesFM3",

    ("chronos2", "tirex2"):
        "Chronos-2 ↔ TiRex-2",

    ("timesfm3", "tirex2"):
        "TimesFM3 ↔ TiRex-2",
}


rows = []

for dataset, label in DOMAINS:

    path = (
        ROOT
        / dataset
        / "pairwise_agreement.csv"
    )

    df = pd.read_csv(path)

    for _, row in df.iterrows():

        pair = (
            row["model_a"],
            row["model_b"],
        )

        rows.append(
            {
                "dataset": dataset,
                "domain": label,
                "model_pair": (
                    PAIR_LABELS[pair]
                ),
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "raw_exact_origin_rho": (
                    row.get(
                        "raw_rho",
                        np.nan,
                    )
                ),
                "between_entity_rho": (
                    row[
                        "between_entity_rho"
                    ]
                ),
                "within_entity_state_rho": (
                    row[
                        "within_state_rho"
                    ]
                ),
                "difficulty_adjusted_state_rho": (
                    row[
                        "adjusted_state_rho"
                    ]
                ),
                "permutation_p": (
                    row[
                        "permutation_p"
                    ]
                ),
                "null_mean": (
                    row["null_mean"]
                ),
                "null_sd": (
                    row["null_sd"]
                ),
            }
        )


all_df = pd.DataFrame(rows)

all_df.to_csv(
    OUT
    / "cross_domain_pairwise_agreement.csv",
    index=False,
)


print("=" * 105)
print("THREE-MODEL CROSS-DOMAIN STATE AGREEMENT")
print("=" * 105)

pivot = all_df.pivot(
    index="domain",
    columns="model_pair",
    values="difficulty_adjusted_state_rho",
)

print()
print(
    pivot.round(3).to_string()
)


values = all_df[
    "difficulty_adjusted_state_rho"
].to_numpy(
    dtype=float
)


print()
print("=" * 105)
print("GLOBAL DESCRIPTIVE SUMMARY")
print("=" * 105)

print(
    "Comparisons:",
    len(values),
)

print(
    "Positive:",
    int(
        np.sum(values > 0)
    ),
    "/",
    len(values),
)

print(
    "Permutation p < 0.05:",
    int(
        np.sum(
            all_df[
                "permutation_p"
            ] < 0.05
        )
    ),
    "/",
    len(values),
)

print(
    f"Mean adjusted rho:   "
    f"{np.mean(values):.3f}"
)

print(
    f"Median adjusted rho: "
    f"{np.median(values):.3f}"
)

print(
    f"Range:               "
    f"{np.min(values):.3f} "
    f"to {np.max(values):.3f}"
)


print()
print("=" * 105)
print("BY ARCHITECTURE PAIR")
print("=" * 105)


pair_summary = (
    all_df
    .groupby(
        "model_pair"
    )[
        "difficulty_adjusted_state_rho"
    ]
    .agg(
        [
            "mean",
            "median",
            "min",
            "max",
        ]
    )
    .reset_index()
)

print(
    pair_summary
    .round(3)
    .to_string(
        index=False
    )
)

pair_summary.to_csv(
    OUT
    / "architecture_pair_summary.csv",
    index=False,
)


print()
print("=" * 105)
print("PERMUTATION INFERENCE")
print("=" * 105)

for _, row in all_df.iterrows():

    p = float(
        row["permutation_p"]
    )

    if p <= 1 / 10001:
        p_text = "<1e-4"
    else:
        p_text = f"{p:.4g}"

    print(
        f"{row['domain']:<16} "
        f"{row['model_pair']:<24} "
        f"rho="
        f"{row['difficulty_adjusted_state_rho']:.3f} "
        f"p={p_text}"
    )


print()
print("Saved to:", OUT)
