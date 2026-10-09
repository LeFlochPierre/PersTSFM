from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_BOOTSTRAP = 50000
N_SIGNFLIP = 100000

INPUT = Path(
    "results/pems_bay/interventions/"
    "chronos2/recent_24h_long_240h/"
    "entity_metrics.csv"
)

OUT = Path(
    "results/pems_bay/interventions/"
    "chronos2/recent_24h_long_240h/"
    "paired_statistics.csv"
)


df = pd.read_csv(INPUT)

wide = df.pivot(
    index="entity_id",
    columns="intervention",
    values="mean_mae",
)

required = [
    "short",
    "same_entity",
    "matched_other_affine",
    "matched_other_raw",
    "same_entity_day_shuffle",
]

missing = [
    x for x in required
    if x not in wide.columns
]

if missing:
    raise RuntimeError(
        f"Missing interventions: {missing}"
    )


comparisons = [
    {
        "name": "total_long_context_gain",
        "better": "same_entity",
        "worse": "short",
    },
    {
        "name": "entity_specific_gain",
        "better": "same_entity",
        "worse": "matched_other_affine",
    },
    {
        "name": "temporal_order_gain",
        "better": "same_entity",
        "worse": "same_entity_day_shuffle",
    },
    {
        "name": "generic_long_context_gain",
        "better": "matched_other_affine",
        "worse": "short",
    },
    {
        "name": "raw_vs_affine_donor",
        "better": "matched_other_affine",
        "worse": "matched_other_raw",
    },
]


rng = np.random.default_rng(SEED)

rows = []

print("=" * 88)
print("PEMS-BAY Chronos-2 context intervention statistics")
print("=" * 88)


for comparison in comparisons:

    better = comparison["better"]
    worse = comparison["worse"]

    # Positive = first condition is better.
    delta = (
        wide[worse]
        - wide[better]
    ).dropna().to_numpy(
        dtype=float
    )

    n = len(delta)

    observed = float(
        np.mean(delta)
    )

    median = float(
        np.median(delta)
    )

    fraction_positive = float(
        np.mean(delta > 0)
    )

    sd = float(
        np.std(
            delta,
            ddof=1,
        )
    )

    if sd > 0:
        cohen_dz = (
            observed / sd
        )
    else:
        cohen_dz = np.nan

    # ---------------------------------------------------------
    # Entity-level paired bootstrap
    # ---------------------------------------------------------

    boot_means = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for b in range(
        N_BOOTSTRAP
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot_means[b] = (
            delta[idx].mean()
        )

    ci_low, ci_high = (
        np.quantile(
            boot_means,
            [0.025, 0.975],
        )
    )

    # ---------------------------------------------------------
    # Paired sign-flip randomization test
    # H0: paired differences are symmetric around zero.
    # ---------------------------------------------------------

    extreme = 0

    chunk_size = 5000

    completed = 0

    while completed < N_SIGNFLIP:

        this_chunk = min(
            chunk_size,
            N_SIGNFLIP - completed,
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(
                this_chunk,
                n,
            ),
        )

        null_means = (
            signs
            * delta[
                None,
                :
            ]
        ).mean(
            axis=1
        )

        extreme += int(
            np.sum(
                np.abs(
                    null_means
                )
                >= abs(
                    observed
                )
            )
        )

        completed += (
            this_chunk
        )

    p_signflip = (
        extreme + 1
    ) / (
        N_SIGNFLIP + 1
    )

    row = {
        "comparison": (
            comparison[
                "name"
            ]
        ),
        "better_condition": better,
        "worse_condition": worse,
        "n_entities": n,
        "mean_mae_advantage": (
            observed
        ),
        "median_mae_advantage": (
            median
        ),
        "bootstrap_ci_low": float(
            ci_low
        ),
        "bootstrap_ci_high": float(
            ci_high
        ),
        "fraction_entities_better": (
            fraction_positive
        ),
        "cohen_dz": float(
            cohen_dz
        ),
        "signflip_p": float(
            p_signflip
        ),
    }

    rows.append(row)

    print()
    print(
        comparison["name"]
    )
    print("-" * 88)

    print(
        f"{better} better than {worse}"
    )

    print(
        f"N sensors:             {n}"
    )

    print(
        "Mean MAE advantage:    "
        f"{observed:.4f}"
    )

    print(
        "Median MAE advantage:  "
        f"{median:.4f}"
    )

    print(
        "95% bootstrap CI:      "
        f"[{ci_low:.4f}, "
        f"{ci_high:.4f}]"
    )

    print(
        "Fraction sensors better: "
        f"{fraction_positive:.3f}"
    )

    print(
        "Cohen dz:              "
        f"{cohen_dz:.3f}"
    )

    print(
        "Sign-flip p:           "
        f"{p_signflip:.6g}"
    )


result = pd.DataFrame(
    rows
)

result.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 88)
print(f"Saved: {OUT}")
print("=" * 88)
