import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_BOOTSTRAP = 50000
N_SIGNFLIP = 100000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "chronos2",
            "timesfm3",
        ],
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


def main():
    args = parse_args()

    folder = (
        Path("results")
        / args.dataset
        / "interventions"
        / args.model
        / (
            f"recent_{args.recent_hours:g}h"
            f"_long_{args.long_hours:g}h"
        )
    )

    input_path = (
        folder
        / "entity_metrics.csv"
    )

    output_path = (
        folder
        / "paired_statistics.csv"
    )

    df = pd.read_csv(
        input_path
    )

    wide = df.pivot(
        index="entity_id",
        columns="intervention",
        values="mean_mae",
    )

    comparisons = [
        {
            "name":
            "total_long_context_gain",
            "better":
            "same_entity",
            "worse":
            "short",
        },
        {
            "name":
            "entity_specific_gain",
            "better":
            "same_entity",
            "worse":
            "matched_other_affine",
        },
        {
            "name":
            "temporal_order_gain",
            "better":
            "same_entity",
            "worse":
            "same_entity_day_shuffle",
        },
        {
            "name":
            "generic_long_context_gain",
            "better":
            "matched_other_affine",
            "worse":
            "short",
        },
        {
            "name":
            "raw_vs_affine_donor",
            "better":
            "matched_other_affine",
            "worse":
            "matched_other_raw",
        },
        {
            "name":
            "identity_vs_temporal_order_control",
            "better":
            "matched_other_affine",
            "worse":
            "same_entity_day_shuffle",
        },
    ]

    rng = np.random.default_rng(
        SEED
    )

    rows = []

    print("=" * 88)
    print(
        f"{args.dataset} context intervention statistics"
    )
    print("=" * 88)

    print()
    print(
        f"Model: {args.model}"
    )

    print(
        f"Recent context: "
        f"{args.recent_hours:g} h"
    )

    print(
        f"Long context: "
        f"{args.long_hours:g} h"
    )

    for comparison in comparisons:

        better = comparison[
            "better"
        ]

        worse = comparison[
            "worse"
        ]

        delta = (
            wide[worse]
            - wide[better]
        ).dropna().to_numpy(
            dtype=float
        )

        n = len(delta)

        mean_delta = float(
            np.mean(delta)
        )

        median_delta = float(
            np.median(delta)
        )

        fraction_better = float(
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
                mean_delta / sd
            )
        else:
            cohen_dz = np.nan

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
                [
                    0.025,
                    0.975,
                ],
            )
        )

        extreme = 0
        completed = 0
        chunk_size = 5000

        while (
            completed
            < N_SIGNFLIP
        ):
            this_chunk = min(
                chunk_size,
                N_SIGNFLIP
                - completed,
            )

            signs = rng.choice(
                [
                    -1.0,
                    1.0,
                ],
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
                        mean_delta
                    )
                )
            )

            completed += (
                this_chunk
            )

        p_value = (
            extreme + 1
        ) / (
            N_SIGNFLIP + 1
        )

        rows.append(
            {
                "comparison":
                comparison["name"],
                "better_condition":
                better,
                "worse_condition":
                worse,
                "n_entities":
                n,
                "mean_mae_advantage":
                mean_delta,
                "median_mae_advantage":
                median_delta,
                "bootstrap_ci_low":
                float(ci_low),
                "bootstrap_ci_high":
                float(ci_high),
                "fraction_entities_better":
                fraction_better,
                "cohen_dz":
                float(cohen_dz),
                "signflip_p":
                float(p_value),
            }
        )

        print()
        print(
            comparison["name"]
        )
        print("-" * 88)

        print(
            f"{better} better than {worse}"
        )

        print(
            f"N entities:            {n}"
        )

        print(
            "Mean MAE advantage:    "
            f"{mean_delta:.4f}"
        )

        print(
            "Median MAE advantage:  "
            f"{median_delta:.4f}"
        )

        print(
            "95% bootstrap CI:      "
            f"[{ci_low:.4f}, "
            f"{ci_high:.4f}]"
        )

        print(
            "Fraction entities better: "
            f"{fraction_better:.3f}"
        )

        print(
            "Cohen dz:              "
            f"{cohen_dz:.3f}"
        )

        print(
            "Sign-flip p:           "
            f"{p_value:.6g}"
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 88)
    print(
        f"Saved: {output_path}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
