from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "results/pems_bay/information_acquisition/"
    "sensor_mechanism_curves.csv"
)

OUTPUT = Path(
    "results/pems_bay/information_acquisition/"
    "acquisition_threshold_robustness.csv"
)

N_BOOTSTRAP = 20000
SEED = 42

MODELS = [
    "chronos2",
    "timesfm3",
]

METRICS = [
    "generic_gain",
    "identity_gap",
]

THRESHOLDS = [
    0.50,
    0.90,
]


def acquisition_context(
    curve,
    context_days,
    fraction,
):
    curve = np.asarray(
        curve,
        dtype=float,
    )

    maximum = float(
        np.max(curve)
    )

    if maximum <= 0:
        return np.nan

    target = (
        fraction
        * maximum
    )

    for value, day in zip(
        curve,
        context_days,
    ):
        if value >= target:
            return float(day)

    return float(
        context_days[-1]
    )


df = pd.read_csv(
    INPUT
)

rng = np.random.default_rng(
    SEED
)

rows = []


print("=" * 100)
print(
    "PEMS-BAY nonparametric "
    "acquisition-scale robustness"
)
print("=" * 100)


for model in MODELS:

    model_df = df[
        df["model"] == model
    ].copy()

    entities = sorted(
        model_df[
            "entity_id"
        ].unique()
    )

    contexts = np.array(
        sorted(
            model_df[
                "context_days"
            ].unique()
        ),
        dtype=float,
    )

    n_entities = len(
        entities
    )

    matrices = {}

    for metric in METRICS:

        matrix = (
            model_df
            .pivot(
                index="entity_id",
                columns="context_days",
                values=metric,
            )
            .reindex(
                index=entities,
                columns=contexts,
            )
        )

        if matrix.isna().any().any():
            raise RuntimeError(
                f"{model} / {metric}: "
                "missing values."
            )

        matrices[
            metric
        ] = matrix.to_numpy(
            dtype=float
        )

    bootstrap_indices = (
        rng.integers(
            0,
            n_entities,
            size=(
                N_BOOTSTRAP,
                n_entities,
            ),
        )
    )

    print()
    print("-" * 100)
    print(f"Model: {model}")
    print("-" * 100)

    for fraction in THRESHOLDS:

        observed = {}
        bootstrap_values = {}

        for metric in METRICS:

            matrix = matrices[
                metric
            ]

            cohort_curve = (
                matrix.mean(
                    axis=0
                )
            )

            observed_context = (
                acquisition_context(
                    cohort_curve,
                    contexts,
                    fraction,
                )
            )

            boot_contexts = np.empty(
                N_BOOTSTRAP,
                dtype=float,
            )

            for b in range(
                N_BOOTSTRAP
            ):
                sample_curve = (
                    matrix[
                        bootstrap_indices[b],
                        :
                    ]
                    .mean(
                        axis=0
                    )
                )

                boot_contexts[
                    b
                ] = acquisition_context(
                    sample_curve,
                    contexts,
                    fraction,
                )

            observed[
                metric
            ] = observed_context

            bootstrap_values[
                metric
            ] = boot_contexts

            low, high = np.quantile(
                boot_contexts,
                [
                    0.025,
                    0.975,
                ],
            )

            rows.append(
                {
                    "model": model,
                    "fraction": fraction,
                    "mechanism": metric,
                    "observed_context_days": (
                        observed_context
                    ),
                    "bootstrap_ci_low": (
                        float(low)
                    ),
                    "bootstrap_ci_high": (
                        float(high)
                    ),
                }
            )

            print()
            print(
                f"{metric} | "
                f"{int(100*fraction)}%"
            )

            print(
                "  observed context: "
                f"{observed_context:.1f} d"
            )

            print(
                "  bootstrap 95% range: "
                f"[{low:.1f}, {high:.1f}] d"
            )

        generic = (
            bootstrap_values[
                "generic_gain"
            ]
        )

        identity = (
            bootstrap_values[
                "identity_gap"
            ]
        )

        probability = float(
            np.mean(
                identity
                > generic
            )
        )

        equal_probability = float(
            np.mean(
                identity
                == generic
            )
        )

        difference = (
            identity
            - generic
        )

        low, high = np.quantile(
            difference,
            [
                0.025,
                0.975,
            ],
        )

        print()
        print(
            f"{int(100*fraction)}% "
            "staged-acquisition comparison"
        )

        print(
            "  P(identity context "
            "> generic context): "
            f"{probability:.4f}"
        )

        print(
            "  P(equal): "
            f"{equal_probability:.4f}"
        )

        print(
            "  difference 95% range: "
            f"[{low:.1f}, {high:.1f}] d"
        )


result = pd.DataFrame(
    rows
)

result.to_csv(
    OUTPUT,
    index=False,
)

print()
print("=" * 100)
print(f"Saved: {OUTPUT}")
print("=" * 100)
