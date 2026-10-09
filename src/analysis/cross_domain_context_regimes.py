from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_ROOT = Path("results")

OUT_DIR = Path(
    "results/context_regime_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
N_BOOTSTRAP = 20000

MODELS = [
    "chronos2",
    "timesfm3",
]

DATASETS = {
    "pems_bay": {
        "short_days": 1.0,
        "recent_hours": 24,
        "long_hours": [
            48,
            72,
            120,
            168,
            240,
            336,
            504,
            672,
        ],
    },
    "electricity": {
        "short_days": 4.0,
        "recent_hours": 96,
        "long_hours": [
            120,
            144,
            168,
            192,
            240,
            288,
            336,
            504,
            744,
        ],
    },
}


def curve_auc(
    x,
    y,
):
    return float(
        np.trapezoid(
            y,
            x,
        )
    )


def load_entity_curves(
    dataset,
    model,
    config,
):
    rows = []

    for long_hours in config[
        "long_hours"
    ]:
        path = (
            RESULTS_ROOT
            / dataset
            / "interventions"
            / model
            / (
                f"recent_{config['recent_hours']}h"
                f"_long_{long_hours}h"
            )
            / "entity_metrics.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        df = pd.read_csv(
            path,
            dtype={
                "entity_id": str,
            },
        )

        wide = df.pivot(
            index="entity_id",
            columns="intervention",
            values="mean_mae",
        )

        required = [
            "short",
            "same_entity",
            "matched_other_affine",
        ]

        missing = [
            name
            for name in required
            if name not in wide.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path}: missing "
                f"{missing}"
            )

        context_days = (
            long_hours
            / 24.0
        )

        extra_days = (
            context_days
            - config[
                "short_days"
            ]
        )

        for entity_id, row in (
            wide.iterrows()
        ):
            short = float(
                row["short"]
            )

            same = float(
                row["same_entity"]
            )

            other = float(
                row[
                    "matched_other_affine"
                ]
            )

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "entity_id": (
                        entity_id
                    ),
                    "context_days": (
                        context_days
                    ),
                    "extra_days": (
                        extra_days
                    ),
                    "total_gain": (
                        short
                        - same
                    ),
                    "generic_gain": (
                        short
                        - other
                    ),
                    "entity_gain": (
                        other
                        - same
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


all_rows = []

for dataset, config in (
    DATASETS.items()
):
    for model in MODELS:
        all_rows.append(
            load_entity_curves(
                dataset=dataset,
                model=model,
                config=config,
            )
        )


curves = pd.concat(
    all_rows,
    ignore_index=True,
)

curve_path = (
    OUT_DIR
    / "entity_mechanism_curves.csv"
)

curves.to_csv(
    curve_path,
    index=False,
)


rng = np.random.default_rng(
    SEED
)

summary_rows = []
bootstrap_rows = []

bootstrap_shares = {}


print("=" * 105)
print(
    "Cross-domain long-context regime comparison"
)
print("=" * 105)


for dataset, config in (
    DATASETS.items()
):

    for model in MODELS:

        subset = curves[
            (
                curves[
                    "dataset"
                ]
                == dataset
            )
            & (
                curves[
                    "model"
                ]
                == model
            )
        ].copy()

        entities = sorted(
            subset[
                "entity_id"
            ].unique()
        )

        extra_days = np.array(
            sorted(
                subset[
                    "extra_days"
                ].unique()
            ),
            dtype=float,
        )

        matrices = {}

        for metric in [
            "total_gain",
            "generic_gain",
            "entity_gain",
        ]:
            matrix = (
                subset
                .pivot(
                    index="entity_id",
                    columns="extra_days",
                    values=metric,
                )
                .reindex(
                    index=entities,
                    columns=extra_days,
                )
            )

            if (
                matrix
                .isna()
                .any()
                .any()
            ):
                raise RuntimeError(
                    f"{dataset} / "
                    f"{model} / "
                    f"{metric}: "
                    "missing values."
                )

            matrices[
                metric
            ] = matrix.to_numpy(
                dtype=float
            )

        cohort = {
            metric:
            matrix.mean(
                axis=0
            )
            for metric, matrix
            in matrices.items()
        }

        total_auc = curve_auc(
            extra_days,
            cohort[
                "total_gain"
            ],
        )

        generic_auc = curve_auc(
            extra_days,
            cohort[
                "generic_gain"
            ],
        )

        entity_auc = curve_auc(
            extra_days,
            cohort[
                "entity_gain"
            ],
        )

        if total_auc <= 0:
            raise RuntimeError(
                f"{dataset} / {model}: "
                "non-positive total AUC."
            )

        entity_share = (
            entity_auc
            / total_auc
        )

        generic_share = (
            generic_auc
            / total_auc
        )

        n_entities = len(
            entities
        )

        boot_entity_share = (
            np.empty(
                N_BOOTSTRAP,
                dtype=float,
            )
        )

        boot_generic_share = (
            np.empty(
                N_BOOTSTRAP,
                dtype=float,
            )
        )

        for b in range(
            N_BOOTSTRAP
        ):
            idx = rng.integers(
                0,
                n_entities,
                size=n_entities,
            )

            boot_curves = {
                metric:
                matrix[
                    idx,
                    :
                ].mean(
                    axis=0
                )
                for metric, matrix
                in matrices.items()
            }

            boot_total = curve_auc(
                extra_days,
                boot_curves[
                    "total_gain"
                ],
            )

            boot_generic = curve_auc(
                extra_days,
                boot_curves[
                    "generic_gain"
                ],
            )

            boot_entity = curve_auc(
                extra_days,
                boot_curves[
                    "entity_gain"
                ],
            )

            if boot_total > 0:
                boot_entity_share[
                    b
                ] = (
                    boot_entity
                    / boot_total
                )

                boot_generic_share[
                    b
                ] = (
                    boot_generic
                    / boot_total
                )
            else:
                boot_entity_share[
                    b
                ] = np.nan

                boot_generic_share[
                    b
                ] = np.nan

        valid_entity = (
            boot_entity_share[
                np.isfinite(
                    boot_entity_share
                )
            ]
        )

        valid_generic = (
            boot_generic_share[
                np.isfinite(
                    boot_generic_share
                )
            ]
        )

        entity_low, entity_high = (
            np.quantile(
                valid_entity,
                [
                    0.025,
                    0.975,
                ],
            )
        )

        generic_low, generic_high = (
            np.quantile(
                valid_generic,
                [
                    0.025,
                    0.975,
                ],
            )
        )

        bootstrap_shares[
            (
                dataset,
                model,
            )
        ] = valid_entity

        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n_entities": (
                    n_entities
                ),
                "extra_history_min_days": (
                    extra_days.min()
                ),
                "extra_history_max_days": (
                    extra_days.max()
                ),
                "total_gain_auc": (
                    total_auc
                ),
                "generic_gain_auc": (
                    generic_auc
                ),
                "entity_gain_auc": (
                    entity_auc
                ),
                "generic_share": (
                    generic_share
                ),
                "generic_share_ci_low": (
                    generic_low
                ),
                "generic_share_ci_high": (
                    generic_high
                ),
                "entity_share": (
                    entity_share
                ),
                "entity_share_ci_low": (
                    entity_low
                ),
                "entity_share_ci_high": (
                    entity_high
                ),
            }
        )

        for b, value in enumerate(
            valid_entity
        ):
            bootstrap_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "bootstrap_index": b,
                    "entity_share": (
                        value
                    ),
                }
            )

        print()
        print("-" * 105)
        print(
            f"{dataset} | {model}"
        )
        print("-" * 105)

        print(
            "Entity share:  "
            f"{entity_share:.3f} "
            f"[{entity_low:.3f}, "
            f"{entity_high:.3f}]"
        )

        print(
            "Generic share: "
            f"{generic_share:.3f} "
            f"[{generic_low:.3f}, "
            f"{generic_high:.3f}]"
        )

        print(
            "Check sum:     "
            f"{entity_share + generic_share:.6f}"
        )


print()
print("=" * 105)
print(
    "Electricity minus PEMS entity-share contrast"
)
print("=" * 105)


contrast_rows = []


for model in MODELS:

    pems = bootstrap_shares[
        (
            "pems_bay",
            model,
        )
    ]

    electricity = bootstrap_shares[
        (
            "electricity",
            model,
        )
    ]

    n = min(
        len(pems),
        len(electricity),
    )

    difference = (
        electricity[:n]
        - pems[:n]
    )

    observed_summary = pd.DataFrame(
        summary_rows
    )

    pems_observed = float(
        observed_summary[
            (
                observed_summary[
                    "dataset"
                ]
                == "pems_bay"
            )
            & (
                observed_summary[
                    "model"
                ]
                == model
            )
        ][
            "entity_share"
        ].iloc[0]
    )

    electricity_observed = float(
        observed_summary[
            (
                observed_summary[
                    "dataset"
                ]
                == "electricity"
            )
            & (
                observed_summary[
                    "model"
                ]
                == model
            )
        ][
            "entity_share"
        ].iloc[0]
    )

    observed_difference = (
        electricity_observed
        - pems_observed
    )

    low, high = np.quantile(
        difference,
        [
            0.025,
            0.975,
        ],
    )

    probability = float(
        np.mean(
            difference > 0
        )
    )

    contrast_rows.append(
        {
            "model": model,
            "electricity_minus_pems": (
                observed_difference
            ),
            "bootstrap_ci_low": (
                float(low)
            ),
            "bootstrap_ci_high": (
                float(high)
            ),
            "probability_electricity_greater": (
                probability
            ),
        }
    )

    print()
    print(model)

    print(
        "  difference: "
        f"{observed_difference:.3f}"
    )

    print(
        "  95% CI:     "
        f"[{low:.3f}, {high:.3f}]"
    )

    print(
        "  P(Electricity > PEMS): "
        f"{probability:.4f}"
    )


summary = pd.DataFrame(
    summary_rows
)

contrasts = pd.DataFrame(
    contrast_rows
)

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


summary_path = (
    OUT_DIR
    / "context_regime_summary.csv"
)

contrast_path = (
    OUT_DIR
    / "cross_domain_contrasts.csv"
)

bootstrap_path = (
    OUT_DIR
    / "bootstrap_entity_shares.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)

contrasts.to_csv(
    contrast_path,
    index=False,
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False,
)


print()
print("=" * 105)
print("Saved")
print("=" * 105)
print(curve_path)
print(summary_path)
print(contrast_path)
print(bootstrap_path)
