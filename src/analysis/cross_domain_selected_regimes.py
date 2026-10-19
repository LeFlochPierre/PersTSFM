from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_ROOT = Path(
    "results"
)

OUT_DIR = Path(
    "results/context_regime_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42
N_BOOTSTRAP = 50000


CONFIGS = [
    {
        "dataset": "pems_bay",
        "model": "chronos2",
        "recent_hours": 24,
        "long_hours": 240,
    },
    {
        "dataset": "pems_bay",
        "model": "timesfm3",
        "recent_hours": 24,
        "long_hours": 504,
    },

    {
        "dataset": "electricity",
        "model": "chronos2",
        "recent_hours": 96,
        "long_hours": 192,
    },
    {
        "dataset": "electricity",
        "model": "timesfm3",
        "recent_hours": 96,
        "long_hours": 240,
    },

    {
        "dataset": "hupa_ucm",
        "model": "chronos2",
        "recent_hours": 24,
        "long_hours": 96,
    },
    {
        "dataset": "hupa_ucm",
        "model": "timesfm3",
        "recent_hours": 24,
        "long_hours": 168,
    },

    {
        "dataset": "vitaldb",
        "model": "chronos2",
        "recent_hours": 0.5,
        "long_hours": 4,
    },
    {
        "dataset": "vitaldb",
        "model": "timesfm3",
        "recent_hours": 0.5,
        "long_hours": 4,
    },

    {
        "dataset": "solar",
        "model": "chronos2",
        "recent_hours": 24,
        "long_hours": 96,
    },
    {
        "dataset": "solar",
        "model": "timesfm3",
        "recent_hours": 24,
        "long_hours": 96,
    },

    {
        "dataset": "physionet2019",
        "model": "chronos2",
        "recent_hours": 6,
        "long_hours": 48,
    },
    {
        "dataset": "physionet2019",
        "model": "timesfm3",
        "recent_hours": 6,
        "long_hours": 48,
    },
]


def result_folder(
    dataset,
    model,
    recent_hours,
    long_hours,
):
    return (
        RESULTS_ROOT
        / dataset
        / "interventions"
        / model
        / (
            f"recent_{recent_hours:g}h"
            f"_long_{long_hours:g}h"
        )
    )


def bootstrap_mean(
    values,
    indices,
):
    return (
        values[
            indices
        ]
        .mean(
            axis=1
        )
    )


rng = np.random.default_rng(
    SEED
)

summary_rows = []
entity_rows = []
bootstrap_rows = []


print("=" * 112)
print(
    "CROSS-DOMAIN SELECTED-CONTEXT REGIME SYNTHESIS"
)
print("=" * 112)


for config in CONFIGS:

    dataset = config[
        "dataset"
    ]

    model = config[
        "model"
    ]

    recent_hours = float(
        config[
            "recent_hours"
        ]
    )

    long_hours = float(
        config[
            "long_hours"
        ]
    )

    folder = result_folder(
        dataset=dataset,
        model=model,
        recent_hours=recent_hours,
        long_hours=long_hours,
    )

    entity_path = (
        folder
        / "entity_metrics.csv"
    )

    stats_path = (
        folder
        / "paired_statistics.csv"
    )

    if not entity_path.exists():
        raise FileNotFoundError(
            entity_path
        )

    df = pd.read_csv(
        entity_path,
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
        condition
        for condition in required
        if condition not in wide.columns
    ]

    if missing:
        raise RuntimeError(
            f"{dataset} / {model}: "
            f"missing conditions "
            f"{missing}"
        )


    if (
        "same_entity_day_shuffle"
        in wide.columns
    ):
        temporal_control = (
            "same_entity_day_shuffle"
        )

    elif (
        "same_entity_block_shuffle"
        in wide.columns
    ):
        temporal_control = (
            "same_entity_block_shuffle"
        )

    else:
        temporal_control = None


    short = (
        wide[
            "short"
        ]
        .to_numpy(
            dtype=float
        )
    )

    same = (
        wide[
            "same_entity"
        ]
        .to_numpy(
            dtype=float
        )
    )

    other = (
        wide[
            "matched_other_affine"
        ]
        .to_numpy(
            dtype=float
        )
    )


    total_gain = (
        short
        - same
    )

    generic_gain = (
        short
        - other
    )

    entity_gain = (
        other
        - same
    )


    if temporal_control is not None:
        shuffled = (
            wide[
                temporal_control
            ]
            .to_numpy(
                dtype=float
            )
        )

        order_gain = (
            shuffled
            - same
        )

    else:
        order_gain = np.full(
            len(wide),
            np.nan,
        )


    # Exact decomposition check.
    residual = (
        total_gain
        - generic_gain
        - entity_gain
    )

    if not np.allclose(
        residual,
        0.0,
        atol=1e-10,
        rtol=1e-8,
    ):
        raise RuntimeError(
            f"{dataset} / {model}: "
            "gain decomposition failed."
        )


    n_entities = len(
        wide
    )

    mean_total = float(
        np.mean(
            total_gain
        )
    )

    mean_generic = float(
        np.mean(
            generic_gain
        )
    )

    mean_entity = float(
        np.mean(
            entity_gain
        )
    )

    mean_order = float(
        np.nanmean(
            order_gain
        )
    )


    if mean_total <= 0:
        entity_ratio = np.nan
        generic_ratio = np.nan
    else:
        entity_ratio = (
            mean_entity
            / mean_total
        )

        generic_ratio = (
            mean_generic
            / mean_total
        )


    # -------------------------------------------------------------
    # Paired entity bootstrap
    # -------------------------------------------------------------

    boot_idx = rng.integers(
        0,
        n_entities,
        size=(
            N_BOOTSTRAP,
            n_entities,
        ),
    )

    boot_total = bootstrap_mean(
        total_gain,
        boot_idx,
    )

    boot_generic = bootstrap_mean(
        generic_gain,
        boot_idx,
    )

    boot_entity = bootstrap_mean(
        entity_gain,
        boot_idx,
    )

    boot_order = bootstrap_mean(
        order_gain,
        boot_idx,
    )


    boot_entity_ratio = np.full(
        N_BOOTSTRAP,
        np.nan,
        dtype=float,
    )

    boot_generic_ratio = np.full(
        N_BOOTSTRAP,
        np.nan,
        dtype=float,
    )

    valid_total = (
        boot_total > 0
    )

    boot_entity_ratio[
        valid_total
    ] = (
        boot_entity[
            valid_total
        ]
        / boot_total[
            valid_total
        ]
    )

    boot_generic_ratio[
        valid_total
    ] = (
        boot_generic[
            valid_total
        ]
        / boot_total[
            valid_total
        ]
    )


    def ci(values):
        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values) == 0:
            return (
                np.nan,
                np.nan,
            )

        low, high = np.quantile(
            values,
            [
                0.025,
                0.975,
            ],
        )

        return (
            float(low),
            float(high),
        )


    total_low, total_high = ci(
        boot_total
    )

    generic_low, generic_high = ci(
        boot_generic
    )

    entity_low, entity_high = ci(
        boot_entity
    )

    order_low, order_high = ci(
        boot_order
    )

    ratio_low, ratio_high = ci(
        boot_entity_ratio
    )

    generic_ratio_low, generic_ratio_high = (
        ci(
            boot_generic_ratio
        )
    )


    # -------------------------------------------------------------
    # Pull identical sign-flip p-values from the existing stats
    # analysis when available.
    # -------------------------------------------------------------

    p_total = np.nan
    p_generic = np.nan
    p_entity = np.nan
    p_order = np.nan

    if stats_path.exists():

        stats = pd.read_csv(
            stats_path
        )

        p_lookup = dict(
            zip(
                stats[
                    "comparison"
                ],
                stats[
                    "signflip_p"
                ],
            )
        )

        p_total = p_lookup.get(
            "total_long_context_gain",
            np.nan,
        )

        p_generic = p_lookup.get(
            "generic_long_context_gain",
            np.nan,
        )

        p_entity = p_lookup.get(
            "entity_specific_gain",
            np.nan,
        )

        p_order = p_lookup.get(
            "temporal_order_gain",
            np.nan,
        )


    # -------------------------------------------------------------
    # Entity-level output
    # -------------------------------------------------------------

    for i, entity_id in enumerate(
        wide.index
    ):
        entity_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "entity_id": str(
                    entity_id
                ),
                "recent_hours": (
                    recent_hours
                ),
                "long_hours": (
                    long_hours
                ),
                "total_gain": float(
                    total_gain[i]
                ),
                "generic_gain": float(
                    generic_gain[i]
                ),
                "entity_gain": float(
                    entity_gain[i]
                ),
                "order_gain": float(
                    order_gain[i]
                ),
            }
        )


    summary_rows.append(
        {
            "dataset": dataset,
            "model": model,
            "n_entities": n_entities,
            "recent_hours": (
                recent_hours
            ),
            "long_hours": (
                long_hours
            ),

            "total_gain": (
                mean_total
            ),
            "total_gain_ci_low": (
                total_low
            ),
            "total_gain_ci_high": (
                total_high
            ),
            "total_gain_signflip_p": (
                p_total
            ),

            "generic_gain": (
                mean_generic
            ),
            "generic_gain_ci_low": (
                generic_low
            ),
            "generic_gain_ci_high": (
                generic_high
            ),
            "generic_gain_signflip_p": (
                p_generic
            ),

            "entity_gain": (
                mean_entity
            ),
            "entity_gain_ci_low": (
                entity_low
            ),
            "entity_gain_ci_high": (
                entity_high
            ),
            "entity_gain_signflip_p": (
                p_entity
            ),

            "order_gain": (
                mean_order
            ),
            "order_gain_ci_low": (
                order_low
            ),
            "order_gain_ci_high": (
                order_high
            ),
            "order_gain_signflip_p": (
                p_order
            ),

            "entity_contribution_ratio": (
                entity_ratio
            ),
            "entity_ratio_ci_low": (
                ratio_low
            ),
            "entity_ratio_ci_high": (
                ratio_high
            ),

            "generic_contribution_ratio": (
                generic_ratio
            ),
            "generic_ratio_ci_low": (
                generic_ratio_low
            ),
            "generic_ratio_ci_high": (
                generic_ratio_high
            ),
        }
    )


    for b in range(
        N_BOOTSTRAP
    ):
        bootstrap_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "bootstrap_index": b,
                "total_gain": float(
                    boot_total[b]
                ),
                "generic_gain": float(
                    boot_generic[b]
                ),
                "entity_gain": float(
                    boot_entity[b]
                ),
                "order_gain": float(
                    boot_order[b]
                ),
                "entity_contribution_ratio": (
                    float(
                        boot_entity_ratio[b]
                    )
                ),
                "generic_contribution_ratio": (
                    float(
                        boot_generic_ratio[b]
                    )
                ),
            }
        )


    print()
    print("-" * 112)

    print(
        dataset,
        "|",
        model,
    )

    print("-" * 112)

    print(
        "N entities:              ",
        n_entities,
    )

    print(
        "Context:                 ",
        f"{recent_hours:g} h",
        "->",
        f"{long_hours:g} h",
    )

    print(
        "Total gain:              ",
        f"{mean_total:.6f}",
        f"[{total_low:.6f}, "
        f"{total_high:.6f}]",
    )

    print(
        "Generic gain:            ",
        f"{mean_generic:.6f}",
        f"[{generic_low:.6f}, "
        f"{generic_high:.6f}]",
    )

    print(
        "Entity gain:             ",
        f"{mean_entity:.6f}",
        f"[{entity_low:.6f}, "
        f"{entity_high:.6f}]",
    )

    print(
        "Order gain:              ",
        f"{mean_order:.6f}",
        f"[{order_low:.6f}, "
        f"{order_high:.6f}]",
    )

    print(
        "Entity contribution ratio:",
        f"{entity_ratio:.3f}",
        f"[{ratio_low:.3f}, "
        f"{ratio_high:.3f}]",
    )

    print(
        "Generic contribution ratio:",
        f"{generic_ratio:.3f}",
        f"[{generic_ratio_low:.3f}, "
        f"{generic_ratio_high:.3f}]",
    )

    print(
        "Ratio sum:               ",
        f"{entity_ratio + generic_ratio:.6f}",
    )


summary = pd.DataFrame(
    summary_rows
)

entities = pd.DataFrame(
    entity_rows
)

bootstraps = pd.DataFrame(
    bootstrap_rows
)


summary_path = (
    OUT_DIR
    / "selected_context_regime_summary.csv"
)

entity_path = (
    OUT_DIR
    / "selected_context_entity_gains.csv"
)

bootstrap_path = (
    OUT_DIR
    / "selected_context_bootstrap.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)

entities.to_csv(
    entity_path,
    index=False,
)

bootstraps.to_csv(
    bootstrap_path,
    index=False,
)


print()
print("=" * 112)
print("SAVED")
print("=" * 112)

print(
    summary_path
)

print(
    entity_path
)

print(
    bootstrap_path
)
