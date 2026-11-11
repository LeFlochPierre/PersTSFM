import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.adapters import (
    ForecastInput,
    build_model,
)


RESOLUTION_MINUTES = 15
STEPS_PER_DAY = 96

SHORT_DAYS = 1
LONG_DAYS = 7
HORIZON_DAYS = 1

SHORT_STEPS = (
    SHORT_DAYS
    * STEPS_PER_DAY
)

LONG_STEPS = (
    LONG_DAYS
    * STEPS_PER_DAY
)

HORIZON_STEPS = (
    HORIZON_DAYS
    * STEPS_PER_DAY
)

N_ENTITIES = 40
TOTAL_DAYS = 56

ORIGIN_DAYS = [
    28,
    35,
    42,
    49,
]

HETEROGENEITY_GRID = [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
]

NOISE_GRID = [
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
]

DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "chronos2",
            "timesfm3",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


def safe_standardize(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    if std < 1e-8:
        std = 1.0

    return (
        (values - mean) / std,
        mean,
        std,
    )


def make_ar1_noise(
    rng,
    n_entities,
    n_steps,
    rho=0.7,
):
    noise = np.zeros(
        (
            n_entities,
            n_steps,
        ),
        dtype=float,
    )

    innovations = rng.normal(
        size=(
            n_entities,
            n_steps,
        )
    )

    innovation_scale = np.sqrt(
        1.0 - rho ** 2
    )

    noise[:, 0] = (
        innovations[:, 0]
    )

    for t in range(
        1,
        n_steps,
    ):
        noise[:, t] = (
            rho
            * noise[
                :,
                t - 1
            ]
            + innovation_scale
            * innovations[
                :,
                t
            ]
        )

    return noise


def build_base_components(
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    n_steps = (
        TOTAL_DAYS
        * STEPS_PER_DAY
    )

    t = np.arange(
        n_steps,
        dtype=float,
    )

    daily_phase = (
        2.0
        * np.pi
        * t
        / STEPS_PER_DAY
    )

    weekly_phase = (
        2.0
        * np.pi
        * t
        / (
            7
            * STEPS_PER_DAY
        )
    )

    shared = (
        50.0
        + 8.0
        * np.sin(
            daily_phase
        )
        + 3.0
        * np.cos(
            2.0
            * daily_phase
        )
        + 4.0
        * np.sin(
            weekly_phase
        )
        + 2.0
        * np.cos(
            weekly_phase
        )
    )

    basis = np.stack(
        [
            np.sin(
                daily_phase
            ),
            np.cos(
                daily_phase
            ),
            np.sin(
                2.0
                * daily_phase
            ),
            np.cos(
                2.0
                * daily_phase
            ),
            np.sin(
                weekly_phase
            ),
            np.cos(
                weekly_phase
            ),
        ],
        axis=0,
    )

    coefficients = rng.normal(
        size=(
            N_ENTITIES,
            basis.shape[0],
        )
    )

    signatures = (
        coefficients
        @ basis
    )

    for i in range(
        N_ENTITIES
    ):
        signature = (
            signatures[i]
            - np.mean(
                signatures[i]
            )
        )

        std = float(
            np.std(signature)
        )

        if std > 1e-8:
            signature = (
                signature / std
            )

        signatures[i] = (
            signature
        )

    noise = make_ar1_noise(
        rng=rng,
        n_entities=N_ENTITIES,
        n_steps=n_steps,
    )

    timestamps = pd.date_range(
        start="2020-01-01",
        periods=n_steps,
        freq="15min",
    )

    return (
        timestamps,
        shared,
        signatures,
        noise,
    )


def make_series(
    shared,
    signatures,
    noise,
    heterogeneity,
    noise_sd,
):
    return (
        shared[
            None,
            :
        ]
        + heterogeneity
        * signatures
        + noise_sd
        * noise
    )


def find_donor(
    values,
    target_id,
    origin_step,
):
    recent_start = (
        origin_step
        - SHORT_STEPS
    )

    target_recent = values[
        target_id,
        recent_start:origin_step,
    ]

    target_z, _, _ = (
        safe_standardize(
            target_recent
        )
    )

    candidates = []

    for donor_id in range(
        N_ENTITIES
    ):
        if donor_id == target_id:
            continue

        donor_recent = values[
            donor_id,
            recent_start:origin_step,
        ]

        donor_z, _, _ = (
            safe_standardize(
                donor_recent
            )
        )

        distance = float(
            np.sqrt(
                np.mean(
                    (
                        target_z
                        - donor_z
                    )
                    ** 2
                )
            )
        )

        candidates.append(
            (
                distance,
                donor_id,
            )
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return candidates[0]


def prepare_cases(
    values,
    timestamps,
):
    cases = []

    for entity_id in range(
        N_ENTITIES
    ):
        for origin_day in (
            ORIGIN_DAYS
        ):
            origin_step = (
                origin_day
                * STEPS_PER_DAY
            )

            long_start = (
                origin_step
                - LONG_STEPS
            )

            recent_start = (
                origin_step
                - SHORT_STEPS
            )

            target_context = values[
                entity_id,
                long_start:origin_step,
            ]

            recent = values[
                entity_id,
                recent_start:origin_step,
            ]

            truth = values[
                entity_id,
                origin_step:
                origin_step
                + HORIZON_STEPS,
            ]

            (
                donor_distance,
                donor_id,
            ) = find_donor(
                values=values,
                target_id=entity_id,
                origin_step=origin_step,
            )

            donor_long = values[
                donor_id,
                long_start:origin_step,
            ]

            donor_recent = values[
                donor_id,
                recent_start:origin_step,
            ]

            donor_old = (
                donor_long[
                    :-SHORT_STEPS
                ]
            )

            (
                _,
                target_mean,
                target_std,
            ) = safe_standardize(
                recent
            )

            (
                _,
                donor_mean,
                donor_std,
            ) = safe_standardize(
                donor_recent
            )

            donor_old_affine = (
                (
                    donor_old
                    - donor_mean
                )
                / donor_std
                * target_std
                + target_mean
            )

            other_context = (
                np.concatenate(
                    [
                        donor_old_affine,
                        recent,
                    ]
                )
            )

            case_id = (
                f"e{entity_id:03d}"
                f"_d{origin_day:02d}"
            )

            cases.append(
                {
                    "case_id": case_id,
                    "entity_id": (
                        entity_id
                    ),
                    "origin_day": (
                        origin_day
                    ),
                    "truth": truth,
                    "short_values": (
                        recent
                    ),
                    "short_timestamps": (
                        timestamps[
                            recent_start:
                            origin_step
                        ]
                    ),
                    "same_values": (
                        target_context
                    ),
                    "long_timestamps": (
                        timestamps[
                            long_start:
                            origin_step
                        ]
                    ),
                    "other_values": (
                        other_context
                    ),
                    "donor_id": (
                        donor_id
                    ),
                    "donor_distance": (
                        donor_distance
                    ),
                }
            )

    return cases


def predict_condition(
    model,
    cases,
    condition,
    batch_size,
):
    if condition == "short":
        values_key = (
            "short_values"
        )

        timestamps_key = (
            "short_timestamps"
        )

        n_context_steps = (
            SHORT_STEPS
        )

    elif condition == "same":
        values_key = (
            "same_values"
        )

        timestamps_key = (
            "long_timestamps"
        )

        n_context_steps = (
            LONG_STEPS
        )

    elif condition == "other":
        values_key = (
            "other_values"
        )

        timestamps_key = (
            "long_timestamps"
        )

        n_context_steps = (
            LONG_STEPS
        )

    else:
        raise ValueError(
            condition
        )

    predictions = {}

    for start in range(
        0,
        len(cases),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(cases),
        )

        batch = cases[
            start:stop
        ]

        inputs = []

        for case in batch:
            inputs.append(
                ForecastInput(
                    case_id=(
                        case[
                            "case_id"
                        ]
                    ),
                    timestamps=pd.DatetimeIndex(
                        case[
                            timestamps_key
                        ]
                    ),
                    values=np.asarray(
                        case[
                            values_key
                        ],
                        dtype=float,
                    ),
                )
            )

        outputs = model.predict_batch(
            inputs=inputs,
            horizon=(
                HORIZON_STEPS
            ),
            resolution_minutes=(
                RESOLUTION_MINUTES
            ),
            context_steps=(
                n_context_steps
            ),
            batch_size=(
                batch_size
            ),
        )

        for output in outputs:
            predictions[
                output.case_id
            ] = np.asarray(
                output.prediction,
                dtype=float,
            )

    return predictions


def main():
    args = parse_args()

    (
        timestamps,
        shared,
        signatures,
        noise,
    ) = build_base_components(
        seed=args.seed,
    )

    model = build_model(
        args.model,
        batch_size=(
            args.batch_size
        ),
    )

    case_rows = []

    for heterogeneity in (
        HETEROGENEITY_GRID
    ):
        for noise_sd in (
            NOISE_GRID
        ):
            print()
            print("=" * 90)
            print(
                f"{args.model} | "
                f"heterogeneity="
                f"{heterogeneity:g} | "
                f"noise={noise_sd:g}"
            )
            print("=" * 90)

            values = make_series(
                shared=shared,
                signatures=signatures,
                noise=noise,
                heterogeneity=(
                    heterogeneity
                ),
                noise_sd=noise_sd,
            )

            cases = prepare_cases(
                values=values,
                timestamps=timestamps,
            )

            predictions = {}

            for condition in [
                "short",
                "same",
                "other",
            ]:
                predictions[
                    condition
                ] = predict_condition(
                    model=model,
                    cases=cases,
                    condition=condition,
                    batch_size=(
                        args.batch_size
                    ),
                )

            for case in cases:
                truth = np.asarray(
                    case["truth"],
                    dtype=float,
                )

                metrics = {}

                for condition in [
                    "short",
                    "same",
                    "other",
                ]:
                    prediction = (
                        predictions[
                            condition
                        ][
                            case[
                                "case_id"
                            ]
                        ]
                    )

                    mae = float(
                        np.mean(
                            np.abs(
                                truth
                                - prediction
                            )
                        )
                    )

                    metrics[
                        condition
                    ] = mae

                case_rows.append(
                    {
                        "model": (
                            args.model
                        ),
                        "seed": (
                            args.seed
                        ),
                        "heterogeneity": (
                            heterogeneity
                        ),
                        "noise_sd": (
                            noise_sd
                        ),
                        "entity_id": (
                            case[
                                "entity_id"
                            ]
                        ),
                        "origin_day": (
                            case[
                                "origin_day"
                            ]
                        ),
                        "donor_id": (
                            case[
                                "donor_id"
                            ]
                        ),
                        "donor_distance": (
                            case[
                                "donor_distance"
                            ]
                        ),
                        "short_mae": (
                            metrics[
                                "short"
                            ]
                        ),
                        "same_mae": (
                            metrics[
                                "same"
                            ]
                        ),
                        "other_mae": (
                            metrics[
                                "other"
                            ]
                        ),
                    }
                )

    cases = pd.DataFrame(
        case_rows
    )

    entity = (
        cases
        .groupby(
            [
                "model",
                "seed",
                "heterogeneity",
                "noise_sd",
                "entity_id",
            ],
            as_index=False,
        )
        .agg(
            short_mae=(
                "short_mae",
                "mean",
            ),
            same_mae=(
                "same_mae",
                "mean",
            ),
            other_mae=(
                "other_mae",
                "mean",
            ),
            mean_donor_distance=(
                "donor_distance",
                "mean",
            ),
        )
    )

    entity[
        "total_gain"
    ] = (
        entity[
            "short_mae"
        ]
        - entity[
            "same_mae"
        ]
    )

    entity[
        "generic_gain"
    ] = (
        entity[
            "short_mae"
        ]
        - entity[
            "other_mae"
        ]
    )

    entity[
        "entity_gain"
    ] = (
        entity[
            "other_mae"
        ]
        - entity[
            "same_mae"
        ]
    )

    summary = (
        entity
        .groupby(
            [
                "model",
                "seed",
                "heterogeneity",
                "noise_sd",
            ],
            as_index=False,
        )
        .agg(
            n_entities=(
                "entity_id",
                "nunique",
            ),
            short_mae=(
                "short_mae",
                "mean",
            ),
            same_mae=(
                "same_mae",
                "mean",
            ),
            other_mae=(
                "other_mae",
                "mean",
            ),
            total_gain=(
                "total_gain",
                "mean",
            ),
            generic_gain=(
                "generic_gain",
                "mean",
            ),
            entity_gain=(
                "entity_gain",
                "mean",
            ),
            donor_distance=(
                "mean_donor_distance",
                "mean",
            ),
        )
    )

    summary[
        "entity_contribution_ratio"
    ] = (
        summary[
            "entity_gain"
        ]
        / summary[
            "total_gain"
        ]
    )

    output_dir = (
        Path("results")
        / "synthetic_phase_diagram_final"
        / args.model
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases.to_csv(
        output_dir
        / "case_metrics.csv",
        index=False,
    )

    entity.to_csv(
        output_dir
        / "entity_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("Synthetic phase-diagram pilot")
    print("=" * 90)

    print(
        summary.round(
            4
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        output_dir,
    )


if __name__ == "__main__":
    main()
