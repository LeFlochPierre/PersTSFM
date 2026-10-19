from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data.registry import (
    context_steps,
    extract_case,
    load_dataset,
)

from src.evaluation.metrics import (
    add_prediction_metrics,
)

from src.models.adapters import (
    ForecastInput,
    build_model,
)


def _safe_standardize(values):
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


def _deterministic_rng(case_id):
    digest = sha256(
        case_id.encode("utf-8")
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return np.random.default_rng(
        seed
    )


def _find_matched_donor(
    dataset,
    target_id,
    recent_index,
    target_recent,
):
    target_z, _, _ = (
        _safe_standardize(
            target_recent
        )
    )

    candidates = []

    for donor_id, donor_series in (
        dataset.series.items()
    ):
        if donor_id == target_id:
            continue

        donor_recent = (
            donor_series.reindex(
                recent_index
            )
        )

        if donor_recent.isna().any():
            continue

        donor_values = (
            donor_recent.to_numpy(
                dtype=float
            )
        )

        donor_z, _, _ = (
            _safe_standardize(
                donor_values
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
                donor_values,
            )
        )

    if not candidates:
        raise RuntimeError(
            f"No valid donor found "
            f"for {target_id}."
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return candidates[0]




def _find_matched_relative_donor(
    dataset,
    target_id,
    target_eval_index,
    target_recent,
    recent_hours,
    long_hours,
    target_region=None,
    target_source_set=None,
):
    """
    Match another entity using relative-time windows.

    Candidate donors are restricted to the same frozen evaluation
    index as the target.

    Optional grouping restrictions are used where required:
    Solar -> same region;
    PhysioNet 2019 -> same source set (A/B).
    """
    spec = dataset.spec

    target_z, _, _ = (
        _safe_standardize(
            target_recent
        )
    )

    n_recent = context_steps(
        recent_hours,
        spec.resolution_minutes,
    )

    candidates = []

    donor_protocol = (
        dataset.protocol[
            dataset.protocol[
                "eval_index"
            ].astype(int)
            == int(
                target_eval_index
            )
        ]
        .copy()
    )

    if (
        target_region is not None
        and "region"
        in donor_protocol.columns
    ):
        donor_protocol = (
            donor_protocol[
                donor_protocol[
                    "region"
                ].astype(str)
                == str(
                    target_region
                )
            ]
        )

    if (
        target_source_set is not None
        and "source_set"
        in donor_protocol.columns
    ):
        donor_protocol = (
            donor_protocol[
                donor_protocol[
                    "source_set"
                ].astype(str)
                == str(
                    target_source_set
                )
            ]
        )

    for _, donor_row in (
        donor_protocol.iterrows()
    ):
        donor_case = extract_case(
            dataset,
            donor_row,
            long_hours,
        )

        donor_id = (
            donor_case[
                "entity_id"
            ]
        )

        if (
            str(donor_id)
            == str(target_id)
        ):
            continue

        donor_context = (
            donor_case[
                "context"
            ]
        )

        donor_old = (
            donor_context.iloc[
                :-n_recent
            ]
        )

        if (
            dataset.spec.key
            == "physionet2019"
        ):
            donor_recent_case = (
                extract_case(
                    dataset,
                    donor_row,
                    recent_hours,
                )
            )

            donor_recent = (
                donor_recent_case[
                    "context"
                ]
            )

        else:
            donor_recent = (
                donor_context.iloc[
                    -n_recent:
                ]
            )

        if (
            len(donor_recent)
            != n_recent
        ):
            continue

        if (
            donor_recent.isna().any()
            or donor_old.isna().any()
        ):
            continue

        donor_recent_values = (
            donor_recent.to_numpy(
                dtype=float
            )
        )

        donor_old_values = (
            donor_old.to_numpy(
                dtype=float
            )
        )

        donor_z, _, _ = (
            _safe_standardize(
                donor_recent_values
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
                str(donor_id),
                donor_recent_values,
                donor_old_values,
                int(
                    donor_case[
                        "eval_index"
                    ]
                ),
            )
        )

    if not candidates:
        message = (
            f"No relative-time donor found "
            f"for {target_id} / "
            f"eval_index={target_eval_index}"
        )

        if target_region is not None:
            message += (
                f" / region={target_region}"
            )

        if target_source_set is not None:
            message += (
                f" / source_set="
                f"{target_source_set}"
            )

        raise RuntimeError(
            message + "."
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return candidates[0]



def _shuffle_temporal_blocks(
    old_values,
    block_steps,
    case_id,
):
    old_values = np.asarray(
        old_values,
        dtype=float,
    )

    block_steps = int(
        block_steps
    )

    if block_steps <= 0:
        raise ValueError(
            "block_steps must be positive."
        )

    if (
        len(old_values)
        % block_steps
        != 0
    ):
        raise RuntimeError(
            "Old-history length is not "
            "divisible by shuffle block size."
        )

    n_blocks = (
        len(old_values)
        // block_steps
    )

    if n_blocks < 2:
        raise RuntimeError(
            "Need at least two temporal "
            "blocks to shuffle."
        )

    blocks = (
        old_values.reshape(
            n_blocks,
            block_steps,
        )
    )

    rng = _deterministic_rng(
        case_id
    )

    order = rng.permutation(
        n_blocks
    )

    if np.array_equal(
        order,
        np.arange(
            n_blocks
        ),
    ):
        order = np.roll(
            order,
            1,
        )

    shuffled = (
        blocks[
            order
        ]
        .reshape(-1)
    )

    return shuffled, order


def _shuffle_daily_blocks(
    old_values,
    steps_per_day,
    case_id,
):
    """
    Backwards-compatible daily-block wrapper.
    """
    return _shuffle_temporal_blocks(
        old_values=old_values,
        block_steps=steps_per_day,
        case_id=case_id,
    )



def _build_case_conditions(
    dataset,
    row,
    recent_hours,
    long_hours,
    dataset_key,
):
    spec = dataset.spec

    if long_hours <= recent_hours:
        raise ValueError(
            "long_hours must be "
            "greater than recent_hours."
        )

    long_case = extract_case(
        dataset,
        row,
        long_hours,
    )

    recent_case = None

    if dataset_key == "physionet2019":
        recent_case = extract_case(
            dataset,
            row,
            recent_hours,
        )

    target_id = (
        long_case["entity_id"]
    )

    case_id = (
        long_case["case_id"]
    )

    full_context = (
        long_case["context"]
    )

    n_recent = context_steps(
        recent_hours,
        spec.resolution_minutes,
    )

    n_long = context_steps(
        long_hours,
        spec.resolution_minutes,
    )

    n_old = (
        n_long
        - n_recent
    )

    if (
        len(full_context)
        != n_long
    ):
        raise RuntimeError(
            f"{case_id}: unexpected "
            "long-context length."
        )

    old = (
        full_context.iloc[
            :n_old
        ]
    )

    if dataset_key == "physionet2019":
        recent = (
            recent_case[
                "context"
            ]
        )

    else:
        recent = (
            full_context.iloc[
                n_old:
            ]
        )

    recent_values = (
        recent.to_numpy(
            dtype=float
        )
    )

    donor_eval_index = np.nan

    # ------------------------------------------------------------
    # Donor selection
    # ------------------------------------------------------------

    relative_donor_datasets = {
        "hupa_ucm",
        "vitaldb",
        "solar",
        "physionet2019",
    }

    if (
        dataset_key
        in relative_donor_datasets
    ):
        target_region = None
        target_source_set = None

        if (
            dataset_key == "solar"
            and "region" in row.index
        ):
            target_region = (
                row["region"]
            )

        if (
            dataset_key
            == "physionet2019"
            and "source_set"
            in row.index
        ):
            target_source_set = (
                row[
                    "source_set"
                ]
            )

        (
            donor_distance,
            donor_id,
            donor_recent_values,
            donor_old_values,
            donor_eval_index,
        ) = _find_matched_relative_donor(
            dataset=dataset,
            target_id=target_id,
            target_eval_index=(
                long_case[
                    "eval_index"
                ]
            ),
            target_recent=recent_values,
            recent_hours=recent_hours,
            long_hours=long_hours,
            target_region=target_region,
            target_source_set=(
                target_source_set
            ),
        )

        if (
            len(donor_old_values)
            != n_old
        ):
            raise RuntimeError(
                f"{case_id}: relative donor "
                "has unexpected old-history "
                f"length "
                f"{len(donor_old_values)}."
            )

    else:
        (
            donor_distance,
            donor_id,
            donor_recent_values,
        ) = _find_matched_donor(
            dataset=dataset,
            target_id=target_id,
            recent_index=recent.index,
            target_recent=recent_values,
        )

        donor_series = (
            dataset.series[
                donor_id
            ]
        )

        donor_old = (
            donor_series.reindex(
                old.index
            )
        )

        if donor_old.isna().any():
            raise RuntimeError(
                f"{case_id}: matched donor "
                f"{donor_id} lacks old history."
            )

        donor_old_values = (
            donor_old.to_numpy(
                dtype=float
            )
        )

    # ------------------------------------------------------------
    # Affine donor normalization
    # ------------------------------------------------------------

    _, target_mean, target_std = (
        _safe_standardize(
            recent_values
        )
    )

    _, donor_mean, donor_std = (
        _safe_standardize(
            donor_recent_values
        )
    )

    donor_old_affine = (
        (
            donor_old_values
            - donor_mean
        )
        / donor_std
        * target_std
        + target_mean
    )

    # ------------------------------------------------------------
    # Boundary diagnostics
    # ------------------------------------------------------------

    same_boundary_jump = abs(
        recent_values[0]
        - old.to_numpy(
            dtype=float
        )[-1]
    )

    raw_boundary_jump = abs(
        recent_values[0]
        - donor_old_values[-1]
    )

    affine_boundary_jump = abs(
        recent_values[0]
        - donor_old_affine[-1]
    )

    same_boundary_jump_norm = (
        same_boundary_jump
        / target_std
    )

    raw_boundary_jump_norm = (
        raw_boundary_jump
        / target_std
    )

    affine_boundary_jump_norm = (
        affine_boundary_jump
        / target_std
    )

    # ------------------------------------------------------------
    # Temporal-order intervention
    # ------------------------------------------------------------

    if dataset_key in {
        "vitaldb",
        "physionet2019",
    }:
        # Acute physiology uses blocks matched
        # to the recent-context duration rather
        # than 24-hour day blocks.
        #
        # VitalDB:     0.5 h blocks
        # PhysioNet:   6 h blocks
        shuffle_block_hours = (
            recent_hours
        )

        shuffle_block_steps = (
            n_recent
        )

        shuffle_intervention = (
            "same_entity_block_shuffle"
        )

    else:
        shuffle_block_hours = 24.0

        shuffle_block_steps = (
            context_steps(
                24,
                spec.resolution_minutes,
            )
        )

        shuffle_intervention = (
            "same_entity_day_shuffle"
        )

    (
        shuffled_old,
        shuffle_order,
    ) = _shuffle_temporal_blocks(
        old_values=(
            old.to_numpy(
                dtype=float
            )
        ),
        block_steps=(
            shuffle_block_steps
        ),
        case_id=case_id,
    )

    # ------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------

    same_entity_values = (
        full_context.to_numpy(
            dtype=float
        )
    )

    if dataset_key == "physionet2019":
        same_entity_values = (
            np.concatenate(
                [
                    old.to_numpy(
                        dtype=float
                    ),
                    recent_values,
                ]
            )
        )

    conditions = {
        "short": {
            "timestamps": (
                recent.index
            ),
            "values": (
                recent_values
            ),
        },
        "same_entity": {
            "timestamps": (
                full_context.index
            ),
            "values": (
                same_entity_values
            ),
        },
        "matched_other_raw": {
            "timestamps": (
                full_context.index
            ),
            "values": np.concatenate(
                [
                    donor_old_values,
                    recent_values,
                ]
            ),
        },
        "matched_other_affine": {
            "timestamps": (
                full_context.index
            ),
            "values": np.concatenate(
                [
                    donor_old_affine,
                    recent_values,
                ]
            ),
        },
        shuffle_intervention: {
            "timestamps": (
                full_context.index
            ),
            "values": np.concatenate(
                [
                    shuffled_old,
                    recent_values,
                ]
            ),
        },
    }

    metadata = {
        "case_id": case_id,
        "entity_id": target_id,
        "eval_index": int(
            long_case[
                "eval_index"
            ]
        ),
        "eval_origin": (
            long_case[
                "eval_origin"
            ]
        ),
        "donor_id": donor_id,
        "donor_eval_index": (
            donor_eval_index
        ),
        "donor_distance": float(
            donor_distance
        ),
        "same_boundary_jump": float(
            same_boundary_jump
        ),
        "raw_boundary_jump": float(
            raw_boundary_jump
        ),
        "affine_boundary_jump": float(
            affine_boundary_jump
        ),
        "same_boundary_jump_norm": float(
            same_boundary_jump_norm
        ),
        "raw_boundary_jump_norm": float(
            raw_boundary_jump_norm
        ),
        "affine_boundary_jump_norm": float(
            affine_boundary_jump_norm
        ),
        "shuffle_intervention": (
            shuffle_intervention
        ),
        "shuffle_block_hours": float(
            shuffle_block_hours
        ),
        "shuffle_order": ",".join(
            str(int(x))
            for x in shuffle_order
        ),
    }

    return (
        conditions,
        long_case["truth"],
        metadata,
    )


def _aggregate_intervention(
    pred,
):
    case_metrics = (
        pred
        .groupby(
            [
                "intervention",
                "entity_id",
                "eval_index",
                "eval_origin",
                "donor_id",
            ],
            as_index=False,
        )
        .agg(
            mae=(
                "abs_error",
                "mean",
            ),
            mse=(
                "squared_error",
                "mean",
            ),
            coverage_80=(
                "covered_80",
                "mean",
            ),
            interval_width_80=(
                "interval_width_80",
                "mean",
            ),
        )
    )

    case_metrics["rmse"] = np.sqrt(
        case_metrics["mse"]
    )

    case_metrics = (
        case_metrics.drop(
            columns="mse"
        )
    )

    short = (
        case_metrics[
            case_metrics[
                "intervention"
            ]
            == "short"
        ]
        [
            [
                "entity_id",
                "eval_index",
                "mae",
            ]
        ]
        .rename(
            columns={
                "mae": "short_mae"
            }
        )
    )

    same = (
        case_metrics[
            case_metrics[
                "intervention"
            ]
            == "same_entity"
        ]
        [
            [
                "entity_id",
                "eval_index",
                "mae",
            ]
        ]
        .rename(
            columns={
                "mae": "same_mae"
            }
        )
    )

    case_metrics = (
        case_metrics
        .merge(
            short,
            on=[
                "entity_id",
                "eval_index",
            ],
            how="left",
            validate="many_to_one",
        )
        .merge(
            same,
            on=[
                "entity_id",
                "eval_index",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    case_metrics[
        "delta_mae_vs_short"
    ] = (
        case_metrics["mae"]
        - case_metrics["short_mae"]
    )

    case_metrics[
        "delta_mae_vs_same"
    ] = (
        case_metrics["mae"]
        - case_metrics["same_mae"]
    )

    entity_metrics = (
        case_metrics
        .groupby(
            [
                "intervention",
                "entity_id",
            ],
            as_index=False,
        )
        .agg(
            mean_mae=(
                "mae",
                "mean",
            ),
            median_mae=(
                "mae",
                "median",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_coverage_80=(
                "coverage_80",
                "mean",
            ),
            mean_delta_mae_vs_short=(
                "delta_mae_vs_short",
                "mean",
            ),
            mean_delta_mae_vs_same=(
                "delta_mae_vs_same",
                "mean",
            ),
        )
    )

    summary = (
        entity_metrics
        .groupby(
            "intervention",
            as_index=False,
        )
        .agg(
            n_entities=(
                "entity_id",
                "nunique",
            ),
            mean_entity_mae=(
                "mean_mae",
                "mean",
            ),
            median_entity_mae=(
                "mean_mae",
                "median",
            ),
            mean_entity_rmse=(
                "mean_rmse",
                "mean",
            ),
            mean_coverage_80=(
                "mean_coverage_80",
                "mean",
            ),
            mean_delta_mae_vs_short=(
                "mean_delta_mae_vs_short",
                "mean",
            ),
            mean_delta_mae_vs_same=(
                "mean_delta_mae_vs_same",
                "mean",
            ),
        )
    )

    values = {
        row["intervention"]:
        float(
            row[
                "mean_entity_mae"
            ]
        )
        for _, row in (
            summary.iterrows()
        )
    }

    short_mae = values[
        "short"
    ]

    same_mae = values[
        "same_entity"
    ]

    total_gain = (
        short_mae
        - same_mae
    )

    summary[
        "fraction_of_same_entity_gain_retained"
    ] = np.nan

    if total_gain > 0:
        summary[
            "fraction_of_same_entity_gain_retained"
        ] = (
            short_mae
            - summary[
                "mean_entity_mae"
            ]
        ) / total_gain

    summary[
        "entity_specific_gap_vs_same"
    ] = (
        summary[
            "mean_entity_mae"
        ]
        - same_mae
    )

    return (
        case_metrics,
        entity_metrics,
        summary,
    )


def run_context_intervention(
    dataset_key,
    model_key,
    recent_hours,
    long_hours,
    split=None,
    max_cases=None,
    one_case_per_entity=False,
    output_root="results",
):
    dataset = load_dataset(
        dataset_key,
        split=split,
    )

    spec = dataset.spec

    supported_datasets = {
        "pems_bay",
        "electricity",
        "hupa_ucm",
        "vitaldb",
        "solar",
        "physionet2019",
    }

    if dataset_key not in supported_datasets:
        raise NotImplementedError(
            "Context interventions are currently "
            "validated for PEMS-BAY, Electricity, "
            "HUPA-UCM, VitalDB, Solar, and "
            "PhysioNet 2019."
        )

    protocol = (
        dataset.protocol.copy()
    )

    if one_case_per_entity:
        protocol = (
            protocol
            .groupby(
                spec.id_col,
                group_keys=False,
            )
            .head(1)
            .reset_index(drop=True)
        )

    if max_cases is not None:
        protocol = (
            protocol.iloc[
                :max_cases
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    if model_key == "chronos2":
        batch_size = (
            spec.chronos_batch_size
        )
    elif model_key == "timesfm3":
        batch_size = (
            spec.timesfm_batch_size
        )
    else:
        raise ValueError(
            f"Unknown model: "
            f"{model_key}"
        )

    model = build_model(
        model_key,
        batch_size=batch_size,
    )

    if dataset_key in {
        "vitaldb",
        "physionet2019",
    }:
        shuffle_condition = (
            "same_entity_block_shuffle"
        )
    else:
        shuffle_condition = (
            "same_entity_day_shuffle"
        )

    conditions = [
        "short",
        "same_entity",
        "matched_other_raw",
        "matched_other_affine",
        shuffle_condition,
    ]

    prepared = {}
    truth_by_case = {}
    metadata_rows = []

    for _, row in (
        protocol.iterrows()
    ):
        (
            case_conditions,
            truth,
            metadata,
        ) = _build_case_conditions(
            dataset=dataset,
            row=row,
            recent_hours=recent_hours,
            long_hours=long_hours,
            dataset_key=dataset_key,
        )

        case_id = metadata[
            "case_id"
        ]

        prepared[
            case_id
        ] = case_conditions

        truth_by_case[
            case_id
        ] = truth

        metadata_rows.append(
            metadata
        )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    rows = []

    start_time = time.time()

    for intervention in conditions:
        print()
        print("=" * 78)
        print(
            f"Intervention: "
            f"{intervention}"
        )
        print("=" * 78)

        if intervention == "short":
            this_context_hours = (
                recent_hours
            )
        else:
            this_context_hours = (
                long_hours
            )

        n_context_steps = (
            context_steps(
                this_context_hours,
                spec.resolution_minutes,
            )
        )

        case_ids = list(
            prepared.keys()
        )

        for start in range(
            0,
            len(case_ids),
            batch_size,
        ):
            stop = min(
                start + batch_size,
                len(case_ids),
            )

            batch_ids = (
                case_ids[
                    start:stop
                ]
            )

            inputs = []

            for case_id in batch_ids:
                item = prepared[
                    case_id
                ][
                    intervention
                ]

                values = np.asarray(
                    item["values"],
                    dtype=float,
                )

                timestamps = (
                    pd.DatetimeIndex(
                        item[
                            "timestamps"
                        ]
                    )
                )

                if (
                    len(values)
                    != n_context_steps
                ):
                    raise RuntimeError(
                        f"{case_id} / "
                        f"{intervention}: "
                        "unexpected context "
                        f"length {len(values)}."
                    )

                inputs.append(
                    ForecastInput(
                        case_id=case_id,
                        timestamps=timestamps,
                        values=values,
                    )
                )

            outputs = (
                model.predict_batch(
                    inputs=inputs,
                    horizon=(
                        spec.forecast_steps
                    ),
                    resolution_minutes=(
                        spec.resolution_minutes
                    ),
                    context_steps=(
                        n_context_steps
                    ),
                    batch_size=(
                        batch_size
                    ),
                )
            )

            for output in outputs:
                case_id = (
                    output.case_id
                )

                truth = (
                    truth_by_case[
                        case_id
                    ]
                )

                meta = (
                    metadata_df[
                        metadata_df[
                            "case_id"
                        ]
                        == case_id
                    ]
                    .iloc[0]
                )

                for step in range(
                    spec.forecast_steps
                ):
                    rows.append(
                        {
                            "dataset": (
                                dataset_key
                            ),
                            "model": (
                                model_key
                            ),
                            "intervention": (
                                intervention
                            ),
                            "recent_hours": (
                                recent_hours
                            ),
                            "long_hours": (
                                long_hours
                            ),
                            "entity_id": (
                                meta[
                                    "entity_id"
                                ]
                            ),
                            "eval_index": int(
                                meta[
                                    "eval_index"
                                ]
                            ),
                            "eval_origin": (
                                meta[
                                    "eval_origin"
                                ]
                            ),
                            "donor_id": (
                                meta[
                                    "donor_id"
                                ]
                            ),
                            "timestamp": (
                                truth.index[
                                    step
                                ]
                            ),
                            "truth": float(
                                truth.iloc[
                                    step
                                ]
                            ),
                            "prediction": float(
                                output.prediction[
                                    step
                                ]
                            ),
                            "q10": float(
                                output.q10[
                                    step
                                ]
                            ),
                            "q50": float(
                                output.q50[
                                    step
                                ]
                            ),
                            "q90": float(
                                output.q90[
                                    step
                                ]
                            ),
                        }
                    )

            print(
                f"  cases "
                f"{start + 1:03d}-"
                f"{stop:03d}"
            )

    pred = pd.DataFrame(
        rows
    )

    pred, crossing_fraction = (
        add_prediction_metrics(
            pred
        )
    )

    (
        case_metrics,
        entity_metrics,
        summary,
    ) = _aggregate_intervention(
        pred
    )

    output_dir = (
        Path(output_root)
        / dataset_key
        / "interventions"
        / model_key
        / (
            f"recent_{recent_hours:g}h"
            f"_long_{long_hours:g}h"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred.to_parquet(
        output_dir
        / "predictions.parquet",
        index=False,
    )

    case_metrics.to_csv(
        output_dir
        / "case_metrics.csv",
        index=False,
    )

    entity_metrics.to_csv(
        output_dir
        / "entity_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    metadata_df.to_csv(
        output_dir
        / "donor_manifest.csv",
        index=False,
    )

    runtime = (
        time.time()
        - start_time
    )

    print()
    print("=" * 78)
    print("Intervention summary")
    print("=" * 78)
    print()

    display = summary.copy()

    for column in [
        "mean_entity_mae",
        "median_entity_mae",
        "mean_entity_rmse",
        "mean_delta_mae_vs_short",
        "mean_delta_mae_vs_same",
        "fraction_of_same_entity_gain_retained",
        "entity_specific_gap_vs_same",
    ]:
        display[column] = (
            display[column]
            .round(4)
        )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "Quantile crossing: "
        f"{100 * crossing_fraction:.4f}%"
    )

    print(
        "Runtime seconds: "
        f"{runtime:.2f}"
    )

    print()
    print(
        f"Saved to: "
        f"{output_dir}"
    )

    return {
        "predictions": pred,
        "case_metrics": (
            case_metrics
        ),
        "entity_metrics": (
            entity_metrics
        ),
        "summary": summary,
        "donor_manifest": (
            metadata_df
        ),
        "output_dir": (
            output_dir
        ),
    }
