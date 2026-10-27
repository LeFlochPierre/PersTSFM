from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data.registry import (
    context_label,
    extract_case,
    load_dataset,
)

from src.evaluation.aggregation import (
    aggregate_context_results,
)

from src.evaluation.metrics import (
    add_prediction_metrics,
)

from src.models.adapters import (
    ForecastInput,
    build_model,
)


def run_context_curve(
    dataset_key,
    model_key,
    split=None,
    context_hours=None,
    max_cases=None,
    sample_entities=None,
    origins_per_entity=None,
    sample_seed=42,
    output_root="results",
):
    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    dataset = load_dataset(
        dataset_key,
        split=split,
    )

    spec = dataset.spec

    protocol = dataset.protocol.copy()

    # Phase-2 balanced sampling: identical entities/origins across models.
    if sample_entities is not None or origins_per_entity is not None:
        if max_cases is not None:
            raise ValueError("Use either balanced sampling or --max-cases, not both")
        from src.evaluation.case_sampling import balanced_subset
        protocol = balanced_subset(
            protocol,
            id_col=spec.id_col,
            n_entities=sample_entities,
            origins_per_entity=origins_per_entity,
            seed=sample_seed,
        )

    if max_cases is not None:
        protocol = (
            protocol
            .iloc[
                :max_cases
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    if context_hours is None:
        contexts = list(
            spec.context_hours
        )
    else:
        contexts = [
            float(x)
            for x in context_hours
        ]

    if (
        spec.reference_context_hours
        not in contexts
    ):
        raise RuntimeError(
            "The reference context "
            f"{spec.reference_context_hours} h "
            "must be included."
        )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    if model_key == "chronos2":
        batch_size = (
            spec.chronos_batch_size
        )

    elif model_key == "timesfm3":
        batch_size = (
            spec.timesfm_batch_size
        )

    elif model_key == "tirex2":
        batch_size = 8

    elif model_key in ("toto2_22m", "toto2_313m"):
        batch_size = 8

    elif model_key == "moirai2_small":
        batch_size = 8

    else:
        raise ValueError(
            f"Unknown model: {model_key}"
        )

    model = build_model(
        model_key,
        batch_size=batch_size,
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    split_name = (
        split
        if split is not None
        else spec.default_split
    )

    output_dir = (
        Path(output_root)
        / dataset_key
        / "context"
        / model_key
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Freeze the evaluation-case identity for cross-model comparisons.
    protocol[[spec.id_col, "eval_index", "eval_origin", "target_start", "target_end"]].to_csv(
        output_dir / "case_selection.csv", index=False
    )

    print("=" * 78)
    print("PersTSFM unified context-curve runner")
    print("=" * 78)

    print()
    print(
        f"Dataset:       {dataset_key}"
    )

    print(
        f"Model:         {model_key}"
    )

    print(
        f"Split:         {split_name}"
    )

    print(
        f"Entities:      "
        f"{protocol[spec.id_col].nunique()}"
    )

    print(
        f"Cases:         {len(protocol)}"
    )

    print(
        f"Batch size:    {batch_size}"
    )

    print(
        "Contexts:      "
        + ", ".join(
            context_label(x)
            for x in contexts
        )
    )

    # -------------------------------------------------------------
    # Forecast
    # -------------------------------------------------------------

    rows = []

    total_start = time.time()

    for context_hours_value in contexts:

        first_case = extract_case(
            dataset,
            protocol.iloc[0],
            context_hours_value,
        )

        n_context_steps = (
            first_case[
                "context_steps"
            ]
        )

        print()
        print("-" * 78)

        print(
            "Context: "
            f"{context_label(context_hours_value)} "
            f"({n_context_steps} steps)"
        )

        print("-" * 78)

        for batch_number, start in enumerate(
            range(
                0,
                len(protocol),
                batch_size,
            ),
            start=1,
        ):
            end = min(
                start + batch_size,
                len(protocol),
            )

            batch_protocol = (
                protocol.iloc[
                    start:end
                ]
            )

            inputs = []
            truth_by_case = {}
            metadata_by_case = {}

            for _, row in (
                batch_protocol.iterrows()
            ):
                case = extract_case(
                    dataset,
                    row,
                    context_hours_value,
                )

                context = case[
                    "context"
                ]

                truth = case[
                    "truth"
                ]

                inputs.append(
                    ForecastInput(
                        case_id=(
                            case[
                                "case_id"
                            ]
                        ),
                        timestamps=(
                            pd.DatetimeIndex(
                                context.index
                            )
                        ),
                        values=(
                            context.to_numpy(
                                dtype=float
                            )
                        ),
                    )
                )

                truth_by_case[
                    case["case_id"]
                ] = truth

                metadata_by_case[
                    case["case_id"]
                ] = {
                    "entity_id": (
                        case[
                            "entity_id"
                        ]
                    ),
                    "eval_index": (
                        case[
                            "eval_index"
                        ]
                    ),
                    "eval_origin": (
                        case[
                            "eval_origin"
                        ]
                    ),
                }

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

            if (
                len(outputs)
                != len(inputs)
            ):
                raise RuntimeError(
                    "Model output count "
                    "does not match input count."
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
                    metadata_by_case[
                        case_id
                    ]
                )

                if (
                    len(output.prediction)
                    != len(truth)
                ):
                    raise RuntimeError(
                        f"{case_id}: "
                        "prediction/target "
                        "length mismatch."
                    )

                for step in range(
                    len(truth)
                ):
                    rows.append(
                        {
                            "dataset": (
                                dataset_key
                            ),
                            "model": (
                                model_key
                            ),
                            "split": (
                                split_name
                            ),
                            "context_hours": float(
                                context_hours_value
                            ),
                            "context_label": (
                                context_label(
                                    context_hours_value
                                )
                            ),
                            "context_steps": int(
                                n_context_steps
                            ),
                            "entity_id": (
                                meta[
                                    "entity_id"
                                ]
                            ),
                            "eval_index": (
                                meta[
                                    "eval_index"
                                ]
                            ),
                            "eval_origin": (
                                meta[
                                    "eval_origin"
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
                f"  batch "
                f"{batch_number:02d} | "
                f"cases "
                f"{start + 1:03d}-"
                f"{end:03d}"
            )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

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
    ) = aggregate_context_results(
        pred,
        reference_context_hours=(
            spec.reference_context_hours
        ),
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    pred_path = (
        output_dir
        / "predictions.parquet"
    )

    case_path = (
        output_dir
        / "case_metrics.csv"
    )

    entity_path = (
        output_dir
        / "entity_metrics.csv"
    )

    summary_path = (
        output_dir
        / "summary.csv"
    )

    pred.to_parquet(
        pred_path,
        index=False,
    )

    case_metrics.to_csv(
        case_path,
        index=False,
    )

    entity_metrics.to_csv(
        entity_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Optional Phase-2 per-batch timing audit
    # -------------------------------------------------------------
    if getattr(model, "batch_timings", None):
        pd.DataFrame(model.batch_timings).to_csv(
            output_dir / "runtime_batches.csv", index=False
        )
        print("Model checkpoint:", model.model_id)
        print("Model load seconds:", round(model.load_seconds, 3))

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    runtime = (
        time.time()
        - total_start
    )

    best = (
        summary
        .sort_values(
            "mean_entity_mae"
        )
        .iloc[0]
    )

    print()
    print("=" * 78)
    print("Context summary")
    print("=" * 78)

    display = summary.copy()

    numeric_cols = [
        "mean_entity_mae",
        "median_entity_mae",
        "mean_entity_rmse",
        "mean_coverage_80",
        "mean_interval_width_80",
        "mean_delta_mae_vs_reference",
        "mean_case_mae",
    ]

    for col in numeric_cols:
        display[col] = (
            display[col]
            .round(4)
        )

    display[
        "entity_better_than_reference_fraction"
    ] = (
        100.0
        * display[
            "entity_better_than_reference_fraction"
        ]
    ).round(1)

    print()
    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "Best context: "
        f"{best['context_label']}"
    )

    print(
        "Best mean entity MAE: "
        f"{best['mean_entity_mae']:.4f}"
    )

    print(
        "Quantile crossing: "
        f"{100 * crossing_fraction:.4f}%"
    )

    print(
        "Runtime seconds: "
        f"{runtime:.2f}"
    )

    print()
    print("Saved:")
    print(f"  {pred_path}")
    print(f"  {case_path}")
    print(f"  {entity_path}")
    print(f"  {summary_path}")

    return {
        "predictions": pred,
        "case_metrics": (
            case_metrics
        ),
        "entity_metrics": (
            entity_metrics
        ),
        "summary": summary,
        "output_dir": output_dir,
    }
