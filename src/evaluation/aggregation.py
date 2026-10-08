import numpy as np
import pandas as pd


def aggregate_context_results(
    pred,
    reference_context_hours,
):
    # -------------------------------------------------------------
    # Forecast-case level
    # -------------------------------------------------------------

    case_metrics = (
        pred
        .groupby(
            [
                "context_hours",
                "context_label",
                "context_steps",
                "entity_id",
                "eval_index",
                "eval_origin",
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
            negative_fraction=(
                "negative_prediction",
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

    # -------------------------------------------------------------
    # Frozen short-context reference
    # -------------------------------------------------------------

    reference = (
        case_metrics[
            case_metrics[
                "context_hours"
            ]
            == reference_context_hours
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
                "mae":
                "reference_mae",
            }
        )
    )

    if len(reference) == 0:
        raise RuntimeError(
            "Reference context "
            f"{reference_context_hours} h "
            "was not evaluated."
        )

    case_metrics = (
        case_metrics.merge(
            reference,
            on=[
                "entity_id",
                "eval_index",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    case_metrics[
        "delta_mae_vs_reference"
    ] = (
        case_metrics["mae"]
        - case_metrics["reference_mae"]
    )

    # -------------------------------------------------------------
    # Entity / participant level
    # -------------------------------------------------------------

    entity_metrics = (
        case_metrics
        .groupby(
            [
                "context_hours",
                "context_label",
                "context_steps",
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
            mean_interval_width_80=(
                "interval_width_80",
                "mean",
            ),
            mean_delta_mae_vs_reference=(
                "delta_mae_vs_reference",
                "mean",
            ),
        )
    )

    # -------------------------------------------------------------
    # Cohort level
    #
    # Entity first, cohort second.
    # -------------------------------------------------------------

    rows = []

    contexts = (
        entity_metrics[
            "context_hours"
        ]
        .drop_duplicates()
        .sort_values()
    )

    for context_hours in contexts:

        entities = entity_metrics[
            entity_metrics[
                "context_hours"
            ]
            == context_hours
        ]

        cases = case_metrics[
            case_metrics[
                "context_hours"
            ]
            == context_hours
        ]

        first = entities.iloc[0]

        rows.append(
            {
                "context_hours": float(
                    context_hours
                ),
                "context_label": first[
                    "context_label"
                ],
                "context_steps": int(
                    first[
                        "context_steps"
                    ]
                ),
                "n_entities": int(
                    len(entities)
                ),
                "n_cases": int(
                    len(cases)
                ),
                "mean_entity_mae": float(
                    entities[
                        "mean_mae"
                    ].mean()
                ),
                "median_entity_mae": float(
                    entities[
                        "mean_mae"
                    ].median()
                ),
                "mean_entity_rmse": float(
                    entities[
                        "mean_rmse"
                    ].mean()
                ),
                "mean_coverage_80": float(
                    entities[
                        "mean_coverage_80"
                    ].mean()
                ),
                "mean_interval_width_80": float(
                    entities[
                        "mean_interval_width_80"
                    ].mean()
                ),
                "mean_delta_mae_vs_reference": float(
                    entities[
                        "mean_delta_mae_vs_reference"
                    ].mean()
                ),
                "entity_better_than_reference_fraction": float(
                    (
                        entities[
                            "mean_delta_mae_vs_reference"
                        ]
                        < 0
                    ).mean()
                ),
                "mean_case_mae": float(
                    cases[
                        "mae"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return (
        case_metrics,
        entity_metrics,
        summary,
    )
