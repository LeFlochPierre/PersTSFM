from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.registry import (
    DATASETS,
    get_dataset_spec,
)


MODELS = (
    "chronos2",
    "timesfm3",
    "tirex2",
)


def context_steps_to_hours(
    context_steps: int,
    resolution_minutes: float,
) -> float:
    return (
        float(context_steps)
        * float(resolution_minutes)
        / 60.0
    )


def load_plan(
    dataset_key: str,
    *,
    plan_root: str | Path = (
        "results/selfconfig/plans"
    ),
) -> dict:
    path = (
        Path(plan_root)
        / f"{dataset_key}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen self-config plan: {path}"
        )

    plan = json.loads(
        path.read_text()
    )

    if plan.get("dataset_name") != dataset_key:
        raise RuntimeError(
            "Plan/dataset mismatch: "
            f"requested={dataset_key!r}, "
            f"plan={plan.get('dataset_name')!r}"
        )

    return plan


def candidates_for_mode(
    plan: dict,
    mode: str,
) -> list[dict]:
    if mode == "proposed":
        key = "proposed_candidates"
    elif mode == "full":
        key = "full_candidates"
    else:
        raise ValueError(
            f"Unknown execution mode: {mode}"
        )

    candidates = list(
        plan.get(key, [])
    )

    if not candidates:
        raise RuntimeError(
            f"No {mode} candidates in plan."
        )

    return candidates


def group_candidates_by_model(
    candidates: Iterable[dict],
    *,
    models: Iterable[str] = MODELS,
) -> dict[str, list[int]]:
    allowed = set(models)

    grouped: dict[
        str,
        list[int],
    ] = {}

    for candidate in candidates:
        model = str(
            candidate["model"]
        )

        if model not in allowed:
            continue

        steps = int(
            candidate["context_steps"]
        )

        grouped.setdefault(
            model,
            [],
        ).append(
            steps
        )

    for model in grouped:
        grouped[model] = sorted(
            set(
                grouped[model]
            )
        )

    return grouped


def add_reference_context_for_legacy_runner(
    requested_steps: Iterable[int],
    *,
    reference_context_hours: float,
    resolution_minutes: float,
) -> tuple[
    list[int],
    int,
    bool,
]:
    requested = sorted(
        set(
            int(x)
            for x in requested_steps
        )
    )

    reference_steps_float = (
        float(reference_context_hours)
        * 60.0
        / float(resolution_minutes)
    )

    reference_steps = int(
        round(
            reference_steps_float
        )
    )

    if abs(
        reference_steps_float
        - reference_steps
    ) > 1e-8:
        raise RuntimeError(
            "Legacy reference context does not "
            "map to an integer number of steps."
        )

    added = (
        reference_steps
        not in requested
    )

    executed = list(
        requested
    )

    if added:
        executed.append(
            reference_steps
        )
        executed.sort()

    return (
        executed,
        reference_steps,
        added,
    )


def _summary_path(
    execution_root: Path,
    dataset_key: str,
    model_key: str,
) -> Path:
    return (
        execution_root
        / dataset_key
        / "context"
        / model_key
        / "summary.csv"
    )


def _read_requested_summary(
    summary_path: Path,
    *,
    dataset_key: str,
    model_key: str,
    mode: str,
    requested_steps: list[int],
    auxiliary_reference_steps: int,
    auxiliary_reference_added: bool,
) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Context runner did not create: "
            f"{summary_path}"
        )

    summary = pd.read_csv(
        summary_path
    )

    required = {
        "context_steps",
        "mean_entity_mae",
    }

    missing = (
        required
        - set(summary.columns)
    )

    if missing:
        raise RuntimeError(
            "Unexpected context summary schema; "
            f"missing columns: {sorted(missing)}"
        )

    summary[
        "context_steps"
    ] = pd.to_numeric(
        summary["context_steps"],
        errors="raise",
    ).astype(int)

    wanted = set(
        int(x)
        for x in requested_steps
    )

    result = (
        summary[
            summary[
                "context_steps"
            ].isin(
                wanted
            )
        ]
        .copy()
        .sort_values(
            "context_steps"
        )
        .reset_index(
            drop=True
        )
    )

    found = set(
        result[
            "context_steps"
        ].tolist()
    )

    if found != wanted:
        raise RuntimeError(
            f"{dataset_key}/{model_key}: "
            "summary does not contain exactly the "
            "requested candidate contexts. "
            f"requested={sorted(wanted)}, "
            f"found={sorted(found)}"
        )

    result.insert(
        0,
        "dataset",
        dataset_key,
    )

    result.insert(
        1,
        "model",
        model_key,
    )

    result.insert(
        2,
        "search_mode",
        mode,
    )

    result[
        "plan_candidate"
    ] = True

    result[
        "auxiliary_reference_added"
    ] = bool(
        auxiliary_reference_added
    )

    result[
        "auxiliary_reference_steps"
    ] = int(
        auxiliary_reference_steps
    )

    return result


def print_execution_plan(
    dataset_key: str,
    *,
    mode: str,
    grouped: dict[
        str,
        list[int],
    ],
    resolution_minutes: float,
    reference_context_hours: float,
) -> None:
    print("=" * 78)
    print(
        "PersTSFM frozen self-configuration "
        "execution plan"
    )
    print("=" * 78)
    print()
    print(
        f"Dataset:       {dataset_key}"
    )
    print(
        f"Search mode:   {mode}"
    )
    print(
        f"Resolution:    "
        f"{resolution_minutes:g} min"
    )
    print()

    for model in MODELS:
        if model not in grouped:
            continue

        requested = grouped[
            model
        ]

        (
            executed,
            reference_steps,
            reference_added,
        ) = (
            add_reference_context_for_legacy_runner(
                requested,
                reference_context_hours=(
                    reference_context_hours
                ),
                resolution_minutes=(
                    resolution_minutes
                ),
            )
        )

        requested_text = ", ".join(
            f"{x} steps "
            f"({context_steps_to_hours(x, resolution_minutes):g} h)"
            for x in requested
        )

        print(
            f"{model:10s} requested: "
            f"{requested_text}"
        )

        if reference_added:
            print(
                f"{'':10s} auxiliary legacy "
                f"reference: {reference_steps} steps "
                f"({reference_context_hours:g} h)"
            )

        print(
            f"{'':10s} actual runner count: "
            f"{len(executed)}"
        )
        print()


def execute_frozen_plan(
    dataset_key: str,
    *,
    mode: str = "proposed",
    models: Iterable[str] = MODELS,
    max_cases: int | None = None,
    plan_root: str | Path = (
        "results/selfconfig/plans"
    ),
    output_root: str | Path = (
        "results/selfconfig/execution"
    ),
    matrix_root: str | Path = (
        "results/selfconfig/benchmark_matrix"
    ),
    dry_run: bool = False,
) -> pd.DataFrame | None:
    if dataset_key not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_key!r}. "
            f"Available: {sorted(DATASETS)}"
        )

    requested_models = tuple(
        models
    )

    unknown_models = (
        set(requested_models)
        - set(MODELS)
    )

    if unknown_models:
        raise ValueError(
            "Unknown models: "
            f"{sorted(unknown_models)}"
        )

    spec = get_dataset_spec(
        dataset_key
    )

    plan = load_plan(
        dataset_key,
        plan_root=plan_root,
    )

    candidates = candidates_for_mode(
        plan,
        mode,
    )

    grouped = group_candidates_by_model(
        candidates,
        models=requested_models,
    )

    if not grouped:
        raise RuntimeError(
            "No candidates remain after model filtering."
        )

    print_execution_plan(
        dataset_key,
        mode=mode,
        grouped=grouped,
        resolution_minutes=(
            spec.resolution_minutes
        ),
        reference_context_hours=(
            spec.reference_context_hours
        ),
    )

    if dry_run:
        print(
            "DRY RUN: no forecasting executed."
        )
        return None

    # Import only for real execution, keeping dry-run/tests light.
    from src.evaluation.context_curve import (
        run_context_curve,
    )

    execution_root = (
        Path(output_root)
        / mode
    )

    matrix_dir = (
        Path(matrix_root)
        / mode
    )

    matrix_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_parts = []
    manifest_models = {}

    for model in MODELS:
        if model not in grouped:
            continue

        requested_steps = grouped[
            model
        ]

        (
            executed_steps,
            reference_steps,
            reference_added,
        ) = (
            add_reference_context_for_legacy_runner(
                requested_steps,
                reference_context_hours=(
                    spec.reference_context_hours
                ),
                resolution_minutes=(
                    spec.resolution_minutes
                ),
            )
        )

        executed_hours = [
            context_steps_to_hours(
                steps,
                spec.resolution_minutes,
            )
            for steps in executed_steps
        ]

        print()
        print("#" * 78)
        print(
            f"EXECUTING: {dataset_key} / "
            f"{model} / {mode}"
        )
        print("#" * 78)

        run_context_curve(
            dataset_key=dataset_key,
            model_key=model,
            split=None,
            context_hours=executed_hours,
            max_cases=max_cases,
            output_root=str(
                execution_root
            ),
        )

        summary_path = _summary_path(
            execution_root,
            dataset_key,
            model,
        )

        part = _read_requested_summary(
            summary_path,
            dataset_key=dataset_key,
            model_key=model,
            mode=mode,
            requested_steps=(
                requested_steps
            ),
            auxiliary_reference_steps=(
                reference_steps
            ),
            auxiliary_reference_added=(
                reference_added
            ),
        )

        matrix_parts.append(
            part
        )

        manifest_models[
            model
        ] = {
            "requested_context_steps": (
                requested_steps
            ),
            "executed_context_steps": (
                executed_steps
            ),
            "auxiliary_reference_steps": (
                reference_steps
            ),
            "auxiliary_reference_added": (
                reference_added
            ),
            "n_requested": len(
                requested_steps
            ),
            "n_executed": len(
                executed_steps
            ),
        }

    matrix = pd.concat(
        matrix_parts,
        ignore_index=True,
    )

    matrix = matrix.sort_values(
        [
            "model",
            "context_steps",
        ]
    ).reset_index(
        drop=True
    )

    matrix_path = (
        matrix_dir
        / f"{dataset_key}.csv"
    )

    matrix.to_csv(
        matrix_path,
        index=False,
    )

    manifest = {
        "dataset": dataset_key,
        "mode": mode,
        "max_cases": max_cases,
        "resolution_minutes": (
            spec.resolution_minutes
        ),
        "forecast_steps": (
            spec.forecast_steps
        ),
        "plan_path": str(
            Path(plan_root)
            / f"{dataset_key}.json"
        ),
        "matrix_path": str(
            matrix_path
        ),
        "legacy_reference_note": (
            "The existing context-curve aggregation "
            "requires its historical reference context. "
            "When absent from the frozen candidate set, "
            "that reference is executed only as an "
            "auxiliary compatibility evaluation and is "
            "excluded from the self-config benchmark "
            "matrix. Final compute-cost experiments must "
            "account for or remove this overhead."
        ),
        "models": manifest_models,
    }

    manifest_path = (
        matrix_dir
        / f"{dataset_key}_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 78)
    print(
        "SELF-CONFIG EXECUTION COMPLETE"
    )
    print("=" * 78)
    print(
        f"Benchmark matrix: {matrix_path}"
    )
    print(
        f"Manifest:         {manifest_path}"
    )
    print(
        f"Rows:             {len(matrix)}"
    )

    print()
    display_cols = [
        c
        for c in [
            "dataset",
            "model",
            "context_steps",
            "context_hours",
            "mean_entity_mae",
            "mean_entity_rmse",
            "mean_case_mae",
        ]
        if c in matrix.columns
    ]

    print(
        matrix[
            display_cols
        ].to_string(
            index=False
        )
    )

    return matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Execute a frozen PersTSFM "
            "self-configuration plan using the "
            "existing unified context-curve runner."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(
            DATASETS
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "proposed",
            "full",
        ],
        default="proposed",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(
            MODELS
        ),
        default=list(
            MODELS
        ),
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--plan-root",
        default=(
            "results/selfconfig/plans"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "results/selfconfig/execution"
        ),
    )

    parser.add_argument(
        "--matrix-root",
        default=(
            "results/selfconfig/benchmark_matrix"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    execute_frozen_plan(
        dataset_key=args.dataset,
        mode=args.mode,
        models=args.models,
        max_cases=args.max_cases,
        plan_root=args.plan_root,
        output_root=args.output_root,
        matrix_root=args.matrix_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
