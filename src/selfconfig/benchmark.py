from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.registry import DATASETS

from .dataset_adapter import (
    load_fingerprint_input,
)
from .fingerprint import (
    FingerprintConfig,
    compute_fingerprint_ragged,
)
from .memory import (
    compute_temporal_memory_fingerprint,
)
from .regime import (
    compute_regime_fingerprint,
)
from .capacity import (
    compute_history_capacity,
)
from .planner import (
    ForecastPlan,
    print_plan,
)
from .rules import apply_rules
from .search_space import (
    CandidateConfig,
    build_search_space,
)


BENCHMARK_COLUMNS = [
    "dataset",
    "model",
    "context_steps",
    "context_multiplier",
    "split",
    "mae",
]


def make_empty_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        columns=BENCHMARK_COLUMNS
    )


def expected_grid(
    dataset: str,
    candidates: Iterable[
        CandidateConfig
    ],
    splits=("val", "test"),
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        for split in splits:
            rows.append(
                {
                    "dataset": dataset,
                    "model": (
                        candidate.model
                    ),
                    "context_steps": (
                        candidate.context_steps
                    ),
                    "context_multiplier": (
                        candidate.context_multiplier
                    ),
                    "split": split,
                    "mae": pd.NA,
                }
            )

    return pd.DataFrame(
        rows,
        columns=BENCHMARK_COLUMNS,
    )


def append_scores(
    path: str | Path,
    scores: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        old = pd.read_csv(path)
        scores = pd.concat(
            [old, scores],
            ignore_index=True,
        )

    scores = scores.drop_duplicates(
        subset=[
            "dataset",
            "model",
            "context_steps",
            "split",
        ],
        keep="last",
    )

    scores.sort_values(
        [
            "dataset",
            "model",
            "context_steps",
            "split",
        ]
    ).to_csv(
        path,
        index=False,
    )


def run_fingerprint_audit(
    *,
    dataset_key: str,
    split: str | None,
    output_root: str,
    max_series: int,
    max_points_per_series: int,
) -> ForecastPlan:
    data = load_fingerprint_input(
        dataset_key,
        split=split,
    )

    fp_cfg = FingerprintConfig(
        max_series=max_series,
        max_points_per_series=(
            max_points_per_series
        ),
        seed=42,
    )

    fingerprint = (
        compute_fingerprint_ragged(
            data.histories,
            dataset_name=dataset_key,
            sampling_interval_seconds=(
                data.resolution_minutes
                * 60.0
            ),
            forecast_horizon_steps=(
                data.forecast_steps
            ),
            n_variables=1,
            config=fp_cfg,
        )
    )

    memory = (
        compute_temporal_memory_fingerprint(
            data.histories,
            horizon_steps=(
                data.forecast_steps
            ),
            sampling_interval_seconds=(
                data.resolution_minutes
                * 60.0
            ),
            dominant_period_steps=(
                fingerprint.dominant_period_steps
            ),
            max_series=max_series,
            max_points_per_series=max(
                max_points_per_series,
                32768,
            ),
            seed=42,
        )
    )

    capacity = (
        compute_history_capacity(
            data.histories,
            horizon_steps=(
                data.forecast_steps
            ),
            support_quantile=0.25,
            resource_limit_steps=4096,
        )
    )

    regime = (
        compute_regime_fingerprint(
            data.histories,
            horizon_steps=(
                data.forecast_steps
            ),
            sampling_interval_seconds=(
                data.resolution_minutes
                * 60.0
            ),
            max_series=max_series,
            seed=42,
        )
    )

    full_candidates = (
        build_search_space(
            horizon_steps=(
                data.forecast_steps
            ),
            fingerprint=fingerprint,
            memory=memory,
            history_limit_steps=(
                capacity.effective_limit_steps
            ),
        )
    )

    rule_result = apply_rules(
        fingerprint,
        full_candidates,
        memory=memory,
        regime=regime,
        history_limit_steps=(
            capacity.effective_limit_steps
        ),
    )

    plan = ForecastPlan(
        dataset_name=dataset_key,
        fingerprint=fingerprint,
        full_candidates=tuple(
            full_candidates
        ),
        proposed_candidates=(
            rule_result.selected
        ),
        rationale=(
            rule_result.rationale
        ),
    )

    print("=" * 78)
    print(
        "PersTSFM self-configuration "
        "fingerprint audit"
    )
    print("=" * 78)
    print(
        f"Dataset:              "
        f"{dataset_key}"
    )
    print(
        f"Split:                "
        f"{data.split}"
    )
    print(
        f"Protocol entities:    "
        f"{data.n_protocol_entities}"
    )
    print(
        f"Histories extracted:  "
        f"{len(data.histories)}"
    )
    print(
        "History boundary:     "
        f"{data.cutoff_method}"
    )
    print(
        "Leakage policy:       "
        "only observations before each "
        "entity's first evaluation origin"
    )
    print()
    print("History capacity:")
    print(
        f"  p25 leakage-safe history: "
        f"{capacity.p25_steps} steps "
        f"({capacity.p25_steps / data.forecast_steps:.1f}×H)"
    )
    print(
        f"  Fixed resource ceiling:   "
        f"{capacity.resource_limit_steps} steps"
    )
    print(
        f"  Effective context limit:  "
        f"{capacity.effective_limit_steps} steps "
        f"({capacity.effective_limit_horizons:.1f}×H)"
    )
    print()
    print("Temporal memory:")
    print(f"  ACF(H):              {memory.acf_h:.3f}")
    print(f"  ACF(2H):             {memory.acf_2h:.3f}")
    print(f"  ACF(4H):             {memory.acf_4h:.3f}")
    print(f"  ACF(8H):             {memory.acf_8h:.3f}")
    print(f"  ACF(16H):            {memory.acf_16h:.3f}")
    print(f"  ACF(32H):            {memory.acf_32h:.3f}")
    print(
        "  ACF(P):              "
        + (
            f"{memory.acf_period:.3f}"
            if memory.acf_period is not None
            else "n/a"
        )
    )
    print(
        "  ACF(2P):             "
        + (
            f"{memory.acf_2period:.3f}"
            if memory.acf_2period is not None
            else "n/a"
        )
    )
    print(
        "  Persistence |ACF|>=.2: "
        + (
            f"{memory.memory_lag_02_steps} steps"
            if memory.memory_lag_02_steps is not None
            else "not reached"
        )
    )
    print(
        "  Persistence |ACF|>=.1: "
        + (
            f"{memory.memory_lag_01_steps} steps"
            if memory.memory_lag_01_steps is not None
            else "not reached"
        )
    )
    print()
    print("Regime / profile structure:")
    print(f"  Recent-vs-prior mean shift:     {regime.recent_vs_prior_mean_shift:.3f}")
    print(f"  Recent-vs-prior scale shift:    {regime.recent_vs_prior_scale_shift:.3f}")
    print(f"  Recent-vs-prior IQR shift:      {regime.recent_vs_prior_iqr_shift:.3f}")
    print(f"  Recent-vs-prior quantile shift: {regime.recent_vs_prior_quantile_shift:.3f}")
    print(f"  Rolling mean variability:       {regime.rolling_mean_variability:.3f}")
    print(f"  Rolling scale variability:      {regime.rolling_scale_variability:.3f}")
    print(f"  Adjacent mean shift:            {regime.adjacent_mean_shift:.3f}")
    print(f"  Window-mean persistence:        {regime.window_mean_persistence:.3f}")
    print(f"  Usable series fraction:         {regime.usable_series_fraction:.3f}")
    print(f"  Median profile windows:         {regime.median_windows_per_series:.1f}")
    print()

    print_plan(plan)

    fp_path, plan_path = plan.write(
        output_root=output_root
    )

    memory_path = (
        Path(output_root)
        / "fingerprints"
        / f"{dataset_key}_memory.json"
    )
    memory_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

    memory_path.write_text(
        json.dumps(
            memory.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    capacity_path = (
        Path(output_root)
        / "fingerprints"
        / f"{dataset_key}_capacity.json"
    )

    capacity_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capacity_path.write_text(
        json.dumps(
            capacity.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    regime_path = (
        Path(output_root)
        / "fingerprints"
        / f"{dataset_key}_regime.json"
    )
    regime_path.parent.mkdir(parents=True, exist_ok=True)
    regime_path.write_text(
        json.dumps(
            regime.to_dict(),
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print()
    print("Outputs:")
    print(
        f"  Fingerprint: {fp_path}"
    )
    print(
        f"  Plan:        {plan_path}"
    )
    print(
        f"  Memory:      {memory_path}"
    )
    print(
        f"  Regime:      {regime_path}"
    )
    print(
        f"  Capacity:    {capacity_path}"
    )

    return plan


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "PersTSFM nnU-Net-style "
            "self-configuration benchmark."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
    )
    parser.add_argument(
        "--split",
        default=None,
    )
    parser.add_argument(
        "--fingerprint-only",
        action="store_true",
        help=(
            "Compute the leakage-safe dataset "
            "fingerprint and rule-based plan."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="results/selfconfig",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max-points-per-series",
        type=int,
        default=8192,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.fingerprint_only:
        raise SystemExit(
            "Phase 1 currently implements "
            "--fingerprint-only. "
            "Execution of candidate TSFM "
            "configurations is the next step."
        )

    run_fingerprint_audit(
        dataset_key=args.dataset,
        split=args.split,
        output_root=args.output_root,
        max_series=args.max_series,
        max_points_per_series=(
            args.max_points_per_series
        ),
    )


if __name__ == "__main__":
    main()
