from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from .fingerprint import DatasetFingerprint, FingerprintConfig, compute_fingerprint
from .rules import RuleConfig, apply_rules
from .search_space import CandidateConfig, build_search_space


@dataclass(frozen=True)
class ForecastPlan:
    dataset_name: str
    fingerprint: DatasetFingerprint
    full_candidates: tuple[CandidateConfig, ...]
    proposed_candidates: tuple[CandidateConfig, ...]
    rationale: tuple[str, ...]

    @property
    def search_reduction(self) -> float:
        if not self.full_candidates:
            return 0.0
        return 1.0 - (len(self.proposed_candidates) / len(self.full_candidates))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "fingerprint": self.fingerprint.to_dict(),
            "full_candidates": [c.to_dict() for c in self.full_candidates],
            "proposed_candidates": [c.to_dict() for c in self.proposed_candidates],
            "rationale": list(self.rationale),
            "search_reduction": self.search_reduction,
        }

    def write(self, output_root: str | Path = "results/selfconfig") -> tuple[Path, Path]:
        root = Path(output_root)
        fp_path = root / "fingerprints" / f"{self.dataset_name}.json"
        plan_path = root / "plans" / f"{self.dataset_name}.json"
        self.fingerprint.to_json(fp_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return fp_path, plan_path


def build_plan(
    values: Any,
    *,
    dataset_name: str,
    sampling_interval_seconds: float,
    forecast_horizon_steps: int,
    n_variables: int = 1,
    models: tuple[str, ...] = (
        "chronos2",
        "timesfm3",
        "tirex2",
    ),
    fingerprint_config: FingerprintConfig | None = None,
    rule_config: RuleConfig | None = None,
) -> ForecastPlan:

    # ---------------------------------------------------------
    # 1. Dataset fingerprint
    # ---------------------------------------------------------

    fp = compute_fingerprint(
        values,
        dataset_name=dataset_name,
        sampling_interval_seconds=(
            sampling_interval_seconds
        ),
        forecast_horizon_steps=(
            forecast_horizon_steps
        ),
        n_variables=n_variables,
        config=fingerprint_config,
    )

    # ---------------------------------------------------------
    # 2. Fingerprint-aware feasible search space
    # ---------------------------------------------------------

    full = build_search_space(
        horizon_steps=(
            forecast_horizon_steps
        ),
        fingerprint=fp,
        models=models,
    )

    # ---------------------------------------------------------
    # 3. nnU-Net-style rule-based pruning
    # ---------------------------------------------------------

    result = apply_rules(
        fp,
        full,
        config=rule_config,
    )

    # ---------------------------------------------------------
    # 4. Final plan
    # ---------------------------------------------------------

    return ForecastPlan(
        dataset_name=dataset_name,
        fingerprint=fp,
        full_candidates=tuple(full),
        proposed_candidates=(
            result.selected
        ),
        rationale=(
            result.rationale
        ),
    )

def print_plan(plan: ForecastPlan) -> None:
    fp = plan.fingerprint
    print(f"=== SELF-CONFIGURATION PLAN: {plan.dataset_name} ===")
    print(f"Series:                  {fp.n_series}")
    print(f"Horizon:                 {fp.forecast_horizon_steps} steps")
    print(f"Sampling interval:       {fp.sampling_interval_seconds:g} s")
    print(f"Missing fraction:        {fp.missing_fraction:.4f}")
    print(f"Lag-1 ACF:               {fp.acf_lag1:.3f}")
    print(f"Horizon ACF:             {fp.acf_horizon:.3f}")
    if fp.dominant_period_steps is None:
        print("Dominant period:          not reliably detected")
    else:
        print(f"Dominant period:          {fp.dominant_period_steps:.1f} steps")
    print(f"Seasonality strength:    {fp.seasonality_strength:.3f}")
    print(f"Spectral entropy:        {fp.spectral_entropy:.3f}")
    print()
    print(f"Full candidates:         {len(plan.full_candidates)}")
    print(f"Proposed candidates:     {len(plan.proposed_candidates)}")
    print(f"Search reduction:        {100.0 * plan.search_reduction:.1f}%")
    print("\nProposed:")
    for c in plan.proposed_candidates:
        print(f"  {c.model:10s} context={c.context_steps:5d} steps ({c.context_multiplier:g}×H)")
    print("\nRationale:")
    for r in plan.rationale:
        print(f"  - {r}")
