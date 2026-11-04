"""Phase B/C: cost-feasible static experts versus an explicitly hindsight oracle."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .cache import Case


def mae(case: Case, action_id: str) -> float:
    pred = case.actions[action_id].prediction
    return sum(abs(y - p) for y, p in zip(case.truth, pred)) / len(case.truth)


def _macro(values: Iterable[tuple[str, float]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for dataset, value in values:
        grouped[dataset].append(value)
    if not grouped:
        raise ValueError("No observations")
    return statistics.mean(statistics.mean(v) for v in grouped.values())


@dataclass(frozen=True)
class Decision:
    budget_ms: float
    strategy: str
    dataset_id: str
    series_id: str
    case_id: str
    action_id: str
    mae: float
    latency_ms: float
    exceeds_budget: bool


def run_feasibility(cases: list[Case], budgets_ms: Iterable[float]) -> tuple[dict, list[Decision]]:
    """Select fixed actions on calibration; evaluate only on evaluation.

    Eligibility uses the *calibration median* latency of an action, not unknown
    evaluation runtime. Runtime overruns are measured, not hidden. The oracle
    uses evaluation truth to pick its action and is NEVER a deployable policy.
    """
    calibration = [c for c in cases if c.split == "calibration"]
    evaluation = [c for c in cases if c.split == "evaluation"]
    if not calibration or not evaluation:
        raise ValueError("Calibration and evaluation cases are both required")
    action_ids = sorted(calibration[0].actions)
    calibration_cost = {
        action: statistics.median(c.actions[action].latency_ms for c in calibration)
        for action in action_ids
    }
    calibration_loss = {
        action: _macro((c.dataset_id, mae(c, action)) for c in calibration)
        for action in action_ids
    }
    report = {
        "schema_version": "direction_a_feasibility_v1",
        "status": "offline_feasibility_only",
        "cost_metric": "measured_per_origin_latency_ms",
        "calibration_cases": len(calibration),
        "evaluation_cases": len(evaluation),
        "calibration_latency_median_ms": calibration_cost,
        "calibration_dataset_macro_mae": calibration_loss,
        "interpretation": (
            "Hindsight oracle uses evaluation truths and is not deployable. "
            "Action feasibility is estimated using calibration median latency; "
            "actual runtime may exceed budgets. No sequential policy is evaluated."
        ),
        "budgets": [],
    }
    decisions: list[Decision] = []
    for budget in budgets_ms:
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("All budgets must be finite, nonnegative milliseconds")
        eligible = [a for a in action_ids if calibration_cost[a] <= budget]
        if not eligible:
            raise ValueError(
                f"Budget {budget}ms too small. Minimum observed calibration median "
                f"is {min(calibration_cost.values()):.4f}ms."
            )
        cheapest = min(eligible, key=lambda a: (calibration_cost[a], a))
        selected = min(eligible, key=lambda a: (calibration_loss[a], calibration_cost[a], a))
        choices = {
            "cheapest": lambda c: cheapest,
            "validation_selected": lambda c: selected,
            "hindsight_oracle": lambda c: min(eligible, key=lambda a: (mae(c, a), calibration_cost[a], a)),
        }
        budget_decisions: list[Decision] = []
        for case in evaluation:
            for strategy, chooser in choices.items():
                action = chooser(case)
                actual_cost = case.actions[action].latency_ms
                decision = Decision(
                    budget_ms=float(budget), strategy=strategy,
                    dataset_id=case.dataset_id, series_id=case.series_id,
                    case_id=case.case_id, action_id=action,
                    mae=mae(case, action), latency_ms=actual_cost,
                    exceeds_budget=actual_cost > budget,
                )
                budget_decisions.append(decision)
        decisions.extend(budget_decisions)
        metrics = {}
        for strategy in choices:
            chosen = [d for d in budget_decisions if d.strategy == strategy]
            per_dataset = {}
            for dataset in sorted(set(d.dataset_id for d in chosen)):
                subset = [d for d in chosen if d.dataset_id == dataset]
                per_dataset[dataset] = {
                    "cases": len(subset),
                    "mae": statistics.mean(d.mae for d in subset),
                    "latency_ms": statistics.mean(d.latency_ms for d in subset),
                    "budget_overruns": sum(d.exceeds_budget for d in subset),
                }
            metrics[strategy] = {
                "macro_mae": statistics.mean(v["mae"] for v in per_dataset.values()),
                "macro_latency_ms": statistics.mean(v["latency_ms"] for v in per_dataset.values()),
                "budget_overruns": sum(d.exceeds_budget for d in chosen),
                "by_dataset": per_dataset,
            }
        fixed_err = metrics["validation_selected"]["macro_mae"]
        oracle_err = metrics["hindsight_oracle"]["macro_mae"]
        # Hindsight oracle cannot be worse if eligible actions are the same.
        if oracle_err > fixed_err + 1e-10:
            raise AssertionError("Oracle must not exceed fixed selection error")
        report["budgets"].append({
            "budget_ms": float(budget),
            "eligible_actions": eligible,
            "cheapest_action": cheapest,
            "validation_selected_action": selected,
            "strategies": metrics,
            "oracle_gap_mae": fixed_err - oracle_err,
            "oracle_gap_percent_of_fixed": 100 * (fixed_err - oracle_err) / fixed_err if fixed_err > 0 else 0.,
        })
    return report, decisions
