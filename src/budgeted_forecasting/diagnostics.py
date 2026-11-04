"""B1.3: evaluation-only expert complementarity and oracle-gap diagnostics.

Never trains a selector. Uses outcomes ONLY for retrospective diagnostics.
Bootstrap resamples *entities*, not horizon steps/forecast origins.
No GPU usage or modification of cached model predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .cache import Case, load_cache
from .feasibility import mae, run_feasibility


def _avg(items):
    if not items:
        raise ValueError("Cannot average empty collection")
    return statistics.mean(items)


def _percentile(items: list[float], p: float) -> float:
    a = sorted(items)
    j = (len(a) - 1) * p
    lo = int(math.floor(j))
    hi = int(math.ceil(j))
    return a[lo] * (hi - j) + a[hi] * (j - lo) if lo != hi else a[lo]


def _wins(cases: list[Case], actions: list[str], tolerance: float = 1e-10) -> dict[str, float]:
    """Fractional allocation if multiple actions tie for minimum MAE."""
    counts = dict.fromkeys(actions, 0.0)
    for c in cases:
        scores = {a: mae(c, a) for a in actions}
        best = min(scores.values())
        winners = [a for a in actions if abs(scores[a] - best) <= tolerance]
        for a in winners:
            counts[a] += 1.0 / len(winners)
    return {a: counts[a] / len(cases) for a in actions}


def _series_gap_rows(evaluation: list[Case], eligible: list[str], selected: str, budget: float):
    by_series: dict[str, list[Case]] = defaultdict(list)
    for c in evaluation:
        by_series[c.series_id].append(c)
    rows = []
    for series, items in sorted(by_series.items()):
        fixed = _avg([mae(c, selected) for c in items])
        oracle = _avg([min(mae(c, a) for a in eligible) for c in items])
        rows.append(dict(budget_ms=budget, series_id=series, evaluation_cases=len(items),
                         fixed_mae=fixed, oracle_mae=oracle, oracle_gap_mae=fixed - oracle))
    return rows


def _bootstrap_by_series(series_rows: list[dict], iterations: int, seed: int):
    """Cluster bootstrap conditional on the fixed/calibrated policy, not a rerouting CI."""
    if iterations < 100:
        raise ValueError("Bootstrap iterations must be >= 100")
    if len(series_rows) < 2:
        return {"status": "insufficient_entities", "n_entities": len(series_rows)}
    rng = random.Random(seed)
    abs_gaps, pct_gaps = [], []
    n = len(series_rows)
    for _ in range(iterations):
        chosen = [series_rows[rng.randrange(n)] for _ in range(n)]
        # Each entity has equal weight; preserve all its correlated origin observations.
        fixed = _avg([r["fixed_mae"] for r in chosen])
        oracle = _avg([r["oracle_mae"] for r in chosen])
        diff = fixed - oracle
        abs_gaps.append(diff)
        pct_gaps.append(100.0 * diff / fixed if fixed > 0 else 0.0)
    return {
        "status": "ok", "n_entities": n, "resamples": iterations, "seed": seed,
        "ci_level": 0.95,
        "oracle_gap_mae_ci_95": [_percentile(abs_gaps, 0.025), _percentile(abs_gaps, 0.975)],
        "oracle_gap_percent_ci_95": [_percentile(pct_gaps, 0.025), _percentile(pct_gaps, 0.975)],
        "note": "Entity-cluster percentile bootstrap, fixed calibration-selected action; "
                "oracle uses evaluation truth. This quantifies sample variability only, "
                "not a deployable gain or guaranteed generalisation.",
    }


def _paired_comparisons(evaluation: list[Case], action_ids: list[str]):
    results = []
    for i, first in enumerate(action_ids):
        for second in action_ids[i + 1:]:
            first_losses = [mae(c, first) for c in evaluation]
            second_losses = [mae(c, second) for c in evaluation]
            win_first = sum(x < y - 1e-10 for x, y in zip(first_losses, second_losses))
            win_second = sum(y < x - 1e-10 for x, y in zip(first_losses, second_losses))
            results.append({
                "action_a": first, "action_b": second,
                "n_cases": len(evaluation), "a_wins": win_first,
                "b_wins": win_second, "ties": len(evaluation) - win_first - win_second,
                "mae_a": _avg(first_losses), "mae_b": _avg(second_losses),
            })
    return results


def analyze(cases: list[Case], budgets: list[float], *, bootstrap: int = 2000, seed: int = 20261102):
    datasets = {c.dataset_id for c in cases}
    if len(datasets) != 1:
        raise ValueError("B1.3 reports one dataset at a time; analyse datasets separately")
    report, _ = run_feasibility(cases, budgets)
    cal = [c for c in cases if c.split == "calibration"]
    evaluation = [c for c in cases if c.split == "evaluation"]
    actions = sorted(cases[0].actions)
    action_stats = {}
    for action in actions:
        action_stats[action] = {
            "calibration_mae": _avg([mae(c, action) for c in cal]),
            "evaluation_mae": _avg([mae(c, action) for c in evaluation]),
            "calibration_latency_median_ms": statistics.median(c.actions[action].latency_ms for c in cal),
            "evaluation_latency_median_ms": statistics.median(c.actions[action].latency_ms for c in evaluation),
            "evaluation_latency_max_ms": max(c.actions[action].latency_ms for c in evaluation),
            "evaluation_overrun_count_at_own_calibration_median": sum(
                c.actions[action].latency_ms > statistics.median(x.actions[action].latency_ms for x in cal)
                for c in evaluation
            ),
            "unrestricted_hindsight_win_fraction": _wins(evaluation, actions)[action],
        }
    per_series = []
    per_origin = []
    budget_stats = []
    for b in report["budgets"]:
        threshold = b["budget_ms"]
        eligible = b["eligible_actions"]
        selected = b["validation_selected_action"]
        series_rows = _series_gap_rows(evaluation, eligible, selected, threshold)
        per_series.extend(series_rows)
        removed_action_contribution = {}
        full_oracle = _avg([min(mae(c, a) for a in eligible) for c in evaluation])
        for excluded in eligible:
            remaining = [a for a in eligible if a != excluded]
            removed_action_contribution[excluded] = (
                _avg([min(mae(c, a) for a in remaining) for c in evaluation]) - full_oracle
                if remaining else None
            )
        for case in evaluation:
            losses = {a: mae(case, a) for a in eligible}
            best = min(losses.values())
            winning = sorted(a for a, loss in losses.items() if abs(loss - best) <= 1e-10)
            per_origin.append({
                "budget_ms": threshold, "dataset_id": case.dataset_id,
                "series_id": case.series_id, "case_id": case.case_id,
                "origin_time": case.origin_time.isoformat(),
                "fixed_action": selected, "fixed_mae": losses[selected],
                "oracle_winning_actions": ";".join(winning),
                "oracle_mae": best, "oracle_gap_mae": losses[selected] - best,
            })
        budget_stats.append({
            "budget_ms": threshold, "eligible_actions": eligible, "selected_action": selected,
            "fixed_mae": b["strategies"]["validation_selected"]["macro_mae"],
            "oracle_mae": b["strategies"]["hindsight_oracle"]["macro_mae"],
            "oracle_gap_percent": b["oracle_gap_percent_of_fixed"],
            "evaluation_budget_overruns": b["strategies"]["validation_selected"]["budget_overruns"],
            "oracle_winner_fraction": _wins(evaluation, eligible),
            "marginal_oracle_mae_contribution_when_action_removed": removed_action_contribution,
            "entity_cluster_bootstrap": _bootstrap_by_series(series_rows, bootstrap, seed),
        })
    output = {
        "schema_version": "direction_a_b13_diagnostics_v1",
        "status": "retrospective_diagnostics_only_no_trained_policy",
        "dataset": next(iter(datasets)), "calibration_cases": len(cal),
        "evaluation_cases": len(evaluation),
        "evaluation_entities": len({c.series_id for c in evaluation}),
        "action_stats": action_stats, "budgets": budget_stats,
        "evaluation_pairwise_comparisons": _paired_comparisons(evaluation, actions),
        "limitations": [
            "Oracle accesses evaluation truth; this is not predictive accuracy of a routing policy.",
            "Entity bootstrap is unstable with few entities and does not address domain shift.",
            "The budget for each action is its independent warm inference cost; costs of action sequences are not modelled.",
            "Calibration median eligibility is not a hard runtime guarantee; inspect actual overruns.",
            "Evaluation results must not be used for training or selecting model hyperparameters.",
        ],
    }
    return output, per_series, per_origin


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PersTSFM B1.3 evaluation-only diagnostics")
    p.add_argument("--input-cache", type=Path, required=True)
    p.add_argument("--feasibility-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261102)
    args = p.parse_args(argv)
    raw = json.loads(args.feasibility_json.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "direction_a_feasibility_v1":
        raise ValueError("Unexpected feasibility report schema")
    budgets = [float(b["budget_ms"]) for b in raw["budgets"]]
    if len(set(budgets)) != len(budgets):
        raise ValueError("Duplicate budgets in feasibility report")
    cases = load_cache(args.input_cache)
    original, _ = run_feasibility(cases, budgets)
    for old, computed in zip(raw["budgets"], original["budgets"]):
        if old["eligible_actions"] != computed["eligible_actions"] or old["validation_selected_action"] != computed["validation_selected_action"]:
            raise ValueError("Feasibility report inconsistent with input cache/actions")
        if not math.isclose(old["strategies"]["validation_selected"]["macro_mae"], computed["strategies"]["validation_selected"]["macro_mae"], abs_tol=1e-8, rel_tol=1e-9):
            raise ValueError("Feasibility report scores mismatch input cache")
        if not math.isclose(old["strategies"]["hindsight_oracle"]["macro_mae"], computed["strategies"]["hindsight_oracle"]["macro_mae"], abs_tol=1e-8, rel_tol=1e-9):
            raise ValueError("Feasibility report oracle mismatch input cache")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory {args.output_dir}")
    summary, series, per_origin = analyze(cases, budgets, bootstrap=args.bootstrap, seed=args.seed)
    summary["source_cache_sha256"] = hashlib.sha256(args.input_cache.read_bytes()).hexdigest()
    summary["source_feasibility_sha256"] = hashlib.sha256(args.feasibility_json.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "per_series_oracle_gap.csv", series)
    _write_csv(args.output_dir / "per_origin_oracle_gap.csv", per_origin)
    print(f"PASS: evaluation-only diagnostics: {summary['evaluation_cases']} evaluation cases / "
          f"{summary['evaluation_entities']} entities, {len(summary['action_stats'])} actions")
    for b in summary["budgets"]:
        ci = b["entity_cluster_bootstrap"].get("oracle_gap_percent_ci_95")
        interval = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else " (insufficient entities)"
        print(f"  {b['budget_ms']:.3f} ms | fixed={b['selected_action']} | "
              f"oracle gap={b['oracle_gap_percent']:.2f}% | entity bootstrap CI{interval}")
    print(f"Files: {args.output_dir}/diagnostics.json, per_series_oracle_gap.csv, per_origin_oracle_gap.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
