"""B2.2 evaluation-only accuracy/cost audit for one-shot forecast routers.

No refitting, no change to cached decisions. Evaluation-best fixed is a
*retrospective diagnostic*, never an action that was known at deployment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .cache import Case, load_cache
from .feasibility import mae, run_feasibility

POLICIES = {"calibration_fixed": "fixed_action", "per_series": "entity_action",
            "ridge": "ridge_action", "hindsight_oracle": "oracle_action"}


def _mean(values):
    a = list(values)
    if not a:
        raise ValueError("Cannot average empty values")
    return statistics.mean(a)


def _match_budget(value: float, budgets: list[float]) -> float:
    matches = [b for b in budgets if math.isclose(value, b, rel_tol=0, abs_tol=1e-6)]
    if len(matches) != 1:
        raise ValueError(f"Unexpected/ambiguous budget in decisions: {value}")
    return matches[0]


def audit(cases: list[Case], report: dict, routing: dict,
          decisions: list[dict]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Check exact decisions and return summary, per-action, frontier, excess rows."""
    if report.get("schema_version") != "direction_a_feasibility_v1":
        raise ValueError("Incorrect feasibility schema")
    if routing.get("schema_version") != "direction_a_b2_predictability_v1":
        raise ValueError("Incorrect routing schema")
    ev = [c for c in cases if c.split == "evaluation"]
    cal = [c for c in cases if c.split == "calibration"]
    if not ev or not cal:
        raise ValueError("Calibration and evaluation required")
    ev_map = {c.case_id: c for c in ev}
    if len(ev_map) != len(ev):
        raise ValueError("Duplicate evaluation case IDs")
    budgets = [float(b["budget_ms"]) for b in report["budgets"]]
    if len(budgets) != len(set(budgets)):
        raise ValueError("Duplicate budgets")
    independent, _ = run_feasibility(cases, budgets)
    if report["budgets"] != independent["budgets"]:
        raise ValueError("Feasibility cache/report disagreement")
    if [float(b["budget_ms"]) for b in routing["budgets"]] != budgets:
        raise ValueError("Routing report budget disagreement")
    grouped = defaultdict(dict)
    for row in decisions:
        budget = _match_budget(float(row["budget_ms"]), budgets)
        case_id = row["case_id"]
        if case_id not in ev_map:
            raise ValueError(f"Decisions include unknown/non-evaluation case {case_id}")
        if case_id in grouped[budget]:
            raise ValueError(f"Duplicate decision for {budget}/{case_id}")
        c = ev_map[case_id]
        if row["series_id"] != c.series_id or row["dataset_id"] != c.dataset_id:
            raise ValueError("Decision series/dataset mismatch")
        for policy, field in POLICIES.items():
            action = row[field]
            if action not in c.actions:
                raise ValueError(f"Unknown decision action {action}")
            key = {"calibration_fixed": "fixed_mae", "per_series": "entity_mae",
                   "ridge": "ridge_mae", "hindsight_oracle": "oracle_mae"}[policy]
            if not math.isclose(float(row[key]), mae(c, action), rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError(f"Decision loss mismatch for {case_id}/{policy}")
        grouped[budget][case_id] = row
    if set(grouped) != set(budgets):
        raise ValueError("Decisions missing one or more budgets")
    if any(set(grouped[b]) != set(ev_map) for b in budgets):
        raise ValueError("Incomplete decision coverage")
    per_action = []
    frontier = []
    excess = []
    for i, budget in enumerate(budgets):
        meta = report["budgets"][i]
        eligible = meta["eligible_actions"]
        fixed = meta["validation_selected_action"]
        for action in eligible:
            per_action.append(dict(budget_ms=budget, action=action,
                calibration_mae=_mean(mae(c, action) for c in cal),
                evaluation_mae=_mean(mae(c, action) for c in ev),
                calibration_median_runtime_ms=statistics.median(c.actions[action].latency_ms for c in cal),
                evaluation_mean_runtime_ms=_mean(c.actions[action].latency_ms for c in ev),
                evaluation_budget_overruns=sum(c.actions[action].latency_ms > budget for c in ev)))
        best_eval = min(eligible, key=lambda a: (_mean(mae(c, a) for c in ev), a))
        for policy, field in POLICIES.items():
            chosen = [(c, grouped[budget][c.case_id][field]) for c in ev]
            frontier.append(dict(budget_ms=budget, policy=policy,
                mean_mae=_mean(mae(c, a) for c, a in chosen),
                mean_runtime_ms=_mean(c.actions[a].latency_ms for c, a in chosen),
                budget_overruns=sum(c.actions[a].latency_ms > budget for c, a in chosen),
                non_fixed_choices=sum(a != fixed for _, a in chosen),
                selection_counts=json.dumps(dict(Counter(a for _, a in chosen)), sort_keys=True)))
        frontier.append(dict(budget_ms=budget, policy="retrospective_best_fixed_DO_NOT_DEPLOY",
            mean_mae=_mean(mae(c, best_eval) for c in ev),
            mean_runtime_ms=_mean(c.actions[best_eval].latency_ms for c in ev),
            budget_overruns=sum(c.actions[best_eval].latency_ms > budget for c in ev),
            non_fixed_choices=sum(best_eval != fixed for _ in ev),
            selection_counts=json.dumps({best_eval: len(ev)})))
        for c in ev:
            d = grouped[budget][c.case_id]
            r = d["ridge_action"]
            excess.append(dict(budget_ms=budget, case_id=c.case_id,
                series_id=c.series_id, fixed_action=fixed, ridge_action=r,
                fixed_mae=mae(c, fixed), ridge_mae=mae(c, r),
                excess_mae=mae(c, r) - mae(c, fixed),
                fixed_runtime_ms=c.actions[fixed].latency_ms,
                ridge_runtime_ms=c.actions[r].latency_ms))
    summary = {"schema_version": "direction_a_b22_cost_audit_v1",
        "status": "evaluation_only_reanalysis_no_policy_training",
        "calibration_cases": len(cal), "evaluation_cases": len(ev),
        "evaluation_entities": len(set(c.series_id for c in ev)),
        "notes": [
            "The retrospective best-fixed comparator uses evaluation labels and is not deployable.",
            "Ridge and other policies were previously developed with this electricity cohort; exploratory only.",
            "Cost is measured expert inference latency; router feature computation and model loading are omitted.",
            "Calibration-median eligibility does not ensure per-origin budget compliance.",
            "One-shot routing is not sequential compute acquisition.",
        ]}
    return summary, per_action, frontier, excess


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-cache", type=Path, required=True)
    p.add_argument("--feasibility-json", type=Path, required=True)
    p.add_argument("--predictability-json", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing {args.output_dir}")
    cases = load_cache(args.input_cache)
    report = json.loads(args.feasibility_json.read_text())
    routing = json.loads(args.predictability_json.read_text())
    with args.decisions.open(newline="", encoding="utf-8") as f:
        decisions = list(csv.DictReader(f))
    summary, per_action, frontier, excess = audit(cases, report, routing, decisions)
    summary["source_sha256"] = {p.name: _digest(p) for p in (
        args.input_cache, args.feasibility_json, args.predictability_json, args.decisions)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / "per_action.csv", per_action)
    _write_csv(args.output_dir / "policy_frontier.csv", frontier)
    _write_csv(args.output_dir / "ridge_excess.csv", excess)
    (args.output_dir / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS B2.2: {len(cases)} cases, {summary['evaluation_entities']} evaluation entities, "
          f"{len(report['budgets'])} compute budgets")
    for b in report["budgets"]:
        budget = float(b["budget_ms"])
        fr = {v["policy"]: v for v in frontier if v["budget_ms"] == budget}
        selected = fr["calibration_fixed"]
        learned = fr["ridge"]
        best = fr["retrospective_best_fixed_DO_NOT_DEPLOY"]
        print(f"  {budget:8.3f} ms | cal-fixed {selected['mean_mae']:.3f} / {selected['mean_runtime_ms']:.3f}ms "
              f"| ridge {learned['mean_mae']:.3f} / {learned['mean_runtime_ms']:.3f}ms "
              f"| eval-best-fixed* {best['mean_mae']:.3f} / {best['mean_runtime_ms']:.3f}ms "
              f"| overrun={learned['budget_overruns']}")
    print("* Evaluation-best-fixed is a hindsight diagnostic, not a deployable policy")
    print("Files: audit.json, per_action.csv, policy_frontier.csv, ridge_excess.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
