"""Direction A B2.0: retrospective history-only forecast-action predictability.

This is a *single-action routing* research baseline, not sequential acquisition.
The policy is fitted once to calibration labels, which B0 validates as available
before every evaluation origin. No evaluation truth is read while choosing an action.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np

from .cache import Case, load_cache
from .feasibility import mae, run_feasibility

HISTORY_STEPS = 14 * 96  # 14 days, 15-minute electricity series
HORIZON = 96
FEATURE_NAMES = (
    "log_scale", "recent_level", "recent_std", "week_std", "recent_change",
    "daily_change", "weekly_change", "trend_1d", "trend_7d",
    "last_vs_daily", "daily_corr", "weekly_corr",
)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) < 1e-10 or float(np.std(b)) < 1e-10:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))


def history_features(history: np.ndarray) -> tuple[np.ndarray, float]:
    """Only the 14 days ending at forecast origin are permitted as input."""
    x = np.asarray(history, dtype=np.float64).reshape(-1)
    if len(x) != HISTORY_STEPS or not np.isfinite(x).all():
        raise ValueError("Expected finite 14-day (1344 point) pre-origin history")
    d = x[-96:]
    w = x[-672:]
    scale = max(float(np.mean(np.abs(d))), float(np.std(d)), 1.0)
    level = float(np.mean(d))
    parts = [
        math.log1p(scale), level / scale,
        float(np.std(d)) / scale, float(np.std(w)) / scale,
        float(np.mean(np.abs(np.diff(d)))) / scale,
        float(np.mean(np.abs(d - x[-192:-96]))) / scale,
        float(np.mean(np.abs(d - x[-768:-672]))) / scale,
        float((np.mean(d[-24:]) - np.mean(d[:24])) / scale),
        float((np.mean(w[-96:]) - np.mean(w[:96])) / scale),
        float((d[-1] - level) / scale),
        _corr(d, x[-192:-96]),
        _corr(d, x[-768:-672]),
    ]
    features = np.asarray(parts, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite context features")
    return features, scale


def load_electricity_histories(cases: list[Case]) -> dict[str, tuple[np.ndarray, float]]:
    """Reconstruct contexts against the existing raw electricity dataset.

    B1 tagged source-naive electricity timestamps as UTC for monotone indexing;
    this deliberately reverses that *tag*, not a geographic timezone conversion.
    The registry is imported lazily to keep pure unit tests CPU-only.
    """
    if any(c.dataset_id != "electricity" or len(c.truth) != HORIZON for c in cases):
        raise ValueError("B2.0 currently supports electricity horizon=96 only")
    import pandas as pd
    from src.data.registry import _load_electricity, get_dataset_spec
    spec = get_dataset_spec("electricity")
    if spec.resolution_minutes != 15 or spec.forecast_steps != HORIZON:
        raise ValueError("Unexpected electricity registry protocol")
    entities = sorted({c.series_id for c in cases})
    series = _load_electricity(spec, entities)
    output = {}
    for case in cases:
        origin = pd.Timestamp(case.origin_time).tz_convert("UTC").tz_localize(None)
        expected = pd.date_range(end=origin, periods=HISTORY_STEPS, freq="15min")
        data = series[case.series_id]
        context = data.reindex(expected)
        if context.isna().any():
            raise ValueError(f"Missing or irregular history: {case.case_id}")
        if data.index.has_duplicates:
            raise ValueError(f"Duplicate raw timestamps: {case.series_id}")
        # The raw data is used only up to and including the forecast origin.
        output[case.case_id] = history_features(context.to_numpy(dtype=np.float64))
    return output


def _fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 10.0):
    """Fixed regularisation, calibration-only standardisation, multioutput ridge."""
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be positive and finite")
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    z = np.clip((X - mu) / sigma, -10.0, 10.0)
    xb = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(xb.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(xb.T @ xb + penalty, xb.T @ Y)
    return mu, sigma, weights


def _predict_ridge(model, x: np.ndarray) -> np.ndarray:
    mu, sigma, weights = model
    z = np.clip((x - mu) / sigma, -10.0, 10.0)
    return np.r_[1.0, z] @ weights


def _cal_cost(cases: list[Case], actions: list[str]) -> dict[str, float]:
    return {a: float(np.median([c.actions[a].latency_ms for c in cases])) for a in actions}


def _cluster_ci(rows: list[dict], *, bootstrap: int, seed: int) -> list[float] | None:
    if bootstrap < 1:
        raise ValueError("bootstrap must be at least one")
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_entity[r["series_id"]].append(r)
    keys = sorted(by_entity)
    if len(keys) < 3:
        return None
    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap):
        chosen = [rng.choice(keys) for _ in keys]
        subset = [r for key in chosen for r in by_entity[key]]
        fixed = np.mean([r["fixed_mae"] for r in subset])
        learned = np.mean([r["ridge_mae"] for r in subset])
        boot.append(100 * (fixed - learned) / fixed if fixed > 0 else 0.0)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def evaluate(cases: list[Case], histories: Mapping[str, tuple[np.ndarray, float]],
             budgets: list[float], *, alpha: float = 10., bootstrap: int = 2000,
             seed: int = 20261102) -> tuple[dict, list[dict]]:
    """Fit only on calibration; produce evaluation decisions and paired comparisons."""
    if len(set(c.case_id for c in cases)) != len(cases):
        raise ValueError("Duplicate case IDs")
    if {c.case_id for c in cases} != set(histories):
        raise ValueError("History keys must match all cached cases exactly")
    cal = [c for c in cases if c.split == "calibration"]
    ev = [c for c in cases if c.split == "evaluation"]
    if not cal or not ev:
        raise ValueError("Need calibration and evaluation")
    if max(c.label_available_at for c in cal) > min(c.origin_time for c in ev):
        raise ValueError("Calibration labels unavailable at evaluation start")
    actions = sorted(cal[0].actions)
    for c in cases:
        if sorted(c.actions) != actions:
            raise ValueError("Action mismatch across cases")
        x, scale = histories[c.case_id]
        if np.asarray(x).shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("Invalid history feature dimensions or values")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Nonpositive history scale")
    X = np.array([histories[c.case_id][0] for c in cal], dtype=float)
    Y = np.array([[mae(c, a) / histories[c.case_id][1] for a in actions] for c in cal])
    ridge = _fit_ridge(X, Y, alpha)
    action_index = {a: i for i, a in enumerate(actions)}
    costs = _cal_cost(cal, actions)
    mean_cal = {a: float(np.mean([mae(c, a) for c in cal])) for a in actions}
    entity_cal = defaultdict(list)
    for c in cal:
        entity_cal[c.series_id].append(c)
    reference, _ = run_feasibility(cases, budgets)
    out_rows = []
    per_budget = []
    for b, original in zip(budgets, reference["budgets"]):
        eligible = original["eligible_actions"]
        fixed = original["validation_selected_action"]
        if not eligible:
            raise ValueError("No eligible actions")
        budget_rows = []
        for c in ev:
            feats, _ = histories[c.case_id]
            expected = _predict_ridge(ridge, feats)
            ridge_action = min(eligible, key=lambda a: (expected[action_index[a]], costs[a], a))
            known = entity_cal.get(c.series_id, [])
            per_entity = (min(eligible, key=lambda a: (np.mean([mae(k, a) for k in known]), costs[a], a))
                          if known else fixed)
            oracle = min(eligible, key=lambda a: (mae(c, a), costs[a], a))
            decision = dict(budget_ms=float(b), dataset_id=c.dataset_id,
                            series_id=c.series_id, case_id=c.case_id,
                            origin_time=c.origin_time.isoformat(), fixed_action=fixed,
                            entity_action=per_entity, ridge_action=ridge_action,
                            oracle_action=oracle, fixed_mae=mae(c, fixed),
                            entity_mae=mae(c, per_entity), ridge_mae=mae(c, ridge_action),
                            oracle_mae=mae(c, oracle),
                            fixed_runtime_ms=c.actions[fixed].latency_ms,
                            entity_runtime_ms=c.actions[per_entity].latency_ms,
                            ridge_runtime_ms=c.actions[ridge_action].latency_ms)
            budget_rows.append(decision)
        out_rows.extend(budget_rows)
        fixed_mae = float(np.mean([r["fixed_mae"] for r in budget_rows]))
        entity_mae = float(np.mean([r["entity_mae"] for r in budget_rows]))
        ridge_mae = float(np.mean([r["ridge_mae"] for r in budget_rows]))
        oracle_mae = float(np.mean([r["oracle_mae"] for r in budget_rows]))
        per_budget.append({
            "budget_ms": float(b), "eligible_actions": eligible,
            "fixed_action": fixed, "fixed_mae": fixed_mae,
            "entity_selected_mae": entity_mae, "ridge_selected_mae": ridge_mae,
            "hindsight_oracle_mae": oracle_mae,
            "entity_gain_percent_vs_fixed": 100 * (fixed_mae - entity_mae) / fixed_mae if fixed_mae > 0 else 0.0,
            "ridge_gain_percent_vs_fixed": 100 * (fixed_mae - ridge_mae) / fixed_mae if fixed_mae > 0 else 0.0,
            "ridge_gain_entity_bootstrap_ci_95": _cluster_ci(budget_rows, bootstrap=bootstrap, seed=seed),
            "ridge_action_counts": dict(Counter(r["ridge_action"] for r in budget_rows)),
            "entity_action_counts": dict(Counter(r["entity_action"] for r in budget_rows)),
            "ridge_actual_budget_overruns": sum(r["ridge_runtime_ms"] > b for r in budget_rows),
            "mean_ridge_runtime_ms": float(np.mean([r["ridge_runtime_ms"] for r in budget_rows])),
            "note": "Single-action history-only routing. Expert costs are calibration-median estimates; "
                    "history feature computation and expert loading overhead not charged.",
        })
    return ({
        "schema_version": "direction_a_b2_predictability_v1",
        "status": "exploratory_history_only_single_action_routing",
        "calibration_cases": len(cal), "evaluation_cases": len(ev),
        "calibration_entities": len({c.series_id for c in cal}),
        "evaluation_entities": len({c.series_id for c in ev}),
        "feature_names": list(FEATURE_NAMES), "history_steps": HISTORY_STEPS,
        "ridge_alpha_fixed_a_priori": float(alpha),
        "mean_calibration_loss_by_action": mean_cal,
        "calibration_median_latency_ms": costs,
        "budgets": per_budget,
        "limitations": [
            "Do not tune features/alpha/budgets after seeing evaluation performance.",
            "Only 24 calibration cases and eight entities in the initial pilot; highly exploratory.",
            "No sequential acquisition: only one TSFM action selected per origin.",
            "Uses known pre-origin histories and calibration labels; never evaluation targets for decision making.",
            "Calibration median costs do not guarantee a runtime budget; feature cost omitted.",
            "Entity bootstrap quantifies sample uncertainty, not generalisation to new domains.",
        ],
    }, out_rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-cache", type=Path, required=True)
    p.add_argument("--feasibility-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=10.)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261102)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory {args.output_dir}")
    raw = json.loads(args.feasibility_json.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "direction_a_feasibility_v1":
        raise ValueError("Wrong feasibility report schema")
    budgets = [float(b["budget_ms"]) for b in raw["budgets"]]
    if len(budgets) != len(set(budgets)):
        raise ValueError("Duplicate budgets")
    cases = load_cache(args.input_cache)
    expected, _ = run_feasibility(cases, budgets)
    for prior, fresh in zip(raw["budgets"], expected["budgets"]):
        if prior["eligible_actions"] != fresh["eligible_actions"] or prior["validation_selected_action"] != fresh["validation_selected_action"]:
            raise ValueError("Feasibility report mismatch with cached actions")
        for s in ("validation_selected", "hindsight_oracle"):
            a = prior["strategies"][s]["macro_mae"]
            b = fresh["strategies"][s]["macro_mae"]
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("Feasibility MAE mismatch with cached outcomes")
    histories = load_electricity_histories(cases)
    report, rows = evaluate(cases, histories, budgets, alpha=args.alpha,
                            bootstrap=args.bootstrap, seed=args.seed)
    report["input_cache_sha256"] = _sha256(args.input_cache)
    report["input_feasibility_sha256"] = _sha256(args.feasibility_json)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "predictability.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: B2 history-only routing: {report['calibration_cases']} calibration / "
          f"{report['evaluation_cases']} evaluation cases; {report['evaluation_entities']} entities")
    for b in report["budgets"]:
        ci = b["ridge_gain_entity_bootstrap_ci_95"]
        ci_s = " (insufficient entities)" if ci is None else f" [CI {ci[0]:.2f}%, {ci[1]:.2f}%]"
        print(f"  {b['budget_ms']:.3f} ms | fixed={b['fixed_mae']:.4f} "
              f"| per-series={b['entity_selected_mae']:.4f} "
              f"| ridge={b['ridge_selected_mae']:.4f} "
              f"| oracle={b['hindsight_oracle_mae']:.4f} "
              f"| ridge gain={b['ridge_gain_percent_vs_fixed']:+.2f}%{ci_s}")
    print(f"Reports: {args.output_dir}/predictability.json, decisions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
