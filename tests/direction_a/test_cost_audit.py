"""B2.2 invariants: never silently misalign decisions and actual cache errors."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.budgeted_forecasting.cache import Case, Candidate
from src.budgeted_forecasting.feasibility import mae, run_feasibility
from src.budgeted_forecasting.cost_audit import audit

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fixture():
    cases = []
    for j in range(4):
        for split, day in (("calibration", 0), ("evaluation", 30)):
            a, b = (2., 4.) if j % 2 == 0 else (4., 2.)
            case_id = f"e{j}_{split}"
            origin = T0 + timedelta(days=day, hours=j)
            cases.append(Case("electricity", f"e{j}", case_id, split,
                origin, origin + timedelta(hours=1), (0.,),
                {"a": Candidate("a", (a,), 1.), "b": Candidate("b", (b,), 3.)}))
    feas, _ = run_feasibility(cases, [1., 3.])
    routing = {"schema_version": "direction_a_b2_predictability_v1",
               "budgets": [{"budget_ms": 1.}, {"budget_ms": 3.}]}
    decisions = []
    for budget in feas["budgets"]:
        b = budget["budget_ms"]
        fixed = budget["validation_selected_action"]
        eligible = budget["eligible_actions"]
        for c in cases:
            if c.split != "evaluation":
                continue
            oracle = min(eligible, key=lambda a: (mae(c, a), a))
            chosen = oracle if b == 3. else fixed
            decisions.append(dict(budget_ms=b, case_id=c.case_id,
                series_id=c.series_id, dataset_id=c.dataset_id,
                fixed_action=fixed, entity_action=fixed, ridge_action=chosen,
                oracle_action=oracle, fixed_mae=mae(c, fixed),
                entity_mae=mae(c, fixed), ridge_mae=mae(c, chosen),
                oracle_mae=mae(c, oracle)))
    return cases, feas, routing, decisions


def test_clean_audit():
    args = fixture()
    summary, actions, frontier, excess = audit(*args)
    assert summary["evaluation_cases"] == 4
    assert len(actions) == 3
    assert len(frontier) == 10
    assert len(excess) == 8


def test_missing_cases_fails():
    cases, feas, routing, decisions = fixture()
    with pytest.raises(ValueError, match="Incomplete"):
        audit(cases, feas, routing, decisions[:-1])


def test_duplicate_decision_fails():
    cases, feas, routing, decisions = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        audit(cases, feas, routing, decisions + decisions[:1])


def test_changed_prediction_detected():
    cases, feas, routing, decisions = fixture()
    bad = list(decisions)
    bad[0] = dict(bad[0], ridge_mae=999.)
    with pytest.raises(ValueError, match="loss mismatch"):
        audit(cases, feas, routing, bad)


def test_changed_report_detected():
    cases, feas, routing, decisions = fixture()
    corrupt = dict(feas)
    corrupt["budgets"] = [dict(x) for x in feas["budgets"]]
    corrupt["budgets"][1]["validation_selected_action"] = "b" if feas["budgets"][1]["validation_selected_action"] == "a" else "a"
    with pytest.raises(ValueError, match="disagreement"):
        audit(cases, corrupt, routing, decisions)


def test_performance_and_runtime_consistency():
    cases, feas, routing, decisions = fixture()
    _, _, frontier, _ = audit(cases, feas, routing, decisions)
    high = [v for v in frontier if v["budget_ms"] == 3.]
    assert next(v for v in high if v["policy"] == "ridge")["mean_mae"] <= next(v for v in high if v["policy"] == "calibration_fixed")["mean_mae"]
    assert next(v for v in high if v["policy"] == "ridge")["mean_runtime_ms"] >= 1.
