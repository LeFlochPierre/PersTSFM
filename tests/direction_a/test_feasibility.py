import json

import pytest

from src.budgeted_forecasting.cache import load_cache
from src.budgeted_forecasting.demo import write_demo
from src.budgeted_forecasting.feasibility import run_feasibility
from src.budgeted_forecasting.cli import main


def demo_cases(tmp_path):
    path = tmp_path / "demo.csv"
    write_demo(path)
    return load_cache(path)


def test_scoring_budget_and_oracle_bound(tmp_path):
    report, decisions = run_feasibility(demo_cases(tmp_path), [2, 10, 30])
    assert len(decisions) == 24 * 3 * 3
    for b in report["budgets"]:
        s = b["strategies"]
        assert s["hindsight_oracle"]["macro_mae"] <= s["validation_selected"]["macro_mae"]
        assert s["cheapest"]["macro_latency_ms"] <= s["validation_selected"]["macro_latency_ms"]
        assert all(d.budget_ms != b["budget_ms"] or not d.exceeds_budget for d in decisions)
    assert report["budgets"][0]["oracle_gap_mae"] == pytest.approx(0)
    assert report["budgets"][-1]["oracle_gap_mae"] > 0


def test_static_action_constant_per_budget(tmp_path):
    _, decisions = run_feasibility(demo_cases(tmp_path), [30])
    for strategy in ("cheapest", "validation_selected"):
        selected = [d.action_id for d in decisions if d.strategy == strategy]
        assert len(set(selected)) == 1


def test_calibration_selection_not_test_oracle(tmp_path):
    report, decisions = run_feasibility(demo_cases(tmp_path), [30])
    selected = report["budgets"][0]["validation_selected_action"]
    assert selected in report["budgets"][0]["eligible_actions"]
    assert all(d.action_id == selected for d in decisions if d.strategy == "validation_selected")
    assert "hindsight" in report["interpretation"].lower()


def test_too_small_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        run_feasibility(demo_cases(tmp_path), [0.01])


def test_negative_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        run_feasibility(demo_cases(tmp_path), [-1])


def test_cli_writes_summary_without_overwriting(tmp_path):
    cache = tmp_path / "demo.csv"
    out = tmp_path / "report"
    assert main(["demo", "--output", str(cache)]) == 0
    assert main(["validate", "--input", str(cache)]) == 0
    assert main(["analyze", "--input", str(cache), "--output-dir", str(out), "--budget-ms", "2", "--budget-ms", "30"]) == 0
    assert (out / "decisions.csv").exists()
    doc = json.loads((out / "feasibility.json").read_text())
    assert doc["status"] == "offline_feasibility_only"
    assert len(doc["budgets"]) == 2
    with pytest.raises(FileExistsError, match="overwrite"):
        main(["demo", "--output", str(cache)])
    with pytest.raises(FileExistsError, match="Existing output"):
        main(["analyze", "--input", str(cache), "--output-dir", str(out), "--budget-ms", "2"])
