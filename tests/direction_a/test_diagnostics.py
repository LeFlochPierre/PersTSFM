import json
import csv
from dataclasses import replace
from pathlib import Path

import pytest

from src.budgeted_forecasting.cache import load_cache
from src.budgeted_forecasting.demo import write_demo
from src.budgeted_forecasting.feasibility import run_feasibility
from src.budgeted_forecasting.diagnostics import _wins, _percentile, analyze, main


def fixtures(tmp_path):
    path = tmp_path / "demo.csv"
    full = tmp_path / "all.csv"
    write_demo(full)
    with full.open(newline="") as source, path.open("w", newline="") as dest:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(row for row in reader if row["dataset_id"] == "synthetic_regular")
    cases = load_cache(path)
    return path, cases


def test_basic_diagnostics(tmp_path):
    _, cases = fixtures(tmp_path)
    report, series, origins = analyze(cases, [2.0, 10.0, 30.0], bootstrap=100)
    assert report["evaluation_cases"] == 12
    assert report["evaluation_entities"] == 3
    assert len(series) == 9
    assert len(origins) == 36
    assert report["budgets"][-1]["oracle_gap_percent"] > 0


def test_diagnostic_matches_feasibility(tmp_path):
    _, cases = fixtures(tmp_path)
    old, _ = run_feasibility(cases, [2, 10, 30])
    report, _, _ = analyze(cases, [2, 10, 30], bootstrap=100)
    for old_budget, b in zip(old["budgets"], report["budgets"]):
        assert b["oracle_gap_percent"] == pytest.approx(old_budget["oracle_gap_percent_of_fixed"])
        assert b["selected_action"] == old_budget["validation_selected_action"]


def test_win_fraction_normalized(tmp_path):
    _, cases = fixtures(tmp_path)
    cases = [c for c in cases if c.split == "evaluation"]
    fractions = _wins(cases, sorted(cases[0].actions))
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert all(0 <= x <= 1 for x in fractions.values())


def test_no_evaluation_training(tmp_path):
    _, cases = fixtures(tmp_path)
    report, _, _ = analyze(cases, [30], bootstrap=100)
    assert report["status"] == "retrospective_diagnostics_only_no_trained_policy"


def test_bootstrap_reproducible(tmp_path):
    _, cases = fixtures(tmp_path)
    a, _, _ = analyze(cases, [30], bootstrap=100, seed=43)
    b, _, _ = analyze(cases, [30], bootstrap=100, seed=43)
    assert a == b


def test_marginal_contribution_nonnegative(tmp_path):
    _, cases = fixtures(tmp_path)
    result, _, _ = analyze(cases, [30], bootstrap=100)
    for value in result["budgets"][0]["marginal_oracle_mae_contribution_when_action_removed"].values():
        assert value is None or value >= -1e-8


def test_percentile():
    assert _percentile([1,2,3,4,5], .5) == pytest.approx(3)
    assert _percentile([4,2,8], 0) == 2
    assert _percentile([4,2,8], 1) == 8


def test_report_cli_roundtrip(tmp_path):
    cache, cases = fixtures(tmp_path)
    feasibility, _ = run_feasibility(cases, [2, 10, 30])
    report_path = tmp_path / "feasibility.json"
    report_path.write_text(json.dumps(feasibility))
    out = tmp_path / "diag"
    argv = ["--input-cache", str(cache), "--feasibility-json", str(report_path),
            "--output-dir", str(out), "--bootstrap", "100"]
    assert main(argv) == 0
    assert (out / "diagnostics.json").exists()
    assert (out / "per_series_oracle_gap.csv").exists()
    assert (out / "per_origin_oracle_gap.csv").exists()
    with pytest.raises(FileExistsError):
        main(argv)


def test_report_mismatch_rejected(tmp_path):
    cache, cases = fixtures(tmp_path)
    feasibility, _ = run_feasibility(cases, [2, 10, 30])
    feasibility["budgets"][-1]["validation_selected_action"] = "nonsense"
    report_path = tmp_path / "feasibility.json"
    report_path.write_text(json.dumps(feasibility))
    with pytest.raises(ValueError, match="inconsistent"):
        main(["--input-cache", str(cache), "--feasibility-json", str(report_path),
              "--output-dir", str(tmp_path / "diag"), "--bootstrap", "100"])
    assert not (tmp_path / "diag").exists()


def test_one_dataset_only(tmp_path):
    _, cases = fixtures(tmp_path)
    with pytest.raises(ValueError, match="one dataset"):
        analyze(cases + [replace(cases[0], dataset_id="another")], [30], bootstrap=100)


def test_bootstrap_too_small_rejected(tmp_path):
    _, cases = fixtures(tmp_path)
    with pytest.raises(ValueError, match=">= 100"):
        analyze([c for c in cases if c.dataset_id == "synthetic_regular"], [30], bootstrap=99)
