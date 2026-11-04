import csv
from datetime import datetime, timezone

import pytest

from src.budgeted_forecasting.cache import load_cache, parse_time
from src.budgeted_forecasting.demo import write_demo


def make_cache(tmp_path):
    p = tmp_path / "predictions.csv"
    write_demo(p)
    return p


def mutate(path, change):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        names = reader.fieldnames
        rows = list(reader)
    change(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def test_demo_loads_complete_action_matrix(tmp_path):
    cases = load_cache(make_cache(tmp_path))
    assert len(cases) == 48
    assert {c.split for c in cases} == {"calibration", "evaluation"}
    assert {len(c.actions) for c in cases} == {3}
    assert {len(c.truth) for c in cases} == {4}
    assert all(c.origin_time < c.label_available_at for c in cases)


def test_duplicate_row_fails(tmp_path):
    path = make_cache(tmp_path)
    mutate(path, lambda rows: rows.append(rows[0].copy()))
    with pytest.raises(ValueError, match="duplicate"):
        load_cache(path)


def test_inconsistent_truth_fails(tmp_path):
    path = make_cache(tmp_path)
    mutate(path, lambda rows: rows[4].__setitem__("truth", "-999"))
    with pytest.raises(ValueError, match="inconsistent truth"):
        load_cache(path)


def test_missing_horizon_step_fails(tmp_path):
    path = make_cache(tmp_path)
    mutate(path, lambda rows: rows.pop(0))
    with pytest.raises(ValueError, match="incomplete horizon"):
        load_cache(path)


def test_missing_action_fails(tmp_path):
    path = make_cache(tmp_path)
    def change(rows):
        for r in rows[:]:
            if r["case_id"] == "synthetic_regular/s0/00" and r["action_id"] == "large_long":
                rows.remove(r)
    mutate(path, change)
    with pytest.raises(ValueError, match="missing or extra actions"):
        load_cache(path)


def test_label_available_must_be_future(tmp_path):
    path = make_cache(tmp_path)
    mutate(path, lambda rows: rows[0].__setitem__("label_available_at", "2020-01-01T00:00:00+00:00"))
    with pytest.raises(ValueError, match="label availability"):
        load_cache(path)


def test_calibration_cannot_leak_into_eval(tmp_path):
    path = make_cache(tmp_path)
    def change(rows):
        for r in rows:
            if r["case_id"] == "synthetic_regular/s0/03":
                r["label_available_at"] = "2025-01-07T00:00:00+00:00"
    mutate(path, change)
    with pytest.raises(ValueError, match="Temporal leakage"):
        load_cache(path)


def test_unzoned_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone"):
        parse_time("2025-01-01T12:00:00")
    assert parse_time("2025-01-01T12:00:00Z") == datetime(2025, 1, 1, 12, tzinfo=timezone.utc)


def test_nonfinite_prediction_fails(tmp_path):
    path = make_cache(tmp_path)
    mutate(path, lambda rows: rows[0].__setitem__("prediction", "nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        load_cache(path)


def test_latency_not_summed_across_steps(tmp_path):
    cases = load_cache(make_cache(tmp_path))
    c = cases[0]
    assert {a.latency_ms for a in c.actions.values()} == {1., 8., 24.}


def test_pooled_calibration_cannot_leak_between_series(tmp_path):
    path = make_cache(tmp_path)
    def change(rows):
        # Label is available before evaluation of s0, but after the first
        # evaluation of s1; a pooled policy would leak across series.
        from datetime import timedelta
        from src.budgeted_forecasting.cache import parse_time
        for r in rows:
            if r["case_id"] == "synthetic_regular/s0/03":
                r["label_available_at"] = "2025-01-07T00:00:00+00:00"
            if r["dataset_id"] == "synthetic_regular" and r["series_id"] == "s0" and r["split"] == "evaluation":
                r["origin_time"] = (parse_time(r["origin_time"]) + timedelta(days=4)).isoformat()
                r["label_available_at"] = (parse_time(r["label_available_at"]) + timedelta(days=4)).isoformat()
    mutate(path, change)
    with pytest.raises(ValueError, match="Temporal leakage"):
        load_cache(path)
