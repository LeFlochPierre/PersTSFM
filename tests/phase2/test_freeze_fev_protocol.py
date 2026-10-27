"""Pure protocol invariants; no fev download required."""
import pytest
from scripts.phase2.freeze_fev_protocol import (
    select_common_cohort, validate_chronology, role_for_window, timestamps_ns
)


def sample_rows(ids=("a", "b"), n=8):
    return [dict(entity_id=entity, window_index=i,
                 role=role_for_window(i, 4), history_length=384,
                 future_start_ns=(i * 24 + 1) * 100,
                 future_end_ns=(i * 24 + 24) * 100)
            for entity in ids for i in range(n)]


def test_common_cohort_across_windows():
    rows = sample_rows()
    rows[0]["history_length"] = 200
    selected, ids = select_common_cohort(rows, [96, 192, 384], 8)
    assert ids == {"b"}
    assert len(selected) == 8


def test_strict_chronology():
    assert validate_chronology(sample_rows(), "calendar_check_required") == "strict_global_calendar"
    assert validate_chronology(sample_rows(), "series_relative_not_global_calendar") == "strict_within_entity_only"


def test_detect_future_overlap():
    rows = sample_rows(ids=("a",))
    rows[1]["future_start_ns"] = rows[0]["future_end_ns"]
    with pytest.raises(RuntimeError, match="overlapping"):
        validate_chronology(rows, "calendar_check_required")


def test_times_are_strict():
    assert timestamps_ns(["2020-01-01T00:00:00", "2020-01-01T01:00:00"], "x").shape == (2,)
    with pytest.raises(RuntimeError, match="non-increasing"):
        timestamps_ns(["2020-01-01T01:00:00", "2020-01-01T01:00:00"], "x")
