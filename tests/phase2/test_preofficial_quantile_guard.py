"""Regression tests for explicit quantile rearrangement with preserved raw forecasts."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.phase2.run_preofficial_calibration import checked_output


def synthetic(crossed=True):
    inp = [SimpleNamespace(case_id='synthetic__1')]
    outputs = [SimpleNamespace(case_id='synthetic__1',
        prediction=np.array([11., 12.]),
        q10=np.array([2., 0.]) if crossed else np.array([0., 0.]),
        q50=np.array([1., 1.]),
        q90=np.array([3., 3.]))]
    row = [{'entity_id': '1', 'lag_before_first_official': 1, 'horizon': 2}]
    future = [np.array([100, 200], dtype=np.int64)]
    return outputs, inp, row, future


def test_strict_rejects_crossing_with_diagnostics():
    with pytest.raises(ValueError, match=r'crossing quantiles at 1/2 steps.*maximum violation=1'):
        checked_output(*synthetic(True), 'moirai2_small', 96, 1, 'task')


def test_opt_in_rearranges_and_preserves_raw_point_forecast():
    events = []
    row, = checked_output(*synthetic(True), 'moirai2_small', 96, 1, 'task',
                          quantile_policy='rearrange', crossing_events=events)
    assert row['prediction'] == [11., 12.]
    assert row['raw_q10'] == [2., 0.]
    assert row['raw_q50'] == [1., 1.]
    assert row['q10'] == [1., 0.]
    assert row['q50'] == [2., 1.]
    assert row['q90'] == [3., 3.]
    assert row['quantile_crossing_steps'] == 1
    assert len(events) == 1 and events[0]['max_pairwise_crossing'] == 1.
    assert row['mean_abs_quantile_correction'] > 0


def test_noncrossing_unchanged_and_diagnostics_empty():
    events = []
    row, = checked_output(*synthetic(False), 'moirai2_small', 96, 1, 'task',
                          quantile_policy='rearrange', crossing_events=events)
    assert row['q10'] == row['raw_q10']
    assert row['q50'] == row['raw_q50']
    assert row['q90'] == row['raw_q90']
    assert row['quantile_crossing_steps'] == 0 and events == []


def test_invalid_quantile_values_rejected_even_with_rearrangement():
    out, inp, rows, future = synthetic()
    out[0].q10[0] = float('nan')
    with pytest.raises(ValueError, match='invalid q10'):
        checked_output(out, inp, rows, future, 'm', 96, 1, 'task',
                       quantile_policy='rearrange')


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match='Unknown quantile'):
        checked_output(*synthetic(), 'm', 96, 1, 'task',quantile_policy='auto')
