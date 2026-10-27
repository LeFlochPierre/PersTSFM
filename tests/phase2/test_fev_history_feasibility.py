"""Offline contract tests; no fev or network access needed."""
import pytest
from scripts.phase2.audit_fev_history_feasibility import (
    calibration_steps_available, feasibility_counts, underlying_family_hint,
    select_rows, summarize_lengths,
)


def test_family_hint_merges_frequencies_without_overmerging_unrelated_names():
    assert underlying_family_hint('ercot_1H') == underlying_family_hint('ercot_1W') == 'ercot'
    assert underlying_family_hint('LOOP_SEATTLE_5T') == 'loop_seattle'
    assert underlying_family_hint('us_consumption_1Q') == 'us_consumption'
    assert underlying_family_hint('hospital_admissions_1D') == 'hospital_admissions'
    assert underlying_family_hint('hospital') == 'hospital'
    assert underlying_family_hint('hospital') != underlying_family_hint('hospital_admissions_1D')


def test_calibration_uses_strictly_pre_official_horizons():
    assert calibration_steps_available(1000, 24, 3) == 928
    assert calibration_steps_available(50, 24, 3) == 0
    assert calibration_steps_available(96, 24, 1) == 72
    with pytest.raises(ValueError):
        calibration_steps_available(100, 0, 3)


def test_context_counts_do_not_claim_later_calibration_feasibility():
    rows = feasibility_counts([100, 200, 1000], (96, 192, 768), 24, 3)
    assert [r['entities_feasible_first_official_window'] for r in rows] == [3, 2, 1]
    assert [r['entities_feasible_all_calibration_origins'] for r in rows] == [2, 1, 1]


def test_median_for_odd_even():
    assert summarize_lengths([400, 20, 30])['median'] == 30
    assert summarize_lengths([8, 10])['median'] == 9


def test_selection_rejects_covariate_tasks():
    rows = [dict(task_id='a', dataset_config='x', target_only_candidate='True'),
            dict(task_id='b', dataset_config='y', target_only_candidate='False')]
    assert [r['task_id'] for r in select_rows(rows, [], ['x'], None)] == ['a']
    with pytest.raises(ValueError, match='Not compatible'):
        select_rows(rows, [], ['y'], None)
    with pytest.raises(ValueError, match='Unknown'):
        select_rows(rows, [], ['missing'], None)


def test_first_window_audit_is_input_only(monkeypatch):
    import json
    import sys
    import types
    from scripts.phase2.audit_fev_history_feasibility import inspect_first_window

    class Window:
        def get_input_data(self):
            return ([{'date': ['2020-01-01'] * 120, 'value': list(range(120))},
                     {'date': ['2020-01-01'] * 240, 'value': list(range(240))}], [])
        def get_ground_truth(self):
            raise AssertionError('Test labels must not be accessed')

    class Task:
        target_columns = ['value']
        timestamp_column = 'date'
        horizon = 24
        num_windows = 5
        eval_metric = 'SQL'
        def __init__(self, **kwargs):
            pass
        def get_window(self, index):
            assert index == 0, 'Do not open later official windows during audit'
            return Window()
        def load_full_dataset(self):
            raise AssertionError('Full dataset access prohibited in history audit')

    monkeypatch.setitem(sys.modules, 'fev', types.SimpleNamespace(Task=Task))
    row = {'official_task_json': json.dumps({'dataset_path':'dummy', 'horizon':24}),
           'task_id':'id1', 'task_name':'demo', 'dataset_config':'demo_1H',
           'family_id_exact':'dummy/demo_1H'}
    result = inspect_first_window(row, (96, 192), 3, 0)
    assert result['entities_official_total'] == 2
    assert result['history_min'] == 120
    assert [r['entities_feasible_all_calibration_origins'] for r in result['context_counts']] == [1, 0]
    assert not result['future_truth_read']
