"""Synthetic invariants; these tests never access fev or official datasets."""
import numpy as np
import pandas as pd
import pytest

from scripts.phase2.export_preofficial_calibration import (
    split_one_history, timestamps_ns, validate_export,
    selected_registry_rows,
)


def fixture(L=100, horizon=12, K=3):
    dates = np.arange(L, dtype='int64') * 3_600_000_000_000
    y = np.arange(L, dtype='float64')
    return split_one_history('A', y, dates, horizon, K,
                             (8, 16, 32, 64, 96), 'task_001')


def test_three_disjoint_historical_horizons():
    inputs, truths, inventory = fixture()
    assert len(inputs) == len(truths) == 3
    assert [r['lag_before_first_official'] for r in inputs] == [3, 2, 1]
    assert [t['target_values'][0] for t in truths] == [64.0, 76.0, 88.0]
    assert [t['target_values'][-1] for t in truths] == [75.0, 87.0, 99.0]
    assert [i['history'][-1] for i in inputs] == [63.0, 75.0, 87.0]
    assert inventory['max_calibration_origins'] == 3
    validate_export(inputs, truths, [inventory])


def test_not_enough_history_for_three_origins():
    inputs, truths, inventory = fixture(L=20, horizon=8)
    assert len(inputs) == len(truths) == 1
    assert inventory['max_calibration_origins'] == 1
    assert inputs[0]['lag_before_first_official'] == 1
    assert inputs[0]['available_history_steps'] == 12


def test_zero_origin_produces_only_inventory():
    inputs, truths, inventory = fixture(K=0)
    assert inputs == truths == []
    assert inventory['max_calibration_origins'] == 0
    validate_export(inputs, truths, [inventory])


def test_never_include_official_future():
    inputs, truths, inv = fixture(L=80, horizon=5)
    for x, t in zip(inputs, truths):
        assert max(t['target_values']) <= 79
        assert max(x['history']) < min(t['target_values'])
        assert t['future_end_ns'] <= inv['last_preofficial_observed_ns']


def test_missing_values_preserved():
    ts = np.arange(75, dtype=np.int64) * 100
    values = np.arange(75, dtype=np.float64)
    values[3] = np.nan
    values[-1] = np.nan
    xs, ys, inv = split_one_history('A', values, ts, 8, 2, (8, 16), 'task')
    assert np.isnan(ys[-1]['target_values'][-1])
    assert inv['missing_historical_values'] == 2
    validate_export(xs, ys, [inv])


def test_infinite_values_rejected():
    ts = np.arange(40)
    y = np.arange(40, dtype=float)
    y[0] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        split_one_history('A', y, ts, 5, 1, (8,), 'task')


def test_duplicate_timestamps_rejected():
    ts = np.arange(40)
    ts[2] = ts[1]
    with pytest.raises(ValueError, match='strictly increasing'):
        split_one_history('A', np.arange(40.), ts, 5, 1, (8,), 'task')


@pytest.mark.parametrize('unit', ['s','ms','us','ns'])
def test_precision_conversion(unit):
    dt = pd.date_range('2020-01-01', periods=3, freq='h', tz='UTC').as_unit(unit)
    ts = timestamps_ns(dt, 'test')
    assert int(ts[1]-ts[0]) == 3_600_000_000_000


def test_cross_case_leakage_detected():
    ins, labs, inv = fixture()
    ins[0]['future_start_ns'] = 1234
    with pytest.raises(ValueError, match='mismatch'):
        validate_export(ins, labs, [inv])


def test_reject_covariate_task():
    rows = [dict(task_id='t',dataset_config='foo',task_name='foo',target_only_candidate='False')]
    with pytest.raises(ValueError,match='Target-only'):
        selected_registry_rows(rows,['foo'],[])


def test_history_beyond_max_context_is_truncated():
    ins, labs, inv = fixture(L=5000, horizon=10, K=1)
    assert len(ins[0]['history']) == 96
    assert ins[0]['available_history_steps'] == 4990
    assert labs[0]['target_values'] == list(np.arange(4990, 5000, dtype=float))


def test_mocked_fev_export_end_to_end(tmp_path, monkeypatch):
    """Exercises filesystem, version manifest, and API boundary without external data."""
    import sys
    import types
    import json
    from scripts.phase2.export_preofficial_calibration import export_task

    ns = pd.date_range('2020-01-01', periods=100, freq='h').as_unit('us')
    history = dict(item_id='A', timestamp=ns, target=np.arange(100, dtype=float))
    calls = []

    class Window:
        def get_input_data(self):
            calls.append('input_only')
            return [history], None

    class Task:
        def __init__(self, **kw):
            self.target_columns = ['target']
            self.timestamp_column = 'timestamp'
            self.id_column = 'item_id'
            self.horizon = kw['horizon']

        def get_window(self, index):
            assert index == 0
            return Window()

    monkeypatch.setitem(sys.modules, 'fev', types.SimpleNamespace(Task=Task))

    # Verify everything except real PyArrow serialization (not installed locally).
    def fake_parquet(self, path, index=False):
        assert index is False
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json(orient='records'))

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fake_parquet)
    row = dict(task_id='fev_0_10_0_999', task_name='fake/h12',
               dataset_config='fake', family_id_exact='fake',
               official_task_json=json.dumps(dict(horizon=12)))
    registry = dict(source_sha256='a' * 64)
    output = tmp_path/'export'
    manifest = export_task(row, (8,16,32), 3, 0, output, registry, tmp_path/'audit')
    assert calls == ['input_only']
    assert manifest['n_cases'] == 3
    assert manifest['eligible_entities_per_budget']['3'] == 1
    assert manifest['official_test_target_values_included'] is False
    assert (output / row['task_id'] / 'historical_calibration_labels.parquet').exists()
    with pytest.raises(FileExistsError):
        export_task(row, (8,16,32), 3, 0, output, registry, tmp_path/'audit')
