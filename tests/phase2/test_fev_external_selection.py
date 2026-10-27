from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC1 = importlib.util.spec_from_file_location('fev_sel_runner', ROOT/'scripts/phase2/run_fev_selection.py')
runner = importlib.util.module_from_spec(SPEC1)
SPEC1.loader.exec_module(runner)
SPEC2 = importlib.util.spec_from_file_location('fev_sel_score', ROOT/'scripts/phase2/score_fev_selection.py')
scorer = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(scorer)


@pytest.fixture
def fake_input_class(monkeypatch):
    @dataclass
    class ForecastInput:
        case_id: str
        timestamps: object
        values: np.ndarray
    adapters = types.ModuleType('src.models.adapters')
    adapters.ForecastInput = ForecastInput
    src = types.ModuleType('src')
    models = types.ModuleType('src.models')
    monkeypatch.setitem(sys.modules, 'src', src)
    monkeypatch.setitem(sys.modules, 'src.models', models)
    monkeypatch.setitem(sys.modules, 'src.models.adapters', adapters)
    return ForecastInput


def record(n=96):
    step = int(3600 * 1e9)
    return dict(case_id='d__w00__x', entity_id='x', window_index=0,
                observed_steps_saved=n,
                history=[float(x) for x in range(n)],
                history_timestamps_ns=np.arange(n, dtype=np.int64) * step,
                future_timestamps_ns=np.arange(n, n+24, dtype=np.int64) * step)


def test_frozen_input_exact_context(fake_input_class):
    data, future = runner.make_input(record(96), 48, 24)
    assert len(data.values) == 48
    assert data.values[0] == 48
    assert len(future) == 24
    assert runner.resolution_minutes(record()) == 60


def test_short_context_rejected(fake_input_class):
    with pytest.raises(ValueError, match='only'):
        runner.make_input(record(25), 48, 24)


def test_future_overlap_rejected(fake_input_class):
    r = record()
    r['future_timestamps_ns'][0] = r['history_timestamps_ns'][-1]
    with pytest.raises(ValueError):
        runner.make_input(r, 48, 24)


def test_bad_cadence_rejected(fake_input_class):
    r = record()
    r['history_timestamps_ns'][-1] += 60 * int(1e9)
    with pytest.raises(ValueError):
        runner.make_input(r, 48, 24)


def test_predictions_are_case_aligned(fake_input_class):
    a, future = runner.make_input(record(), 48, 24)
    out = types.SimpleNamespace(case_id=a.case_id, prediction=np.ones(24),
                                q10=np.zeros(24), q50=np.ones(24), q90=2*np.ones(24))
    rows = runner.validate_and_export([out], [a], [record()], [future], 24,
                                      model='test', context=48, task='d')
    assert rows[0]['role'] == 'selection'
    assert len(rows[0]['prediction']) == 24


def test_missing_prediction_rejected(fake_input_class):
    a, future = runner.make_input(record(), 48, 24)
    out = types.SimpleNamespace(case_id='wrong', prediction=np.ones(24),
                                q10=np.zeros(24), q50=np.ones(24), q90=2*np.ones(24))
    with pytest.raises(RuntimeError):
        runner.validate_and_export([out], [a], [record()], [future], 24,
                                   model='test', context=48, task='d')


def test_seasonal_scale_and_mae():
    assert np.isclose(scorer.seasonal_scale([1, 2, 4, 7], 1), 2)
    mae, mase = scorer.evaluate_one({'prediction':[2., 4.]}, [1, 3], [1, 2, 4, 7], horizon=2, seasonality=1)
    assert mae == 1
    assert mase == 0.5
    assert np.isnan(scorer.seasonal_scale([1, 1, 1], 1))
