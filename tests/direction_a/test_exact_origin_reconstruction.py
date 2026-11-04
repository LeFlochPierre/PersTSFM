"""Regression for B1.2 exact-origin reconstruction; no real data or GPU needed."""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.budgeted_forecasting.cache import Candidate, Case
from src.budgeted_forecasting import extend_atlas
from src.budgeted_forecasting.real_atlas import OriginCase


def synthetic_test_inputs(tmp_path, monkeypatch, *, remove_one=False, duplicate_one=False, wrong_truth=False):
    # Uneven case coverage, with a later origin NOT in the cache.
    entries = [
        ('e1', 'calibration', '2026-01-20T00:00:00'),
        ('e2', 'calibration', '2026-01-21T00:00:00'),
        ('e2', 'calibration', '2026-01-22T00:00:00'),
        ('e1', 'evaluation', '2026-01-28T00:00:00'),
        ('e2', 'evaluation', '2026-01-29T00:00:00'),
    ]
    cases = []
    rows = []
    mapping = {}
    for idx, (entity, split, when) in enumerate(entries):
        native_origin = pd.Timestamp(when)
        origin = native_origin.tz_localize('UTC').to_pydatetime()
        available = (native_origin + pd.Timedelta(days=1) + pd.Timedelta(minutes=15)).tz_localize('UTC').to_pydatetime()
        cid = f'electricity::{entity}::{native_origin.isoformat()}'
        truth = tuple(float(idx + 1) for _ in range(96))
        case = Case('electricity', entity, cid, split, origin, available,
                    truth, {'seasonal_naive': Candidate('seasonal_naive', truth, 1.)})
        cases.append(case)
        mapping[(entity, native_origin)] = case
        rows.append(dict(entity=entity, eval_origin=native_origin, eval_index=idx,
                         target_start=native_origin + pd.Timedelta(minutes=15),
                         target_end=native_origin + pd.Timedelta(days=1)))
    # Extra future origin changes counts-based resampling but must not affect exact matching.
    rows.append(dict(entity='e1', eval_origin=pd.Timestamp('2026-01-30'),
                     eval_index=100, target_start=pd.Timestamp('2026-01-30T00:15'),
                     target_end=pd.Timestamp('2026-01-31T00:00')))
    if remove_one:
        rows = [r for r in rows if not (r['entity'] == 'e1' and r['eval_origin'] == pd.Timestamp('2026-01-28'))]
    if duplicate_one:
        rows.append(rows[0].copy())
    protocol = pd.DataFrame(rows)
    meta = tmp_path / 'manifest.json'
    meta.write_text(json.dumps(dict(dataset='electricity', horizon_steps=96,
                                    entities=['e1', 'e2'], seed=42,
                                    protocol_split='validation')))

    spec = SimpleNamespace(key='electricity', id_col='entity', resolution_minutes=15, forecast_steps=96)
    fake_registry = types.ModuleType('src.data.registry')
    fake_registry.LoadedDataset = lambda spec, protocol, series: SimpleNamespace(
        spec=spec, protocol=protocol, series=series)
    fake_registry._load_electricity = lambda spec, ids: {key: 1 for key in ids}
    fake_registry.get_dataset_spec = lambda name: spec
    fake_registry.load_protocol = lambda spec, split: protocol.copy()
    monkeypatch.setitem(sys.modules, 'src.data.registry', fake_registry)
    monkeypatch.setattr(extend_atlas, 'load_cache', lambda path: cases)

    def make_cases(loaded, selected):
        outputs = []
        for _, row in selected.iterrows():
            key = (row['entity'], pd.Timestamp(row['eval_origin']))
            ref = mapping[key]
            index = pd.date_range(end=key[1], periods=1344, freq='15min')
            truth = np.asarray(ref.truth, dtype=float)
            if wrong_truth and ref.case_id == cases[0].case_id:
                truth[0] += .1
            outputs.append(OriginCase(
                case_id=ref.case_id, entity_id=ref.series_id, split=row['split'],
                origin=key[1].tz_localize('UTC'),
                label_available_at=pd.Timestamp(ref.label_available_at),
                context=pd.Series(np.ones(len(index)), index=index), truth=truth))
        return outputs

    monkeypatch.setattr(extend_atlas, 'prepare_cases', make_cases)
    return tmp_path / 'baseline.csv', meta


def test_exact_origin_set_preserved_with_uneven_coverage_and_extra_origin(tmp_path, monkeypatch):
    cache, meta = synthetic_test_inputs(tmp_path, monkeypatch)
    base, real, info, details = extend_atlas.load_matched_dataset(cache, meta)
    assert len(real) == len(base) == 5
    assert {x.case_id for x in real} == {x.case_id for x in base}
    assert details['selection'].startswith('exact_origin_keys')
    assert details['calibration_cases'] == 3
    assert details['evaluation_cases'] == 2


def test_missing_exact_origin_rejected(tmp_path, monkeypatch):
    cache, meta = synthetic_test_inputs(tmp_path, monkeypatch, remove_one=True)
    with pytest.raises(ValueError, match='Baseline origins missing'):
        extend_atlas.load_matched_dataset(cache, meta)


def test_duplicate_protocol_origin_rejected(tmp_path, monkeypatch):
    cache, meta = synthetic_test_inputs(tmp_path, monkeypatch, duplicate_one=True)
    with pytest.raises(ValueError, match='Duplicate entity/origin'):
        extend_atlas.load_matched_dataset(cache, meta)


def test_truth_mutation_rejected(tmp_path, monkeypatch):
    cache, meta = synthetic_test_inputs(tmp_path, monkeypatch, wrong_truth=True)
    with pytest.raises(ValueError, match='Ground-truth mismatch'):
        extend_atlas.load_matched_dataset(cache, meta)
