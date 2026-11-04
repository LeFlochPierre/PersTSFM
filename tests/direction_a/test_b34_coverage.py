"""Metadata-only, coverage-aware B3 selection regression tests."""
from __future__ import annotations
import pandas as pd
import pytest
from src.budgeted_forecasting.coverage import select_coverage_aware


def protocol():
    # Z is sporadic at the latest time; A/B/C appear in both blocks.
    blocks = ["2026-01-01", "2026-01-02", "2026-01-03",
              "2026-01-08", "2026-01-09", "2026-01-10"]
    rows=[]
    for entity in ("A", "B", "C"):
        for i,t in enumerate(blocks):
            ts=pd.Timestamp(t)
            rows.append(dict(entity=entity,eval_origin=ts,
                             target_end=ts+pd.Timedelta(hours=2),eval_index=i))
    ts=pd.Timestamp("2026-01-10")
    rows.append(dict(entity="Z",eval_origin=ts,target_end=ts+pd.Timedelta(hours=2),eval_index=0))
    return pd.DataFrame(rows)


def test_excludes_sparse_entity_without_outcome_features():
    a, meta = select_coverage_aware(protocol(), "entity", entities=2,
        calibration_origins=2,evaluation_origins=2,seed=42)
    assert set(a.entity).issubset({"A", "B", "C"})
    assert meta["eligible_entity_count"]==3
    assert meta["rejected_later_endpoints"]==0
    assert a.query("split == 'calibration'").target_end.max() < a.query("split == 'evaluation'").eval_origin.min()


def test_deterministic_over_order_and_seed():
    frame=protocol()
    x,m=select_coverage_aware(frame,"entity",entities=2,calibration_origins=2,evaluation_origins=2,seed=2026)
    y,n=select_coverage_aware(frame.sample(frac=1,random_state=33),"entity",entities=2,calibration_origins=2,evaluation_origins=2,seed=2026)
    assert x.to_dict('records')==y.to_dict('records')
    assert m==n


def test_no_partial_cohort():
    with pytest.raises(ValueError,match='maximum eligible together'):
        select_coverage_aware(protocol(),"entity",entities=4,calibration_origins=2,evaluation_origins=2,seed=1)


def test_duplicate_origins_rejected():
    p=protocol()
    with pytest.raises(ValueError,match='Duplicate entity/origin'):
        select_coverage_aware(pd.concat([p,p.iloc[:1]],ignore_index=True),"entity",entities=2,calibration_origins=2,evaluation_origins=2,seed=1)


def test_latest_feasible_block_chosen():
    p=protocol()
    # Make most recent window exclusive to Z, so last feasible evaluated block is earlier.
    future=[pd.Timestamp("2026-01-17"), pd.Timestamp("2026-01-24")]
    p=pd.concat([p,pd.DataFrame([dict(entity='Z',eval_origin=t,target_end=t+pd.Timedelta(hours=2),eval_index=i+1) for i,t in enumerate(future)])],ignore_index=True)
    _,m=select_coverage_aware(p,"entity",entities=2,calibration_origins=2,evaluation_origins=2,seed=0)
    assert m['rejected_later_endpoints']==1
    assert m['evaluation_slots']==['2026-01-10T00:00:00','2026-01-17T00:00:00']


def test_b3_freeze_uses_opt_in_coverage_aware(tmp_path, monkeypatch):
    """Integration: new mode freezes a valid cohort without touching default API."""
    from types import SimpleNamespace
    from src.budgeted_forecasting import capability_atlas as atlas
    source = protocol()
    protocol_path = tmp_path / 'solar_like_protocol.csv'
    source.to_csv(protocol_path, index=False)
    raw_path = tmp_path / 'raw_marker.csv'
    raw_path.write_text('marker\n')
    spec = SimpleNamespace(key='solar_like', id_col='entity',
                           protocol_path=protocol_path, raw_path=raw_path,
                           loader_kind='dummy', resolution_minutes=60,
                           forecast_steps=2, context_hours=(24,))
    registry = SimpleNamespace(
        get_dataset_spec=lambda dataset: spec,
        load_protocol=lambda spec, split: source.copy(),
    )
    monkeypatch.setattr(atlas, '_registry', lambda: registry)
    monkeypatch.setattr(atlas, '_load_selected', lambda s, rows: rows)

    def fake_extract(selected, row, max_context_hours):
        rec = dict(case_id=f"{row.entity}:{row.eval_origin}",
                   series_id=str(row.entity),
                   origin_time=pd.Timestamp(row.eval_origin).tz_localize('UTC').isoformat(),
                   label_available_at=(pd.Timestamp(row.target_end)+pd.Timedelta(hours=1)).tz_localize('UTC').isoformat(),
                   split=row['split'], eval_index=int(row.eval_index),
                   truth_sha256='truth',context_sha256='context')
        return rec, (None, None)
    monkeypatch.setattr(atlas, '_extract', fake_extract)
    payload = atlas.freeze_cohort('solar_like', 'validation', 2, 2, 2, 24, 12,
                                  tmp_path / 'frozen', coverage_aware=True)
    assert payload['selection']=='coverage_aware_global_times'
    assert len({r['series_id'] for r in payload['cases']})==2
    assert payload['selection_details']['eligible_entity_count']==3
