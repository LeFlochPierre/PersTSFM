"""B3 solar-style asynchronous per-entity protocol selection tests."""
from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import pytest
from src.budgeted_forecasting.per_entity_origins import select_per_entity_origins


def staggered_protocol(n_entities: int = 10):
    rows = []
    for entity in range(n_entities):
        name = f"WA_{entity:03d}"
        # Each entity uses its own forecast timestamps, yet the overall
        # calibration/evaluation periods are shared and chronologically safe.
        for j, day in enumerate((1, 2, 3, 8, 9, 10)):
            t = pd.Timestamp(f"2026-01-{day:02d}") + pd.Timedelta(minutes=60*entity)
            rows.append(dict(entity=name, eval_origin=t,
                             target_end=t+pd.Timedelta(hours=1), eval_index=j))
    return pd.DataFrame(rows)


def select(frame=None, entities=8, seed=22):
    return select_per_entity_origins(
        staggered_protocol() if frame is None else frame,
        'entity', entities=entities, calibration_origins=3,
        evaluation_origins=3, seed=seed, resolution_minutes=60)


def test_selects_eight_balanced_from_staggered_origins():
    selected, meta = select()
    assert len(selected) == 48
    assert selected.entity.nunique() == 8
    counts = selected.groupby(['entity', 'split']).size().unstack()
    assert set(counts.calibration) == {3}
    assert set(counts.evaluation) == {3}
    assert meta['eligible_entity_count'] >= 8
    assert meta['selection'] == 'latest_feasible_global_cutoff_entity_specific_origins'
    assert selected[selected.split=='calibration'].target_end.max() + pd.Timedelta(hours=1) <= (
        selected[selected.split=='evaluation'].eval_origin.min())


def test_deterministic_with_order_changes():
    frame = staggered_protocol()
    a, ma = select(frame)
    b, mb = select(frame.sample(frac=1, random_state=12))
    assert a.to_dict('records') == b.to_dict('records')
    assert ma == mb


def test_not_dependent_on_target_or_prediction_columns():
    frame=staggered_protocol()
    frame['truth']=range(len(frame))
    frame['prediction']=0.0
    a, _ = select(frame)
    frame['truth'] = -9999
    frame['prediction'] = 10000
    b, _ = select(frame)
    pd.testing.assert_frame_equal(
        a.drop(columns=['truth','prediction']),
        b.drop(columns=['truth','prediction']))


def test_rejects_duplicate_entity_origin_and_index():
    frame=staggered_protocol()
    with pytest.raises(ValueError, match='Duplicate entity/origin'):
        select(pd.concat([frame,frame.iloc[[0]]],ignore_index=True))
    frame2=frame.copy()
    frame2.loc[0,'eval_index']=frame2.loc[1,'eval_index']
    # row 1 is a different entity, so deliberately duplicate within entity:
    frame2.loc[0,'eval_index']=frame2.loc[2,'eval_index']
    # Correct index of the second row from first entity, if present:
    frame2.loc[0,'eval_index']=frame2.loc[frame2.entity==frame2.loc[0,'entity'],'eval_index'].iloc[1]
    with pytest.raises(ValueError, match='Duplicate entity/eval_index'):
        select(frame2)


def test_reports_maximum_feasible_if_too_many_entities():
    frame = staggered_protocol(6)
    with pytest.raises(ValueError, match='maximum feasible: 6'):
        select(frame, entities=8)


def test_frozen_b3_integration(tmp_path, monkeypatch):
    from src.budgeted_forecasting import capability_atlas as atlas
    frame = staggered_protocol()
    raw = tmp_path / 'data.csv'
    raw.write_text('marker\n')
    protocol = tmp_path / 'protocol.csv'
    frame.to_csv(protocol, index=False)
    spec = SimpleNamespace(key='solar_like', id_col='entity',
        raw_path=raw, protocol_path=protocol, context_hours=(96,),
        resolution_minutes=60, forecast_steps=1)
    reg = SimpleNamespace(get_dataset_spec=lambda key: spec,
        load_protocol=lambda s, split: frame.copy())
    monkeypatch.setattr(atlas,'_registry',lambda: reg)
    monkeypatch.setattr(atlas,'_load_selected',lambda s,selected: selected)
    def extract(selected,row,hours):
        t=pd.Timestamp(row['eval_origin'])
        rec = dict(case_id=f"solar_like::{row['entity']}::{t.isoformat()}",
            series_id=str(row['entity']), origin_time=t.tz_localize('UTC').isoformat(),
            label_available_at=(pd.Timestamp(row['target_end'])+pd.Timedelta(hours=1)).tz_localize('UTC').isoformat(),
            split=row['split'], eval_index=int(row.eval_index),
            truth_sha256='fake_truth', context_sha256='fake_context')
        return rec, (None,None)
    monkeypatch.setattr(atlas,'_extract',extract)
    result = atlas.freeze_cohort('solar_like','validation',8,3,3,96,22,
                                 tmp_path/'output',per_entity_origins=True)
    assert len(result['cases'])==48
    assert result['selection']=='per_entity_origins_pooled_cutoff'
    assert result['selection_details']['evaluation_span_days'] >= 0
    assert len({x['series_id'] for x in result['cases']}) == 8


def test_mutually_exclusive_selector_flags(tmp_path,monkeypatch):
    from src.budgeted_forecasting import capability_atlas as atlas
    spec=SimpleNamespace(key='dummy',context_hours=(96,))
    reg=SimpleNamespace(get_dataset_spec=lambda key:spec,
        load_protocol=lambda s,split:staggered_protocol())
    monkeypatch.setattr(atlas,'_registry',lambda:reg)
    with pytest.raises(ValueError,match='not both'):
        atlas.freeze_cohort('dummy','validation',8,3,3,96,22,tmp_path,
                             per_entity_origins=True, coverage_aware=True)
