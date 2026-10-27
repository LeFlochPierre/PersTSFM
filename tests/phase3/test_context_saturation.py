from datetime import datetime, timedelta
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT=Path(__file__).resolve().parents[2] / 'scripts' / 'phase3' / 'context_saturation.py'
spec=importlib.util.spec_from_file_location('context_saturation',SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample(n=30):
    return {'id':'test_entity',
            'y':np.arange(n,dtype=np.float64),
            'time':pd.date_range('2020-01-01',periods=n,freq='h')}


def test_older_and_later_cutoffs_do_not_overlap_targets():
    w=mod.case_windows(sample(),horizon=3,context=6,reveal_lag1=True)
    np.testing.assert_array_equal(w[2]['values'],[18,19,20,21,22,23])
    np.testing.assert_array_equal(w[2]['truth'],[24,25,26])
    np.testing.assert_array_equal(w[1]['values'],[21,22,23,24,25,26])
    np.testing.assert_array_equal(w[1]['truth'],[27,28,29])
    assert w[2]['times'][-1] < w[2]['future_times'][0]


def test_context_feasibility_guard():
    with pytest.raises(ValueError):
        mod.case_windows(sample(),horizon=3,context=25)
    with pytest.raises(RuntimeError):
        mod.check_and_plan([sample()],horizon=3,contexts=[6,25])
    with pytest.raises(ValueError):
        mod.check_and_plan([sample()],horizon=3,contexts=[6,6])


def test_nan_guard_checks_exact_model_input():
    r=sample()
    r['y'][19]=np.nan
    with pytest.raises(ValueError):
        mod.case_windows(r,horizon=3,context=6)


def test_raw_probabilistic_metrics():
    q=np.zeros((2,3,4),dtype=float)
    q[:,:,0]=np.array([[1,2,3],[2,3,4]])
    q[:,:,1]=q[:,:,0]-1
    q[:,:,2]=q[:,:,0]
    q[:,:,3]=q[:,:,0]+1
    truth=q[:,:,0].copy()
    result=mod.score(q,truth)
    assert result['mae']==pytest.approx(0.0)
    assert result['pinball3']>=0
    with pytest.raises(ValueError):
        mod.score(q,truth[:1])


def test_context_selection_uses_validation_only():
    data=[{'context':96,'validation_mae':4.,'later_mae':1.},
          {'context':768,'validation_mae':3.,'later_mae':30.},
          {'context':8192,'validation_mae':3.,'later_mae':0.}]
    assert mod.select_context(data)==768


def test_preofficial_sha_deterministic_and_cadence():
    r=sample()
    assert mod.sha256_histories([r])==mod.sha256_histories([r])
    assert mod.cadence_minutes([r])==60
    r2=sample();r2['y'][0]=-11
    assert mod.sha256_histories([r])!=mod.sha256_histories([r2])


def test_take_record_timestamps_and_format():
    t=pd.date_range('2020-01-01',periods=30,freq='h')
    r=mod.take_record({'id':'x','ts':list(t),'target':np.arange(30)},entity_key='id',time_key='ts',target_key='target')
    assert r['id']=='x' and len(r['time'])==30


def test_snapshot_roundtrip_without_fev(tmp_path):
    a=sample(35)
    a['id']='a'
    b=sample(35)
    b['id']='b'
    b['y']=b['y']*2
    path=tmp_path/'full_history.npz'
    mod.save_snapshot([a,b],'M_DENSE_1H',3,path)
    name,h,loaded=mod.read_snapshot(path)
    assert name=='M_DENSE_1H' and h==3
    assert [r['id'] for r in loaded]==['a','b']
    np.testing.assert_array_equal(loaded[1]['y'],b['y'])
    assert mod.sha256_histories(loaded)==mod.sha256_histories([a,b])
    with pytest.raises(FileExistsError):
        mod.save_snapshot([a,b],'M_DENSE_1H',3,path)


def test_snapshot_rejects_corruption(tmp_path):
    path=tmp_path/'bad.npz'
    r=sample(35)
    mod.save_snapshot([r],'M_DENSE_1H',3,path)
    with np.load(path,allow_pickle=False) as z:
        kw={k:z[k] for k in z.files}
    kw['values']=kw['values'].copy()
    kw['values'][1]=999.
    np.savez_compressed(path,**kw)
    with pytest.raises(RuntimeError,match='hash mismatch'):
        mod.read_snapshot(path)


def test_lag1_truth_masked_by_default():
    w=mod.case_windows(sample(),horizon=3,context=6)
    assert w[2]['truth'] is not None
    assert w[1]['truth'] is None


def test_later_scoring_works_without_fev_import(tmp_path):
    import json
    from types import SimpleNamespace
    r=sample()
    snap=tmp_path/'input.npz'
    mod.save_snapshot([r],'M_DENSE_1H',3,snap)
    folder=tmp_path/'locked_run'
    folder.mkdir()
    pred=np.zeros((1,2,1,3,4),dtype=np.float32)
    pred[0,1,0,:,0]=[27,28,29]
    pred[0,1,0,:,1:]=np.array([[26,27,28],[27,28,29],[28,29,30]])
    np.savez_compressed(folder/'predictions.npz',forecast=pred,
                        contexts=np.array([6],dtype=np.int32))
    (folder/'validation.csv').write_text('context,validation_mae\n6,0.0\n')
    manifest={
        'schema':'pertsfm_3_3c_v1','task':'M_DENSE_1H','horizon':3,
        'ids':['test_entity'],'contexts':[6],
        'history_sha256':mod.sha256_histories([r]),
        'snapshot_path':str(snap),'snapshot_sha256':mod.sha256_file(snap),
    }
    (folder/'manifest.json').write_text(json.dumps(manifest))
    selected={
        'schema':'pertsfm_3_3c_selection_v1','chosen_context':6,
        'locked_before_lag1_scoring':True,
        'manifest_sha256':mod.sha256_file(folder/'manifest.json'),
        'predictions_sha256':mod.sha256_file(folder/'predictions.npz'),
        'validation_sha256':mod.sha256_file(folder/'validation.csv')
    }
    (folder/'selection.json').write_text(json.dumps(selected))
    mod.later(SimpleNamespace(score_later=folder))
    result=pd.read_csv(folder/'later.csv')
    assert result.iloc[0].later_mae == pytest.approx(0.0)
    with pytest.raises(FileExistsError):
        mod.later(SimpleNamespace(score_later=folder))


def test_timestamp_us_ns_are_equivalent_for_cadence_and_hash():
    base=sample(36)
    one_ns=dict(base, time=base['time'].as_unit('ns'))
    one_us=dict(base, time=base['time'].as_unit('us'))
    assert one_ns['time'].asi8[0] == one_us['time'].asi8[0]*1000
    assert mod.cadence_minutes([one_ns]) == 60
    assert mod.cadence_minutes([one_us]) == 60
    assert mod.sha256_histories([one_ns]) == mod.sha256_histories([one_us])


def test_microsecond_snapshot_roundtrip_preserves_calendar_dates(tmp_path):
    r=sample(36)
    r['time']=r['time'].as_unit('us')
    p=tmp_path/'corrected.npz'
    mod.save_snapshot([r], 'M_DENSE_1H', 3, p)
    name, horizon, loaded=mod.read_snapshot(p)
    assert name == 'M_DENSE_1H' and horizon == 3
    assert loaded[0]['time'][0].year == 2020
    np.testing.assert_array_equal(mod.timestamp_ns(loaded[0]['time']),
                                  mod.timestamp_ns(r['time']))
    assert mod.cadence_minutes(loaded) == 60


def test_legacy_unversioned_snapshot_refused(tmp_path):
    r=sample(36)
    p=tmp_path/'old.npz'
    mod.save_snapshot([r], 'M_DENSE_1H', 3, p)
    with np.load(p,allow_pickle=False) as z:
        values={k:z[k] for k in z.files if k not in ('snapshot_schema','timestamp_unit')}
    np.savez_compressed(p,**values)
    with pytest.raises(RuntimeError,match='Legacy/unknown snapshot'):
        mod.read_snapshot(p)


def test_irregular_microsecond_cadence_rejected():
    r=sample(30)
    time=r['time'].as_unit('us').tolist()
    time[10]+=timedelta(minutes=2)
    r['time']=pd.DatetimeIndex(time).as_unit('us')
    with pytest.raises(RuntimeError,match='irregular'):
        mod.cadence_minutes([r])
