"""GPU-free scientific and integration regression tests for Phase 3.4 core."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.experiments.adaptation_benchmark.history import (
    Series,Snapshot,make_series,save_snapshot,load_snapshot,timestamp_ns,history_hash,
    case_window,training_pools,cadence_minutes,file_hash,
)
from src.experiments.adaptation_benchmark.metrics import metrics,organise_forecast
from src.experiments.adaptation_benchmark.features import compute_features
from src.experiments.adaptation_benchmark.design import (
    normalise,build_plan,choose_strategy,load_batch,
)
from src.experiments.adaptation_benchmark.experiment import prepare,run_one,score_later_run
from src.experiments.adaptation_benchmark.artifacts import read_trial


def cohort(n=3, length=64, horizon=4):
    records=[]
    for i in range(n):
        t=pd.date_range('2020-01-01',periods=length,freq='h')
        y=np.arange(length,dtype=np.float64)+i*5 + np.sin(np.arange(length)/3)*.1
        records.append(make_series(f'entity_{i}',y,t))
    return Snapshot('DEMO_1H',horizon,tuple(records))


def settings(path, **overrides):
    config={'task':'DEMO_1H','model':'chronos2','snapshot':str(path),
            'contexts':[6,16],'lora_contexts':[6], 'histories':['full'],
            'budgets':[2,4], 'seeds':[3,5], 'max_entities':0,
            'batch_size':2,'train_batch_size':2,'learning_rate':1e-5,
            'invalid_policy':'error','selection_metric':'validation_mae'}
    config.update(overrides)
    return config


class DummyBackend:
    """Deterministic Chronos interface with no torch, for protocol tests."""
    max_context=8192
    gpu='CPU MOCK'
    model_name='fake-chronos'

    def __init__(self, fail_on=None):
        self.state=(0,0,0)
        self.calls=[]
        self.fail_on=fail_on

    def fit_lora(self,records,*,horizon,context,history,steps,seed,batch_size,learning_rate,output_dir):
        self.state=(history,steps,seed)
        self.calls.append(('fit',steps,seed))
        if self.fail_on==('fit',steps,seed):
            raise RuntimeError('simulated GPU crash')
        assert all(len(v)==history for v in training_pools(records,horizon,history))
        return 0.01,50.0

    def reset(self):
        self.state=(0,0,0)

    def forecast(self,records,*,horizon,context,lag,batch_size,cadence):
        self.calls.append(('forecast',context,lag))
        out=np.empty((len(records),horizon,4),dtype=np.float32)
        for i,r in enumerate(records):
            x,_,label=case_window(r,horizon,context,lag)
            if lag==1:
                assert label is None  # lag1 target must not be read by runner
            adjustment = 0. if self.state[1]==0 else .15 + self.state[1]/30 + self.state[2]/1000
            point=np.full(horizon,x[-1]+adjustment,dtype=np.float32)
            out[i]=np.stack([point,point-.4,point,point+.4],axis=-1)
        return out,0,0.002


def test_timestamp_units_and_cadence():
    snap=cohort()
    assert cadence_minutes(snap.records)==60
    t=snap.records[0].time
    assert np.array_equal(timestamp_ns(t.as_unit('us')),timestamp_ns(t.as_unit('ns')))


def test_snapshot_roundtrip_matches_canonical_hash(tmp_path):
    snap=cohort()
    path=tmp_path/'full.npz'
    save_snapshot(snap,path)
    read=load_snapshot(path)
    assert read.task==snap.task and read.horizon==snap.horizon
    assert history_hash(read.records)==history_hash(snap.records)
    with pytest.raises(FileExistsError):
        save_snapshot(snap,path)


def test_snapshot_rejects_legacy_timestamp_schema(tmp_path):
    p=tmp_path/'bad.npz'
    np.savez_compressed(p,ids=np.asarray(['a']), offsets=[0,2],values=[1.,2.],timestamps_ns=[100,200])
    with pytest.raises(RuntimeError,match='legacy'):
        load_snapshot(p)


def test_snapshot_hash_tampering(tmp_path):
    snap=cohort()
    path=tmp_path/'full.npz'
    save_snapshot(snap,path)
    with np.load(path,allow_pickle=False) as z:
        arrs={x:z[x] for x in z.files}
    arrs['values']=arrs['values'].copy()
    arrs['values'][0]+=3
    np.savez_compressed(path,**arrs)
    with pytest.raises(RuntimeError,match='hash mismatch'):
        load_snapshot(path)


def test_features_are_strictly_pre_lag2():
    snap=cohort()
    before,_=compute_features(snap.records,4,60)
    snap.records[0].y[-8:]=1e12  # mutate BOTH score windows
    after,_=compute_features(snap.records,4,60)
    pd.testing.assert_frame_equal(before,after)


def test_causal_windows_and_no_lag1_truth():
    snap=cohort(length=40)
    r=snap.records[0]
    a,t,y=case_window(r,4,16,2)
    b,u,later=case_window(r,4,16,1)
    assert y.tolist()==r.y[-8:-4].tolist()
    assert later is None
    np.testing.assert_allclose(a,r.y[-24:-8].astype(np.float32))
    np.testing.assert_allclose(b,r.y[-20:-4].astype(np.float32))
    assert t[-1]<r.time[-8] and u[-1]<r.time[-4]
    assert case_window(r,4,16,1,reveal_later=True)[2].tolist()==r.y[-4:].tolist()


def test_training_pool_causally_excludes_both_target_windows():
    snap=cohort(length=64)
    pools=training_pools(snap.records,4,32)
    np.testing.assert_allclose(pools[0],snap.records[0].y[-40:-8].astype(np.float32))
    before=pools[0].copy()
    snap.records[0].y[-8:]=1e6
    np.testing.assert_allclose(training_pools(snap.records,4,32)[0],before)


def test_training_history_excess_rejected():
    with pytest.raises(ValueError,match='extends'):
        training_pools(cohort().records,4,60)


def test_metric_parity_entity_macro():
    target=np.array([[1.,2.],[4.,5.]])
    pred=np.zeros((2,2,4),dtype=float)
    pred[:,:,0]=[[1,4],[2,3]]
    pred[:,:,1:]=pred[:,:,0,None]
    res=metrics(pred,target)
    assert res['mae']==pytest.approx((1+2)/2)
    assert res['pinball3']>=0


def test_quantile_rearrange_point_unchanged():
    arr,cross=organise_forecast([1,2],[2,3],[1,2],[0,1],2)
    assert cross==2
    assert arr[:,0].tolist()==[1.,2.]
    assert np.all(np.diff(arr[:,1:],axis=1)>=0)


def test_selection_aggregates_seeds_instead_of_cherry_picking():
    rows=[{'strategy':'zero_c6','validation_mae':3.,'validation_pinball3':2.,'steps':0,'context':6,'history':0},
          {'strategy':'lora_c6_h24_s2','validation_mae':1.,'validation_pinball3':1.,'steps':2,'context':6,'history':24},
          {'strategy':'lora_c6_h24_s2','validation_mae':7.,'validation_pinball3':4.,'steps':2,'context':6,'history':24}]
    winner,summaries=choose_strategy(rows)
    assert winner=='zero_c6'
    assert next(x for x in summaries if x['strategy']=='lora_c6_h24_s2')['mean']==4.
    rows[1]['validation_mae']=3.
    rows[2]['validation_mae']=3.
    assert choose_strategy(rows)[0]=='zero_c6'


def test_selection_is_independent_of_later_metrics():
    rows=[{'strategy':'zero_c6','validation_mae':1.,'validation_pinball3':1.,'steps':0,'context':6,'history':0},
          {'strategy':'lora_c6_h24_s2','validation_mae':2.,'validation_pinball3':2.,'steps':2,'context':6,'history':24}]
    ref=choose_strategy(rows)[0]
    rows[0]['later_mae']=99999
    rows[1]['later_mae']=-100
    assert choose_strategy(rows)[0]==ref


def test_bad_configuration_rejected():
    cfg=settings('dummy.npz',lora_contexts=[20])
    with pytest.raises(ValueError,match='LoRA contexts'):
        normalise(cfg)
    cfg=settings('dummy.npz',histories=[2])
    with pytest.raises(ValueError,match='LoRA history'):
        build_plan(cfg,cohort(),Path('dummy.npz'))


def test_missing_data_requires_predeclared_drop_policy(tmp_path):
    snap=cohort()
    snap.records[1].y[-30]=float('nan')
    p=tmp_path/'series.npz'
    save_snapshot(snap,p)
    with pytest.raises(ValueError,match='incompatible'):
        build_plan(settings(p),load_snapshot(p),p)
    plan=build_plan(settings(p,invalid_policy='drop'),load_snapshot(p),p)
    assert len(plan.records)==2 and plan.excluded[0]['id']=='entity_1'


def test_plan_is_deterministic(tmp_path):
    p=tmp_path/'series.npz'
    save_snapshot(cohort(),p)
    first=prepare(settings(p))
    second=prepare(settings(p))
    assert [t.key for t in first.trials]==[t.key for t in second.trials]
    assert len(first.trials)==6  # 2 zero-shot + 2 budgets x 2 seeds


def test_full_integration_lock_score_and_idempotence(tmp_path):
    p=tmp_path/'series.npz'
    save_snapshot(cohort(),p)
    config=settings(p)
    run=run_one(config,output_root=tmp_path/'out',backend=DummyBackend())
    assert (run/'selection.json').is_file()
    assert (run/'entity_features.csv').is_file() and (run/'dataset_features.csv').is_file()
    assert not (run/'later.csv').exists()
    valid=pd.read_csv(run/'validation.csv')
    assert len(valid)==6
    strat=pd.read_csv(run/'validation_strategies.csv')
    assert len(strat)==4
    group=score_later_run(run)
    assert len(group)==4 and (run/'later_entity.csv').is_file()
    # Re-running must print existing results and never overwrite.
    old_bytes=(run/'later.csv').read_bytes()
    score_later_run(run)
    assert (run/'later.csv').read_bytes()==old_bytes


def test_lock_tamper_detection(tmp_path):
    p=tmp_path/'series.npz'
    save_snapshot(cohort(),p)
    run=run_one(settings(p),output_root=tmp_path/'runs',backend=DummyBackend())
    with (run/'validation.csv').open('a') as f:
        f.write('tampered\n')
    with pytest.raises(RuntimeError,match='altered'):
        score_later_run(run)


def test_snapshot_mutation_detected_after_lock(tmp_path):
    p=tmp_path/'series.npz'
    save_snapshot(cohort(),p)
    run=run_one(settings(p),output_root=tmp_path/'runs',backend=DummyBackend())
    with p.open('ab') as f: f.write(b'tamper')
    with pytest.raises(RuntimeError,match='Snapshot bytes'):
        score_later_run(run)


def test_resume_only_missing_trials(tmp_path):
    p=tmp_path/'series.npz'
    save_snapshot(cohort(),p)
    config=settings(p)
    with pytest.raises(RuntimeError,match='simulated GPU crash'):
        run_one(config,output_root=tmp_path/'runs',backend=DummyBackend(fail_on=('fit',2,5)))
    dirs=list((tmp_path/'runs').iterdir())
    assert len(dirs)==1
    run=dirs[0]
    assert (run/'trials/zero_c6.npz').exists()
    assert (run/'trials/lora_c6_h56_s2_seed3.npz').exists()
    dummy=DummyBackend()
    resumed=run_one(config,output_root=tmp_path/'runs',resume=run,backend=dummy)
    assert resumed==run
    assert len(pd.read_csv(run/'validation.csv'))==6
    assert [x for x in dummy.calls if x[0]=='fit']==[('fit',2,5),('fit',4,3),('fit',4,5)]
    with pytest.raises(RuntimeError,match='already locked'):
        run_one(config,output_root=tmp_path/'runs',resume=run,backend=DummyBackend())


def test_corrupt_cached_trial_rejected(tmp_path):
    p=tmp_path/'series.npz'
    save_snapshot(cohort(),p)
    config=settings(p)
    with pytest.raises(RuntimeError,match='simulated GPU crash'):
        run_one(config,output_root=tmp_path/'runs',backend=DummyBackend(fail_on=('fit',2,5)))
    run=next((tmp_path/'runs').iterdir())
    path=run/'trials/zero_c6.npz'
    with path.open('ab') as f: f.write(b'mutated')
    with pytest.raises(RuntimeError,match='Corrupt cached trial'):
        run_one(config,output_root=tmp_path/'runs',resume=run,backend=DummyBackend())


def test_resume_rejects_modified_config(tmp_path):
    p=tmp_path/'series.npz'
    save_snapshot(cohort(),p)
    config=settings(p)
    with pytest.raises(RuntimeError,match='simulated GPU crash'):
        run_one(config,output_root=tmp_path/'runs',backend=DummyBackend(fail_on=('fit',2,5)))
    run=next((tmp_path/'runs').iterdir())
    with pytest.raises(RuntimeError,match='Resume mismatch'):
        run_one(settings(p,learning_rate=2e-5),output_root=tmp_path/'runs',resume=run,backend=DummyBackend())


def test_batch_config_load_and_schema(tmp_path):
    f=tmp_path/'batch.json'
    f.write_text(json.dumps({'schema':'pertsfm_phase34_batch_v1','experiments':[{'task':'hi'}]}))
    assert load_batch(f)[0]['task']=='hi'
    f.write_text(json.dumps({'schema':'wrong','experiments':[]}))
    with pytest.raises(ValueError,match='schema'):
        load_batch(f)
