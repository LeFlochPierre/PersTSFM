"""CPU-only tests: rolling temporal boundaries, frozen strategy, sampler audit."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.experiments.adaptation_benchmark.history import (
    Snapshot,make_series,save_snapshot,load_snapshot,training_pools,case_window,file_hash,
)
from src.experiments.adaptation_benchmark.rolling import (
    SCHEMA,make_rolling_plan,describe_rolling,initialize_suite,load_suite,run_suite,score_suite,
)
from src.experiments.adaptation_benchmark.sampler_audit import pool_window_audit,inspect_sampler_source
from src.experiments.adaptation_benchmark.uncertainty import paired_entity_bootstrap
from src.experiments.adaptation_benchmark.experiment import run_one
from src.experiments.adaptation_benchmark.reporting import report_suite


def data(path,n=3,length=100,h=4):
    records=[]
    for i in range(n):
        t=pd.date_range('2022-01-01',periods=length,freq='h')
        y=np.asarray(np.arange(length)+i,dtype=float)
        records.append(make_series(f'e{i}',y,t))
    save_snapshot(Snapshot('TEST_1H',h,tuple(records)),path)
    return path


def conf(path,**kwargs):
    exp={'task':'TEST_1H','snapshot':str(path),'model':'chronos2',
         'max_entities':0,'contexts':[6,16],'lora_contexts':[6],
         'histories':['full'],'budgets':[2],'seeds':[42,43],
         'batch_size':2,'train_batch_size':2,'learning_rate':1e-5,
         'invalid_policy':'error','selection_metric':'validation_mae'}
    opt={'folds':3,'stride_horizons':1,'require_aligned_ends':True}
    exp.update(kwargs.get('experiment',{}))
    opt.update(kwargs.get('rolling',{}))
    return {'schema':SCHEMA,'experiment':exp,'rolling':opt}


class FakeBackend:
    max_context=8192
    gpu='CPU-FAKE'
    model_name='fake'
    def __init__(self):
        self.state=(0,0)
    def reset(self): self.state=(0,0)
    def fit_lora(self,records,*,horizon,context,history,steps,seed,batch_size,learning_rate,output_dir):
        assert all(len(y)==history for y in training_pools(records,horizon,history))
        self.state=(steps,seed)
        return .01,1.
    def forecast(self,records,*,horizon,context,lag,batch_size,cadence):
        x=[]
        for r in records:
            inp,t,target=case_window(r,horizon,context,lag)
            if lag==1: assert target is None
            offset=(.1*self.state[0])
            pred=np.full(horizon,inp[-1]+offset,dtype=np.float32)
            x.append(np.stack((pred,pred-1,pred,pred+1),axis=1))
        return np.stack(x),0,.01


def test_rolling_fold_strict_prevalidation(tmp_path):
    p=data(tmp_path/'original.npz',length=100)
    plan=make_rolling_plan(conf(p))
    assert [len(s.records[0].y) for s in plan.folds]==[92,96,100]
    assert all([r.id for r in f.records]==['e0','e1','e2'] for f in plan.folds)
    for i,fold in enumerate(plan.folds):
        r=fold.records[0]; h=fold.horizon
        assert len(training_pools(fold.records,h,len(r.y)-2*h)[0])==len(r.y)-2*h
        assert r.time[-2*h-1]<r.time[-2*h]<r.time[-h]<r.time[-1]
        assert case_window(r,h,6,1)[2] is None
        if i:
            older=plan.folds[i-1].records[0]
            assert older.time[-1]<r.time[-h]


def test_rolling_source_not_mutated_and_old_future_not_available(tmp_path):
    path=data(tmp_path/'source.npz')
    before=file_hash(path)
    p=make_rolling_plan(conf(path))
    oldest=p.folds[0].records[0]
    newest=p.folds[-1].records[0]
    assert len(oldest.y)<len(newest.y)
    assert np.max(oldest.y)<np.max(newest.y)
    assert file_hash(path)==before


def test_rolling_no_overlapping_evaluation(tmp_path):
    path=data(tmp_path/'source.npz')
    plan=make_rolling_plan(conf(path,rolling={'folds':4,'stride_horizons':2}))
    for a,b in zip(plan.folds,plan.folds[1:]):
        assert a.records[0].time[-1]<b.records[0].time[-b.horizon]


def test_unaligned_end_rejected(tmp_path):
    path=data(tmp_path/'source.npz')
    snap=load_snapshot(path)
    r=snap.records[1]
    bad=Snapshot(snap.task,snap.horizon,(snap.records[0],make_series(r.id,r.y,r.time+pd.Timedelta(hours=1)),snap.records[2]))
    p=tmp_path/'bad.npz'; save_snapshot(bad,p)
    with pytest.raises(ValueError,match='Unaligned'):
        make_rolling_plan(conf(p))


def test_invalid_rolling_options(tmp_path):
    path=data(tmp_path/'source.npz')
    with pytest.raises(ValueError,match='stride'):
        make_rolling_plan(conf(path,rolling={'stride_horizons':0}))
    with pytest.raises(ValueError,match='folds'):
        make_rolling_plan(conf(path,rolling={'folds':1}))


def test_fixed_complete_cohort_drop_policy(tmp_path):
    path=data(tmp_path/'source.npz')
    snap=load_snapshot(path)
    rec=list(snap.records)
    rec[1].y[80]=np.nan  # within earliest fold's pre-val training prefix
    p=tmp_path/'with_nan.npz';save_snapshot(Snapshot(snap.task,snap.horizon,tuple(rec)),p)
    with pytest.raises(ValueError):make_rolling_plan(conf(p))
    plan=make_rolling_plan(conf(p,experiment={'invalid_policy':'drop'}))
    assert plan.entities==('e0','e2')
    assert plan.excluded==('e1',)
    assert all([r.id for r in f.records]==['e0','e2'] for f in plan.folds)


def test_suite_freeze_snapshots_and_integrity(tmp_path):
    path=data(tmp_path/'source.npz')
    plan=make_rolling_plan(conf(path))
    out=initialize_suite(plan,tmp_path/'roll')
    _,metadata=load_suite(out)
    assert len(metadata['folds'])==3
    assert len(metadata['entity_ids'])==3
    assert len(load_snapshot(Path(metadata['folds'][0]['snapshot'])).records[0].y)==92
    with pytest.raises(FileExistsError):initialize_suite(plan,out)
    with Path(metadata['folds'][0]['snapshot']).open('ab') as f:f.write(b'bad')
    with pytest.raises(RuntimeError,match='changed'):
        load_suite(out)


def _fake_suite(tmp_path):
    orig=data(tmp_path/'source.npz')
    suite=initialize_suite(make_rolling_plan(conf(orig)),tmp_path/'suite')
    _,meta=load_suite(suite)
    # Write runs using GPU-free fake, then populate suite references.
    for item in meta['folds']:
        folder=suite/'runs'/f"fold_{item['index']:02d}"
        cfg=json.loads(Path(item['config']).read_text())['experiments'][0]
        run=run_one(cfg,output_root=folder,backend=FakeBackend())
        ref=suite/'folds'/f"fold_{item['index']:02d}"/'run_ref.json'
        ref.write_text(json.dumps({'run':str(run),'selection_sha256':file_hash(run/'selection.json')}))
    return suite,meta


def test_score_requires_every_fold_locked(tmp_path):
    path=data(tmp_path/'source.npz')
    suite=initialize_suite(make_rolling_plan(conf(path)),tmp_path/'suite')
    with pytest.raises(RuntimeError,match='not locked'):
        score_suite(suite)
    assert not (suite/'suite_freeze.json').exists()


def test_full_fake_rolling_suite_locked_before_score(tmp_path):
    suite,meta=_fake_suite(tmp_path)
    assert not (suite/'rolling_summary.csv').exists()
    assert all(not list((suite/'runs'/f"fold_{i:02d}").rglob('later.csv')) for i in range(3))
    df=score_suite(suite)
    assert len(df)==3
    assert (suite/'suite_freeze.json').exists()
    assert (suite/'rolling_summary.csv').exists()
    assert list(df.fold)==[0,1,2]
    old=(suite/'rolling_summary.csv').read_bytes()
    score_suite(suite)
    assert (suite/'rolling_summary.csv').read_bytes()==old
    with pytest.raises(RuntimeError,match='already frozen'):
        run_suite(suite)


def test_selected_lock_tampering_rejected_before_scoring(tmp_path):
    suite,meta=_fake_suite(tmp_path)
    item=meta['folds'][1]
    ref=suite/'folds'/f"fold_{item['index']:02d}"/'run_ref.json'
    info=json.loads(ref.read_text())
    with (Path(info['run'])/'selection.json').open('a') as f:f.write(' ')
    with pytest.raises(RuntimeError,match='altered'):
        score_suite(suite)
    assert not (suite/'suite_freeze.json').exists()


def test_sampler_pool_eligible_window_count_and_before_validation(tmp_path):
    path=data(tmp_path/'source.npz')
    plan=make_rolling_plan(conf(path))
    df=pool_window_audit(plan.folds[0].records,4,context=6,history=40)
    assert len(df)==3
    assert (df.eligible_context_target_windows==31).all()
    assert (df.finite_training_points==40).all()
    assert df.actual_draws_observed.eq(False).all()
    with pytest.raises(ValueError,match='history'):
        pool_window_audit(plan.folds[0].records,4,context=6,history=6)
    assert inspect_sampler_source(None)['realized_sample_trace']=='NOT_OBSERVED'


def test_paired_entity_bootstrap_identical_and_positive_gain():
    base=np.array([3.,4.,5.,7.])
    ci=paired_entity_bootstrap(base,base*.9,replicates=1000,seed=12)
    assert ci['gain_pct']==pytest.approx(10.)
    assert ci['ci_lower_pct']==pytest.approx(10.)
    assert ci['ci_upper_pct']==pytest.approx(10.)
    assert ci['unit']=='paired_entity'
    with pytest.raises(ValueError):paired_entity_bootstrap([1.],[1.])


def test_suite_report_rejects_modified_frozen_selection(tmp_path):
    suite,meta=_fake_suite(tmp_path)
    score_suite(suite)
    item=meta['folds'][0]
    ref=json.loads((suite/'folds'/f"fold_{item['index']:02d}"/'run_ref.json').read_text())
    with (Path(ref['run'])/'selection.json').open('a') as f: f.write(' ')
    with pytest.raises(RuntimeError,match='changed'):
        report_suite(suite)

class FakeChronosMode:
    TRAIN='train'


class FakeChronosDataset:
    """Exact audited RNG call signature from Chronos 2, stripped to CPU numpy."""
    def __init__(self):
        self.inputs=[{'context':np.ones((1,40))},{'context':np.ones((1,60))}]
        self.prediction_length=4
        self.context_length=6
        self.min_past=4
        self.mode=FakeChronosMode.TRAIN

    def _construct_slice(self,input_idx):
        full_length=self.inputs[input_idx]['context'].shape[-1]
        slice_idx = np.random.randint(self.min_past, full_length - self.prediction_length + 1)
        return slice_idx


def test_actual_sampler_trace_records_real_draws_without_rng_change(tmp_path):
    from src.experiments.adaptation_benchmark.sampler_audit import trace_chronos_draws,save_observed_draws
    ds=FakeChronosDataset()
    np.random.seed(12)
    reference=[ds._construct_slice(i%2) for i in range(8)]
    np.random.seed(12)
    before=np.random.randint
    with trace_chronos_draws(dataset_class=FakeChronosDataset) as trace:
        observed=[ds._construct_slice(i%2) for i in range(8)]
    assert np.random.randint is before
    assert observed==reference
    assert len(trace.rows)==8
    assert all(r['target_end_exclusive']<=r['pool_length'] for r in trace.rows)
    assert {r['input_index'] for r in trace.rows}=={0,1}
    info=save_observed_draws(trace,tmp_path/'draws',expected_steps=2)
    assert info['draws_observed']==8
    assert info['distinct_entity_indices']==2
    with pytest.raises(FileExistsError):save_observed_draws(trace,tmp_path/'draws')


def test_actual_sampler_trace_restores_rng_on_exception():
    from src.experiments.adaptation_benchmark.sampler_audit import trace_chronos_draws
    old=np.random.randint
    with pytest.raises(RuntimeError,match='intentional'):
        with trace_chronos_draws(dataset_class=FakeChronosDataset):
            raise RuntimeError('intentional')
    assert np.random.randint is old


def test_sampler_audit_rejects_changed_implementation():
    from src.experiments.adaptation_benchmark.sampler_audit import trace_chronos_draws
    class ChangedSampler:
        def _construct_slice(self,input_idx):
            return 2
    with pytest.raises(RuntimeError,match='differs'):
        with trace_chronos_draws(dataset_class=ChangedSampler):pass
