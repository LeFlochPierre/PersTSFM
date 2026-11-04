"""Phase 3.6 CPU regression tests: chronology, audit, cache, locked scoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.experiments.adaptation_benchmark.history import Snapshot, make_series, save_snapshot, case_window
from src.experiments.adaptation_benchmark.capability import (
    SCHEMA,validate_config,make_plan,run,score_later, _trial_specs
)
from src.experiments.adaptation_benchmark.capability_models import (
    validate_lr_trace,trainable_summary,lora_weight_change
)


def snapshot(tmp_path, n=3):
    t=pd.date_range('2020-01-01',periods=80,freq='h')
    records=[make_series(f'e{i}',np.arange(80,dtype=float)*.3+i+np.sin(np.arange(80)/5),t) for i in range(n)]
    p=tmp_path/'history.npz'
    save_snapshot(Snapshot('TEST_1H',4,tuple(records)),p)
    return p


def config(path,**changes):
    d={'schema':SCHEMA,'task':'TEST_1H','snapshot':str(path),'max_entities':0,
       'context':8,'zero_contexts':[8,16], 'history':'full','batch_size':2,
       'trace_sampler':True,'reload_check':True,
       'variants':[{'name':'lora_s2_seed3','mode':'lora','steps':2,'batch_size':2,
                    'learning_rate':1e-5,'seed':3}]}
    d.update(changes)
    return d


class MockBackend:
    max_context=8192;model_name='FAKE';gpu='MOCK'
    def __init__(self,fail=None):
        self.state=0;self.calls=[];self.fail=fail
    def reset(self):self.state=0
    def forecast(self,records,*,horizon,context,lag,batch_size,cadence):
        self.calls.append(('forecast',lag,self.state,context))
        y=np.empty((len(records),horizon,4),dtype=np.float32)
        for i,r in enumerate(records):
            x,t,target=case_window(r,horizon,context,lag)
            if lag==1:assert target is None
            point=np.full(horizon,x[-1]+self.state*.02,dtype=np.float32)
            y[i]=np.stack([point,point-.1,point,point+.1],axis=-1)
        return y,0,.02
    def fit_audited(self,records,*,horizon,context,history,mode,steps,seed,batch_size,
                    learning_rate,output_dir,trace_sampler,reload_check,cadence,forecast_batch_size):
        assert history==len(records[0].y)-2*horizon
        assert mode=='lora'
        assert trace_sampler and reload_check
        self.calls.append(('fit',mode,steps,seed))
        if self.fail:
            raise RuntimeError('interrupted GPU')
        self.state=steps
        return .1,20.,{'verified_weight_update':True,'verified_checkpoint_reload':True,
                        'trainable_parameters':10,'trained_parameters_fraction':.01,
                        'observed_steps':steps,'checkpoint':'fake'}


def test_config_sane(tmp_path):
    c=validate_config(config(snapshot(tmp_path)))
    assert c['zero_contexts']==[8,16]
    assert len(_trial_specs(c))==3

@pytest.mark.parametrize('broken',[
    {'zero_contexts':[16]}, {'context':8193},{'history':1},
    {'variants':[{'name':'x','mode':'lora','steps':0,'batch_size':8,'learning_rate':1e-5,'seed':1}]},
    {'variants':[{'name':'x','mode':'full','steps':2,'batch_size':8,'learning_rate':0,'seed':1}]},
    {'variants':[{'name':'x','mode':'lora','steps':2,'batch_size':8,'learning_rate':1e-5,'seed':1},
                 {'name':'x','mode':'lora','steps':2,'batch_size':8,'learning_rate':1e-5,'seed':2}]},
    {'reload_check':'no'}, {'selection_metric':'later_mae'},
])
def test_invalid_config(tmp_path,broken):
    with pytest.raises((ValueError, TypeError)):
        make_plan(config(snapshot(tmp_path),**broken))


def test_plan_uses_full_pre_validation_pool(tmp_path):
    plan,cfg,h=make_plan(config(snapshot(tmp_path)))
    assert h==72
    assert len(plan.records)==3
    assert plan.snapshot.horizon==4


def test_lr_guard():
    ok=[{'step':i,'learning_rate':1e-5*(1-i/3)} for i in range(1,4)]
    result=validate_lr_trace(ok,3)
    assert result['observed_steps']==3
    with pytest.raises(RuntimeError):validate_lr_trace(ok,4)
    with pytest.raises(RuntimeError):validate_lr_trace([{'step':2,'learning_rate':1e-5}],1)


def test_trainable_and_lora_weight_checks_cpu():
    import torch
    from torch import nn
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('lora_B',nn.Parameter(torch.tensor([0.1,0.])))
            self.register_parameter('base',nn.Parameter(torch.ones(3),requires_grad=False))
    f=Fake()
    assert trainable_summary(f)['trainable_parameters']==2
    assert lora_weight_change(f)['verified_weight_update']
    f.lora_B.data.zero_()
    assert not lora_weight_change(f)['verified_weight_update']


def test_end_to_end_lock_and_cache(tmp_path):
    cfg=config(snapshot(tmp_path))
    b=MockBackend()
    out=run(cfg,output_root=tmp_path/'results',backend=b)
    assert (out/'selection.json').exists()
    lock=json.loads((out/'selection.json').read_text())
    assert lock['locked_before_later_scoring']
    assert (out/'validation.csv').exists()
    assert not (out/'later.csv').exists()
    assert any(x[0]=='fit' for x in b.calls)
    with pytest.raises(RuntimeError,match='locked'):
        run(cfg,output_root=tmp_path/'results',resume=out,backend=b)
    df=score_later(out)
    assert len(df)==3
    assert (out/'later.csv').exists()
    assert len(score_later(out))==3


def test_unfinished_resume_avoids_completed_trials(tmp_path):
    cfg=config(snapshot(tmp_path))
    b=MockBackend(fail=True)
    with pytest.raises(RuntimeError,match='interrupted GPU'):
        run(cfg,output_root=tmp_path/'results',backend=b)
    paths=list((tmp_path/'results').iterdir())
    assert len(paths)==1
    assert (paths[0]/'trials'/'zero_c8.json').exists()
    resumed=MockBackend()
    out=run(cfg,output_root=tmp_path/'results',resume=paths[0],backend=resumed)
    assert (out/'selection.json').exists()
    # Only the LoRA trial requires new inference: both zero-shot trials are cached
    assert len([x for x in resumed.calls if x[0]=='forecast'])==2


def test_freeze_tamper_guard(tmp_path):
    cfg=config(snapshot(tmp_path))
    out=run(cfg,output_root=tmp_path/'results',backend=MockBackend())
    with (out/'validation.csv').open('a') as f:f.write('tamper')
    with pytest.raises(RuntimeError,match='Locked input modified'):
        score_later(out)


def test_lag1_not_scored_by_run(tmp_path,monkeypatch):
    from src.experiments.adaptation_benchmark import capability as cap
    real=cap.case_window
    def guarded(r,h,c,lag,*,reveal_later=False):
        if lag==1 and reveal_later:
            raise AssertionError('Lag1 label read before strategy lock')
        return real(r,h,c,lag,reveal_later=reveal_later)
    monkeypatch.setattr(cap,'case_window',guarded)
    out=run(config(snapshot(tmp_path)),output_root=tmp_path/'results',backend=MockBackend())
    assert (out/'selection.json').exists()
    with pytest.raises(AssertionError):score_later(out)


def test_multi_seed_grouped_before_selection(tmp_path):
    p=snapshot(tmp_path)
    variants=[{'name':f'lora_s2_seed{s}','mode':'lora','steps':2,
               'batch_size':2,'learning_rate':1e-5,'seed':s} for s in (3,5)]
    c=config(p,variants=variants)
    out=run(c,output_root=tmp_path/'results',backend=MockBackend())
    table=pd.read_csv(out/'validation_strategies.csv')
    trained=table[table.steps==2]
    assert len(trained)==1
    assert trained.iloc[0]['n']==2
    df=score_later(out)
    assert len(df)==3  # two zero-shot strategies and one seed-mean LoRA
    assert df.loc[df.steps==2,'replicates'].iloc[0]==2
    assert len(pd.read_csv(out/'later_trials.csv'))==4


def test_mismatched_seed_sets_rejected(tmp_path):
    p=snapshot(tmp_path)
    variants=[
        {'name':'a','mode':'lora','steps':2,'batch_size':2,'learning_rate':1e-5,'seed':3},
        {'name':'b','mode':'lora','steps':4,'batch_size':2,'learning_rate':1e-5,'seed':5},
    ]
    with pytest.raises(ValueError,match='identical training-seed'):
        make_plan(config(p,variants=variants))


def test_baseline_parity_guard_before_training(tmp_path):
    p=snapshot(tmp_path)
    b=MockBackend()
    with pytest.raises(RuntimeError,match='parity FAILED'):
        run(config(p,zero_shot_parity={'8':100000.0}),output_root=tmp_path/'results',backend=b)
    assert not any(c[0]=='fit' for c in b.calls)
