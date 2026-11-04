"""No GPU, FEV network or official target dependencies."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.experiments.adaptation_benchmark.atlas import (
    read_config,plan_dataset,compile_plan,capability_config,
    summarise_validation,_bootstrap_gain, summarize_root)
from src.experiments.adaptation_benchmark.history import (
    Snapshot,make_series,save_snapshot,load_snapshot,history_hash,case_window)
from src.experiments.adaptation_benchmark.capability import make_plan


def setup(tmp_path,n=24,length=1400,h=24):
    time=pd.date_range('2026-01-01',periods=length,freq='h')
    records=[]
    for k in range(n):
        y=10+np.arange(length)*.001+np.sin(np.arange(length)*.26)+k/10
        records.append(make_series(f'entity_{k:03d}',y,time))
    snap=tmp_path/'history.npz'
    save_snapshot(Snapshot('SYNTH',h,tuple(records)),snap)
    entry={'key':'synth','task':'SYNTH','family':'synthetic','role':'development',
           'snapshot':str(snap),'preferred_context':384,'max_zero_context':1536,
           'minimum_entities':8}
    cfg={'schema':'pertsfm_phase37a_atlas_v1','datasets':[entry],
         'screening':{'steps':[100,500], 'learning_rates':[1e-5,3e-5],
                      'seeds':[42,43],'train_batch_size':32,'inference_batch_size':8,
                      'sample_entities':16,'sample_seed':371}}
    return cfg,entry


def test_config_validation(tmp_path):
    cfg,_=setup(tmp_path)
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg))
    assert read_config(p)==cfg
    cfg['datasets'].append(dict(cfg['datasets'][0]))
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='duplicated'):
        read_config(p)


def test_hashed_sample_plan_and_core_contract(tmp_path):
    cfg,entry=setup(tmp_path)
    p=plan_dataset(entry,cfg)
    assert len(p.records)==16 and p.n_total==24
    assert p.lora_context==384
    assert p.zero_contexts==(192,384)
    # Sorted-input prefix of 16 IDs should not be our selected cohort.
    assert [r.id for r in p.records]!=[r.id for r in p.snapshot.records[:16]]
    p2=plan_dataset(entry,cfg)
    assert [r.id for r in p.records]==[r.id for r in p2.records]
    out,meta=compile_plan(p,cfg,tmp_path/'atlas')
    again,_=compile_plan(p,cfg,tmp_path/'atlas')
    assert out==again
    assert history_hash(load_snapshot(out).records)==meta['selected_history_sha256']
    settings=capability_config(p,cfg,out)
    core,normalized,history=make_plan(settings)
    assert len(core.records)==16 and history==1352
    assert len(settings['variants'])==8
    assert all(v['seed'] in [42,43] for v in settings['variants'])
    assert settings['zero_contexts']==[192,384]


def test_context_fallback_and_missing(tmp_path):
    cfg,entry=setup(tmp_path,length=170,h=24)
    p=plan_dataset(entry,cfg)
    assert p.lora_context==48 # length - 4*H =74, so 48 is max
    assert p.zero_contexts==(24,48)
    entry['snapshot']='missing.npz'
    with pytest.raises(FileNotFoundError):plan_dataset(entry,cfg,cwd=tmp_path)


def test_manifest_refuses_unexpected_cohort_change(tmp_path):
    cfg,e=setup(tmp_path)
    p=plan_dataset(e,cfg)
    compile_plan(p,cfg,tmp_path/'atlas')
    cfg['screening']['sample_seed']=372
    p2=plan_dataset(e,cfg)
    with pytest.raises(RuntimeError,match='provenance changed'):
        compile_plan(p2,cfg,tmp_path/'atlas')


def test_cohort_validation_does_not_use_later_target(tmp_path):
    cfg,e=setup(tmp_path)
    snap=load_snapshot(e['snapshot'])
    original=plan_dataset(e,cfg)
    # Modify ONLY later (lag1) truth; data becomes different, but cohort and
    # selected context rules do not change. Lag1 truth is not read in plan.
    adjusted=[]
    for r in snap.records:
        y=r.y.copy()
        y[-snap.horizon:]=np.nan
        adjusted.append(make_series(r.id,y,r.time))
    alt=tmp_path/'altered.npz'
    save_snapshot(Snapshot('SYNTH',snap.horizon,tuple(adjusted)),alt)
    e2={**e,'snapshot':str(alt)}
    later=plan_dataset(e2,cfg)
    assert [r.id for r in later.records]==[r.id for r in original.records]
    assert later.zero_contexts==original.zero_contexts
    # But the later truth cannot be revealed as a valid score.
    with pytest.raises(ValueError):case_window(later.records[0],snap.horizon,24,1,reveal_later=True)


def test_paired_gain_and_summary(tmp_path):
    root=tmp_path/'runs'/'demo'/'run_0';root.mkdir(parents=True)
    ids=['a','b','c']
    rows=[{'trial_key':'zero_c96','strategy':'zero_c96','mode':'zero','context':96,
           'steps':0,'validation_mae':10.,'train_seconds':0.},
          {'trial_key':'zero_c384','strategy':'zero_c384','mode':'zero','context':384,
           'steps':0,'validation_mae':8.,'train_seconds':0.},
          {'trial_key':'lora_one','strategy':'lora_c96_steps100_lr1e-05_b32',
           'mode':'lora','context':96,'steps':100,'validation_mae':7.,'train_seconds':11.},
          {'trial_key':'lora_two','strategy':'lora_c96_steps100_lr1e-05_b32',
           'mode':'lora','context':96,'steps':100,'validation_mae':7.,'train_seconds':11.}]
    pd.DataFrame(rows).to_csv(root/'validation.csv',index=False)
    all_entities=[]
    for row in rows:
        for eid in ids:
            all_entities.append({'trial_key':row['trial_key'],'entity_id':eid,
               'validation_mae':row['validation_mae']})
    pd.DataFrame(all_entities).to_csv(root/'validation_entity.csv',index=False)
    (root/'selection.json').write_text(json.dumps({'winner':rows[2]['strategy']}))
    (root/'manifest.json').write_text(json.dumps({'horizon':24}))
    z=summarise_validation(root,{'family':'test','source_sha256':'abc','lora_context':96})
    assert z['lora_gain_best_zero_pct']==pytest.approx(12.5)
    assert z['lora_gain_matched_zero_pct']==pytest.approx(30.)
    assert z['n_entities']==3 and z['entity_win_fraction']==1.
    assert not z['screening_threshold_met'] # fewer than 16 entities
    assert z['selected_is_lora'] is True
    assert np.allclose(_bootstrap_gain([8,8,8],[7,7,7],1),[12.5,12.5])
    registry={'demo':{'run_dir':str(root),'plan':{'family':'test','source_sha256':'abc',
                  'lora_context':96},'sample_seed':1}}
    (tmp_path/'registry.json').write_text(json.dumps(registry))
    summary=summarize_root(tmp_path)
    assert len(summary)==1 and (tmp_path/'atlas_validation_summary.csv').exists()


def test_missing_seed_entity_is_blocked(tmp_path):
    root=tmp_path/'run';root.mkdir()
    rows=[{'trial_key':'zero_c96','strategy':'zero_c96','mode':'zero','context':96,
           'steps':0,'validation_mae':10.,'train_seconds':0.},
          {'trial_key':'lora_one','strategy':'lora_c96_steps100_lr1e-05_b32',
           'mode':'lora','context':96,'steps':100,'validation_mae':9.,'train_seconds':11.}]
    pd.DataFrame(rows).to_csv(root/'validation.csv',index=False)
    pd.DataFrame([{'trial_key':'zero_c96','entity_id':'a','validation_mae':10.},
                  {'trial_key':'lora_one','entity_id':'b','validation_mae':9.}]).to_csv(root/'validation_entity.csv',index=False)
    (root/'selection.json').write_text(json.dumps({'winner':rows[1]['strategy']}))
    (root/'manifest.json').write_text(json.dumps({'horizon':24}))
    with pytest.raises(RuntimeError,match='Unmatched per-entity'):
        summarise_validation(root)


def test_simple_reference_forecasts_are_causal(tmp_path):
    from src.experiments.adaptation_benchmark.atlas import naive_validation_references
    cfg,e=setup(tmp_path,n=8,length=300,h=24)
    original=load_snapshot(e['snapshot'])
    first=naive_validation_references(original)
    assert set(first.model)=={'last_value','seasonal_naive'}
    assert set(first.seasonal_lag)=={1,24}
    records=[]
    for r in original.records:
        values=r.y.copy();values[-original.horizon:]=1e20
        records.append(make_series(r.id,values,r.time))
    alternate=Snapshot(original.task,original.horizon,tuple(records))
    second=naive_validation_references(alternate)
    pd.testing.assert_frame_equal(first,second)


def test_real_phase36_runner_with_atlas_config_and_mock_backend(tmp_path):
    from src.experiments.adaptation_benchmark.capability import run as phase36_run
    cfg,entry=setup(tmp_path,n=18,length=850,h=24)
    cfg['screening'].update(steps=[2],learning_rates=[1e-5],seeds=[42],sample_entities=12)
    p=plan_dataset(entry,cfg)
    snap,meta=compile_plan(p,cfg,tmp_path/'output')
    config=capability_config(p,cfg,snap)
    class Fake:
        max_context=8192
        model_name='fake'
        gpu='cpu'
        def __init__(self):self.offset=0
        def reset(self):self.offset=0
        def fit_audited(self,records,*,horizon,context,history,mode,steps,seed,
                        batch_size,learning_rate,output_dir,trace_sampler,reload_check,cadence,forecast_batch_size):
            self.offset=.01
            return .1,0,{'verified_weight_update':True,'verified_checkpoint_reload':True,
                         'observed_steps':steps,'checkpoint':'fake'}
        def forecast(self,records,*,horizon,context,lag,batch_size,cadence):
            arr=[]
            for record in records:
                x,_,target=case_window(record,horizon,context,lag)
                assert target is None if lag==1 else target is not None
                pred=np.full(horizon,x[-1]+self.offset,dtype=np.float32)
                arr.append(np.stack([pred,pred-.1,pred,pred+.1],axis=-1))
            return np.stack(arr),0,.02
    run=phase36_run(config,output_root=tmp_path/'fake_runs',backend=Fake())
    assert (run/'selection.json').is_file()
    assert not (run/'later.csv').exists()
    result=summarise_validation(run,meta)
    assert result['n_entities']==12
    assert np.isfinite(result['best_naive_mae'])
    assert (run/'atlas_naive_validation.csv').exists()
    assert not (run/'later.csv').exists()
