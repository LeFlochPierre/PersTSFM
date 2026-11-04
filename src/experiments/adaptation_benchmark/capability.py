"""Phase 3.6: adaptation-capability audit on historical FEV input prefixes.

Additive to Phase 3.4/3.5. Explicit trials (not a combinatorial grid), exact
pre-lag2 training, lag2 strategy selection, protected lag1 scoring. No FEV
API or torch imported here. The separate model backend owns GPU operations.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import re
import numpy as np
import pandas as pd
from .history import load_snapshot, file_hash, history_hash, case_window
from .design import build_plan, choose_strategy
from .metrics import metrics
from .artifacts import write_json, atomic_npz, save_trial, read_trial

SCHEMA = 'pertsfm_phase36_capability_v1'
LOCK = 'pertsfm_phase36_lock_v1'


def _positive_int(value, key):
    if type(value) is not int or value < 1:
        raise ValueError(f'{key} must be a positive integer')
    return value


def validate_config(raw):
    if raw.get('schema') != SCHEMA:
        raise ValueError(f'Expected schema {SCHEMA}')
    cfg = dict(raw)
    for k in ('task', 'snapshot', 'variants'):
        if k not in cfg:
            raise ValueError(f'Missing {k}')
    cfg['context'] = _positive_int(cfg.get('context',768), 'context')
    cfg['zero_contexts'] = list(cfg.get('zero_contexts',[cfg['context']]))
    if not cfg['zero_contexts'] or any(type(c) is not int or c < 2 for c in cfg['zero_contexts']):
        raise ValueError('Invalid zero contexts')
    cfg['zero_contexts'] = sorted(set(cfg['zero_contexts']))
    if cfg['context'] not in cfg['zero_contexts']:
        raise ValueError('A zero-shot baseline at the exact adaptation context is compulsory')
    if max(cfg['zero_contexts'])>8192 or cfg['context']>8192:
        raise ValueError('Context exceeds model limit')
    cfg['max_entities'] = int(cfg.get('max_entities',0))
    cfg['batch_size'] = _positive_int(cfg.get('batch_size',8),'batch_size')
    if cfg['max_entities']<0:
        raise ValueError('max_entities must be >= 0')
    cfg['selection_metric'] = cfg.get('selection_metric','validation_mae')
    if cfg['selection_metric'] not in ('validation_mae','validation_pinball3'):
        raise ValueError('Invalid selection metric')
    cfg['trace_sampler'] = cfg.get('trace_sampler',True)
    cfg['reload_check'] = cfg.get('reload_check',True)
    if type(cfg['trace_sampler']) is not bool or cfg['reload_check'] is not True:
        raise ValueError('trace_sampler must be bool and reload_check must be true for an audited run')
    cfg['invalid_policy'] = cfg.get('invalid_policy','error')
    if cfg['invalid_policy'] not in ('error','drop'):
        raise ValueError('Invalid policy')
    cfg['history'] = cfg.get('history','full')
    if cfg['history']!='full':
        _positive_int(cfg['history'],'history')
    if not isinstance(cfg['variants'],list) or not cfg['variants']:
        raise ValueError('Provide explicit nonempty variants')
    variants=[]
    for v in cfg['variants']:
        if not isinstance(v,dict):
            raise ValueError('Variants must be objects')
        x=dict(v)
        if x.get('mode') not in ('lora','full'):
            raise ValueError('Unsupported adaptation mode')
        x['steps'] = _positive_int(x.get('steps'),'steps')
        x['batch_size'] = _positive_int(x.get('batch_size'),'batch_size')
        x['seed'] = _positive_int(x.get('seed'),'seed')
        lr=float(x.get('learning_rate',0))
        if not np.isfinite(lr) or lr<=0 or lr>=1:
            raise ValueError('Invalid learning rate')
        x['learning_rate']=lr
        if 'name' not in x or not re.fullmatch(r'[A-Za-z0-9_]+',str(x['name'])):
            raise ValueError('Variant name must be a safe distinct ID [A-Za-z0-9_]+')
        variants.append(x)
    if len({v['name'] for v in variants}) != len(variants):
        raise ValueError('Duplicate trial names')
    if any(v['name'].startswith('zero_') for v in variants):
        raise ValueError('Variant name collides with zero-shot namespace')
    # Strategy identity deliberately excludes random seed; the seed is a replicate,
    # not a selectable hyperparameter. Enforce matched seed sets across all
    # adaptation strategies so validation cannot cherry-pick a lucky seed.
    by_strategy={}
    for x in variants:
        strategy=(x['mode'],x['steps'],x['batch_size'],x['learning_rate'])
        if x['seed'] in by_strategy.setdefault(strategy,set()):
            raise ValueError(f'Duplicate seed for strategy {strategy}')
        by_strategy[strategy].add(x['seed'])
    if len({tuple(sorted(v)) for v in by_strategy.values()}) != 1:
        raise ValueError('All adaptation strategies must use identical training-seed sets')
    cfg['variants']=variants
    cfg['zero_shot_parity']=dict(cfg.get('zero_shot_parity',{}))
    cfg['zero_shot_atol']=float(cfg.get('zero_shot_atol',0.02))
    if cfg['zero_shot_atol'] <= 0 or not np.isfinite(cfg['zero_shot_atol']):
        raise ValueError('Invalid zero-shot parity tolerance')
    for k,v in cfg['zero_shot_parity'].items():
        if str(k) not in {str(x) for x in cfg['zero_contexts']} or not np.isfinite(float(v)):
            raise ValueError('Invalid zero-shot parity requirement')
    return cfg


def make_plan(raw):
    cfg=validate_config(raw)
    snapshot=Path(cfg['snapshot']).resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    # Reuse validated Phase 3.4 cohort feasibility, strictly pre-lag2 histories.
    # Dummy grid only checks feasibility; we run the explicit variant list below.
    dummy={
        'model':'chronos2','task':cfg['task'],'snapshot':str(snapshot),
        'contexts':cfg['zero_contexts'], 'lora_contexts':[cfg['context']],
        'histories':[cfg['history']],'budgets':[min(v['steps'] for v in cfg['variants'])],
        'seeds':[min(v['seed'] for v in cfg['variants'])],
        'max_entities':cfg['max_entities'],'batch_size':cfg['batch_size'],
        'train_batch_size':min(v['batch_size'] for v in cfg['variants']),
        'learning_rate':cfg['variants'][0]['learning_rate'],
        'invalid_policy':cfg['invalid_policy'], 'selection_metric':cfg['selection_metric'],
    }
    plan=build_plan(dummy,load_snapshot(snapshot),snapshot)
    history=min(len(r.y)-2*plan.snapshot.horizon for r in plan.records) if cfg['history']=='full' else cfg['history']
    cfg['snapshot']=str(snapshot)
    return plan,cfg,int(history)


def print_plan(plan,cfg,history):
    print('='*105)
    print('PERTSFM PHASE 3.6 | CHRONOS-2 ADAPTATION CAPABILITY AUDIT | HISTORICAL DEV ONLY')
    print('='*105)
    print(f"Dataset={plan.task} n={len(plan.records)} H={plan.snapshot.horizon} "
          f"train_history={history} C_lora={cfg['context']} zero_C={cfg['zero_contexts']}")
    print(f"Trials={len(cfg['variants']) + len(cfg['zero_contexts'])} "
          f"trace_sampler={cfg['trace_sampler']} reload_check={cfg['reload_check']}")
    for x in cfg['variants']:
        print(f"  {x['name']:35s} mode={x['mode']:4s} steps={x['steps']:4d} "
              f"train_batch={x['batch_size']:3d} lr={x['learning_rate']:.1e} seed={x['seed']}")
    print('Training ends BEFORE validation (lag2). Later lag1 labels are NOT used for choosing strategies.')
    print('This is DEVELOPMENT evidence, not an official FEV test or untouched family holdout.')


def manifest_for(plan,cfg,history):
    return {'schema':SCHEMA, 'task':plan.task,'horizon':plan.snapshot.horizon,
            'config':cfg, 'history':history, 'cadence':plan.cadence,
            'ids':[r.id for r in plan.records],
            'history_sha256':history_hash(plan.records),
            'snapshot_sha256':file_hash(Path(cfg['snapshot'])),
            'protocol':'lag2 validation strategy selection; lag1 scored ONLY in separate command',
            'interpretation':'Frozen adapter from pre-lag2 history, zero-shot can use newer lag1 context'}


def _trial_specs(cfg):
    return ([{'name':f'zero_c{c}','strategy':f'zero_c{c}',
              'context':c,'mode':'zero','steps':0} for c in cfg['zero_contexts']]
            + [{**v,'context':cfg['context'],
                'strategy':(f"{v['mode']}_c{cfg['context']}_steps{v['steps']}_"
                            f"lr{v['learning_rate']:.8g}_b{v['batch_size']}")}
               for v in cfg['variants']])


def run(config, *, output_root, resume=None, backend=None):
    plan,cfg,history=make_plan(config)
    print_plan(plan,cfg,history)
    output_root=Path(output_root)
    dest=Path(resume).resolve() if resume else (output_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+f'_{plan.task}_n{len(plan.records)}')).resolve()
    manifest=manifest_for(plan,cfg,history)
    if resume:
        if json.loads((dest/'manifest.json').read_text())!=manifest:
            raise RuntimeError('Resumption configuration or cohort changed')
        if (dest/'selection.json').exists():
            raise RuntimeError('Selection already locked')
    else:
        dest.mkdir(parents=True,exist_ok=False)
        write_json(dest/'manifest.json',manifest)
    if backend is None:
        from .capability_models import AuditedChronos2Backend
        backend=AuditedChronos2Backend()
    if max(cfg['zero_contexts'])>backend.max_context:
        raise RuntimeError('Loaded model context limit differs from plan')
    env={'python':platform.python_version(),'backend':backend.__class__.__name__,
         'model_name':backend.model_name, 'gpu':backend.gpu,'max_context':backend.max_context,
         'torch':str(getattr(getattr(backend,'torch',None),'__version__','none'))}
    if (dest/'environment.json').exists():
        if json.loads((dest/'environment.json').read_text())!=env:
            raise RuntimeError('Software/backend environment changed on resume')
    else:
        write_json(dest/'environment.json',env)
    h=plan.snapshot.horizon
    valid_truth=np.stack([case_window(r,h,2,2)[2] for r in plan.records])
    rows=[];pred_list=[];keys=[];entity_rows=[]
    specs=_trial_specs(cfg)
    try:
        for pos,spec in enumerate(specs,1):
            key=spec['name']
            cached=read_trial(dest,key,(2,len(plan.records),h,4))
            if cached is None:
                if spec['mode']=='zero':
                    backend.reset()
                    audit={'verified_weight_update':True,'verified_checkpoint_reload':True,
                           'trainable_parameters':0,'trained_parameters_fraction':0.,
                           'observed_steps':0,'checkpoint':''}
                    train_s=0.; peak=0.
                else:
                    # Every trial restarts from a fresh copy of the pretrained checkpoint.
                    train_s,peak,audit=backend.fit_audited(plan.records,horizon=h,context=cfg['context'],
                        history=history,mode=spec['mode'],steps=spec['steps'],seed=spec['seed'],
                        batch_size=spec['batch_size'],learning_rate=spec['learning_rate'],
                        output_dir=dest/'adapters'/key,trace_sampler=cfg['trace_sampler'],
                        reload_check=cfg['reload_check'],cadence=plan.cadence,
                        forecast_batch_size=cfg['batch_size'])
                    if not (audit['verified_weight_update'] and audit['verified_checkpoint_reload']):
                        raise RuntimeError('Adaptation verification failed; forecasts not accepted')
                    if audit['observed_steps']!=spec['steps']:
                        raise RuntimeError('Incomplete step accounting')
                try:
                    pred2,cross2,sec2=backend.forecast(plan.records,horizon=h,context=spec['context'],lag=2,
                        batch_size=cfg['batch_size'],cadence=plan.cadence)
                    pred1,cross1,sec1=backend.forecast(plan.records,horizon=h,context=spec['context'],lag=1,
                        batch_size=cfg['batch_size'],cadence=plan.cadence)
                finally:
                    backend.reset()
                both=np.stack([pred2,pred1])
                meta={'trial_key':key,'strategy':spec['strategy'],'mode':spec['mode'],
                      'context':spec['context'],'steps':spec['steps'],'seed':int(spec.get('seed',-1)),
                      'history':history if spec['mode']!='zero' else 0,
                      'learning_rate':float(spec.get('learning_rate',0)),
                      'train_batch_size':int(spec.get('batch_size',0)),
                      'train_seconds':train_s,'peak_allocated_mib':peak,
                      'validation_seconds':sec2,'later_seconds':sec1,
                      'validation_crossing_steps':cross2,'later_crossing_steps':cross1,
                      'audit':audit}
                save_trial(dest,key,both,meta)
            else:
                both,meta=cached
                print(f'[{pos}/{len(specs)}] CACHE {key}',flush=True)
            evaluated=metrics(both[0],valid_truth)
            if spec['mode']=='zero' and str(spec['context']) in cfg['zero_shot_parity']:
                expected=float(cfg['zero_shot_parity'][str(spec['context'])])
                if abs(evaluated['mae']-expected)>cfg['zero_shot_atol']:
                    raise RuntimeError(f'Zero-shot parity FAILED at C={spec["context"]}: {evaluated["mae"]:.5f} != {expected:.5f}')
            row={k:meta[k] for k in ('trial_key','strategy','mode','context','steps','seed','history',
                  'learning_rate','train_batch_size','train_seconds','peak_allocated_mib',
                  'validation_seconds','later_seconds','validation_crossing_steps')}
            row.update(validation_mae=evaluated['mae'],validation_pinball3=evaluated['pinball3'],
                       verified_weight_update=meta['audit']['verified_weight_update'],
                       verified_checkpoint_reload=meta['audit']['verified_checkpoint_reload'])
            rows.append(row);pred_list.append(both);keys.append(key)
            for i,r in enumerate(plan.records):
                entity_rows.append({'trial_key':key,'entity_id':r.id,
                                    'validation_mae':float(evaluated['entity_mae'][i]),
                                    'validation_pinball3':float(evaluated['entity_pinball3'][i])})
            print(f'[{pos}/{len(specs)}] {key:36s} val_MAE={evaluated["mae"]:.5f} '
                  f'train={meta["train_seconds"]:.1f}s delta_verified={meta["audit"]["verified_weight_update"]}',flush=True)
    finally:
        backend.reset()
    valid=pd.DataFrame(rows)
    valid.to_csv(dest/'validation.csv',index=False)
    pd.DataFrame(entity_rows).to_csv(dest/'validation_entity.csv',index=False)
    atomic_npz(dest/'forecasts.npz',forecasts=np.stack(pred_list),trial_keys=np.asarray(keys,dtype=np.str_))
    winner,summary=choose_strategy(rows,cfg['selection_metric'])
    pd.DataFrame(summary).to_csv(dest/'validation_strategies.csv',index=False)
    lock={'schema':LOCK,'winner':winner,'selection_metric':cfg['selection_metric'],
          'locked_before_later_scoring':True}
    for f in ('manifest.json','validation.csv','validation_entity.csv','validation_strategies.csv',
              'forecasts.npz','environment.json'):
        lock[f+'_sha256']=file_hash(dest/f)
    write_json(dest/'selection.json',lock)
    print('\nVALIDATION ONLY | explicit adaptation capability trials:')
    print(valid[['strategy','mode','steps','learning_rate','train_batch_size','validation_mae',
                 'train_seconds']].sort_values('validation_mae').to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f'LOCKED: {winner}\nPASS: {dest}')
    print(f'NEXT: python -m scripts.phase3.adaptation_capability score-later --run {dest}')
    return dest


def score_later(run_dir):
    root=Path(run_dir).resolve()
    m=json.loads((root/'manifest.json').read_text())
    lock=json.loads((root/'selection.json').read_text())
    if m.get('schema')!=SCHEMA or lock.get('schema')!=LOCK or not lock.get('locked_before_later_scoring'):
        raise RuntimeError('Missing valid selection lock')
    for f in ('manifest.json','validation.csv','validation_entity.csv','validation_strategies.csv',
              'forecasts.npz','environment.json'):
        if file_hash(root/f)!=lock[f+'_sha256']:
            raise RuntimeError(f'Locked input modified: {f}')
    if file_hash(Path(m['config']['snapshot'])) != m['snapshot_sha256']:
        raise RuntimeError('Snapshot changed')
    if (root/'later.csv').exists():
        print('ALREADY SCORED; refusing overwrite. Existing scores:')
        existing=pd.read_csv(root/'later.csv')
        print(existing.to_string(index=False))
        return existing
    snap=load_snapshot(Path(m['config']['snapshot']))
    by_id={r.id:r for r in snap.records}
    recs=[by_id[i] for i in m['ids']]
    if snap.task!=m['task'] or snap.horizon!=m['horizon'] or history_hash(recs)!=m['history_sha256']:
        raise RuntimeError('Historical cohort changed')
    truth=np.stack([case_window(r,snap.horizon,2,1,reveal_later=True)[2] for r in recs])
    with np.load(root/'forecasts.npz',allow_pickle=False) as z:
        y=np.asarray(z['forecasts'],dtype=np.float32)
        keys=z['trial_keys'].astype(str).tolist()
    valid=pd.read_csv(root/'validation.csv')
    if (keys!=valid.trial_key.tolist() or len(keys)!=len(set(keys)) or
        y.shape!=(len(keys),2,len(recs),snap.horizon,4)):
        raise RuntimeError('Locked predictions mismatch')
    rows=[]; entities=[]
    for i,r in enumerate(valid.to_dict('records')):
        scored=metrics(y[i,1],truth)
        rows.append({'strategy':r['strategy'],'mode':r['mode'],'context':r['context'],
                     'steps':r['steps'],'train_seconds':r['train_seconds'],
                     'validation_mae':r['validation_mae'],
                     'later_mae':scored['mae'],'later_pinball3':scored['pinball3']})
        for j,record in enumerate(recs):
            entities.append({'trial_key':r['trial_key'],'entity_id':record.id,
                             'later_mae':float(scored['entity_mae'][j])})
    trials=pd.DataFrame(rows)
    # Aggregate multiple independent seeds BEFORE ranking strategies.
    df=trials.groupby('strategy',as_index=False,sort=False).agg(
        mode=('mode','first'),context=('context','first'),steps=('steps','first'),
        replicates=('later_mae','count'),train_seconds=('train_seconds','mean'),
        validation_mae=('validation_mae','mean'),later_mae=('later_mae','mean'),
        later_mae_sd=('later_mae','std'),later_pinball3=('later_pinball3','mean'))
    df['later_mae_sd']=df['later_mae_sd'].fillna(0.)
    if lock['winner'] not in set(df.strategy):
        raise RuntimeError('Chosen strategy missing')
    trials.to_csv(root/'later_trials.csv',index=False)
    df.to_csv(root/'later.csv',index=False)
    pd.DataFrame(entities).to_csv(root/'later_entity.csv',index=False)
    oracle=df.loc[df.later_mae.idxmin()]
    print('\nPHASE 3.6 | LATER HISTORICAL DEVELOPMENT EVALUATION')
    print(df.sort_values('later_mae').to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f"LOCKED WINNER: {lock['winner']} later MAE={float(df.loc[df.strategy==lock['winner'],'later_mae'].iloc[0]):.5f}")
    print(f'HINDSIGHT BEST (diagnostic only): {oracle.strategy}, MAE={oracle.later_mae:.5f}')
    print('Historical preofficial later labels are NOT independent final test data.')
    return df
