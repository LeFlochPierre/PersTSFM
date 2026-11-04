"""Chronos-2 cross-dataset benchmark: plan, train/cache, lock, score-later.

GPU training is isolated in models.py; tests can use a deterministic fake backend.
All model-selection choices depend only on lag2 labels. Later period labels
are accessed ONLY in score_later_run(), a separate explicit command.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import time
import numpy as np
import pandas as pd
from .history import load_snapshot, history_hash, file_hash, case_window
from .design import build_plan, choose_strategy
from .features import compute_features
from .metrics import metrics
from .artifacts import write_json, atomic_npz, save_trial, read_trial

SCHEMA = 'pertsfm_phase34_runner_v1'
LOCK_SCHEMA = 'pertsfm_phase34_lock_v1'


def prepare(config: dict):
    snap_path = Path(config['snapshot']).resolve()
    if not snap_path.is_file():
        raise FileNotFoundError(snap_path)
    return build_plan(config, load_snapshot(snap_path), snap_path)


def describe(plan):
    n,h=len(plan.records),plan.snapshot.horizon
    print('='*104,flush=True)
    print('PERTSFM 3.4 | MODULAR CHRONOS-2 ADAPTATION BENCHMARK | PREOFFICIAL HISTORIES ONLY',flush=True)
    print('='*104,flush=True)
    print(f'Task={plan.task} | matched entities={n} | excluded={len(plan.excluded)} | H={h} | cadence={plan.cadence}min',flush=True)
    print(f'History lengths min={min(len(r.y) for r in plan.records)} | pre-lag2 pool min={min(len(r.y)-2*h for r in plan.records)}',flush=True)
    print(f"Contexts={plan.config['contexts']} | LoRA contexts={plan.config['lora_contexts']}",flush=True)
    print(f"Budgets={plan.config['budgets']} | Seeds={plan.config['seeds']} | candidates={len(plan.trials)}",flush=True)
    print(f"Training histories={sorted({t.history for t in plan.trials if t.steps})}",flush=True)
    print('SELECT on lag2 only; lag1 predictions saved unscored; official FEV test labels never accessed.',flush=True)
    print('NOTE: Passing a full history pool to Chronos.fit does NOT prove uniform sampling across it.',flush=True)
    if plan.excluded:
        print('EXCLUSIONS (predeclared validity only):',flush=True)
        for x in plan.excluded[:12]:
            print(f"  {x['id']}: {x['reason']}",flush=True)
        if len(plan.excluded)>12: print(f'  ... {len(plan.excluded)-12} more listed in manifest',flush=True)


def manifest_for_plan(plan):
    cfg=dict(plan.config)
    cfg['snapshot']=str(Path(cfg['snapshot']).resolve())
    return {
        'schema':SCHEMA, 'task':plan.task, 'model':'amazon/chronos-2',
        'horizon':plan.snapshot.horizon, 'cadence_minutes':plan.cadence,
        'snapshot_path':plan.snapshot_path, 'snapshot_sha256':file_hash(Path(plan.snapshot_path)),
        'history_sha256':history_hash(plan.records),
        'ids':[r.id for r in plan.records], 'excluded':list(plan.excluded),
        'config':cfg, 'trials':[vars(t) for t in plan.trials],
        'selection_protocol':'train before lag2; choose by lag2 seed-mean; lag1 separately scored after lock',
        'metric_protocol':'entity macro MAE / mean pinball q10,q50,q90; quantiles rearranged; point unchanged',
        'training_sampling_audit':'Chronos2.fit pool supplied; internal sampled training windows not independently audited',
    }


def _verify_manifest(plan, path):
    existing=json.loads((path/'manifest.json').read_text())
    if existing != manifest_for_plan(plan):
        raise RuntimeError('Resume mismatch: experiment config, cohort, snapshot or trial grid changed')
    if (path/'selection.json').exists():
        raise RuntimeError('Run already locked; score using score-later, not --resume')
    return existing


def _validation_truth(plan):
    # Only lag2 truths accessed inside the whole training/evaluation procedure.
    return np.stack([case_window(r,plan.snapshot.horizon,2,2)[2] for r in plan.records])


def _write_final(run_dir, plan, rows, pred_list, trial_keys, validation_entities):
    valid=pd.DataFrame(rows)
    if len(valid)!=len(plan.trials) or valid['trial_key'].duplicated().any():
        raise RuntimeError('Incomplete/duplicated validation trials')
    # Keep a compact, compatible tensor for separate lag1 evaluation.
    atomic_npz(run_dir/'forecasts.npz', forecasts=np.stack(pred_list),
               trial_keys=np.asarray(trial_keys,dtype=np.str_))
    valid.to_csv(run_dir/'validation.csv',index=False)
    pd.DataFrame(validation_entities).to_csv(run_dir/'validation_entity.csv',index=False)
    winner, summary = choose_strategy(rows,plan.config['selection_metric'])
    pd.DataFrame(summary).to_csv(run_dir/'validation_strategies.csv',index=False)
    lock = {
        'schema':LOCK_SCHEMA,'strategy':winner,'selection_metric':plan.config['selection_metric'],
        'locked_before_lag1_scoring':True,
        **{key:file_hash(run_dir/name) for key,name in [
            ('manifest_sha256','manifest.json'),('validation_sha256','validation.csv'),
            ('forecasts_sha256','forecasts.npz'),('validation_strategies_sha256','validation_strategies.csv'),
            ('validation_entity_sha256','validation_entity.csv'),
            ('entity_features_sha256','entity_features.csv'),
            ('dataset_features_sha256','dataset_features.csv'),
            ('environment_sha256','environment.json')]}
    }
    write_json(run_dir/'selection.json',lock)
    print('\nVALIDATION ONLY | strategies averaged over available training seeds:',flush=True)
    print(pd.DataFrame(summary)[['strategy','mean','std','n','steps','context','history']].to_string(index=False,float_format=lambda x:f'{x:.5f}'),flush=True)
    print(f'LOCKED STRATEGY: {winner}',flush=True)
    print(f'PASS: {run_dir}',flush=True)
    print(f'NEXT: python -m scripts.phase3.cross_dataset_adaptation score-later --run {run_dir}',flush=True)
    return winner


def run_one(config: dict, *, output_root: Path, resume: Path | None = None, backend=None):
    """Run one dataset; supplying backend enables fast GPU-free integration tests."""
    plan=prepare(config)
    describe(plan)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=Path(resume) if resume else Path(output_root)/f'{stamp}_{plan.task}_n{len(plan.records)}'
    if resume:
        _verify_manifest(plan,dest)
        print(f'RESUMING {dest}',flush=True)
    else:
        dest.mkdir(parents=True,exist_ok=False)
        write_json(dest/'manifest.json',manifest_for_plan(plan))
    if backend is None:
        from .models import Chronos2Backend
        tic=time.perf_counter()
        backend=Chronos2Backend()
        print(f'Loaded {backend.model_name} on {backend.gpu} in {time.perf_counter()-tic:.2f}s',flush=True)
    if max(plan.config['contexts']) > backend.max_context:
        raise RuntimeError(f'Model actual max_context={backend.max_context} smaller than requested')
    env={
        'python':platform.python_version(),
        'backend':backend.__class__.__name__,
        'model_name':str(backend.model_name),
        'gpu':str(backend.gpu),
        'model_max_context':int(backend.max_context),
        'torch':str(getattr(getattr(backend,'torch',None),'__version__','none')),
        'transformers':str(getattr(getattr(backend,'transformers',None),'__version__','none')),
        'peft':str(getattr(getattr(backend,'peft',None),'__version__','none')),
    }
    if (dest/'environment.json').exists():
        if json.loads((dest/'environment.json').read_text())!=env:
            raise RuntimeError('Resume software, checkpoint backend, GPU or context capability differs')
    else:
        write_json(dest/'environment.json',env)
    # Fingerprints are computed ONLY from training prefixes, never from lag2 or lag1.
    # Generate once and keep byte-stable on resume (cannot silently overwrite).
    if not (dest/'entity_features.csv').exists() and not (dest/'dataset_features.csv').exists():
        entity_features,dataset_features=compute_features(plan.records,plan.snapshot.horizon,plan.cadence)
        entity_features.to_csv(dest/'entity_features.csv',index=False)
        dataset_features.to_csv(dest/'dataset_features.csv',index=False)
    elif not (dest/'entity_features.csv').exists() or not (dest/'dataset_features.csv').exists():
        raise RuntimeError('Incomplete fingerprint artifacts; inspect before resuming')
    horizon=plan.snapshot.horizon
    truth=_validation_truth(plan)
    forecasts=[]
    keys=[]
    rows=[]
    entity_rows=[]
    for number,trial in enumerate(plan.trials,1):
        key=trial.key
        cached=read_trial(dest,key,(2,len(plan.records),horizon,4))
        if cached is not None:
            both,meta=cached
            print(f'[{number}/{len(plan.trials)}] CACHE {key}',flush=True)
        else:
            if trial.steps:
                train_sec,peak_mib=backend.fit_lora(plan.records,horizon=horizon,
                    context=trial.context,history=trial.history,steps=trial.steps,seed=trial.seed,
                    batch_size=plan.config['train_batch_size'],
                    learning_rate=plan.config['learning_rate'],
                    output_dir=dest/'adapters'/key/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            else:
                backend.reset()
                train_sec,peak_mib=0.,0.
            try:
                lag2,cross2,t2=backend.forecast(plan.records,horizon=horizon,context=trial.context,
                    lag=2,batch_size=plan.config['batch_size'],cadence=plan.cadence)
                lag1,cross1,t1=backend.forecast(plan.records,horizon=horizon,context=trial.context,
                    lag=1,batch_size=plan.config['batch_size'],cadence=plan.cadence)
            finally:
                backend.reset()
            both=np.stack([lag2,lag1],axis=0)
            meta={'trial_key':key,'strategy':trial.strategy,'context':trial.context,
                  'history':trial.history,'steps':trial.steps,'seed':trial.seed,
                  'train_seconds':train_sec,'peak_allocated_mib':peak_mib,
                  'validation_infer_seconds':t2,'later_infer_seconds':t1,
                  'lag2_crossing_steps':cross2,'lag1_crossing_steps':cross1}
            # Store forecasts BEFORE validation metrics to enable interrupted-run replay.
            save_trial(dest,key,both,meta)
        evaluated=metrics(both[0],truth)
        row={**{k:meta[k] for k in ('trial_key','strategy','context','history','steps','seed',
                'train_seconds','peak_allocated_mib','validation_infer_seconds','later_infer_seconds',
                'lag2_crossing_steps','lag1_crossing_steps')},
             'validation_mae':evaluated['mae'],'validation_pinball3':evaluated['pinball3']}
        rows.append(row)
        for i,r in enumerate(plan.records):
            entity_rows.append({'trial_key':key,'strategy':trial.strategy,'seed':trial.seed,
                'entity_id':r.id,'validation_mae':float(evaluated['entity_mae'][i]),
                'validation_pinball3':float(evaluated['entity_pinball3'][i])})
        forecasts.append(both)
        keys.append(key)
        print(f"[{number}/{len(plan.trials)}] {key:35s} VAL MAE={evaluated['mae']:.5f} "
              f"pinball={evaluated['pinball3']:.5f} train={meta['train_seconds']:.2f}s",flush=True)
        expected=plan.config.get('zero_shot_parity',{})
        if not trial.steps and str(trial.context) in expected:
            if abs(evaluated['mae']-float(expected[str(trial.context)]))>float(plan.config.get('zero_shot_atol',.02)):
                raise RuntimeError(f'{key}: zero-shot MAE parity FAILED, expected {expected[str(trial.context)]}')
    _write_final(dest,plan,rows,forecasts,keys,entity_rows)
    return dest


def score_later_run(run_dir: Path):
    root=Path(run_dir)
    lock=json.loads((root/'selection.json').read_text())
    meta=json.loads((root/'manifest.json').read_text())
    if (lock.get('schema')!=LOCK_SCHEMA or meta.get('schema')!=SCHEMA or
        lock.get('locked_before_lag1_scoring') is not True):
        raise RuntimeError('Missing valid locked lag2 selection')
    for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.csv'),
                     ('forecasts_sha256','forecasts.npz'),('validation_strategies_sha256','validation_strategies.csv'),
                     ('validation_entity_sha256','validation_entity.csv'),
                     ('entity_features_sha256','entity_features.csv'),
                     ('dataset_features_sha256','dataset_features.csv'),
                     ('environment_sha256','environment.json')]:
        if file_hash(root/name)!=lock[key]:
            raise RuntimeError(f'Frozen selection input altered: {name}')
    snapshot_path=Path(meta['snapshot_path'])
    if file_hash(snapshot_path)!=meta['snapshot_sha256']:
        raise RuntimeError('Snapshot bytes changed since strategy lock')
    if (root/'later.csv').exists():
        print(f'ALREADY SCORED: {root}/later.csv (will not overwrite)',flush=True)
        existing=pd.read_csv(root/'later.csv')
        print(existing[['strategy','later_mae','gain_vs_zero_same_context_pct']].to_string(index=False),flush=True)
        return existing
    snap=load_snapshot(snapshot_path)
    by_id={r.id:r for r in snap.records}
    records=[by_id[i] for i in meta['ids']]
    if (snap.task!=meta['task'] or snap.horizon!=meta['horizon'] or
        history_hash(records)!=meta['history_sha256']):
        raise RuntimeError('Chronological cohort changed since lock')
    truth=np.stack([case_window(r,snap.horizon,2,1,reveal_later=True)[2] for r in records])
    with np.load(root/'forecasts.npz',allow_pickle=False) as z:
        forecasts=np.asarray(z['forecasts'],dtype=np.float32)
        keys=z['trial_keys'].astype(str).tolist()
    valid=pd.read_csv(root/'validation.csv')
    if (keys!=valid['trial_key'].tolist() or len(set(keys))!=len(keys) or
        forecasts.shape!=(len(keys),2,len(records),snap.horizon,4)):
        raise RuntimeError('Saved predictions do not match frozen validation rows')
    all_rows,entity_rows=[],[]
    for idx,row in enumerate(valid.to_dict('records')):
        scored=metrics(forecasts[idx,1],truth)
        all_rows.append({'trial_key':row['trial_key'],'strategy':row['strategy'],
                         'seed':int(row['seed']),'context':int(row['context']),
                         'history':int(row['history']),'steps':int(row['steps']),
                         'later_mae':scored['mae'],'later_pinball3':scored['pinball3']})
        for j,r in enumerate(records):
            entity_rows.append({'trial_key':row['trial_key'],'strategy':row['strategy'],'seed':int(row['seed']),
                                'entity_id':r.id,'later_mae':float(scored['entity_mae'][j]),
                                'later_pinball3':float(scored['entity_pinball3'][j])})
    result=pd.DataFrame(all_rows)
    groups=result.groupby('strategy',sort=False).agg(
        context=('context','first'),history=('history','first'),steps=('steps','first'),
        replicates=('trial_key','count'),later_mae=('later_mae','mean'),
        later_mae_sd=('later_mae','std'),later_pinball3=('later_pinball3','mean')).reset_index()
    groups['later_mae_sd']=groups['later_mae_sd'].fillna(0.)
    by_zero={int(row.context):float(row.later_mae) for row in groups.itertuples() if int(row.steps)==0}
    groups['gain_vs_zero_same_context_pct']=groups.apply(
        lambda r:100*(by_zero[int(r['context'])]-float(r['later_mae']))/by_zero[int(r['context'])],axis=1)
    # Record both (a) frozen selected policy and (b) hindsight diagnostic only.
    if lock['strategy'] not in set(groups.strategy):
        raise RuntimeError('Chosen strategy absent')
    result.to_csv(root/'later_trials.csv',index=False)
    pd.DataFrame(entity_rows).to_csv(root/'later_entity.csv',index=False)
    groups.to_csv(root/'later.csv',index=False)
    chosen=groups.loc[groups.strategy==lock['strategy']].iloc[0]
    oracle=groups.loc[groups.later_mae.idxmin()]
    print('\nPHASE 3.4 | LATER HISTORICAL EVALUATION (LAG2 CHOICE FROZEN)',flush=True)
    print(groups[['strategy','later_mae','later_mae_sd','gain_vs_zero_same_context_pct']].to_string(index=False,float_format=lambda x:f'{x:.5f}'),flush=True)
    print(f"LOCKED: {chosen.strategy} | later MAE={chosen.later_mae:.5f}",flush=True)
    print(f"HINDSIGHT ONLY: {oracle.strategy} | later MAE={oracle.later_mae:.5f}",flush=True)
    print('Warning: lag1 results must not be reused for hyperparameter/policy selection.',flush=True)
    return groups
