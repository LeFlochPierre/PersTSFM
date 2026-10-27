#!/usr/bin/env python3
"""PersTSFM Phase 3.3D: leakage-controlled LoRA training-history ablation.

Read ONLY a Phase 3.3C full pre-official .npz snapshot. Train each configuration
from the SAME pretrained Chronos-2 checkpoint with independent restarts.
Lag2 truth selects (history, steps), lag1 truth is inaccessible to selection
until `--score-later` reads the locked predictions and snapshot.

Important: full train history includes all samples ending BEFORE lag2 targets;
model input context for every adaptation example/forecast remains fixed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts.phase3.context_saturation import (
    read_snapshot, sha256_file, sha256_histories, case_windows,
    cadence_minutes, infer, score,
)

DEFAULT_SNAPSHOT = Path('results/phase3/context_saturation_inputs/mdense_preofficial_full_ns_v2.npz')
DEFAULT_OUTPUT = Path('results/phase3/lora_history_saturation')


def validate_grid(records, horizon, context, histories, budgets):
    if not records or horizon < 1 or context < 2:
        raise ValueError('Invalid empty cohort/horizon/context')
    if (not histories or len(histories) != len(set(histories)) or
        min(histories) < context + 2*horizon):
        raise ValueError('Training histories must be distinct and >= context + 2*horizon')
    if not budgets or len(budgets) != len(set(budgets)) or min(budgets) <= 0:
        raise ValueError('Budgets must be distinct positive gradient-step counts')
    if any(len(r['y']) < context + 2*horizon for r in records):
        raise RuntimeError('Insufficient history for lag2 and lag1 forecasting')
    available = {r['id']: len(r['y']) - 2*horizon for r in records}
    if max(histories) > min(available.values()):
        raise RuntimeError(f'History grid exceeds minimum strictly pre-lag2 length: {min(available.values())}')
    # case_windows checks finite, exact causal split and timestamp order.
    for r in records:
        case_windows(r, horizon, context)
        full_train = np.asarray(r['y'][:-2*horizon], dtype=np.float64)
        if not np.isfinite(full_train[-max(histories):]).all():
            raise RuntimeError(f"{r['id']}: training pool contains NaN/inf")
    return available


def training_series(records, horizon, history_steps):
    """The final observation in every training pool is before lag2's first label."""
    result=[]
    for r in records:
        prefix=np.asarray(r['y'][:-2*horizon], dtype=np.float32)
        if history_steps > len(prefix):
            raise RuntimeError('Training history reaches into validation or is unavailable')
        pool=prefix[-history_steps:].copy()
        if len(pool)!=history_steps or not np.isfinite(pool).all():
            raise RuntimeError('Nonfinite/mismatched training history')
        result.append(pool)
    return result


def config_name(history_steps, steps):
    return 'zero_shot' if steps == 0 else f'lora_h{history_steps}_s{steps}'


def choose_validation(summary: pd.DataFrame) -> str:
    if summary.empty or summary.model.duplicated().any():
        raise ValueError('Bad validation summary')
    if 'zero_shot' not in set(summary.model):
        raise RuntimeError('Missing zero-shot candidate')
    if not np.isfinite(summary.validation_mae.to_numpy(dtype=float)).all():
        raise RuntimeError('Invalid validation losses')
    # In exact ties: zero-shot, then fewer steps, then smaller training pool.
    return str(min(summary.to_dict('records'), key=lambda r: (
        r['validation_mae'], r['steps'], r['train_history_steps'], r['model']))['model'])


def require_snapshot(path: Path, expected_task: str, max_entities: int):
    if not path.is_file():
        raise FileNotFoundError(f'Full-history snapshot not found: {path}')
    name,horizon,records=read_snapshot(path)
    if name != expected_task:
        raise RuntimeError(f'Snapshot task mismatch: {name} != {expected_task}')
    if max_entities:
        records=records[:max_entities]
    return horizon,records


def run(args):
    if args.max_entities < 0 or args.batch_size < 1 or args.train_batch_size < 1:
        raise ValueError('Bad cohort/batch arguments')
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError('Invalid learning rate')
    if args.seed < 0:
        raise ValueError('Invalid seed')
    if args.expected_zero_shot_mae is not None and (
        not np.isfinite(args.expected_zero_shot_mae) or args.zero_shot_atol <= 0
    ):
        raise ValueError('Invalid zero-shot parity specification')
    path=Path(args.snapshot).resolve()
    horizon,records=require_snapshot(path,args.task,args.max_entities)
    grid_hist=sorted(args.histories)
    grid_steps=sorted(args.budgets)
    available=validate_grid(records,horizon,args.context,grid_hist,grid_steps)
    print('='*106,flush=True)
    print('PERTSFM PHASE 3.3D | LORA TRAINING-HISTORY SATURATION | HISTORICAL INPUTS ONLY',flush=True)
    print('='*106,flush=True)
    print(f'{args.task} | entities={len(records)} | horizon={horizon} | inference context={args.context}',flush=True)
    print(f'Pre-lag2 available lengths: {min(available.values())}..{max(available.values())}',flush=True)
    print(f'Train-history grid={grid_hist} | independent LoRA step budgets={grid_steps}',flush=True)
    print(f'Snapshot SHA256={sha256_file(path)}',flush=True)
    print('TRAIN: <= lag2 cutoff | SELECT: lag2 | EVALUATE: lag1 only after choice locked',flush=True)
    print('NO official FEV test labels accessed.',flush=True)
    if not args.execute:
        print('PASS: DRY RUN (no model loaded, no training, no labels scored)',flush=True)
        return

    import torch
    import transformers
    import peft
    from transformers.utils import is_peft_available
    from peft import PeftModel
    from src.models.adapters import Chronos2Model
    if not torch.cuda.is_available() or not is_peft_available():
        raise RuntimeError('CUDA and PEFT are REQUIRED; refuse CPU/full-finetuning fallback')

    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=Path(args.output_root)/f'{stamp}_{args.task}_n{len(records)}'
    dest.mkdir(parents=True, exist_ok=False)
    cohort_sha=sha256_histories(records)
    manifest={
        'schema':'pertsfm_3_3d_lora_history_v1', 'task':args.task, 'model_id':'amazon/chronos-2',
        'horizon':horizon, 'n_entities':len(records), 'ids':[r['id'] for r in records],
        'context':args.context, 'histories':grid_hist, 'budgets':grid_steps,
        'train_batch_size':args.train_batch_size, 'inference_batch_size':args.batch_size,
        'learning_rate':args.learning_rate, 'seed':args.seed,
        'snapshot_path':str(path), 'snapshot_sha256':sha256_file(path),
        'history_sha256':cohort_sha,
        'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
        'transformers':transformers.__version__, 'peft':peft.__version__,
        'protocol':'train strictly before lag2; validate lag2; evaluate lag1 after lock',
        'training_mode':'independent_restarts_not_incremental',
        'forecast_quantile_policy':'rearrange q10,q50,q90; preserve point prediction',
    }
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cadence=cadence_minutes(records)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)
    adapter=Chronos2Model()
    pretrained=adapter.pipeline
    if args.context > int(pretrained.model_context_length):
        raise RuntimeError('Requested context exceeds actual model context')

    # Forecast outputs: [config, historical lag(2,1), entity, horizon, point/q10/q50/q90].
    names=['zero_shot']+[config_name(h,s) for h in grid_hist for s in grid_steps]
    predictions=np.empty((len(names),2,len(records),horizon,4),dtype=np.float32)
    metric_rows=[]
    for idx,name in enumerate(names):
        hist=0 if idx==0 else int(name.split('_')[1].removeprefix('h'))
        steps=0 if idx==0 else int(name.split('_')[2].removeprefix('s'))
        if steps:
            adapter.pipeline=pretrained
            random.seed(args.seed)
            np.random.seed(args.seed)
            transformers.set_seed(args.seed)
            train=training_series(records,horizon,hist)
            torch.cuda.reset_peak_memory_stats(0)
            tic=time.perf_counter()
            ft=pretrained.fit(
                inputs=train, prediction_length=horizon, validation_inputs=None,
                finetune_mode='lora', context_length=args.context,
                learning_rate=args.learning_rate, num_steps=steps,
                batch_size=args.train_batch_size,
                output_dir=dest/'adapters'/name,
                remove_printer_callback=True,disable_data_parallel=True,
            )
            torch.cuda.synchronize()
            training_time=time.perf_counter()-tic
            if not isinstance(ft.model,PeftModel):
                raise RuntimeError('Expected PEFT LoRA model; unsafe full fine-tuning fallback')
            train_peak=int(torch.cuda.max_memory_allocated(0))
            adapter.pipeline=ft
        else:
            training_time=0.0
            train_peak=0
            adapter.pipeline=pretrained
        valid_pred, crossing, inference_time=infer(adapter,records,horizon,args.context,2,args.batch_size,cadence)
        later_pred, later_crossing, later_time=infer(adapter,records,horizon,args.context,1,args.batch_size,cadence)
        predictions[idx,0]=valid_pred
        predictions[idx,1]=later_pred
        truth=np.stack([case_windows(r,horizon,args.context)[2]['truth'] for r in records])
        scored=score(valid_pred,truth)
        if idx == 0 and args.expected_zero_shot_mae is not None:
            discrepancy=abs(scored['mae']-args.expected_zero_shot_mae)
            if discrepancy > args.zero_shot_atol:
                raise RuntimeError(
                    f'Zero-shot parity FAILED before LoRA training: got {scored["mae"]:.6f}, '
                    f'expected {args.expected_zero_shot_mae:.6f}, tolerance {args.zero_shot_atol}. '
                    'Check cohort, timestamps, preprocessing and checkpoint.'
                )
            print(f'PASS: zero-shot validation parity (error {discrepancy:.6f})',flush=True)
        row={
            'model':name,'train_history_steps':hist,'steps':steps,
            'training_wall_seconds':round(training_time,6),
            'train_peak_allocated_mib':round(train_peak/1048576,4),
            'validation_mae':scored['mae'],'validation_pinball_3':scored['pinball3'],
            'validation_crossing_steps':crossing,
            'lag2_inference_seconds':inference_time,'lag1_inference_seconds':later_time,
            'lag1_crossing_steps':later_crossing,
        }
        metric_rows.append(row)
        print(f"{name:22s} | val MAE={scored['mae']:.5f} | pinball3={scored['pinball3']:.5f} "
              f"| train={training_time:.1f}s | crossings={crossing}",flush=True)
        if steps:
            adapter.pipeline=pretrained
            del ft
            gc.collect()
            torch.cuda.empty_cache()

    np.savez_compressed(dest/'forecasts.npz', forecasts=predictions, names=np.asarray(names,dtype='U64'))
    summary=pd.DataFrame(metric_rows)
    summary.to_csv(dest/'validation.csv',index=False)
    winner=choose_validation(summary)
    selection={
        'schema':'pertsfm_3_3d_lock_v1','selected_model':winner,
        'selection_metric':'lag2_entity_macro_mae','locked_before_lag1_scoring':True,
        'manifest_sha256':sha256_file(dest/'manifest.json'),
        'validation_sha256':sha256_file(dest/'validation.csv'),
        'forecasts_sha256':sha256_file(dest/'forecasts.npz'),
    }
    (dest/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('\nVALIDATION ONLY (all LoRA runs are independent restarts):',flush=True)
    print(summary[['model','train_history_steps','steps','validation_mae','validation_pinball_3',
                   'training_wall_seconds','validation_crossing_steps']].to_string(index=False,float_format=lambda x:f'{x:.5f}'),flush=True)
    print(f'LOCKED LAG2 CHOICE: {winner}',flush=True)
    print(f'PASS: {dest}',flush=True)
    print(f'NEXT: python scripts/phase3/lora_history_saturation.py --score-later {dest}',flush=True)


def score_later(path: Path):
    root=Path(path)
    meta=json.loads((root/'manifest.json').read_text())
    locked=json.loads((root/'selection.json').read_text())
    if (meta.get('schema')!='pertsfm_3_3d_lora_history_v1' or
        locked.get('schema')!='pertsfm_3_3d_lock_v1' or
        locked.get('locked_before_lag1_scoring') is not True):
        raise RuntimeError('Invalid/unlocked run')
    if (root/'later.csv').exists():
        raise FileExistsError('Lag1 already scored; refusing overwrite')
    for file,key in [('manifest.json','manifest_sha256'),('validation.csv','validation_sha256'),
                     ('forecasts.npz','forecasts_sha256')]:
        if sha256_file(root/file)!=locked[key]:
            raise RuntimeError(f'Frozen selection input changed: {file}')
    snap=Path(meta['snapshot_path'])
    if sha256_file(snap)!=meta['snapshot_sha256']:
        raise RuntimeError('Historical snapshot changed since selection')
    name,horizon,records=read_snapshot(snap)
    records=[r for r in records if r['id'] in set(meta['ids'])]
    if (name!=meta['task'] or horizon!=meta['horizon'] or
        [r['id'] for r in records]!=meta['ids'] or
        sha256_histories(records)!=meta['history_sha256']):
        raise RuntimeError('Snapshot cohort/history hash mismatch')
    with np.load(root/'forecasts.npz',allow_pickle=False) as z:
        pred=np.asarray(z['forecasts'],dtype=np.float64)
        names=z['names'].astype(str).tolist()
    if pred.shape!=(len(names),2,len(records),horizon,4) or len(set(names))!=len(names):
        raise RuntimeError('Unexpected prediction shape/names')
    if locked['selected_model'] not in names:
        raise RuntimeError('Selected configuration missing from forecast catalog')
    valid=pd.read_csv(root/'validation.csv')
    if set(valid.model)!=set(names):
        raise RuntimeError('Validation and predictions disagree on configurations')
    truth=np.stack([case_windows(r,horizon,int(meta['context']),reveal_lag1=True)[1]['truth'] for r in records])
    summaries=[]
    for index,n in enumerate(names):
        scored=score(pred[index,1],truth)
        summaries.append({'model':n,'later_mae':scored['mae'],'later_pinball_3':scored['pinball3']})
    result=pd.DataFrame(summaries).merge(valid,on='model',validate='one_to_one')
    baseline=float(result.loc[result.model=='zero_shot','later_mae'].iloc[0])
    result['improvement_vs_zero_shot_pct']=100*(baseline-result.later_mae)/baseline
    result.to_csv(root/'later.csv',index=False)
    chosen=result.loc[result.model==locked['selected_model']].iloc[0]
    hindsight=result.loc[result.later_mae.idxmin()]
    print('='*104)
    print('PHASE 3.3D | CHRONOLOGICAL LAG1 EVALUATION | SELECTION WAS LOCKED ON LAG2')
    print('='*104)
    print(result[['model','validation_mae','later_mae','improvement_vs_zero_shot_pct',
                  'training_wall_seconds']].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f'LOCKED CHOICE: {chosen.model} | lag1 MAE={chosen.later_mae:.5f}')
    print(f'HINDSIGHT BEST (diagnostic only): {hindsight.model} | lag1 MAE={hindsight.later_mae:.5f}')
    print('CAUTION: Later-period scores must not influence hyperparameter or budget tuning.')
    print(f'PASS: {root}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',default='M_DENSE_1H')
    p.add_argument('--snapshot',type=Path,default=DEFAULT_SNAPSHOT)
    p.add_argument('--context',type=int,default=768)
    p.add_argument('--histories',nargs='+',type=int,default=[3072,6144,15504])
    p.add_argument('--budgets',nargs='+',type=int,default=[20,100,500])
    p.add_argument('--max-entities',type=int,default=0,help='0=full cohort, 4=smoke')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--train-batch-size',type=int,default=8)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--expected-zero-shot-mae',type=float,default=None,
                   help='Optional pre-LoRA parity guard, e.g. 53.89950 for M-DENSE full cohort at C768')
    p.add_argument('--zero-shot-atol',type=float,default=0.02)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--score-later',type=Path)
    args=p.parse_args()
    if args.score_later:
        if args.execute:
            raise ValueError('Cannot score later and train in one invocation')
        score_later(args.score_later)
    else:
        run(args)

if __name__=='__main__':
    main()
