#!/usr/bin/env python3
"""Phase 2.11: historical, pre-official, matched-context temporal replay.

Reads only previously scored PRE-OFFICIAL calibration_case_metrics.csv and
calibration_score.json, never official evaluation windows or future labels.

Temporal decision: older lags 3/2 select context, younger lag 1 evaluates.
All comparisons use exactly matching case IDs across contexts. All outcomes
are exploratory development evidence, not official FEV benchmark results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('results/phase2/preofficial_forecasts/fev_bench_v0_10_0')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_matched(root, task_id, model, contexts, k, metric):
    arrays, provenance = [], []
    for context in contexts:
        folder = root / task_id / model / f'context_{context}' / f'k{k}' / 'full'
        path = folder / 'calibration_case_metrics.csv'
        score_path = folder / 'calibration_score.json'
        if not path.is_file() or not score_path.is_file():
            raise FileNotFoundError(f'Missing full-cohort historical calibration score: {path}')
        score = json.loads(score_path.read_text())
        checks = {'task_id':task_id, 'model':model, 'context_steps':context,'k':k,'cohort':'FULL'}
        for key, expected in checks.items():
            if score.get(key) != expected:
                raise ValueError(f'{score_path}: {key}={score.get(key)!r}; expected {expected!r}')
        if score.get('role') != 'HISTORICAL_CALIBRATION_ONLY':
            raise ValueError(f'{score_path}: not historical calibration scoring')
        df = pd.read_csv(path, dtype={'case_id':'string','entity_id':'string'})
        required = {'case_id', 'entity_id', 'lag_before_first_official', metric}
        if not required.issubset(df):
            raise ValueError(f'{path}: missing columns {required.difference(df.columns)}')
        if df.case_id.duplicated().any():
            raise ValueError(f'{path}: duplicated case IDs')
        if len(df) != int(score['n_cases']):
            raise ValueError(f'{path}: mismatched scored case count')
        df['lag_before_first_official'] = pd.to_numeric(df['lag_before_first_official'],errors='raise').astype(int)
        if not bool(df['lag_before_first_official'].isin(range(1,k+1)).all()):
            raise ValueError(f'{path}: unsupported lag')
        if df[metric].isna().any() or not np.isfinite(df[metric].to_numpy(float)).all():
            raise ValueError(f'{path}: nonfinite scores')
        if (df[metric] < 0).any():
            raise ValueError(f'{path}: negative loss')
        df = df[['case_id','entity_id','lag_before_first_official',metric]].copy()
        df['context_steps'] = context
        arrays.append(df)
        provenance.append({'context_steps':context,'case_metrics_sha256':digest(path),
                           'score_sha256':digest(score_path), 'n_cases':len(df)})
    ref = arrays[0].set_index('case_id')[['entity_id','lag_before_first_official']].sort_index()
    for df in arrays[1:]:
        check = df.set_index('case_id')[['entity_id','lag_before_first_official']].sort_index()
        if not ref.equals(check):
            raise ValueError('Case IDs, entities, or lags differ across contexts; not comparable')
    seen = ref.groupby('entity_id')['lag_before_first_official'].apply(lambda x:sorted(x.tolist()))
    if any(v != list(range(1,k+1)) for v in seen):
        raise ValueError('Incomplete or duplicate lag-origin set per entity')
    wide = arrays[0].drop(columns=['context_steps']).set_index('case_id').rename(columns={metric:contexts[0]})
    for ctx, df in zip(contexts[1:], arrays[1:]):
        wide[ctx] = df.set_index('case_id').loc[wide.index,metric]
    return wide.reset_index(), provenance


def evaluate(wide, contexts, metric, bootstrap, seed):
    context_cols = list(contexts)
    by_lag = []
    for lag in sorted(wide.lag_before_first_official.unique(),reverse=True):
        part = wide[wide.lag_before_first_official == lag]
        avg = part[context_cols].mean()
        for ctx in contexts:
            by_lag.append({'lag_before_first_official':int(lag),'context_steps':int(ctx),
                           f'mean_case_{metric}':float(avg[ctx]),'n_entities':part.entity_id.nunique(),
                           'n_cases':len(part)})
    per_lag = pd.DataFrame(by_lag)
    dev = wide[wide.lag_before_first_official >= 2]
    future = wide[wide.lag_before_first_official == 1]
    if len(dev) == 0 or len(future) == 0:
        raise ValueError('Need at least two historical lags to hold out lag 1')
    winner = lambda data: int(min(contexts,key=lambda c:(data[c].mean(), c)))
    one = wide[wide.lag_before_first_official == int(wide.lag_before_first_official.max())]
    decisions = [('Oldest lag only',winner(one)), ('Older lags (2+)',winner(dev)),
                 ('Always shortest',contexts[0]),('Always longest',contexts[-1]),
                 ('Hindsight lag-1 oracle',winner(future))]
    lag1_oracle = float(min(future[c].mean() for c in contexts))
    policies = pd.DataFrame([{'policy':label,'selected_context_steps':ctx,
                              f'heldout_lag1_{metric}':float(future[ctx].mean()),
                              'lag1_regret_pct':float(100*(future[ctx].mean()/lag1_oracle-1))}
                             for label,ctx in decisions])
    # Bootstrap paired loss differences using entities as resampling clusters.
    # A single task's entities may themselves be correlated: intervals are exploratory.
    rng = np.random.default_rng(seed)
    later_by_entity = future.groupby('entity_id')[context_cols].mean().sort_index()
    A = later_by_entity.to_numpy(float)
    sample = rng.integers(0,len(A),size=(bootstrap,len(A)))
    boot = A[sample].mean(axis=1)
    long_idx = len(contexts)-1
    pairs = []
    for j,context in enumerate(contexts[:-1]):
        d = boot[:,j]-boot[:,long_idx]
        observed = float(A[:,j].mean()-A[:,long_idx].mean())
        low,high = np.quantile(d,[.025,.975])
        pairs.append({'context_a':int(context),'context_b':int(contexts[-1]),
                      'delta_a_minus_b_lag1':observed,'ci025':float(low),'ci975':float(high),
                      'bootstrap_fraction_a_better':float(np.mean(d<0))})
    return per_lag, policies, pd.DataFrame(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--task-id',default='fev_0_10_0_044')
    ap.add_argument('--model',default='moirai2_small')
    ap.add_argument('--contexts',type=int,nargs='+',default=[96,384,1536])
    ap.add_argument('--k',type=int,default=3)
    ap.add_argument('--metric',choices=['mae','mean_pinball_3'],default='mae')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output-dir',type=Path,default=Path('results/phase2/temporal_replay'))
    args = ap.parse_args()
    contexts = sorted(set(args.contexts))
    if len(contexts)<2 or args.k < 2 or args.bootstrap<100:
        ap.error('Need >=2 contexts, >=2 calibration lags, and >=100 bootstrap draws')
    wide,provenance = read_matched(args.root,args.task_id,args.model,contexts,args.k,args.metric)
    per_lag,policy,paired = evaluate(wide,contexts,args.metric,args.bootstrap,args.seed)
    output = args.output_dir / args.task_id / args.model / f'k{args.k}' / args.metric
    if output.exists():
        raise FileExistsError(f'Analysis output already exists: {output}')
    output.mkdir(parents=True, exist_ok=False)
    per_lag.to_csv(output/'by_lag.csv',index=False)
    policy.to_csv(output/'policy_replay.csv',index=False)
    paired.to_csv(output/'paired_bootstrap.csv',index=False)
    (output/'manifest.json').write_text(json.dumps({'scope':'HISTORICAL_PREOFFICIAL_ONLY',
        'official_targets_accessed':False,'task_id':args.task_id,'model':args.model,
        'contexts':contexts,'k':args.k,'metric':args.metric,'n_entities':int(wide.entity_id.nunique()),
        'n_cases_per_context':len(wide),'bootstrap_draws':args.bootstrap,'seed':args.seed,
        'source_scores':provenance},indent=2)+'\n')
    print('='*90)
    print('PERTSFM — PRE-OFFICIAL TEMPORAL REPLAY | NO OFFICIAL EVALUATION LABELS')
    print('='*90)
    print(f'{args.task_id} | {args.model} | contexts={contexts} | {len(wide)} matched cases | '
          f'{wide.entity_id.nunique()} entities | metric={args.metric}')
    print('\nBY HISTORICAL LAG (lag 3 oldest, lag 1 latest):')
    print(per_lag.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nSELECTION ON OLDER ORIGINS -> EVALUATION ON LAG 1:')
    print(policy.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f'\nPAIRED LAG-1 ENTITY BOOTSTRAP (A minus longest={contexts[-1]}; negative favours A):')
    print(paired.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nSaved:',output)
    print('Exploratory within-task evidence only; bootstrap entities may be correlated.')


if __name__=='__main__':
    main()
