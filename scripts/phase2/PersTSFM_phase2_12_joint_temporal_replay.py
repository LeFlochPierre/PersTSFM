#!/usr/bin/env python3
"""PersTSFM Phase 2.12 — matched MODEL x CONTEXT, pre-official temporal replay.

Read-only: consumes Phase 2.10 scored historical calibration cases (K>=2).
Never imports fev, does not read official evaluation target data, never performs inference.
Uses older historical lags for configuration selection and lag 1 as later replay.
Validation cost counts are *configuration/case evaluations*, not measured GPU time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('results/phase2/preofficial_forecasts/fev_bench_v0_10_0')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_task(root, task_id, models, contexts, k, metric):
    """Strictly require full, identical, complete historical cases for every configuration."""
    meta = None
    values = {}
    prov = []
    for model in models:
        for ctx in contexts:
            d = root / task_id / model / f'context_{ctx}' / f'k{k}' / 'full'
            path, score_path = d / 'calibration_case_metrics.csv', d / 'calibration_score.json'
            if not path.is_file() or not score_path.is_file():
                raise FileNotFoundError(f'Missing complete historical calibration scores: {path}')
            s = json.loads(score_path.read_text())
            expected = dict(task_id=task_id, model=model, context_steps=ctx, k=k,
                            cohort='FULL', role='HISTORICAL_CALIBRATION_ONLY')
            for key, val in expected.items():
                if s.get(key) != val:
                    raise ValueError(f'{score_path}: {key} expected {val!r}, found {s.get(key)!r}')
            df = pd.read_csv(path, dtype={'case_id':'string','entity_id':'string'})
            cols = {'case_id', 'entity_id', 'lag_before_first_official', metric}
            if not cols.issubset(df.columns):
                raise ValueError(f'{path}: missing {cols - set(df.columns)}')
            if len(df) != int(s.get('n_cases', -1)) or df.case_id.isna().any() or df.case_id.duplicated().any():
                raise ValueError(f'{path}: incomplete or duplicate scored cases')
            df['lag_before_first_official'] = pd.to_numeric(df.lag_before_first_official,errors='raise').astype(int)
            scores = pd.to_numeric(df[metric],errors='raise').to_numpy(float)
            if not np.isfinite(scores).all() or np.any(scores < 0):
                raise ValueError(f'{path}: invalid historical scores')
            m = df[['case_id','entity_id','lag_before_first_official']].sort_values('case_id').reset_index(drop=True)
            if meta is None:
                meta = m
                counts = meta.groupby('entity_id')['lag_before_first_official'].apply(lambda x: sorted(x.tolist()))
                if not len(counts) or any(v != list(range(1,k+1)) for v in counts):
                    raise ValueError('Incomplete historical lag set for at least one entity')
            elif not meta.equals(m):
                raise ValueError(f'Case-ID/entity/lag mismatch across models and contexts: {model}@{ctx}')
            key = f'{model}@{ctx}'
            values[key] = df.assign(_score=scores).sort_values('case_id')['_score'].to_numpy(float)
            prov.append(dict(model=model, context=ctx, case_metrics_sha256=sha256(path),
                             score_sha256=sha256(score_path), n_cases=len(df)))
    return meta, pd.DataFrame(values, index=meta.case_id.astype(str).to_numpy()), prov


def analyze(meta, scores, models, contexts, bootstrap, seed):
    m = meta.copy()
    m.index = m.case_id.astype(str)
    config_keys = list(scores.columns)
    all_lags = sorted(m.lag_before_first_official.unique(), reverse=True)
    if len(all_lags) < 2 or min(all_lags) != 1:
        raise ValueError('Need earlier historical lags and lag 1 to evaluate future replay')
    later_ids = m.loc[m.lag_before_first_official == 1].index
    older_ids = m.loc[m.lag_before_first_official >= 2].index
    oldest_ids = m.loc[m.lag_before_first_official == max(all_lags)].index
    if len(later_ids) == 0 or len(older_ids) == 0:
        raise ValueError('No historical selection/evaluation separation')
    older = scores.loc[older_ids]
    future = scores.loc[later_ids]
    oldest = scores.loc[oldest_ids]
    def choose(rows, candidates):
        return min(candidates, key=lambda key:(float(rows[key].mean()), key))
    low, high = min(contexts), max(contexts)
    proposals = [
        ('Full grid: older lags', config_keys, older),
        ('Full grid: oldest lag', config_keys, oldest),
        ('Longest-only search', [f'{x}@{high}' for x in models], older),
        ('Shortest-only search', [f'{x}@{low}' for x in models], older),
    ]
    if len(contexts)>=3:
        middle = contexts[len(contexts)//2]
        proposals.append(('Middle-only search', [f'{x}@{middle}' for x in models], older))
    oracle = choose(future, config_keys)
    selections = []
    for label, choices, seen in proposals:
        pick = choose(seen, choices)
        selections.append((label,pick,len(choices),len(seen)*len(choices)))
    for model in models:
        selections.append((f'Fixed {model} longest',f'{model}@{high}',0,0))
    selections.append(('Hindsight lag-1 oracle',oracle,None,None))
    oracle_loss = float(future[oracle].mean())
    if oracle_loss<=0:
        raise ValueError('Cannot calculate relative regret with nonpositive hindsight loss')
    policies = pd.DataFrame([dict(policy=label,selected_model=pick.rsplit('@',1)[0],
                    selected_context_steps=int(pick.rsplit('@',1)[1]),selected_configuration=pick,
                    configurations_probed=num_cfg,validation_case_forecasts=num_cases,
                    heldout_lag1_loss=float(future[pick].mean()),
                    regret_pct=100*(float(future[pick].mean())/oracle_loss-1))
                    for label,pick,num_cfg,num_cases in selections])
    by_lag = []
    for lag in all_lags:
        ids = m.loc[m.lag_before_first_official == lag].index
        means = scores.loc[ids].mean()
        for name in config_keys:
            model, ctx = name.rsplit('@',1)
            by_lag.append(dict(lag_before_first_official=int(lag),model=model,
                     context_steps=int(ctx),mean_case_loss=float(means[name]),
                     n_cases=len(ids),n_entities=int(m.loc[ids,'entity_id'].nunique())))
    by_lag = pd.DataFrame(by_lag)
    # Paired bootstrap on entities, not independent time points. Correlated entities
    # may substantially understate real uncertainty; exploratory only.
    entities = m.loc[later_ids,'entity_id']
    later_by_entity = future.assign(entity_id=entities).groupby('entity_id')[config_keys].mean().sort_index()
    if len(later_by_entity)<2:
        raise ValueError('Need >=2 entities for exploratory bootstrap')
    array = later_by_entity.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0,len(array),size=(bootstrap,len(array)))
    boot_means = array[draws].mean(axis=1)
    column = {key:j for j,key in enumerate(config_keys)}
    baseline_pick = policies.loc[policies.policy=='Longest-only search','selected_configuration'].iloc[0]
    ref = column[baseline_pick]
    pairs=[]
    for policy in ['Full grid: older lags','Shortest-only search','Middle-only search']:
        row=policies.loc[policies.policy==policy]
        if row.empty: continue
        chosen = row.selected_configuration.iloc[0]
        delta = boot_means[:,column[chosen]]-boot_means[:,ref]
        interval = np.quantile(delta,[.025,.975])
        pairs.append(dict(policy=policy,selected_configuration=chosen,
            baseline_configuration=baseline_pick,
            later_loss_diff=float(future[chosen].mean()-future[baseline_pick].mean()),
            ci025=float(interval[0]),ci975=float(interval[1]),
            fraction_policy_beats_long_only=float((delta<0).mean())))
    return by_lag,policies,pd.DataFrame(pairs)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--task-id',required=True)
    ap.add_argument('--models',nargs='+',default=['chronos2','moirai2_small'])
    ap.add_argument('--contexts',nargs='+',type=int,required=True)
    ap.add_argument('--k',type=int,default=3)
    ap.add_argument('--metric',choices=['mae','mean_pinball_3'],default='mae')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output-dir',type=Path,default=Path('results/phase2/joint_temporal_replay'))
    args=ap.parse_args()
    models=list(dict.fromkeys(args.models))
    contexts=sorted(set(args.contexts))
    if len(models)<2 or len(contexts)<2 or args.k<2 or args.bootstrap<100:
        ap.error('Need at least two models, two contexts, K>=2 and 100 bootstrap draws')
    if any(c<=0 for c in contexts): ap.error('Contexts must be positive')
    meta,scores,provenance=load_task(args.root,args.task_id,models,contexts,args.k,args.metric)
    by_lag,policy,paired=analyze(meta,scores,models,contexts,args.bootstrap,args.seed)
    output = args.output_dir / args.task_id / f'k{args.k}' / args.metric
    if output.exists(): raise FileExistsError(f'Refusing overwrite: {output}')
    output.mkdir(parents=True,exist_ok=False)
    by_lag.to_csv(output/'by_lag.csv',index=False)
    policy.to_csv(output/'joint_policy_replay.csv',index=False)
    paired.to_csv(output/'paired_entity_bootstrap.csv',index=False)
    manifest=dict(scope='PRE_OFFICIAL_HISTORICAL_REPLAY_ONLY',official_test_labels_accessed=False,
        task_id=args.task_id,models=models,contexts=contexts,k=args.k,metric=args.metric,
        n_entities=int(meta.entity_id.nunique()),n_cases_per_configuration=len(meta),
        bootstrap=args.bootstrap,seed=args.seed,source_scores=provenance,
        cautions=['Task-level retrospective analysis, not meta-learning generalisation',
                  'Number of configurations and validation case forecasts are not GPU runtime costs',
                  'Entity bootstrap may be optimistic under cross-entity correlation'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('='*100)
    print('PERTSFM JOINT MODEL x CONTEXT TEMPORAL REPLAY — NO OFFICIAL EVALUATION LABELS')
    print('='*100)
    print(f"{args.task_id} | models={models} contexts={contexts} | "
          f"{len(meta)} matched historical cases / configuration | {meta.entity_id.nunique()} entities | metric={args.metric}")
    print('\nBY LAG (3 oldest, 1 latest):')
    print(by_lag.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nOLDER LAGS SELECT; LAG 1 EVALUATES (0 configurations = fixed baseline):')
    print(policy.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nPAIRED ENTITY BOOTSTRAP: policy minus LONGEST-ONLY search (negative favours policy):')
    print(paired.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nSaved:',output)
    print('Exploratory within-task evidence only, no official FEV test labels accessed.')


if __name__=='__main__':main()
