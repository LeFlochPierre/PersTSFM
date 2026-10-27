#!/usr/bin/env python3
"""PersTSFM Phase 3.1: chronological offline adaptive-racing feasibility.

Input: scored pre-official phase2 case catalogs; imports phase3.0 reader and QC.
All configuration choices see ONLY lag2. Lag1 is used ONLY for evaluation.
Costs: configuration x entity x historical-origin forecast calls; NOT GPU time.
This is an exploratory, hand-designed baseline, NOT a trained selector.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.phase3.PersTSFM_phase3_0_probe_value import read_catalogs, task_matrix, sha256


def config_indices(names):
    models = sorted({model for model, _ in names})
    by_model = {
        model: [i for i, (m, _) in enumerate(names) if m == model]
        for model in models
    }
    middle = [inds[len(inds)//2] for inds in by_model.values()]
    longest = [inds[-1] for inds in by_model.values()]
    shortest = [inds[0] for inds in by_model.values()]
    return models, by_model, middle, longest, shortest


def choose(arr, idxs):
    # Stable tie-break by previously sorted model/context index.
    return min(idxs, key=lambda i: (float(arr[i]), i))


def analyze_one(pivot, names, metric, task, n_target, repeats, seed, shortlist_size):
    entity_ids = sorted({entity for entity, _ in pivot.index})
    old = pivot.xs(2, level=1).loc[entity_ids, names].to_numpy(float)
    future = pivot.xs(1, level=1).loc[entity_ids, names].to_numpy(float).mean(axis=0)
    if not (np.isfinite(old).all() and np.isfinite(future).all()):
        raise ValueError('Nonfinite losses')
    n_ent, n_cfg = old.shape
    oracle = float(future.min())
    if oracle <= 0:
        raise ValueError('Nonpositive lag1 oracle loss')
    models, model_idxs, mids, longs, shorts = config_indices(names)
    rng = np.random.default_rng(seed)
    rows = []
    trials = 1 if n_target == n_ent else repeats

    def emit(trial, strategy, idx, calls, nconfigs):
        rows.append(dict(
            task_id=task, metric=metric, target_probe_entities=n_target,
            repeat=trial, strategy=strategy, selected=f'{names[idx][0]}@{names[idx][1]}',
            model=names[idx][0], context_steps=names[idx][1],
            later_loss=float(future[idx]), oracle_lag1_loss=oracle,
            regret_pct=float(100*(future[idx]/oracle-1)),
            probe_case_forecasts=int(calls), unique_configurations=int(nconfigs),
        ))

    for rep in range(trials):
        perm = rng.permutation(n_ent)
        target = perm[:n_target]
        seed_two = target[:min(2,n_target)]
        means = old[target].mean(axis=0)
        initial = old[seed_two].mean(axis=0)
        # Fixed defaults require no probe forecasts.
        for model in ('chronos2', 'toto2_313m'):
            if model in model_idxs:
                idx=model_idxs[model][-1]
                emit(rep, f'fixed_{model}', idx, 0, 0)
        # Static baselines: identical randomly selected entities, lag 2.
        emit(rep, 'full_grid', choose(means, list(range(n_cfg))), n_target*n_cfg, n_cfg)
        emit(rep, 'longest_only', choose(means, longs), n_target*len(longs), len(longs))
        emit(rep, 'middle_only', choose(means, mids), n_target*len(mids), len(mids))
        # Adaptive: probe middle context for all models on TWO entities;
        # shortlist top s models; probe all contexts of those models on n_target entities.
        ranked_middle = sorted(mids, key=lambda i: (float(initial[i]), i))
        s = min(shortlist_size, len(ranked_middle))
        shortlisted = [names[idx][0] for idx in ranked_middle[:s]]
        options = sorted(i for model in shortlisted for i in model_idxs[model])
        chosen = choose(means, options)
        # Count UNIQUE case/config pairs actually observed, without counting
        # the repeated first-two-entity middle probes for shortlisted models.
        calls = len(mids)*len(seed_two) + len(options)*n_target - s*len(seed_two)
        unique = len(set(mids).union(options))
        emit(rep, f'adaptive_top{s}', chosen, calls, unique)
        # Cost-matched longest baseline, allowed to inspect more entities when
        # cheaper than adaptive. Crucially, never exceed the adaptive call budget.
        n_long = min(n_ent, calls // len(longs))
        if n_long >= 1:
            comparator = old[perm[:n_long]].mean(axis=0)
            emit(rep, f'longest_cost_matched_top{s}', choose(comparator, longs),
                 n_long*len(longs), len(longs))
        # Cost-matched full-grid baseline, if at least one case per config fits.
        n_full = min(n_ent, calls // n_cfg)
        if n_full >= 1:
            comparator=old[perm[:n_full]].mean(axis=0)
            emit(rep, f'full_cost_matched_top{s}', choose(comparator, list(range(n_cfg))),
                 n_full*n_cfg, n_cfg)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog-root', default='results/phase2/cross_task_matrix')
    ap.add_argument('--repeats',type=int,default=200)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--budgets',type=int,nargs='+',default=[4,8,16,32])
    ap.add_argument('--output-root',default='results/phase3/adaptive_racing')
    args=ap.parse_args()
    if args.repeats<1 or not args.budgets or any(b<2 for b in args.budgets):
        ap.error('repeats >=1 and budgets >=2 are required')
    catalogs=read_catalogs(args.catalog_root,[])
    frames=[]
    sources=[]
    print('='*94)
    print('PERTSFM PHASE 3.1 | ADAPTIVE CANDIDATE SHORTLIST — EXPLORATORY HISTORICAL REPLAY')
    print('Select on lag 2 only; evaluate on lag 1 only. Cost proxy = forecast calls, NOT GPU seconds.')
    print('='*94)
    for task in sorted(catalogs):
        quality, df, root, csv, scsv = catalogs[task]
        sources.append({'task_id':task,'directory':str(root),'sha256':sha256(csv),'K':int(df.k.iloc[0])})
        for metric in ('mae','mean_pinball_3'):
            piv, names, k = task_matrix(df,metric)
            n_ent=len({ent for ent,_ in piv.index})
            for budget in args.budgets:
                if budget>n_ent:continue
                for shortlist in (1,2):
                    a=analyze_one(piv,names,metric,task,budget,args.repeats,
                                  args.seed+budget*101,shortlist)
                    # Fixed/static rows are repeated across shortlist loops,
                    # retain only once per (task,metric,budget,repeat,strategy).
                    frames.append(a)
            print(f' {df.dataset_config.iloc[0]:26s} | {metric:14s} | '
                  f'{n_ent:3d} entities | {len(names):2d} configs | K={k}')
    trials=pd.concat(frames,ignore_index=True)
    trials=trials.drop_duplicates(['task_id','metric','target_probe_entities','repeat','strategy'])
    summary=(trials.groupby(['task_id','metric','target_probe_entities','strategy'],sort=True)
             .agg(trials=('regret_pct','size'),mean_regret_pct=('regret_pct','mean'),
                  p90_regret_pct=('regret_pct',lambda x:float(np.quantile(x,.90))),
                  within_1pct=('regret_pct',lambda x:float((x<=1).mean())),
                  within_5pct=('regret_pct',lambda x:float((x<=5).mean())),
                  mean_calls=('probe_case_forecasts','mean'),
                  mean_distinct_configs=('unique_configurations','mean'))
             .reset_index())
    ts=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=Path(args.output_root)/ts
    dest.mkdir(parents=True,exist_ok=False)
    trials.to_csv(dest/'trials.csv',index=False)
    summary.to_csv(dest/'summary.csv',index=False)
    (dest/'manifest.json').write_text(json.dumps({
        'phase':'3.1','repeats':args.repeats,'seed':args.seed,'budgets':args.budgets,
        'sources':sources,'older_selection_lag':2,'later_evaluation_lag':1,
        'official_labels_accessed':False,
        'cost_proxy':'unique configuration x historical-entity forecast calls',
        'warning':'This is not GPU seconds; model loads and context lengths unmeasured. '
                  'Only four development families; not source-heldout evidence.',
    },indent=2)+'\n')
    print('\nMAE — 4-entity probe (and matched forecast-call competitors)')
    small=summary[(summary.metric=='mae')&(summary.target_probe_entities==4)]
    if not small.empty:
        print(small[['task_id','strategy','mean_regret_pct','p90_regret_pct','mean_calls']]
              .to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print('\nPASS:',dest)
    print('Saved summary.csv, trials.csv, manifest.json')
    print('Do NOT claim GPU savings or transfer across datasets from this exploratory replay.')

if __name__=='__main__':main()
