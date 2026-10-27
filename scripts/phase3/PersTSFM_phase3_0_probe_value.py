#!/usr/bin/env python3
"""PersTSFM Phase 3.0: historical low-fidelity probes and personalization headroom.

Offline/read-only replay on scored PRE-OFFICIAL historical case catalogs. Does not
import FEV, run foundation models, or access official evaluation labels.

Selection: most recent available *older* lags (2, then 3, etc.); evaluation: lag 1.
Probe fidelity: common randomly chosen entities x older historical origins.
Budget proxy: model-configuration forecast calls; NOT GPU seconds.

Outputs are exploratory, not source-family-heldout generalization results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ('mae', 'mean_pinball_3')
REQ = {'task_id', 'dataset_config', 'model', 'context_steps', 'k',
       'case_id', 'entity_id', 'lag_before_first_official', *METRICS}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_catalogs(root, user_dirs):
    dirs = [Path(p) for p in user_dirs] if user_dirs else sorted(
        {p.parent for p in Path(root).glob('*/case_catalog.csv')})
    if not dirs:
        raise FileNotFoundError('No case catalogs found')
    options = {}
    for path in dirs:
        csv = path / 'case_catalog.csv'
        score_csv = path / 'score_catalog.csv'
        if not csv.is_file() or not score_csv.is_file():
            if user_dirs:
                raise FileNotFoundError(f'Missing case/score catalog in {path}')
            continue
        raw = pd.read_csv(csv, dtype={'entity_id': str, 'case_id': str, 'task_id': str})
        missing = REQ.difference(raw.columns)
        if missing:
            raise ValueError(f'{csv} missing {sorted(missing)}')
        if raw.empty:
            continue
        scores = pd.read_csv(score_csv, dtype={'task_id': str})
        for (task, k), grp in raw.groupby(['task_id', 'k'], sort=True):
            k = int(k)
            if k < 2:
                continue
            nconfig = grp[['model', 'context_steps']].drop_duplicates().shape[0]
            matched_scores = scores[(scores['task_id'] == task) & (scores['k'] == k)]
            if nconfig != len(matched_scores):
                raise ValueError(f'{csv}: {task} K={k} config count disagrees with score catalog')
            key = str(task)
            # Select the densest available historical grid for each task.
            # Tie-break by K, then latest directory path. Never merge different cohorts.
            quality = (nconfig, k, str(path))
            if key not in options or quality > options[key][0]:
                options[key] = (quality, grp.copy(), path, csv, score_csv)
    if not options:
        raise ValueError('No usable K>=2 scored tasks found')
    return options


def task_matrix(data, metric):
    df = data.copy()
    for field in ['context_steps', 'k', 'lag_before_first_official']:
        df[field] = pd.to_numeric(df[field], errors='raise').astype(int)
    df[metric] = pd.to_numeric(df[metric], errors='raise')
    if not np.isfinite(df[metric].to_numpy()).all() or (df[metric] < 0).any():
        raise ValueError('Nonfinite or negative losses')
    key = ['model', 'context_steps', 'case_id']
    if df.duplicated(key).any():
        raise ValueError('Duplicate configuration/case')
    if df['dataset_config'].nunique() != 1 or df['k'].nunique() != 1:
        raise ValueError('Expected one task, one K')
    k = int(df.k.iloc[0])
    expected_lags = set(range(1, k + 1))
    names = sorted(set(zip(df.model.astype(str), df.context_steps.astype(int))))
    signatures = []
    for m, c in names:
        sub = df[(df.model == m) & (df.context_steps == c)]
        signature = frozenset(zip(sub.case_id, sub.entity_id, sub.lag_before_first_official))
        signatures.append(signature)
    if len(set(signatures)) != 1:
        raise ValueError('Unmatched case/entity/lag across configurations')
    by_entity = df[(df.model == names[0][0]) & (df.context_steps == names[0][1])]
    for entity, group in by_entity.groupby('entity_id'):
        if set(group.lag_before_first_official) != expected_lags or len(group) != k:
            raise ValueError(f'Incomplete historical lag set: {entity}')
    if by_entity['case_id'].duplicated().any():
        raise ValueError('Duplicate case ID in reference configuration')
    pivot = df.pivot(index=['entity_id', 'lag_before_first_official'],
                     columns=['model', 'context_steps'], values=metric).sort_index()
    if pivot.isna().any().any():
        raise ValueError('Missing configuration case scores')
    return pivot, names, k


def choices(names):
    models = sorted({m for m, _ in names})
    model_ctxt = {m: sorted(c for model, c in names if model == m) for m in models}
    return {
        'full_grid': list(names),
        'longest_per_model': [(m, model_ctxt[m][-1]) for m in models],
        'middle_per_model': [(m, model_ctxt[m][len(model_ctxt[m]) // 2]) for m in models],
        'shortest_per_model': [(m, model_ctxt[m][0]) for m in models],
    }


def select(loss, opts):
    return min(opts, key=lambda key: (float(loss.loc[key]), key))


def sel_name(config):
    return f'{config[0]}@{config[1]}'


def simulated_probes(pivot, names, k, metric, reps, seed, budgets):
    entities = sorted({e for e, _ in pivot.index})
    n_entities = len(entities)
    older_lags = list(range(2, k + 1))  # nearest-to-future first
    future = pivot.xs(1, level=1).loc[entities]
    future_mean = future.mean(axis=0)
    oracle = select(future_mean, names)
    oracle_loss = float(future_mean.loc[oracle])
    if oracle_loss <= 0:
        raise ValueError('Nonpositive lag-1 oracle loss')
    policy_candidates = choices(names)
    rng = np.random.default_rng(seed)
    rows = []

    def emit(strategy, n, num_lags, trial, probe_cases, choice, rank_rho=np.nan, zero_probe=False):
        eval_loss = float(future_mean.loc[choice])
        base_strategy = strategy.removesuffix('_all_older')
        n_cfg = 0 if zero_probe else len(policy_candidates[base_strategy])
        rows.append(dict(metric=metric,
                         strategy=strategy, sampled_entities=n,
                         sampled_older_lags=num_lags, repeat=trial,
                         probed_configs=n_cfg,
                         probe_case_forecasts=0 if zero_probe else n_cfg * probe_cases,
                         selected=sel_name(choice), later_loss=eval_loss,
                         later_regret_pct=100 * (eval_loss / oracle_loss - 1),
                         selection_loss_rank_rho=rank_rho,
                         hindsight_oracle=sel_name(oracle)))

    # Strong zero-probe fixed defaults: deployment only, no calibration.
    for m in ['chronos2', 'toto2_313m']:
        candidates = [p for p in names if p[0] == m]
        if candidates:
            fixed = max(candidates, key=lambda z: z[1])
            rows.append(dict(metric=metric,
                             strategy=f'fixed_{m}_longest', sampled_entities=0,
                             sampled_older_lags=0, repeat=0, probed_configs=0,
                             probe_case_forecasts=0, selected=sel_name(fixed),
                             later_loss=float(future_mean.loc[fixed]),
                             later_regret_pct=100*(float(future_mean.loc[fixed])/oracle_loss-1),
                             selection_loss_rank_rho=np.nan,
                             hindsight_oracle=sel_name(oracle)))

    # Full older evidence is the historical search reference (not the lag1 oracle).
    entire = pivot.loc[pd.IndexSlice[:, older_lags], :].mean(axis=0)
    for strategy, opts in policy_candidates.items():
        best = select(entire, opts)
        emit(strategy + '_all_older', n_entities, len(older_lags), 0,
             n_entities * len(older_lags), best,
             rank_rho=entire.corr(future_mean, method='spearman'))

    probe_counts = sorted({int(n) for n in budgets if int(n) >= 2 and int(n) <= n_entities} | {n_entities})
    for n in probe_counts:
        n_reps = 1 if n == n_entities else reps
        for n_lags in range(1, len(older_lags) + 1):
            lags = older_lags[:n_lags]
            for trial in range(n_reps):
                sample = sorted(rng.choice(entities, n, replace=False).tolist())
                probe = pivot.loc[pd.MultiIndex.from_product(
                    [sample, lags], names=pivot.index.names)].mean(axis=0)
                rho = float(probe.corr(future_mean, method='spearman'))
                for strategy, opts in policy_candidates.items():
                    best = select(probe, opts)
                    emit(strategy, n, n_lags, trial, n * n_lags, best, rank_rho=rho)
    return pd.DataFrame(rows)


def personalization(pivot, names, k, metric):
    entities = sorted({e for e, _ in pivot.index})
    historical = pivot.loc[pd.IndexSlice[:, list(range(2, k + 1))], :]
    older = historical.groupby(level=0).mean().loc[entities]
    future = pivot.xs(1, level=1).loc[entities]
    group_prior = older.mean(axis=0)
    global_choice = select(group_prior, names)

    # Normalized profiles enable shrinkage without comparing regional magnitudes.
    prior_scale = max(float(np.median(group_prior)), 1e-12)
    prior = group_prior / prior_scale
    later_global = future.loc[:, global_choice].to_numpy(float)
    entity_oracle = future.min(axis=1).to_numpy(float)
    rows = []
    for alpha in [0., .25, .5, .75, 1.]:
        chosen = []
        later = []
        for entity in entities:
            local = older.loc[entity]
            local_scale = max(float(np.median(local)), 1e-12)
            blend = (1-alpha)*prior + alpha*(local/local_scale)
            choice = select(blend, names)
            chosen.append(choice)
            later.append(float(future.loc[entity, choice]))
        later = np.array(later)
        global_avg = float(later_global.mean())
        rows.append(dict(metric=metric, alpha=alpha, n_entities=len(entities),
                         global_selected=sel_name(global_choice),
                         global_later_loss=global_avg,
                         personalized_later_loss=float(later.mean()),
                         improvement_over_global_pct=100*(1-float(later.mean())/global_avg)
                         if global_avg else np.nan,
                         oracle_entity_later_loss=float(entity_oracle.mean()),
                         hindsight_headroom_pct=100*(1-float(entity_oracle.mean())/global_avg)
                         if global_avg else np.nan,
                         fraction_entities_improved=float((later < later_global).mean()),
                         n_unique_configurations=len(set(chosen)),
                         note='Historical selection only; lag1 oracle is hindsight upper bound'))
    return pd.DataFrame(rows)


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--catalog-root', default='results/phase2/cross_task_matrix')
    pa.add_argument('--catalog-dir', action='append', default=[], help='Can repeat to explicitly select catalogs')
    pa.add_argument('--repeats', type=int, default=200)
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--entity-budgets', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64, 128])
    pa.add_argument('--output-root', default='results/phase3/probe_value')
    args = pa.parse_args()
    if args.repeats < 1:
        pa.error('repeats must be >=1')
    options = read_catalogs(args.catalog_root, args.catalog_dir)
    all_probes, all_personal, sources = [], [], []
    print('='*95)
    print('PERTSFM PHASE 3.0 | VALUE OF CHEAP PROBES + PERSONALIZATION HEADROOM')
    print('Historical calibration only; forecast calls are a proxy, NOT GPU seconds')
    print('='*95)
    for task in sorted(options):
        quality, df, catalog, csv, score_csv = options[task]
        for metric in METRICS:
            pivot, names, k = task_matrix(df, metric)
            n_entities = len({x[0] for x in pivot.index})
            print(f'\n{task} / {df.dataset_config.iloc[0]} | K={k} | {metric} | '
                  f'entities={n_entities} | configurations={len(names)}')
            probes = simulated_probes(pivot, names, k, metric,
                                      args.repeats, args.seed, args.entity_budgets)
            probes['task_id'] = task
            probes['dataset_config'] = str(df.dataset_config.iloc[0])
            personals = personalization(pivot, names, k, metric)
            personals['task_id'] = task
            personals['dataset_config'] = str(df.dataset_config.iloc[0])
            all_probes.append(probes)
            all_personal.append(personals)
            latest = (probes[probes.strategy.isin(['full_grid_all_older', 'longest_per_model_all_older',
                                                  'middle_per_model_all_older', 'shortest_per_model_all_older'])]
                        .sort_values('strategy'))
            print('Full-cohort historical selection -> lag1:')
            print(latest[['strategy','selected','later_regret_pct','probe_case_forecasts']]
                  .to_string(index=False, float_format=lambda x: f'{x:.3f}'))
            p4 = probes[(probes.strategy.isin(['full_grid','longest_per_model'])) &
                        (probes.sampled_entities == min(n_entities, 4)) &
                        (probes.sampled_older_lags == 1)]
            if len(p4):
                print('4-entity or full-cohort, one-origin probe (mean and p90 regret):')
                print(p4.groupby('strategy').agg(
                    mean_regret=('later_regret_pct','mean'),
                    p90_regret=('later_regret_pct',lambda x: np.quantile(x,.9)),
                    evals=('probe_case_forecasts','first')).round(3).to_string())
            print('Personalisation (later lag1; positive improvement is better):')
            print(personals[['alpha','improvement_over_global_pct','hindsight_headroom_pct',
                             'fraction_entities_improved']].round(3).to_string(index=False))
        sources.append({'task_id': task, 'k': int(df.k.iloc[0]),
                        'configurations': int(quality[0]), 'directory': str(catalog),
                        'case_catalog_sha256': sha256(csv), 'score_catalog_sha256': sha256(score_csv)})
    output = Path(args.output_root) / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(all_probes, ignore_index=True).to_csv(output/'probe_trials.csv', index=False)
    pd.concat(all_personal, ignore_index=True).to_csv(output/'personalization.csv', index=False)
    pr = pd.concat(all_probes, ignore_index=True)
    summary = (pr.groupby(['task_id','dataset_config','metric','strategy',
                           'sampled_entities','sampled_older_lags','probed_configs','probe_case_forecasts'],
                          dropna=False).agg(
                              n_trials=('later_regret_pct','size'),
                              mean_regret_pct=('later_regret_pct','mean'),
                              p90_regret_pct=('later_regret_pct',lambda x: np.quantile(x,.9)),
                              frac_within_1pct=('later_regret_pct',lambda x: float((x<=1).mean())),
                              frac_within_5pct=('later_regret_pct',lambda x: float((x<=5).mean())),
                              mean_probe_future_rho=('selection_loss_rank_rho','mean'))
               .reset_index())
    summary.to_csv(output/'probe_summary.csv', index=False)
    (output/'manifest.json').write_text(json.dumps({
        'phase': '3.0', 'source_catalogs': sources, 'seed': args.seed,
        'repeats': args.repeats, 'entity_budgets': args.entity_budgets,
        'selection_lags': 'latest historical lags >=2',
        'evaluation_lag': 1, 'metric_labels': list(METRICS),
        'cost_units': 'configuration-case forecast calls (not GPU time)',
        'official_test_labels_accessed': False,
        'scope': 'Exploratory within-task replay; no held-out family validation',
        'hindsight_entity_oracle': 'Diagnostic upper bound only'
    }, indent=2)+'\n')
    print('\n'+'='*95)
    print('DATASETS INCLUDED:')
    for x in sources:
        print(f" {x['task_id']} | K={x['k']} | {x['configurations']} configurations | {x['directory']}")
    print(f'PASS: {output}')
    print('Read probe_summary.csv, probe_trials.csv, personalization.csv, manifest.json')
    print('Reminder: model loading/batching make forecast-count savings != wall-clock savings.')


if __name__ == '__main__':
    main()
