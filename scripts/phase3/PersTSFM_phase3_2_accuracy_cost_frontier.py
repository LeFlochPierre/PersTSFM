#!/usr/bin/env python3
"""PersTSFM Phase 3.2: descriptive retrospective accuracy vs measured GPU-inference cost.

Does not access official benchmark targets. It does not estimate the cost of an
adaptive policy or make prospective claims. Measured inference is warm (16 cases,
batch size 8); model load times are independent subprocess measurements.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_SCORES = Path('results/phase2/cross_task_matrix/20261027T150118877258Z_plan/score_catalog.csv')
DEFAULT_COSTS = Path('results/phase3/gpu_cost/20261027T164324669647Z/measurements.csv')

def calculate(scores, costs):
    key = ['model', 'context_steps']
    required_score = set(key + ['task_id', 'dataset_config', 'k', 'n_cases',
                                'calibration_mean_case_mae', 'calibration_mean_case_pinball_3'])
    required_cost = set(key + ['dataset_config', 'batch_size', 'n_entities',
                               'inference_median_seconds', 'load_seconds_import_and_build', 'peak_allocated_bytes'])
    if required_score.difference(scores.columns):
        raise ValueError(f'Missing score columns {sorted(required_score.difference(scores.columns))}')
    if required_cost.difference(costs.columns):
        raise ValueError(f'Missing cost columns {sorted(required_cost.difference(costs.columns))}')
    scores = scores.loc[(scores.dataset_config == 'M_DENSE_1H') & (scores.k == 3)].copy()
    costs = costs.loc[(costs.dataset_config == 'M_DENSE_1H')].copy()
    if scores.empty or costs.empty:
        raise ValueError('Missing M_DENSE_1H K=3 scores or GPU measurements')
    if scores.duplicated(key).any() or costs.duplicated(key).any():
        raise ValueError('Duplicate configurations in score or cost catalog')
    if (costs.n_entities != 16).any() or (costs.batch_size != 8).any():
        raise ValueError('Expected same batch size 8 and entity cohort size 16')
    merged = costs.merge(scores, on=['dataset_config'] + key, how='left', validate='one_to_one', indicator=True)
    if not merged._merge.eq('both').all():
        missing = merged.loc[~merged._merge.eq('both'), key].to_dict('records')
        raise ValueError(f'Missing accuracy results for cost-measured configurations: {missing}')
    merged = merged.drop(columns=['_merge'])
    measures = ['inference_median_seconds', 'load_seconds_import_and_build',
                'peak_allocated_bytes', 'calibration_mean_case_mae', 'calibration_mean_case_pinball_3']
    if not np.isfinite(merged[measures].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite scores or cost measurements')
    if (merged[measures] < 0).any().any() or (merged.inference_median_seconds == 0).any():
        raise ValueError('Negative or zero-invalid score/cost data')
    merged['warm_ms_16'] = merged.inference_median_seconds * 1000
    merged['mem_mib'] = merged.peak_allocated_bytes / 1048576
    merged['mae'] = merged.calibration_mean_case_mae
    merged['pinball'] = merged.calibration_mean_case_pinball_3
    best = merged.mae.min()
    merged['mae_gap_pct'] = 100 * (merged.mae / best - 1)
    # Pareto dominance: another measured configuration is no worse on both
    # historical MAE and warm inference, and strictly better on at least one.
    e = merged[['mae', 'warm_ms_16']].to_numpy(float)
    dominated = ((e[:, None, :] >= e[None, :, :]).all(axis=2) &
                 (e[:, None, :] > e[None, :, :]).any(axis=2)).any(axis=1)
    merged['pareto_undominated'] = ~dominated
    return merged.sort_values(['warm_ms_16', 'mae']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores', type=Path, default=DEFAULT_SCORES)
    ap.add_argument('--costs', type=Path, default=DEFAULT_COSTS)
    ap.add_argument('--output-dir', type=Path, default=Path('results/phase3/accuracy_cost_frontier'))
    args = ap.parse_args()
    if not args.scores.is_file() or not args.costs.is_file():
        raise SystemExit(f'Missing scores/cost file: {args.scores} / {args.costs}')
    merged = calculate(pd.read_csv(args.scores), pd.read_csv(args.costs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / 'mdense_matched_accuracy_gpu_cost.csv', index=False)
    cols = ['model', 'context_steps', 'mae', 'mae_gap_pct', 'pinball',
            'warm_ms_16', 'load_seconds_import_and_build', 'mem_mib']
    print('=' * 104)
    print('PERTSFM PHASE 3.2 | M-DENSE ACCURACY–GPU COST | RETROSPECTIVE ONLY')
    print('K=3 historical forecasting loss vs GPU benchmark on 16 entities, batch=8')
    print('Measured warm latency excludes model loading and external preprocessing')
    print('=' * 104)
    print(f'Matched measured configurations: {len(merged)}')
    print('\nALL CONFIGURATIONS (ascending warm latency):')
    print(merged[cols].to_string(index=False, float_format=lambda z: f'{z:.3f}'))
    print('\nPARETO NON-DOMINATED (historical MAE vs warm latency):')
    print(merged.loc[merged.pareto_undominated, cols].to_string(index=False, float_format=lambda z: f'{z:.3f}'))
    print('\nHINDSIGHT MINIMUM WARM LATENCY AT SELECTED HISTORICAL MAE TOLERANCES:')
    for tolerance in [0, 1, 2, 5, 10, 20]:
        feasible = merged.loc[merged.mae_gap_pct <= tolerance + 1e-8]
        winner = feasible.sort_values(['warm_ms_16', 'mae']).iloc[0]
        print(f'  <= {tolerance:2d}% MAE penalty: {winner.model}@{int(winner.context_steps)} | '
              f'MAE={winner.mae:.3f} | warm={winner.warm_ms_16:.2f} ms/16 | '
              f'gap={winner.mae_gap_pct:.2f}%')
    print(f'\nSaved: {args.output_dir / "mdense_matched_accuracy_gpu_cost.csv"}')
    print('CAUTION: Hindsight frontiers are not an actionable selection policy. No official test labels used.')
    print('CAUTION: Forecast call proxy != GPU time; loading, caching, model switches and batching need a full cost model.')

if __name__ == '__main__':
    main()
