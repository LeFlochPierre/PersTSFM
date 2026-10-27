"""Exploratory Phase-2 three-context selection audit (development-only).

Uses already-saved per-model summary.csv files. No inference, no tuning, no
access to held-out data. This is a *pilot analogue* of V1.2, NOT the frozen
original V1.2 policy: the pilot's 3 contexts are not necessarily its
fingerprint-derived SHORT/STRUCTURAL/CAPACITY candidates.

Usage: python -m scripts.phase2.policy_pilot_audit
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_MODELS = (
    'chronos2', 'timesfm3', 'tirex2',
    'toto2_22m', 'toto2_313m', 'moirai2_small',
)


def load_matrix(root: Path, required_models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or not (dataset_dir / 'context').is_dir():
            continue
        for model in required_models:
            path = dataset_dir / 'context' / model / 'summary.csv'
            if not path.is_file():
                raise FileNotFoundError(f'Incomplete benchmark: missing {path}')
            summary = pd.read_csv(path)
            for col in ('context_steps', 'mean_entity_mae'):
                if col not in summary.columns:
                    raise ValueError(f'{path}: missing required column {col}')
            part = summary.loc[:, ['context_steps', 'mean_entity_mae']].copy()
            part['dataset'] = dataset_dir.name
            part['model'] = model
            rows.append(part)
    if not rows:
        raise RuntimeError(f'No dataset context summaries found under {root}')
    df = pd.concat(rows, ignore_index=True)
    if not np.isfinite(df['mean_entity_mae'].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite MAE in input summaries')
    if (df['mean_entity_mae'] < 0).any():
        raise ValueError('Negative MAE in input summaries')
    if df.duplicated(['dataset', 'model', 'context_steps']).any():
        raise ValueError('Duplicate (dataset,model,context_steps) rows')
    counts = df.groupby(['dataset','model']).size()
    if not counts.eq(3).all():
        raise ValueError('This pilot analogue expects exactly three contexts per dataset/model; got:\n'+ counts.to_string())
    return df.sort_values(['dataset','model','context_steps']).reset_index(drop=True)


def best(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(['mean_entity_mae','model','context_steps']).iloc[0]


def simulate(df: pd.DataFrame, band: float) -> dict:
    by_model = {
        model: sub.sort_values('context_steps').reset_index(drop=True)
        for model,sub in df.groupby('model', sort=True)
    }
    first_two = pd.concat([sub.iloc[:2] for sub in by_model.values()], ignore_index=True)
    mid = pd.concat([sub.iloc[[1]] for sub in by_model.values()], ignore_index=True)
    best_mid = float(mid['mean_entity_mae'].min())
    competitive = set(mid.loc[mid['mean_entity_mae'] <= best_mid*(1+band+1e-12), 'model'])
    last = pd.concat([by_model[m].iloc[[2]] for m in sorted(competitive)], ignore_index=True)
    observed = pd.concat([first_two, last], ignore_index=True)
    if observed.duplicated(['model','context_steps']).any():
        raise ValueError('Duplicate candidate evaluated')
    winner = best(observed)
    oracle = best(df)
    regret = 100*(float(winner['mean_entity_mae'])/float(oracle['mean_entity_mae'])-1) if oracle['mean_entity_mae']>0 else 0.
    return dict(dataset=str(oracle['dataset']), band_pct=100*band,
                selected_model=winner['model'], selected_context=int(winner['context_steps']),
                selected_mae=float(winner['mean_entity_mae']), oracle_model=oracle['model'],
                oracle_context=int(oracle['context_steps']), oracle_mae=float(oracle['mean_entity_mae']),
                regret_pct=max(0., regret), evaluations=len(observed), full_evaluations=len(df),
                competitive_models=len(competitive), avoided_configurations=len(df)-len(observed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/phase2/development_pilot'))
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    args = parser.parse_args()
    root = args.root.resolve()
    matrix = load_matrix(root, tuple(args.models))
    rows, near = [], []
    for ds, group in matrix.groupby('dataset', sort=True):
        oracle = best(group)
        for band in (0.0,0.01,0.02,0.05,0.10):
            rows.append(simulate(group, band))
        for epsilon in (0.01,0.02,0.05):
            n = int((group['mean_entity_mae'] <= oracle['mean_entity_mae']*(1+epsilon+1e-12)).sum())
            near.append(dict(dataset=ds, tolerance_pct=100*epsilon,
                             near_optimal_configurations=n, full_evaluations=len(group)))
    replay = pd.DataFrame(rows)
    near_df = pd.DataFrame(near)
    out_dir = root / 'analysis'
    out_dir.mkdir(parents=True,exist_ok=True)
    replay.to_csv(out_dir/'pilot_policy_analogue.csv', index=False)
    near_df.to_csv(out_dir/'pilot_near_optimal_counts.csv', index=False)
    print('='*90)
    print('PersTSFM: THREE-CONTEXT PILOT POLICY ANALOGUE — DEVELOPMENT ONLY')
    print('='*90)
    print('CAUTION: Pilot SHORT/MIDDLE/LONG are not original V1.2 fingerprint anchors.')
    print('Metrics are from 8 sampled entities/dataset; no compute-cost estimates are used.')
    print(f'Datasets: {matrix.dataset.nunique()} | Checkpoints: {len(args.models)} | Configurations: {len(matrix)}')
    chosen = replay[np.isclose(replay.band_pct,2.)]
    cols=['dataset','oracle_model','oracle_context','selected_model','selected_context',
          'regret_pct','evaluations','full_evaluations']
    print('\nPilot two-stage selection at 2% competitive band:')
    print(chosen[cols].to_string(index=False, formatters={'regret_pct':lambda x:f'{x:.3f}%'}))
    print('\nSensitivity (macro mean and worst dataset regret):')
    sensitivity=replay.groupby('band_pct',as_index=False).agg(
        mean_regret_pct=('regret_pct','mean'), max_regret_pct=('regret_pct','max'),
        total_evaluations=('evaluations','sum'),total_full=('full_evaluations','sum'))
    sensitivity['reduction_pct']=100*(1-sensitivity.total_evaluations/sensitivity.total_full)
    print(sensitivity.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print('\nConfigurations within 1%, 2%, 5% of each dataset pilot oracle:')
    print(near_df.pivot(index='dataset',columns='tolerance_pct',values='near_optimal_configurations').to_string())
    print(f'\nSaved: {out_dir}/pilot_policy_analogue.csv and pilot_near_optimal_counts.csv')


if __name__ == '__main__':
    main()
