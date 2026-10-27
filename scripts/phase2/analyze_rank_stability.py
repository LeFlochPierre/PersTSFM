"""PersTSFM Phase 2: paired entity-bootstrap audit of context-dependent ranks.

Read-only analysis of Phase-2 development-pilot case_metrics.csv files.
Works with per-entity, per-origin case metrics from the project runner.

Outputs under development_pilot/analysis/rank_stability/.
These are exploratory development estimates, NOT held-out confidence claims.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["entity_id", "eval_index", "eval_origin"]
REQUIRED = set(KEYS + ["context_steps", "mae"])


def load_dataset(root: Path, dataset: str):
    model_dirs = sorted((root / dataset / "context").glob("*/case_metrics.csv"))
    if len(model_dirs) < 2:
        raise ValueError(f"{dataset}: need >=2 model case_metrics.csv files; got {len(model_dirs)}")
    frames = {}
    grids = {}
    keys = {}
    for path in model_dirs:
        model = path.parent.name
        df = pd.read_csv(path, dtype={k: str for k in KEYS})
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        df["context_steps"] = pd.to_numeric(df["context_steps"], errors="raise").astype(int)
        df["mae"] = pd.to_numeric(df["mae"], errors="raise")
        if not np.isfinite(df["mae"]).all():
            raise ValueError(f"{path}: nonfinite errors")
        if df.duplicated(KEYS + ["context_steps"]).any():
            raise ValueError(f"{path}: duplicate (case, context)")
        grids[model] = tuple(sorted(df.context_steps.unique()))
        for context, g in df.groupby("context_steps"):
            index = pd.MultiIndex.from_frame(g[KEYS])
            key_set = set(index.tolist())
            keys[(model, context)] = key_set
        frames[model] = df

    if len(set(grids.values())) != 1:
        raise ValueError(f"{dataset}: context grids differ: {grids}")
    ref = next(iter(keys.values()))
    if any(v != ref for v in keys.values()):
        raise ValueError(f"{dataset}: cases differ across models/contexts; refusing unpaired comparison")

    ctxs = sorted(next(iter(grids.values())))
    models = sorted(frames)
    entity_ids = sorted({k[0] for k in ref})
    cube = np.empty((len(entity_ids), len(models), len(ctxs)), dtype=float)
    n_cases = len(ref)
    for mi, model in enumerate(models):
        df = frames[model]
        grouped = df.groupby(["context_steps", "entity_id"], sort=False)["mae"].mean()
        for ci, ctx in enumerate(ctxs):
            values = grouped.loc[ctx].reindex(entity_ids)
            if values.isna().any():
                raise ValueError(f"{dataset}/{model}/{ctx}: missing entity")
            cube[:, mi, ci] = values.to_numpy(dtype=float)
    return models, ctxs, entity_ids, n_cases, cube


def analyze(root: Path, n_boot: int, seed: int, reversal_threshold: float,
            margin_threshold_pct: float):
    rng = np.random.default_rng(seed)
    summaries, reversals, gains, winners = [], [], [], []
    directories = sorted(d for d in root.iterdir()
                         if d.is_dir() and (d / 'context').is_dir())
    if not directories:
        raise ValueError(f"No dataset/context results found under {root}")

    for directory in directories:
        dataset = directory.name
        models, contexts, entities, n_cases, cube = load_dataset(root, dataset)
        n_ent, n_models, n_ctx = cube.shape
        obs = cube.mean(axis=0)
        sampled_idx = rng.integers(0, n_ent, size=(n_boot, n_ent))
        boot = cube[sampled_idx].mean(axis=1)  # [resample, model, context]

        # Global best configuration among this *pilot grid*, not exhaustive search.
        best_flat = int(np.argmin(obs))
        best_mi, best_ci = np.unravel_index(best_flat, obs.shape)
        boot_winner = np.argmin(boot.reshape(n_boot, -1), axis=1)
        win_prob = float(np.mean(boot_winner == best_flat))
        for mi, model in enumerate(models):
            for ci, ctx in enumerate(contexts):
                winners.append(dict(dataset=dataset, model=model, context_steps=int(ctx),
                                    mae=float(obs[mi, ci]),
                                    bootstrap_best_probability=float(np.mean(
                                        boot_winner == (mi*n_ctx + ci)))))

        for mi, model in enumerate(models):
            g = 100.0 * (obs[mi, 0] - obs[mi, -1]) / max(obs[mi, 0], 1e-12)
            bg = 100.0 * (boot[:, mi, 0] - boot[:, mi, -1]) / np.maximum(boot[:, mi, 0], 1e-12)
            gains.append(dict(dataset=dataset, model=model, short_context_steps=int(contexts[0]),
                              long_context_steps=int(contexts[-1]), short_mae=float(obs[mi,0]),
                              long_mae=float(obs[mi,-1]), gain_pct=float(g),
                              gain_ci025_pct=float(np.quantile(bg, 0.025)),
                              gain_ci975_pct=float(np.quantile(bg, 0.975)),
                              gain_positive_probability=float(np.mean(bg > 0)),
                              best_context_steps=int(contexts[np.argmin(obs[mi])])))

        flip_count = 0
        stable_flip_count = 0
        for ia, ib in combinations(range(n_models), 2):
            short_diff = obs[ia, 0] - obs[ib, 0]
            long_diff = obs[ia, -1] - obs[ib, -1]
            flipped = bool(short_diff * long_diff < 0)
            bs_short = boot[:, ia, 0] - boot[:, ib, 0]
            bs_long = boot[:, ia, -1] - boot[:, ib, -1]
            bflip = float(np.mean(bs_short * bs_long < 0))
            ms = 100 * abs(short_diff) / max(min(obs[ia,0],obs[ib,0]), 1e-12)
            ml = 100 * abs(long_diff) / max(min(obs[ia,-1],obs[ib,-1]), 1e-12)
            stable = bool(flipped and bflip >= reversal_threshold
                          and ms >= margin_threshold_pct and ml >= margin_threshold_pct)
            flip_count += int(flipped)
            stable_flip_count += int(stable)
            reversals.append(dict(dataset=dataset, model_a=models[ia], model_b=models[ib],
                                  short_context_steps=int(contexts[0]),
                                  long_context_steps=int(contexts[-1]),
                                  reversal=flipped, bootstrap_flip_probability=bflip,
                                  margin_short_pct=ms, margin_long_pct=ml,
                                  strong_reversal_exploratory=stable))
        summaries.append(dict(dataset=dataset, n_models=n_models, n_entities=n_ent,
                              n_cases=n_cases, n_contexts=n_ctx,
                              best_pilot_model=models[best_mi],
                              best_pilot_context_steps=int(contexts[best_ci]),
                              best_pilot_mae=float(obs[best_mi,best_ci]),
                              best_pilot_bootstrap_win_probability=win_prob,
                              observed_pairwise_reversals=flip_count,
                              strong_pairwise_reversals_exploratory=stable_flip_count,
                              total_model_pairs=n_models * (n_models - 1) // 2))

    return tuple(pd.DataFrame(rows) for rows in (summaries, reversals, gains, winners))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='results/phase2/development_pilot')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flip-probability', type=float, default=0.9)
    parser.add_argument('--minimum-margin-pct', type=float, default=1.0)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be >= 100')
    root = Path(args.output_root)
    if not root.is_dir():
        parser.error(f'Benchmark directory missing: {root}')
    summary, flips, gains, wins = analyze(root, args.bootstrap, args.seed,
                                        args.flip_probability, args.minimum_margin_pct)
    dest = root / 'analysis' / 'rank_stability'
    dest.mkdir(parents=True, exist_ok=True)
    for filename, df in [
        ('dataset_summary.csv', summary), ('pairwise_reversals.csv', flips),
        ('context_gain_uncertainty.csv', gains), ('configuration_win_probabilities.csv', wins),
    ]:
        df.to_csv(dest / filename, index=False)
    print('='*80)
    print('PersTSFM Phase-2: PAIRED ENTITY BOOTSTRAP — DEVELOPMENT ONLY')
    print('='*80)
    print(f'Bootstrap draws: {args.bootstrap} | seed: {args.seed}')
    print(f'Strong reversal (exploratory): flip probability >= {args.flip_probability:.0%}, '
          f'endpoint model gap >= {args.minimum_margin_pct:g}% at BOTH contexts')
    cols = ['dataset', 'best_pilot_model', 'best_pilot_context_steps',
            'best_pilot_bootstrap_win_probability', 'observed_pairwise_reversals',
            'strong_pairwise_reversals_exploratory']
    print('\nDataset-level stability:')
    print(summary[cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'\nObserved reversals: {int(flips.reversal.sum())}/{len(flips)}')
    print(f'Strong exploratory reversals: {int(flips.strong_reversal_exploratory.sum())}/{len(flips)}')
    print('\nContext gains (95% paired entity-bootstrap percentiles):')
    print(gains[['dataset', 'model', 'gain_pct', 'gain_ci025_pct', 'gain_ci975_pct']]
          .to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print(f'\nSaved: {dest}')
    print('Caution: 8 sampled entities/dataset and reused development data; '
          'uncertainty is exploratory, and does not establish generalisation.')


if __name__ == '__main__':
    main()
