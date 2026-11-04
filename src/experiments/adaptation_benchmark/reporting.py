"""Rolling fold analysis from AFTER frozen later scoring only."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .history import file_hash
from .uncertainty import paired_entity_bootstrap


def _seed_average_entities(path: Path, strategy: str):
    entity=pd.read_csv(path/'later_entity.csv')
    sub=entity.loc[entity.strategy==strategy]
    if sub.empty:
        raise RuntimeError(f'Missing later-entity scores for {strategy}')
    if sub.duplicated(['trial_key','entity_id']).any():
        raise RuntimeError('Duplicate trial/entity pair')
    return sub.groupby('entity_id')['later_mae'].mean().sort_index()


def report_suite(suite: Path, *, replicates=2000, seed=42):
    suite=Path(suite).resolve()
    manifest=json.loads((suite/'suite_manifest.json').read_text())
    freeze=json.loads((suite/'suite_freeze.json').read_text())
    if file_hash(suite/'suite_manifest.json')!=freeze['suite_manifest_sha256']:
        raise RuntimeError('Suite manifest altered after freeze')
    result=[]
    for ref in freeze['runs']:
        path=Path(ref['run'])
        if file_hash(path/'selection.json')!=ref['selection_sha256']:
            raise RuntimeError('Selection changed since suite freeze')
        if not (path/'later.csv').exists():
            raise RuntimeError(f'Fold {ref["index"]} not yet scored')
        lag2=pd.read_csv(path/'validation_strategies.csv')
        later=pd.read_csv(path/'later.csv')
        locked=json.loads((path/'selection.json').read_text())['strategy']
        if locked not in later.strategy.to_list():
            raise RuntimeError('Locked strategy missing from later results')
        best_zero=lag2.loc[lag2.steps==0].sort_values(['mean','context']).iloc[0]['strategy']
        if best_zero not in later.strategy.to_list():
            raise RuntimeError('Validation-chosen zero-shot comparator missing')
        chosen=later.loc[later.strategy==locked].iloc[0]
        baseline=later.loc[later.strategy==best_zero].iloc[0]
        oracle=later.loc[later.later_mae.idxmin()]
        base=_seed_average_entities(path,best_zero)
        alt=_seed_average_entities(path,locked)
        if not base.index.equals(alt.index):
            raise RuntimeError('Non-matching entity sets; paired analysis invalid')
        ci=paired_entity_bootstrap(base.to_numpy(),alt.to_numpy(),replicates=replicates,seed=seed+ref['index'])
        result.append({'fold':ref['index'], 'selected':locked, 'best_validation_zero':best_zero,
            'later_selected_mae':float(chosen.later_mae),
            'later_validation_zero_mae':float(baseline.later_mae),
            'later_oracle_mae_diagnostic':float(oracle.later_mae),
            'later_oracle_strategy_diagnostic':str(oracle.strategy),
            'selection_regret_pct':float(100*(chosen.later_mae-oracle.later_mae)/oracle.later_mae),
            'gain_vs_validation_zero_pct':ci['gain_pct'],
            'gain_ci_2p5':ci['ci_lower_pct'],'gain_ci_97p5':ci['ci_upper_pct'],
            'n_entities':ci['n_entities']})
    df=pd.DataFrame(result).sort_values('fold')
    output=suite/'rolling_summary.csv'
    if output.exists():
        existing=pd.read_csv(output)
        # A second invocation is safe but must not silently rewrite results.
        if (existing.shape!=df.shape or list(existing.columns)!=list(df.columns) or
            not existing['selected'].equals(df['selected']) or
            not np.allclose(existing.select_dtypes(include=[np.number]).to_numpy(dtype=float),
                            df.select_dtypes(include=[np.number]).to_numpy(dtype=float),
                            rtol=1e-10, atol=1e-10)):
            raise RuntimeError('Existing rolling summary differs; manual audit required')
    else:
        df.to_csv(output,index=False)
    print('\n'+'='*112)
    print('PHASE 3.5 | ROLLING HISTORICAL HOLDOUTS | ALL SELECTIONS FROZEN BEFORE LATER SCORING')
    print('='*112)
    print(df[['fold','selected','later_selected_mae','later_validation_zero_mae',
              'gain_vs_validation_zero_pct','gain_ci_2p5','gain_ci_97p5','selection_regret_pct']]
              .to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print(f"FOLDS={len(df)} | mean selected later MAE={df.later_selected_mae.mean():.5f} | "
          f"mean paired gain vs lag2-chosen zero-shot={df.gain_vs_validation_zero_pct.mean():+.3f}% | "
          f"mean hindsight regret={df.selection_regret_pct.mean():.3f}%")
    print('CIs: descriptive paired-entity bootstrap within each fold; sensors/folds are correlated, not independent datasets.')
    print('Hindsight oracle is DIAGNOSTIC ONLY. Historical holdouts are NOT official FEV tests.')
    print(f'PASS: {output}')
    return df
