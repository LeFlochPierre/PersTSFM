#!/usr/bin/env python3
"""Score TSFM forecasts against *historical*, pre-official FEV calibration labels.

No fev import, no official evaluation-window access, and no reference to any
get_ground_truth() method. Diagnostic MAE and mean pinball of q10/q50/q90 only.
Not benchmark SQL, MASE, or an official fev result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase2.run_preofficial_calibration import (INPUT_ROOT, choose_cases,
                                                          load_frozen_inputs, sha256)


def score_join(run: dict, inp: pd.DataFrame, truth: pd.DataFrame, pred: pd.DataFrame):
    """Require an exact match on case IDs, metadata and forecast timestamps."""
    expected = set(inp.case_id.astype(str))
    if any(df.case_id.duplicated().any() for df in (inp, truth, pred)):
        raise ValueError('Duplicate case ID in inputs, labels or forecasts')
    if not expected or set(truth.case_id.astype(str)) != expected or set(pred.case_id.astype(str)) != expected:
        raise ValueError('Case ID mismatch; partial or mixed-cohort predictions forbidden')
    y_map = truth.set_index('case_id').to_dict(orient='index')
    p_map = pred.set_index('case_id').to_dict(orient='index')
    case_scores = []
    for r in inp.to_dict(orient='records'):
        case = str(r['case_id'])
        y, p = y_map[case], p_map[case]
        if (str(y['entity_id']) != str(r['entity_id']) or str(p['entity_id']) != str(r['entity_id']) or
            int(y['lag_before_first_official']) != int(r['lag_before_first_official']) or
            int(p['lag_before_first_official']) != int(r['lag_before_first_official']) or
            str(p['task_id']) != str(r['task_id']) or
            int(p['context_steps']) != int(run['context_steps']) or
            int(p['k']) != int(run['k']) or str(p['model']) != run['model']):
            raise ValueError(f'{case}: metadata mismatch')
        expected_times = np.asarray(r['future_timestamps_ns'], dtype=np.int64)
        y_times = np.asarray(y['target_timestamps_ns'], dtype=np.int64)
        p_times = np.asarray(p['future_timestamps_ns'], dtype=np.int64)
        if not np.array_equal(expected_times, y_times) or not np.array_equal(expected_times, p_times):
            raise ValueError(f'{case}: target/forecast timestamps mismatch')
        obs = np.asarray(y['target_values'], dtype=np.float64)
        if obs.shape != (int(r['horizon']),) or not np.isfinite(obs).all():
            raise ValueError(f'{case}: calibration label missing/nonfinite; cannot score')
        predicted = np.asarray(p['prediction'], dtype=np.float64)
        if predicted.shape != obs.shape or not np.isfinite(predicted).all():
            raise ValueError(f'{case}: invalid point forecast')
        pinballs = {}
        for q, name in ((.1,'q10'),(.5,'q50'),(.9,'q90')):
            qhat = np.asarray(p[name], dtype=np.float64)
            if qhat.shape != obs.shape or not np.isfinite(qhat).all():
                raise ValueError(f'{case}: invalid quantile {name}')
            residual = obs - qhat
            pinballs[name] = float(np.mean(np.maximum(q * residual, (q - 1) * residual)))
        quantiles = [np.asarray(p[name], dtype=np.float64) for name in ('q10','q50','q90')]
        if np.any(quantiles[0] > quantiles[1]+1e-6) or np.any(quantiles[1] > quantiles[2]+1e-6):
            raise ValueError(f'{case}: crossing quantiles')
        case_scores.append(dict(case_id=case, entity_id=str(r['entity_id']),
            lag_before_first_official=int(r['lag_before_first_official']),
            mae=float(np.abs(predicted-obs).mean()),
            mean_pinball_3=float(np.mean(list(pinballs.values()))), **{f'pinball_{key}': v for key,v in pinballs.items()}))
    return pd.DataFrame(case_scores)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    ap.add_argument('--allow-smoke', action='store_true', help='Explicitly score incomplete smoke run; not comparable')
    args=ap.parse_args()
    path=args.predictions
    run_path=path.with_name('run_manifest.json')
    run=json.loads(run_path.read_text(encoding='utf-8'))
    if run.get('schema') != 'pertsfm_preofficial_calibration_predictions_v1' or run.get('role') != 'PREOFFICIAL_HISTORICAL_CALIBRATION':
        raise RuntimeError('Expected historical calibration predictions only')
    if run['cohort'] != 'FULL' and not args.allow_smoke:
        raise RuntimeError('Refusing subset/smoke benchmark scoring without --allow-smoke')
    root=args.input_root / run['task_id']
    manifest, full=load_frozen_inputs(root)
    if (sha256(root/'manifest.json') != run['input_manifest_sha256'] or
        manifest['input_sha256'] != run['input_sha256'] or
        sha256(path) != run['predictions_sha256']):
        raise RuntimeError('Input or predictions provenance hash mismatch')
    truth_path=root/'historical_calibration_labels.parquet'
    if sha256(truth_path) != manifest['labels_sha256']:
        raise RuntimeError('Historical labels hash mismatch')
    context, k=int(run['context_steps']),int(run['k'])
    if run['cohort']=='FULL':
        selected=choose_cases(full,k=k,context=context)
    else:
        pred_ids=set(pd.read_parquet(path,columns=['case_id']).case_id.astype(str))
        subset=full.loc[full.case_id.astype(str).isin(pred_ids)]
        selected=choose_cases(subset,k=k,context=context)
    if int(run['n_cases']) != len(selected) or int(run['n_entities']) != selected.entity_id.nunique():
        raise RuntimeError('Run case/entity counts differ from frozen selection')
    truth=pd.read_parquet(truth_path)
    truth=truth.loc[truth.case_id.astype(str).isin(set(selected.case_id.astype(str)))].copy()
    predictions=pd.read_parquet(path)
    result=score_join(run,selected,truth,predictions)
    summary=dict(schema='pertsfm_preofficial_calibration_score_v1',role='HISTORICAL_CALIBRATION_ONLY',
        task_id=run['task_id'], dataset_config=manifest['dataset_config'], model=run['model'],
        context_steps=context,k=k,cohort=run['cohort'],n_cases=len(result),
        n_entities=int(result.entity_id.nunique()),
        calibration_mean_case_mae=float(result.mae.mean()),
        calibration_mean_case_pinball_3=float(result.mean_pinball_3.mean()),
        inference_seconds=run['inference_seconds'],model_load_seconds=run['model_load_seconds'],
        note='Historical calibration ONLY, 3 quantiles; NOT official FEV SQL/leaderboard/MASE.\n'
             'Losses may only be compared on identical task, K, and case cohort.')
    out_json=path.with_name('calibration_score.json')
    out_csv=path.with_name('calibration_case_metrics.csv')
    if out_json.exists() or out_csv.exists():
        raise FileExistsError('Refusing score overwrite')
    result.to_csv(out_csv,index=False)
    out_json.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('='*83)
    print('PRE-OFFICIAL HISTORICAL CALIBRATION (NOT OFFICIAL FEV TEST)')
    print('='*83)
    print(f"{manifest['dataset_config']} | {run['model']} | context={context} | K={k} | cases={len(result)}")
    print(f"Calibration MAE: {summary['calibration_mean_case_mae']:.6f}")
    print(f"Calibration mean pinball (q10/50/90): {summary['calibration_mean_case_pinball_3']:.6f}")
    print(f"Measured inference {run['inference_seconds']:.3f}s (exploratory timing)")
    print('PASS:',out_json)


if __name__=='__main__':
    main()
