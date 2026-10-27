#!/usr/bin/env python3
"""Run TSFM inference on pre-official historical calibration input files ONLY.

No fev import or label-file reads. Requires Phase 2.9 exporter v1. First iteration
supports strictly regular, fully finite histories. Do not use for official FEV
benchmark scores or for irregular calendar series (month/quarter/year).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_ROOT = Path('results/phase2/preofficial_calibration/fev_bench_v0_10_0')
OUTPUT_ROOT = Path('results/phase2/preofficial_forecasts/fev_bench_v0_10_0')
NS_PER_MINUTE = 60_000_000_000


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_frozen_inputs(root: Path):
    manifest_path, data_path = root / 'manifest.json', root / 'calibration_inputs.parquet'
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (m.get('schema') != 'pertsfm_preofficial_calibration_v1' or
        m.get('role') != 'HISTORICAL_CALIBRATION_ONLY_NOT_OFFICIAL_EVALUATION' or
        m.get('official_ground_truth_accessed') is not False or
        m.get('official_test_target_values_included') is not False or
        m.get('timestamp_unit') != 'ns'):
        raise RuntimeError('Unrecognised or non-historical calibration protocol')
    if sha256(data_path) != m['input_sha256']:
        raise RuntimeError('Historical input hash differs from manifest')
    df = pd.read_parquet(data_path)
    if len(df) != int(m['n_cases']) or df['case_id'].duplicated().any():
        raise RuntimeError('Calibration case IDs/count mismatch')
    if set(df['task_id'].astype(str)) != {m['task_id']}:
        raise RuntimeError('Wrong task ID in inputs')
    return m, df


def choose_cases(df: pd.DataFrame, *, k: int, context: int, max_entities: int = 0):
    """Require identical K origin set for every included entity; no silent exclusions."""
    if k not in (1, 2, 3) or context < 2 or max_entities < 0:
        raise ValueError('K must be 1..3, context>=2, entity limit >=0')
    chosen = df.loc[df['lag_before_first_official'].astype(int).between(1, k)].copy()
    if chosen.empty:
        raise ValueError('No historical calibration cases at requested K')
    if max_entities:
        keep = sorted(chosen['entity_id'].astype(str).unique())[:max_entities]
        chosen = chosen.loc[chosen['entity_id'].astype(str).isin(keep)].copy()
    if chosen.empty:
        raise ValueError('No cases after entity cap')
    counts = chosen.groupby('entity_id')['lag_before_first_official'].agg(list)
    if any(sorted(int(x) for x in lags) != list(range(1, k + 1)) for lags in counts):
        raise ValueError('Not all entities have K historical calibration horizons; choose another task/K')
    offered = chosen['feasible_contexts'].map(lambda arr: context in set(int(c) for c in arr))
    if not offered.all():
        raise ValueError(f'Context {context} infeasible for {int((~offered).sum())} cases; cohort not reduced silently')
    # Only the final `context` points are actually passed to the model.
    # Missing points outside that slice must not disqualify a forecast.
    for row in chosen.to_dict(orient='records'):
        suffix = np.asarray(row['history'], dtype=np.float64)[-context:]
        if len(suffix) != context or not np.isfinite(suffix).all():
            raise ValueError(
                f"{row['case_id']}: Missing input observations inside selected context={context}"
            )
    return chosen.sort_values(['entity_id', 'lag_before_first_official'], ascending=[True, False]).reset_index(drop=True)


def make_forecast_input(row: dict, context: int):
    """Validate time alignment and prepare typed data, independent of TSFM libraries."""
    y = np.asarray(row['history'], dtype=np.float64)
    ts = np.asarray(row['history_timestamps_ns'], dtype=np.int64)
    future = np.asarray(row['future_timestamps_ns'], dtype=np.int64)
    h = int(row['horizon'])
    if y.ndim != 1 or ts.ndim != 1 or future.ndim != 1:
        raise ValueError('Expected 1-D values and timestamps')
    if (len(y) != len(ts) or len(y) != int(row['observed_steps_saved']) or
        len(y) < context or len(future) != h or h <= 0 or
        not np.isfinite(y[-context:]).all()):
        raise ValueError(f"{row['case_id']}: invalid length/nonfinite history")
    if not np.all(np.diff(ts) > 0) or not np.all(np.diff(future) > 0) or ts[-1] >= future[0]:
        raise ValueError(f"{row['case_id']}: chronology error")
    recent = ts[-context:]
    dt = np.concatenate([np.diff(recent), [future[0] - recent[-1]], np.diff(future)])
    if len(dt) == 0 or np.any(dt != dt[0]) or dt[0] <= 0 or int(dt[0]) % NS_PER_MINUTE:
        raise ValueError(f"{row['case_id']}: irregular or sub-minute cadence; calendar series unsupported")
    return y[-context:].astype(np.float32), recent, future, int(dt[0] // NS_PER_MINUTE)


def checked_output(outputs, inputs, rows, futures, model, context, k, task_id,
                   *, quantile_policy='strict', crossing_events=None):
    """Validate quantiles; optionally rearrange crossed quantiles with provenance.

    Rearrangement is explicitly opt-in and applied per forecast time step.
    Raw q10/q50/q90 are stored for reproducibility. The point forecast remains
    untouched, so MAE is comparable across the strict and rearranged policies.
    """
    if quantile_policy not in ('strict', 'rearrange'):
        raise ValueError('Unknown quantile crossing policy')
    if len(outputs) != len(inputs):
        raise ValueError('Model returned wrong number of predictions')
    mapping = {str(o.case_id): o for o in outputs}
    if len(mapping) != len(outputs) or set(mapping) != {str(x.case_id) for x in inputs}:
        raise ValueError('Model returned duplicated/unexpected case IDs')
    saved = []
    for i, row, future in zip(inputs, rows, futures):
        obj = mapping[i.case_id]
        pred = {}
        for name in ('prediction', 'q10', 'q50', 'q90'):
            v = np.asarray(getattr(obj, name), dtype=np.float64)
            if v.shape != (len(future),) or not np.isfinite(v).all():
                raise ValueError(f'{i.case_id}: invalid {name} output')
            pred[name] = v.tolist()
        raw = np.stack([np.asarray(pred[name], dtype=np.float64)
                        for name in ('q10', 'q50', 'q90')], axis=0)
        pair_violations = np.maximum(raw[0] - raw[1], raw[1] - raw[2])
        crossed = pair_violations > 1e-6
        n_crossed = int(np.count_nonzero(crossed))
        max_crossing = max(0.0, float(np.max(pair_violations)))
        if n_crossed and quantile_policy == 'strict':
            raise ValueError(
                f'{i.case_id}: crossing quantiles at {n_crossed}/{len(future)} steps '
                f'(maximum violation={max_crossing:.9g} target units); '
                'inspect the case before opting into --quantile-policy rearrange'
            )
        # Rearrangement sorts the requested 0.1,0.5,0.9 quantiles *at each horizon*.
        # Never modify the independent point prediction; this keeps MAE intact.
        adjusted = np.sort(raw, axis=0) if quantile_policy == 'rearrange' else raw
        correction = float(np.mean(np.abs(adjusted - raw)))
        if crossing_events is not None and n_crossed:
            crossing_events.append(dict(case_id=str(i.case_id),
                                        n_crossing_steps=n_crossed,
                                        max_pairwise_crossing=max_crossing,
                                        mean_abs_quantile_correction=correction))
        saved.append(dict(task_id=task_id, case_id=i.case_id,
                          entity_id=str(row['entity_id']),
                          lag_before_first_official=int(row['lag_before_first_official']),
                          horizon=int(row['horizon']), context_steps=context, k=k,
                          model=model, future_timestamps_ns=future.tolist(),
                          prediction=pred['prediction'],
                          q10=adjusted[0].tolist(), q50=adjusted[1].tolist(),
                          q90=adjusted[2].tolist(),
                          raw_q10=raw[0].tolist(), raw_q50=raw[1].tolist(),
                          raw_q90=raw[2].tolist(),
                          quantile_crossing_steps=n_crossed,
                          max_pairwise_quantile_crossing=max_crossing,
                          mean_abs_quantile_correction=correction,
                          quantile_policy=quantile_policy))
    return saved

def synchronize_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task-id', required=True, help='Frozen exporter directory (e.g. fev_0_10_0_044)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--context', type=int, required=True)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--max-entities', type=int, default=0, help='0=full exported cohort; nonzero for smoke only')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--quantile-policy', choices=('strict', 'rearrange'), default='strict',
                    help='Default strict; opt in to time-wise monotone rearrangement, retaining raw quantiles')
    ap.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    ap.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('batch size must be positive')
    base = args.input_root / args.task_id
    manifest, df = load_frozen_inputs(base)
    if manifest['task_id'] != args.task_id or args.context not in manifest['contexts_steps']:
        raise RuntimeError('Task/context does not match frozen manifest')
    if args.k > int(manifest['max_origins']):
        raise ValueError('Requested more origins than exporter supplied')
    if args.model == 'tirex2' and args.context > 2048:
        raise ValueError('TiRex-2 context exceeds its supported 2048 steps')
    df = choose_cases(df, k=args.k, context=args.context, max_entities=args.max_entities)
    suffix = 'full' if not args.max_entities else f'smoke_e{args.max_entities}'
    dest = args.output_root / args.task_id / args.model / f'context_{args.context}' / f'k{args.k}' / suffix
    pred_path, run_path = dest / 'predictions.parquet', dest / 'run_manifest.json'
    if dest.exists():
        raise FileExistsError(f'Output exists: {dest}; no overwrite, even for partial runs')
    from src.models.adapters import build_model, ForecastInput
    print(f"CALIBRATION ONLY | {manifest['dataset_config']} | {args.task_id} | {args.model} | context={args.context} K={args.k} cases={len(df)}", flush=True)
    t0 = time.perf_counter()
    model = build_model(args.model, batch_size=args.batch_size)
    synchronize_cuda()
    load_sec = time.perf_counter() - t0
    print(f'Model load: {load_sec:.3f}s', flush=True)
    forecasts, batch_times, crossing_events = [], [], []
    for start in range(0, len(df), args.batch_size):
        rows = df.iloc[start:start + args.batch_size].to_dict(orient='records')
        inputs, future_stamps, cadences = [], [], set()
        for row in rows:
            hist, dates, future, cadence = make_forecast_input(row, args.context)
            inputs.append(ForecastInput(case_id=str(row['case_id']),
                timestamps=pd.DatetimeIndex(pd.to_datetime(dates, unit='ns', utc=True)).tz_localize(None),
                values=hist))
            future_stamps.append(future)
            cadences.add(cadence)
        if len(cadences) != 1:
            raise RuntimeError('Mixed sampling intervals within TSFM batch')
        synchronize_cuda()
        tic = time.perf_counter()
        outputs = model.predict_batch(inputs=inputs, horizon=int(manifest['horizon']),
                                      resolution_minutes=cadences.pop(), context_steps=args.context,
                                      batch_size=args.batch_size)
        synchronize_cuda()
        elapsed = time.perf_counter() - tic
        previous_crossings = len(crossing_events)
        forecasts.extend(checked_output(outputs, inputs, rows, future_stamps, args.model, args.context, args.k, args.task_id,
                                        quantile_policy=args.quantile_policy, crossing_events=crossing_events))
        for event in crossing_events[previous_crossings:]:
            print(f"  QUANTILE CROSSING: {event['case_id']}, "
                  f"steps={event['n_crossing_steps']}, "
                  f"max_violation={event['max_pairwise_crossing']:.6g}, "
                  f"mean_correction={event['mean_abs_quantile_correction']:.6g}", flush=True)
        batch_times.append(elapsed)
        print(f'  batch {len(batch_times)} | cases={len(forecasts)}/{len(df)} | inference={elapsed:.3f}s', flush=True)
    if len(forecasts) != len(df):
        raise RuntimeError('Predictions incomplete')
    dest.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(forecasts).to_parquet(pred_path, index=False)
    run = dict(schema='pertsfm_preofficial_calibration_predictions_v1',
               role='PREOFFICIAL_HISTORICAL_CALIBRATION', task_id=args.task_id,
               dataset_config=manifest['dataset_config'], model=args.model,
               context_steps=args.context, k=args.k, cohort='FULL' if not args.max_entities else 'SMOKE',
               n_cases=len(forecasts), n_entities=int(df.entity_id.nunique()),
               input_sha256=manifest['input_sha256'],
               input_manifest_sha256=sha256(base/'manifest.json'),
               predictions_sha256=sha256(pred_path), model_load_seconds=load_sec,
               inference_seconds=float(sum(batch_times)),
               quantile_policy=args.quantile_policy,
               n_cases_with_quantile_crossing=len(crossing_events),
               n_crossing_horizon_steps=sum(e['n_crossing_steps'] for e in crossing_events),
               max_pairwise_quantile_crossing=max((e['max_pairwise_crossing'] for e in crossing_events), default=0.0),
               mean_abs_quantile_correction=float(np.mean([x['mean_abs_quantile_correction'] for x in forecasts])),
               quantile_crossing_examples=crossing_events[:20],
               warning='Exploratory timing: no warm-up/repeats. No official test data used.')
    run_path.write_text(json.dumps(run, indent=2)+'\n', encoding='utf-8')
    print(f'PASS | saved={pred_path} | cases={len(forecasts)} | '          f'crossed_cases={len(crossing_events)} | quantile_policy={args.quantile_policy}', flush=True)


if __name__ == '__main__':
    main()
