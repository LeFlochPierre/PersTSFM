#!/usr/bin/env python3
"""PersTSFM Phase 3.3A — leakage-controlled Chronos-2 LoRA feasibility pilot.

Training: history strictly before lag 2; validation: lag 2; later: lag 1.
The official FEV test window is never accessed.
Run --execute first, then --score-later <saved-run-dir> separately.
Each LoRA budget is restarted from the same base model; not incremental training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase2.run_preofficial_calibration import (
    load_frozen_inputs, choose_cases, make_forecast_input, sha256,
)

INPUT_ROOT = Path("results/phase2/preofficial_calibration/fev_bench_v0_10_0")
OUTPUT_ROOT = Path("results/phase3/lora_feasibility")
TASK_ID = "fev_0_10_0_044"
QUANTILES = (("q10", .1), ("q50", .5), ("q90", .9))


def task_pair_checks(rows: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check exact entity match and causal rolling-window overlap.

    ROLLING_HISTORY_OVERLAP_V2: Phase 2 freezes only the last max_context
    samples before EACH lag. Hence lag2 and lag1 histories generally have
    equal lengths and shift forward by `horizon` instead of being prefixes.

    Correct chronology:
      lag2 history -> lag2 target (= last h values in lag1 input) -> lag1 target.
    The common history must agree exactly on timestamps AND observed values.
    """
    older = rows.loc[rows.lag_before_first_official.astype(int) == 2].sort_values('entity_id').reset_index(drop=True)
    later = rows.loc[rows.lag_before_first_official.astype(int) == 1].sort_values('entity_id').reset_index(drop=True)
    if (len(older) == 0 or len(older) != len(later) or
        older.entity_id.astype(str).tolist() != later.entity_id.astype(str).tolist()):
        raise RuntimeError('Not an exact entity-matched lag2/lag1 historical replay')
    horizon = int(manifest['horizon'])
    if horizon < 1:
        raise RuntimeError('Invalid horizon')
    for a, b in zip(older.to_dict('records'), later.to_dict('records')):
        t_a = np.asarray(a['history_timestamps_ns'], dtype=np.int64)
        t_b = np.asarray(b['history_timestamps_ns'], dtype=np.int64)
        y_a = np.asarray(a['history'], dtype=float)
        y_b = np.asarray(b['history'], dtype=float)
        f_a = np.asarray(a['future_timestamps_ns'], dtype=np.int64)
        f_b = np.asarray(b['future_timestamps_ns'], dtype=np.int64)
        # In later saved context: a suffix of old history, followed by old
        # target (h steps). The early old context may have been truncated.
        common = len(t_b) - horizon
        if (len(t_a) == 0 or len(t_b) <= horizon or
            len(t_a) != len(y_a) or len(t_b) != len(y_b) or
            len(f_a) != horizon or len(f_b) != horizon or
            common < 1 or common > len(t_a) or
            not np.all(np.diff(t_a) > 0) or
            not np.all(np.diff(t_b) > 0) or
            not np.all(np.diff(f_a) > 0) or
            not np.all(np.diff(f_b) > 0) or
            not np.array_equal(t_a[-common:], t_b[:common]) or
            not np.array_equal(y_a[-common:], y_b[:common]) or
            not np.array_equal(f_a, t_b[common:]) or
            t_a[-1] >= f_a[0] or
            f_a[-1] > t_b[-1] or
            t_b[-1] >= f_b[0]):
            raise RuntimeError(f"Unsafe historical chronology for entity {a['entity_id']}")
    return older, later


def load_historical_input(root: Path, task_id: str, context: int, max_entities: int):
    manifest, inp = load_frozen_inputs(root / task_id)
    if manifest['task_id'] != task_id or manifest['dataset_config'] != 'M_DENSE_1H':
        raise RuntimeError('Phase 3.3A deliberately restricted to frozen M_DENSE_1H')
    if manifest['coverage'] != 'FULL':
        raise RuntimeError('Input exporter was sampled, not full cohort')
    if manifest['max_origins'] < 2 or context not in manifest['contexts_steps']:
        raise RuntimeError('K>=2 and requested context must exist in the export')
    rows = choose_cases(inp, k=2, context=context, max_entities=max_entities)
    older, later = task_pair_checks(rows, manifest)
    for r in rows.to_dict('records'):
        make_forecast_input(r, context)
    return manifest, older, later


def load_truths(root: Path, manifest: dict, lag: int, expected: pd.DataFrame) -> dict:
    """Caller controls which labels are accessed; do not call with lag 1 during training."""
    if lag not in (1, 2):
        raise ValueError('Only lag 1 or 2 supported')
    p = root / manifest['task_id'] / 'historical_calibration_labels.parquet'
    if sha256(p) != manifest['labels_sha256']:
        raise RuntimeError('Historical label file hash mismatch')
    labels = pd.read_parquet(p)
    labels = labels.loc[labels.lag_before_first_official.astype(int) == lag].copy()
    expected_cases = set(expected.case_id.astype(str))
    labels = labels.loc[labels.case_id.astype(str).isin(expected_cases)]
    if (len(labels) != len(expected) or labels.case_id.duplicated().any() or
        set(labels.case_id.astype(str)) != expected_cases):
        raise RuntimeError('Historical labels not exact matched cases')
    mapping = {}
    for row in labels.to_dict('records'):
        case = str(row['case_id'])
        ref = expected.loc[expected.case_id.astype(str) == case].iloc[0]
        if (str(row['entity_id']) != str(ref['entity_id']) or
            int(row['horizon']) != int(ref['horizon']) or
            not np.array_equal(np.asarray(row['target_timestamps_ns'], np.int64),
                               np.asarray(ref['future_timestamps_ns'], np.int64))):
            raise RuntimeError(f'Bad label alignment: {case}')
        target = np.asarray(row['target_values'], dtype=float)
        if target.shape != (int(manifest['horizon']),) or not np.isfinite(target).all():
            raise RuntimeError(f'Nonfinite historical target for {case}')
        mapping[case] = target
    return mapping


def build_forecast_inputs(df: pd.DataFrame, context: int):
    from src.models.adapters import ForecastInput
    prepared = []
    cadence_set = set()
    for r in df.to_dict('records'):
        hist, ts, _future, cadence = make_forecast_input(r, context)
        stamps = pd.DatetimeIndex(pd.to_datetime(ts, unit='ns', utc=True)).tz_localize(None)
        prepared.append(ForecastInput(case_id=str(r['case_id']), timestamps=stamps, values=hist))
        cadence_set.add(cadence)
    if len(cadence_set) != 1:
        raise RuntimeError('Mixed time-series cadences')
    return prepared, cadence_set.pop()


def training_series(older: pd.DataFrame, train_history_steps: int, context: int, horizon: int):
    """Only prefixes that end strictly before lag 2 targets."""
    if train_history_steps < context + 2*horizon:
        raise ValueError('Not enough train history for independent fitting windows')
    out = []
    for r in older.to_dict('records'):
        values = np.asarray(r['history'], dtype=np.float32)
        if len(values) < train_history_steps:
            raise RuntimeError(
                f"Requested {train_history_steps} training observations but only "
                f"{len(values)} saved in frozen lag2 history for {r['entity_id']}. "
                "Phase 2 exporter caps each history at max_context; choose a "
                "smaller --train-history-steps or re-export with a larger cap."
            )
        series = values[-train_history_steps:].copy()
        if (len(series) < context + 2*horizon or
            not np.isfinite(series).all()):
            raise RuntimeError(f"Insufficient finite training prefix for {r['entity_id']}")
        out.append(series)
    return out


def forecast(adapter, inputs: list, table: pd.DataFrame, *, context: int, horizon: int,
             cadence: int, batch_size: int, label: str, torch):
    if len(inputs) != len(table):
        raise ValueError('Forecast case count mismatch')
    output = []
    start = time.perf_counter()
    for i in range(0, len(inputs), batch_size):
        part = inputs[i:i+batch_size]
        raw = adapter.predict_batch(part, horizon=horizon, resolution_minutes=cadence,
                                    context_steps=context, batch_size=batch_size)
        mapping = {str(x.case_id): x for x in raw}
        if len(mapping) != len(part) or set(mapping) != {str(x.case_id) for x in part}:
            raise RuntimeError('Bad forecast ID set')
        for input_case, r in zip(part, table.iloc[i:i+batch_size].to_dict('records')):
            model_out = mapping[str(input_case.case_id)]
            pred = dict(model=label, case_id=str(input_case.case_id), entity_id=str(r['entity_id']),
                        lag=int(r['lag_before_first_official']),
                        future_timestamps_ns=np.asarray(r['future_timestamps_ns'], dtype=np.int64).tolist())
            for name, raw_value in [('prediction', model_out.prediction),
                                    ('q10', model_out.q10), ('q50', model_out.q50), ('q90', model_out.q90)]:
                arr = np.asarray(raw_value, dtype=float)
                if arr.shape != (horizon,) or not np.isfinite(arr).all():
                    raise RuntimeError(f'Bad {name} output: {input_case.case_id}')
                pred[name] = arr.tolist()
            qmat = np.stack([pred['q10'], pred['q50'], pred['q90']], axis=-1)
            pred['crossing_elements'] = int(np.any(np.diff(qmat, axis=1) < -1e-6, axis=1).sum())
            # Match Phase 2.13 rearrange policy; leave mean/point forecast untouched.
            sorted_q = np.sort(qmat, axis=-1)
            pred['q10'], pred['q50'], pred['q90'] = [sorted_q[:,j].tolist() for j in range(3)]
            output.append(pred)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return output, float(time.perf_counter() - start)


def score_rows(pred: list[dict], truth: dict, *, expected: pd.DataFrame) -> pd.DataFrame:
    """Only score the requested lag, with exact timestamps and IDs."""
    by_case = {str(r['case_id']): r for r in pred}
    expected_cases = set(expected.case_id.astype(str))
    if len(by_case) != len(pred) or set(by_case) != expected_cases or set(truth) != expected_cases:
        raise ValueError('Prediction/truth/expected case sets not identical')
    rows = []
    for r in expected.to_dict('records'):
        case = str(r['case_id'])
        x = by_case[case]
        y = np.asarray(truth[case], dtype=np.float64)
        if (str(x['entity_id']) != str(r['entity_id']) or
            int(x['lag']) != int(r['lag_before_first_official']) or
            not np.array_equal(np.asarray(x['future_timestamps_ns'], dtype=np.int64),
                               np.asarray(r['future_timestamps_ns'], dtype=np.int64))):
            raise ValueError(f'Metadata mismatch: {case}')
        p = np.asarray(x['prediction'], dtype=np.float64)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f'Invalid point forecasts: {case}')
        pinballs = []
        for key, q in QUANTILES:
            z = np.asarray(x[key], dtype=np.float64)
            if z.shape != y.shape or not np.isfinite(z).all():
                raise ValueError(f'Invalid quantile: {case} {key}')
            error = y-z
            pinballs.append(np.maximum(q*error, (q-1)*error).mean())
        rows.append(dict(model=x['model'], case_id=case, entity_id=str(r['entity_id']),
                         mae=float(np.abs(y-p).mean()),
                         mean_pinball_3=float(np.mean(pinballs)),
                         crossing_elements=int(x['crossing_elements'])))
    return pd.DataFrame(rows)


def best_budget(validation: pd.DataFrame, metric: str = 'mae') -> str:
    if metric not in ('mae', 'mean_pinball_3'):
        raise ValueError(metric)
    scores = validation.groupby('model')[metric].mean()
    # For exact ties, prefer zero adaptation, then fewer steps.
    def key(name):
        return (float(scores[name]), 0 if name == 'zero_shot' else int(name.removeprefix('lora_')),
                name)
    return min(scores.index, key=key)


def save_jsonl(path: Path, rows: list[dict]):
    with path.open('x', encoding='utf-8') as stream:
        for x in rows:
            stream.write(json.dumps(x, allow_nan=False) + '\n')


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id', default=TASK_ID)
    p.add_argument('--context', type=int, default=768)
    p.add_argument('--max-entities', type=int, default=4,
                   help='4=smoke subset by default; 0=full cohort (scientific pilot)')
    p.add_argument('--train-history-steps', type=int, default=3072)
    p.add_argument('--budgets', type=int, nargs='+', default=[20,100,500])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--train-batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    p.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    p.add_argument('--execute', action='store_true', help='Actually load and fine-tune Chronos-2')
    p.add_argument('--score-later', type=Path, help='Score saved lag1 forecasts after selection locked')
    return p.parse_args()


def preflight(args):
    if args.context < 2 or args.max_entities < 0 or args.batch_size < 1 or args.train_batch_size < 1:
        raise ValueError('Invalid context/entity/batch arguments')
    if not args.budgets or any(n < 1 for n in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError('Budgets must be distinct positive step counts')
    if args.learning_rate <= 0:
        raise ValueError('Learning rate must be positive')
    manifest, older, later = load_historical_input(args.input_root, args.task_id,
                                                     args.context, args.max_entities)
    train = training_series(older, args.train_history_steps, args.context, int(manifest['horizon']))
    print('='*101)
    print('PERTSFM PHASE 3.3A | CHRONOS-2 LORA ADAPTATION | PREOFFICIAL HISTORICAL ONLY')
    print('='*101)
    print(f"{manifest['dataset_config']} | task={manifest['task_id']} | context={args.context} | "
          f"horizon={manifest['horizon']} | entities={len(older)} | budgets={sorted(args.budgets)}")
    print(f'TRAIN: lag2 input histories (capped at {args.train_history_steps} steps)')
    print('VALIDATE: lag2 targets | EVALUATE LATER: lag1 targets only after selection locked')
    print(f'COHORT: {"FULL" if not args.max_entities else "SMOKE/SUBSET"} | train lengths '
          f'{min(map(len,train))}..{max(map(len,train))}')
    print(f"Historical input SHA256: {manifest['input_sha256']}")
    print('NO OFFICIAL FEV TEST ACCESSED. NO TRAINING unless --execute supplied.')
    return manifest, older, later, train


def run(args):
    manifest, older, later, train = preflight(args)
    if not args.execute:
        print('PASS: DRY RUN; no model load, GPU work, or labels read.')
        return

    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers.utils import is_peft_available
    from src.models.adapters import Chronos2Model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refuse unintentional CPU training')
    if not is_peft_available():
        raise RuntimeError('PEFT unavailable to Transformers: risk of silent full fine-tuning')
    device = torch.cuda.get_device_name(0)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = args.output_root / f'{stamp}_chronos2_lora_{args.task_id}_e{len(older)}'
    dest.mkdir(parents=True, exist_ok=False)
    meta = dict(schema='pertsfm_phase3_3a_v1', task_id=args.task_id,
                dataset_config=manifest['dataset_config'], context=args.context,
                horizon=int(manifest['horizon']), n_entities=len(older),
                cohort='FULL' if args.max_entities == 0 else 'SMOKE_SUBSET',
                model_id='amazon/chronos-2', seed=args.seed, budgets=sorted(args.budgets),
                train_history_steps=args.train_history_steps, train_batch_size=args.train_batch_size,
                inference_batch_size=args.batch_size, learning_rate=args.learning_rate,
                input_sha256=manifest['input_sha256'], labels_sha256=manifest['labels_sha256'],
                gpu=device, torch_version=torch.__version__, transformers_version=transformers.__version__,
                peft_version=peft.__version__,
                disclaimer='Independent LoRA restarts; not incremental continuation; retrospective historical only')
    (dest/'run_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    # Load lag-2 labels ONLY. Lag-1 truth is never loaded in run(), even for smoke.
    validation_truth = load_truths(args.input_root, manifest, 2, older)
    older_inputs, cad1 = build_forecast_inputs(older, args.context)
    later_inputs, cad2 = build_forecast_inputs(later, args.context)
    if cad1 != cad2:
        raise RuntimeError('Historical cadence changed between lags')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    load_start = time.perf_counter()
    adapter = Chronos2Model()
    torch.cuda.synchronize()
    load_sec = time.perf_counter()-load_start
    pretrained = adapter.pipeline
    print(f'MODEL LOADED | {load_sec:.2f}s | {device}', flush=True)
    all_pred, step_data = [], []
    for budget in [0] + sorted(args.budgets):
        name = 'zero_shot' if budget == 0 else f'lora_{budget}'
        if budget > 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            transformers.set_seed(args.seed)
            torch.cuda.reset_peak_memory_stats(0)
            t0 = time.perf_counter()
            adapter.pipeline = pretrained
            ft = pretrained.fit(
                inputs=train, prediction_length=int(manifest['horizon']),
                validation_inputs=None, finetune_mode='lora',
                context_length=args.context, learning_rate=args.learning_rate,
                num_steps=budget, batch_size=args.train_batch_size,
                output_dir=dest/'adapters'/name,
                remove_printer_callback=True, disable_data_parallel=True,
            )
            torch.cuda.synchronize()
            train_seconds = time.perf_counter()-t0
            if not isinstance(ft.model, PeftModel):
                raise RuntimeError('LoRA was requested but returned model is not PEFT; refuse full FT fallback')
            adapter.pipeline = ft
            train_peak = int(torch.cuda.max_memory_allocated(0))
        else:
            train_seconds, train_peak = 0.0, 0
            adapter.pipeline = pretrained
        pred_valid, val_seconds = forecast(adapter, older_inputs, older,
            context=args.context, horizon=int(manifest['horizon']), cadence=cad1,
            batch_size=args.batch_size, label=name, torch=torch)
        pred_later, later_seconds = forecast(adapter, later_inputs, later,
            context=args.context, horizon=int(manifest['horizon']), cadence=cad2,
            batch_size=args.batch_size, label=name, torch=torch)
        scores = score_rows(pred_valid, validation_truth, expected=older)
        all_pred.extend(pred_valid + pred_later)
        step_data.append(dict(model=name, steps=budget, train_wall_seconds=train_seconds,
                              train_peak_allocated_mib=train_peak/1048576,
                              val_forecast_seconds=val_seconds,
                              later_forecast_seconds=later_seconds,
                              validation_mae=float(scores.mae.mean()),
                              validation_pinball_3=float(scores.mean_pinball_3.mean()),
                              quantile_crossing_elements=sum(x['crossing_elements'] for x in pred_valid)))
        print(f'{name:12s} | VAL MAE={scores.mae.mean():.5f} | pinball3={scores.mean_pinball_3.mean():.5f} '
              f'| train={train_seconds:.1f}s | peak allocated={train_peak/1048576:.1f}MiB',flush=True)
        if budget > 0:
            adapter.pipeline = pretrained
            del ft
            gc.collect()
            torch.cuda.empty_cache()
    table = pd.DataFrame(step_data).sort_values('steps').reset_index(drop=True)
    # Only lag-2 validation scores available for budget selection.
    score_dataframe = pd.concat([
        score_rows([p for p in all_pred if p['lag'] == 2 and p['model'] == name],
                   validation_truth, expected=older)
        for name in table.model
    ],ignore_index=True)
    selection = best_budget(score_dataframe, metric='mae')
    table.to_csv(dest/'validation_summary.csv',index=False)
    score_dataframe.to_csv(dest/'validation_case_metrics.csv',index=False)
    save_jsonl(dest/'predictions.jsonl',all_pred)
    locked = dict(schema='pertsfm_phase3_3a_selection_v1',
                  selected_model=selection, selection_metric='lag2_macro_case_mae',
                  available_models=table.model.tolist(),
                  predictions_sha256=sha256(dest/'predictions.jsonl'),
                  validation_summary_sha256=sha256(dest/'validation_summary.csv'),
                  validation_case_metrics_sha256=sha256(dest/'validation_case_metrics.csv'),
                  selected_before_lag1_truth_access=True)
    (dest/'selection.json').write_text(json.dumps(locked,indent=2)+'\n',encoding='utf-8')
    print('\nVALIDATION ONLY | frozen input vs LoRA independent restarts')
    print(table.to_string(index=False,float_format=lambda x: f'{x:.4f}'))
    print(f'LOCKED SELECTION ON LAG2: {selection}')
    print(f'PASS: {dest}')
    print('Next: run --score-later with the path above to unlock lag1 historical scores.')


def score_later(args):
    run = args.score_later
    manifest_run = json.loads((run/'run_manifest.json').read_text())
    selected = json.loads((run/'selection.json').read_text())
    if manifest_run['schema'] != 'pertsfm_phase3_3a_v1' or selected['schema'] != 'pertsfm_phase3_3a_selection_v1':
        raise ValueError('Incompatible saved Phase3.3A run')
    if not selected['selected_before_lag1_truth_access']:
        raise RuntimeError('Selection not locked prior to later truth')
    for file, key in [('predictions.jsonl','predictions_sha256'),
                      ('validation_summary.csv','validation_summary_sha256'),
                      ('validation_case_metrics.csv','validation_case_metrics_sha256')]:
        if sha256(run/file) != selected[key]:
            raise RuntimeError('Saved selection/prediction files were modified')
    output_file = run/'later_summary.csv'
    if output_file.exists():
        raise FileExistsError('Later scores already exist; refusing overwrite')
    max_entities = 0 if manifest_run['cohort'] == 'FULL' else int(manifest_run['n_entities'])
    manifest, older, later = load_historical_input(args.input_root, manifest_run['task_id'],
                                                   int(manifest_run['context']), max_entities)
    if (manifest['input_sha256'] != manifest_run['input_sha256'] or
        manifest['labels_sha256'] != manifest_run['labels_sha256'] or len(later) != manifest_run['n_entities']):
        raise RuntimeError('Frozen input/labels changed since selection')
    truth_later = load_truths(args.input_root, manifest, 1, later)
    preds = [json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines() if line.strip()]
    names = list(manifest_run['budgets'])
    names = ['zero_shot'] + [f'lora_{n}' for n in names]
    if set(names) != set(selected['available_models']):
        raise RuntimeError('Saved model grid differs from locked selection')
    tables = []
    for name in names:
        part = [p for p in preds if p['lag'] == 1 and p['model'] == name]
        frame = score_rows(part,truth_later,expected=later)
        tables.append(frame)
    all_scores = pd.concat(tables,ignore_index=True)
    summary = all_scores.groupby('model',as_index=False).agg(
        later_mae=('mae','mean'),later_pinball_3=('mean_pinball_3','mean'),
        crossing_elements=('crossing_elements','sum'))
    summary['steps'] = summary.model.map({'zero_shot':0, **{f'lora_{n}':n for n in manifest_run['budgets']}})
    summary = summary.sort_values('steps')
    base = float(summary.loc[summary.model=='zero_shot','later_mae'].iloc[0])
    summary['improvement_vs_zero_shot_pct'] = 100 * (base - summary.later_mae) / base
    summary.to_csv(output_file,index=False)
    all_scores.to_csv(run/'later_case_metrics.csv',index=False)
    winner = selected['selected_model']
    chosen = summary.loc[summary.model==winner].iloc[0]
    oracle = summary.loc[summary.later_mae.idxmin()]
    print('='*90)
    print('PHASE 3.3A | PREOFFICIAL HISTORICAL LAG1 EVALUATION | CHOICE FROZEN ON LAG2')
    print('='*90)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print(f'LOCKED CHOICE: {winner} | LAG1 MAE={chosen.later_mae:.5f} | '
          f'improvement vs zero-shot={chosen.improvement_vs_zero_shot_pct:+.2f}%')
    print(f'HINDSIGHT ORACLE (diagnostic only): {oracle.model} | LAG1 MAE={oracle.later_mae:.5f}')
    print('Independent LoRA runs; later outcomes MUST NOT be used to adjust budget/hyperparameters.')
    print(f'PASS: {run}')


def main():
    args = get_args()
    if args.score_later:
        if args.execute:
            raise ValueError('Do not combine --score-later and --execute')
        score_later(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
