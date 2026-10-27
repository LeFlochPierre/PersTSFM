#!/usr/bin/env python3
"""Phase 2.9: export disjoint PRE-OFFICIAL calibration cases from fev-bench histories.

Safety boundary: the ONLY dataset access is get_window(0).get_input_data(),
then ONLY the `past` item. Never calls get_ground_truth(), loads later official
windows, or reads any official evaluation labels. Calibration labels are slices
of the already-observed historical target values before window zero.

Expected fev==0.10.0, numpy, pandas, pyarrow. Benchmark registry from Phase 2.6.

Examples:
 python -m scripts.phase2.export_preofficial_calibration --configs us_consumption_1M
 python -m scripts.phase2.export_preofficial_calibration --configs M_DENSE_1H hospital_admissions_1D
 python -m scripts.phase2.export_preofficial_calibration --configs hierarchical_sales_1D --entity-limit 64
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REGISTRY = Path('results/phase2/benchmark_registry/fev_bench_v0_10_0')
HISTORY_AUDIT = Path('results/phase2/benchmark_feasibility/fev_bench_v0_10_0')
OUTPUT = Path('results/phase2/preofficial_calibration/fev_bench_v0_10_0')
DEFAULT_CONTEXTS = (8, 16, 32, 64, 96, 192, 384, 768, 1536, 3072)
REQUIRED_FEV = '0.10.0'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamps_ns(dates, desc: str) -> np.ndarray:
    """Normalize all pandas timestamp resolutions to UTC *nanoseconds*."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).as_unit('ns')
    if len(idx) == 0 or idx.hasnans:
        raise ValueError(f'{desc}: missing/empty timestamps')
    ns = np.asarray(idx.asi8, dtype=np.int64)
    if len(ns) > 1 and not np.all(np.diff(ns) > 0):
        raise ValueError(f'{desc}: duplicate or nonmonotonic timestamps')
    return ns


def split_one_history(
    entity: str,
    history: np.ndarray,
    timestamps: np.ndarray,
    horizon: int,
    max_origins: int,
    contexts: tuple[int, ...],
    task_id: str,
) -> tuple[list[dict], list[dict], dict]:
    """Build historical probes strictly before official cutoff; no test values needed.

    Probe `lag` uses X = series[:L-lag*h] and y = series[L-lag*h:L-(lag-1)*h].
    The label for lag=1 ends at the last observation BEFORE official test.
    Cases are ordered oldest-to-newest (highest lag first).
    """
    if not entity or horizon <= 0 or max_origins < 0:
        raise ValueError('Entity, horizon, and max_origins must be valid')
    if not contexts or min(contexts) <= 0 or tuple(sorted(set(contexts))) != contexts:
        raise ValueError('Contexts must be positive, sorted and unique')
    y = np.asarray(history, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.int64)
    if y.ndim != 1 or ts.ndim != 1 or len(y) != len(ts) or not len(y):
        raise ValueError(f'{entity}: nonempty 1-D matching target/timestamps required')
    if np.isinf(y).any():
        raise ValueError(f'{entity}: infinite target values cannot be exported safely')
    if len(ts) > 1 and not bool(np.all(np.diff(ts) > 0)):
        raise ValueError(f'{entity}: timestamps not strictly increasing')
    min_context, max_context = contexts[0], contexts[-1]
    L = len(y)
    inputs, truths = [], []
    allowable_by_lag = {}
    for lag in range(max_origins, 0, -1):
        cutoff = L - lag * horizon
        if cutoff < min_context:
            allowable_by_lag[lag] = []
            continue
        label_end = cutoff + horizon
        assert label_end <= L
        offered = [c for c in contexts if c <= cutoff]
        allowable_by_lag[lag] = offered
        begin = max(0, cutoff - max_context)
        hist = y[begin:cutoff]
        hist_ts = ts[begin:cutoff]
        targets = y[cutoff:label_end]
        tgt_ts = ts[cutoff:label_end]
        if len(targets) != horizon or len(hist) < min_context:
            raise AssertionError('Incorrect calibration slice lengths')
        if hist_ts[-1] >= tgt_ts[0] or tgt_ts[-1] > ts[-1]:
            raise AssertionError('Calibration predictions are not prior to official cutoff')
        case_id = f'{task_id}__{entity}__lag{lag:02d}'
        common = dict(task_id=task_id, entity_id=entity, case_id=case_id,
                      lag_before_first_official=lag, horizon=horizon,
                      future_start_ns=int(tgt_ts[0]),
                      future_end_ns=int(tgt_ts[-1]))
        inputs.append({**common, 'available_history_steps': cutoff,
                       'observed_steps_saved': len(hist),
                       'history_missing_saved': int(np.isnan(hist).sum()),
                       'feasible_contexts': offered,
                       'history': hist.tolist(),
                       'history_timestamps_ns': hist_ts.tolist(),
                       'future_timestamps_ns': tgt_ts.tolist()})
        truths.append({**common, 'target_values': targets.tolist(),
                       'target_missing': int(np.isnan(targets).sum()),
                       'target_timestamps_ns': tgt_ts.tolist()})
    # Independent case targets must NEVER overlap. They may be immediately adjacent.
    timeline = sorted(inputs, key=lambda r: r['future_start_ns'])
    for prev, nxt in zip(timeline, timeline[1:]):
        if prev['future_end_ns'] >= nxt['future_start_ns']:
            raise AssertionError(f'{entity}: calibration targets overlap')
    # K is supportable on this entity only if all K most-recent lags are feasible.
    k_supported = max((k for k in range(max_origins + 1)
                       if all(allowable_by_lag[j] for j in range(1, k + 1))), default=0)
    inventory = dict(task_id=task_id, entity_id=entity, history_steps=L,
                     missing_historical_values=int(np.isnan(y).sum()),
                     first_observed_ns=int(ts[0]),
                     last_preofficial_observed_ns=int(ts[-1]),
                     max_calibration_origins=k_supported,
                     official_available_contexts=[c for c in contexts if c <= L],
                     # These contexts are feasible for ALL origins in budget K.
                     common_contexts_by_budget={
                         str(k): [c for c in contexts if c <= L - k*horizon]
                         for k in range(max_origins+1)
                     })
    return inputs, truths, inventory


def validate_export(inputs: list[dict], truths: list[dict], inventory: list[dict]):
    if len(inputs) != len(truths):
        raise ValueError('Mismatched inputs/labels')
    x_by = {r['case_id']: r for r in inputs}
    y_by = {r['case_id']: r for r in truths}
    if len(x_by) != len(inputs) or len(y_by) != len(truths) or x_by.keys() != y_by.keys():
        raise ValueError('Missing or duplicated calibration case IDs')
    ids = [r['entity_id'] for r in inventory]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicated entity IDs')
    for case_id, r in x_by.items():
        t = y_by[case_id]
        if r['future_start_ns'] != t['future_start_ns'] or r['future_end_ns'] != t['future_end_ns']:
            raise ValueError('Truth/input timestamp mismatch')
        if r['history_timestamps_ns'][-1] >= t['target_timestamps_ns'][0]:
            raise ValueError('History/label leakage')
        if len(t['target_values']) != r['horizon']:
            raise ValueError('Incorrect label horizon')
        if list(r['future_timestamps_ns']) != list(t['target_timestamps_ns']):
            raise ValueError('Future timestamps do not match labels')
    by_entity = defaultdict(list)
    for r in inputs:
        by_entity[r['entity_id']].append(r)
    for r in inventory:
        cases = by_entity[r['entity_id']]
        for c in cases:
            if c['future_end_ns'] > r['last_preofficial_observed_ns']:
                raise ValueError('Calibration target exceeds preofficial history cutoff')
        ordered = sorted(cases, key=lambda c: c['future_start_ns'])
        for older, newer in zip(ordered, ordered[1:]):
            if older['future_end_ns'] >= newer['future_start_ns']:
                raise ValueError('Overlapping calibration label horizons')


def load_registry(registry: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((registry / 'benchmark_manifest.json').read_text(encoding='utf-8'))
    source_sha = sha256_bytes((registry / 'official_tasks.yaml').read_bytes())
    if source_sha != manifest['source_sha256'] or manifest['task_count'] != 100:
        raise RuntimeError('Official fev-bench v0.10.0 registry has changed')
    with (registry / 'task_catalog.csv').open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 100:
        raise RuntimeError('Expected 100 official task specifications')
    return rows, manifest


def selected_registry_rows(rows: list[dict], configs: list[str], task_ids: list[str]) -> list[dict]:
    if not configs and not task_ids:
        raise ValueError('Pass --configs or --task-ids explicitly')
    if unknown := (set(configs) - {r['dataset_config'] for r in rows}):
        raise ValueError(f'Unknown config(s): {sorted(unknown)}')
    if unknown := (set(task_ids) - {r['task_id'] for r in rows}):
        raise ValueError(f'Unknown task IDs: {sorted(unknown)}')
    selection = [r for r in rows if r['dataset_config'] in configs or r['task_id'] in task_ids]
    bad = [r['task_name'] for r in selection if str(r['target_only_candidate']).lower() != 'true']
    if bad:
        raise ValueError(f'Target-only exporter cannot reproduce covariate/multivariate tasks: {bad}')
    return selection


def export_task(task_row: dict, contexts: tuple[int, ...], max_origins: int,
                entity_limit: int, output_root: Path,
                registry_manifest: dict, history_audit: Path) -> dict:
    import fev  # only in data environment
    task_id = task_row['task_id']
    out = output_root / task_id
    if out.exists():
        raise FileExistsError(f'{out} already exists; refusing overwrite or mixing revisions')
    spec = json.loads(task_row['official_task_json'])
    task = fev.Task(**spec)
    if len(task.target_columns) != 1:
        raise RuntimeError(f'{task_id}: multivariate task not supported')
    target = task.target_columns[0]
    window = task.get_window(0)
    past, _future_known = window.get_input_data()  # future KNOWN features never inspected
    count_total = len(past)
    if count_total == 0:
        raise RuntimeError('No histories in official first input window')
    n_entities = min(entity_limit, count_total) if entity_limit else count_total
    inputs, truths, inventory = [], [], []
    seen = set()
    for index in range(n_entities):
        record = past[index]
        ident = str(record[task.id_column])
        if ident in seen:
            raise ValueError(f'Duplicate entity {ident}')
        seen.add(ident)
        observations = np.asarray(record[target], dtype=np.float64)
        stamps = timestamps_ns(record[task.timestamp_column], f'{task_id}:{ident}')
        x, y, meta = split_one_history(
            ident, observations, stamps, int(task.horizon), max_origins, contexts, task_id
        )
        inputs.extend(x)
        truths.extend(y)
        inventory.append(meta)
    validate_export(inputs, truths, inventory)
    inventory.sort(key=lambda r: r['entity_id'])
    inputs.sort(key=lambda r: (r['entity_id'], -r['lag_before_first_official']))
    truths.sort(key=lambda r: (r['entity_id'], -r['lag_before_first_official']))

    # Cross-check earlier Phase 2.7 census where available; warn on sampled audits.
    prior_profile = history_audit / 'task_profiles' / f'{task_id}.json'
    if prior_profile.exists():
        old = json.loads(prior_profile.read_text(encoding='utf-8'))
        if (old.get('coverage') == 'FULL' and not entity_limit and
            (int(old['entities_official_total']) != count_total or
             int(old['history_min']) != min(r['history_steps'] for r in inventory))):
            raise RuntimeError('Mismatch against prior FULL history audit: possible dataset revision drift')
    hist_counts = [r['history_steps'] for r in inventory]
    n_k = {str(k): sum(r['max_calibration_origins'] >= k for r in inventory)
           for k in range(max_origins+1)}
    by_lag = {str(lag): sum(r['lag_before_first_official'] == lag for r in inputs)
              for lag in range(1, max_origins+1)}
    manifest = {
        'schema': 'pertsfm_preofficial_calibration_v1',
        'task_id': task_id,
        'task_name': task_row['task_name'],
        'dataset_config': task_row['dataset_config'],
        'official_task_json_sha256': sha256_bytes(task_row['official_task_json'].encode('utf-8')),
        'official_registry_source_sha256': registry_manifest['source_sha256'],
        'fev_version': REQUIRED_FEV,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'HISTORICAL_CALIBRATION_ONLY_NOT_OFFICIAL_EVALUATION',
        'only_window_accessed': 0,
        'only_api_used': 'get_window(0).get_input_data() (past portion only)',
        'official_ground_truth_accessed': False,
        'official_test_target_values_included': False,
        'calibration_labels_source': 'historical observations in first official input window',
        'horizon': int(task.horizon),
        'target_column': target,
        'id_column': task.id_column,
        'timestamp_column': task.timestamp_column,
        'timestamp_unit': 'ns',
        'contexts_steps': list(contexts),
        'max_origins': max_origins,
        'entity_count_total': count_total,
        'entity_count_exported': n_entities,
        'coverage': 'FULL' if n_entities == count_total else 'SAMPLED_NOT_FULL_CENSUS',
        'history_steps_min': min(hist_counts),
        'history_steps_max': max(hist_counts),
        'missing_historical_values': sum(r['missing_historical_values'] for r in inventory),
        'n_cases': len(inputs),
        'cases_per_lag': by_lag,
        'eligible_entities_per_budget': n_k,
        'family_id_exact': task_row['family_id_exact'],
        'source_family_review_required': True,
        'dataset_revision': 'NOT_PINNED; pin before publication',
        'limitations': [
            'Calibration cases come only from historical observations before the first official test window.',
            'K=0 means no validation forecasts and is represented in task inventory only.',
            'Each series is independently time-ordered; globally synchronized cutoffs are not claimed.',
            'Missing data are preserved; downstream adapter and metric validity require separate QC.',
            'No official SQL/quantile metric equivalence is claimed.',
            'Cross-frequency dataset families require manual source-lineage audit before train/test splitting.',
            'This export does not score models, measure GPU costs, or access any official test labels.',
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{task_id}.', dir=out.parent))
    try:
        input_path = staging / 'calibration_inputs.parquet'
        truth_path = staging / 'historical_calibration_labels.parquet'
        inventory_path = staging / 'entity_inventory.json'
        # Explicit empty schemas permit K=0 / short-history inventory-only exports.
        input_columns = [
            'task_id', 'entity_id', 'case_id', 'lag_before_first_official',
            'horizon', 'future_start_ns', 'future_end_ns',
            'available_history_steps', 'observed_steps_saved',
            'history_missing_saved', 'feasible_contexts', 'history',
            'history_timestamps_ns', 'future_timestamps_ns',
        ]
        truth_columns = [
            'task_id', 'entity_id', 'case_id', 'lag_before_first_official',
            'horizon', 'future_start_ns', 'future_end_ns',
            'target_values', 'target_missing', 'target_timestamps_ns',
        ]
        pd.DataFrame(inputs, columns=input_columns).to_parquet(input_path, index=False)
        pd.DataFrame(truths, columns=truth_columns).to_parquet(truth_path, index=False)
        inventory_path.write_text(json.dumps(inventory, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        manifest['input_sha256'] = sha256_bytes(input_path.read_bytes())
        manifest['labels_sha256'] = sha256_bytes(truth_path.read_bytes())
        manifest['inventory_sha256'] = sha256_bytes(inventory_path.read_bytes())
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        os.rename(staging, out)  # atomic placement within filesystem; never overwrites
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f'  {task_row["dataset_config"]}: {n_entities}/{count_total} entities, '
          f'{len(inputs)} cases, history {min(hist_counts)}..{max(hist_counts)}, '
          f'missing={manifest["missing_historical_values"]}', flush=True)
    print('  eligible entities by K:', n_k, '| cases per lag:', by_lag, flush=True)
    print(f'  PASS: strictly pre-official historical calibration only | {out}', flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', nargs='*', default=[])
    p.add_argument('--task-ids', nargs='*', default=[])
    p.add_argument('--registry', type=Path, default=REGISTRY)
    p.add_argument('--history-audit', type=Path, default=HISTORY_AUDIT)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--max-origins', type=int, default=3)
    p.add_argument('--contexts', type=int, nargs='+', default=list(DEFAULT_CONTEXTS))
    p.add_argument('--entity-limit', type=int, default=0,
                   help='0=complete cohort, N=first N entities (sampled, flagged)')
    args = p.parse_args()
    if args.entity_limit < 0 or not 0 <= args.max_origins <= 10:
        p.error('entity limit must be >=0, max origins must be 0..10')
    contexts = tuple(sorted(set(args.contexts)))
    if not contexts or any(x < 1 for x in contexts):
        p.error('Must supply positive context lengths')
    current = importlib.metadata.version('fev')
    if current != REQUIRED_FEV:
        raise RuntimeError(f'Requires fev=={REQUIRED_FEV}, got {current}')
    catalog, registry_manifest = load_registry(args.registry)
    selected = selected_registry_rows(catalog, args.configs, args.task_ids)
    print('='*93)
    print('PersTSFM PHASE 2.9 — PRE-OFFICIAL HISTORICAL CALIBRATION EXPORT')
    print('='*93)
    print(f'Official registry: v0.10.0 ({registry_manifest["source_sha256"][:12]}...) | '
          f'tasks={len(selected)} | max K={args.max_origins} | contexts={list(contexts)}')
    print('Only get_window(0).get_input_data() historical inputs; NO official future labels.', flush=True)
    failures = []
    for row in selected:
        try:
            export_task(row, contexts, args.max_origins, args.entity_limit,
                        args.output, registry_manifest, args.history_audit)
        except Exception as exc:
            print(f'FAIL {row["dataset_config"]}: {type(exc).__name__}: {exc}', flush=True)
            failures.append(row['dataset_config'])
    print('='*93)
    print(f'Exported {len(selected)-len(failures)}/{len(selected)} tasks; failures={failures}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
