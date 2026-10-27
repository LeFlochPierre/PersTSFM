#!/usr/bin/env python3
"""PersTSFM Phase 2.8: offline K=0..3 pre-official calibration capacity audit.

Operates ONLY on input-history summaries produced by Phase 2.7; accesses no FEV
windows, datasets, forecasts, or benchmark labels. The oldest of K disjoint
historical calibration horizons has L-K*h observations of context. K=0 means
there are *no calibration probes*; the model/policy needs to choose based on
metadata/prior knowledge, not observed calibration losses.

The result is exact for *common-context feasibility of every inspected series*
given the historical length data, but does not establish practical adapter
support, usable metrics, model-input data quality, or official benchmark scores.
Do not use provisional family hints to assign final train/test splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('results/phase2/benchmark_feasibility/fev_bench_v0_10_0')
OUT = Path('results/phase2/calibration_budget/fev_bench_v0_10_0')


def common_feasibility(history_min: int, horizon: int, contexts: list[int], k: int) -> dict:
    if history_min < 0 or horizon <= 0 or k < 0 or not contexts or any(x <= 0 for x in contexts):
        raise ValueError('Expected nonnegative history, positive horizon/contexts, and nonnegative K')
    if contexts != sorted(set(contexts)):
        raise ValueError('Contexts must be strictly increasing without duplicates')
    available = max(0, history_min - k * horizon)
    permitted = [x for x in contexts if x <= available]
    return {
        'available_steps_at_oldest_calibration_origin': available,
        'n_common_feasible_contexts': len(permitted),
        'max_common_feasible_context': max(permitted) if permitted else None,
        'feasible_contexts': permitted,
        'can_support_k_calibration_origins': bool(permitted) if k else False,
    }


def build_matrix(profiles: list[dict], contexts: list[int], ks: list[int]) -> list[dict]:
    if not ks or sorted(set(ks)) != ks or any(k < 0 for k in ks):
        raise ValueError('K values must be nonnegative, strictly increasing and unique')
    rows = []
    ids = set()
    for p in profiles:
        tid = p['task_id']
        if tid in ids:
            raise ValueError(f'Duplicate task_id in history profiles: {tid}')
        ids.add(tid)
        lo = int(p['history_min'])
        hi = int(p['history_max'])
        if lo > hi:
            raise ValueError(f'Invalid history range in task {tid}')
        n = int(p['entities_inspected'])
        total = int(p['entities_official_total'])
        is_full = p.get('coverage') == 'FULL' and n == total
        for k in ks:
            r = common_feasibility(lo, int(p['official_horizon']), contexts, k)
            rows.append({
                'task_id': tid,
                'dataset_config': p['dataset_config'],
                'family_hint_provisional': p.get('underlying_family_hint', ''),
                'source_family_review_required': True,
                'full_cohort_verified': is_full,
                'entities_inspected': n,
                'entities_official_total': total,
                'history_min': lo,
                'history_max': hi,
                'official_horizon': int(p['official_horizon']),
                'k_calibration_origins': k,
                'no_probes_mode': k == 0,
                'available_steps_at_oldest_calibration_origin': r['available_steps_at_oldest_calibration_origin'],
                'n_common_feasible_contexts': r['n_common_feasible_contexts'],
                'max_common_feasible_context': r['max_common_feasible_context'],
                'feasible_contexts_json': json.dumps(r['feasible_contexts']),
                'can_support_k_calibration_origins': r['can_support_k_calibration_origins'],
                'feasibility_basis': 'FULL_COHORT' if is_full else 'SAMPLED_ONLY',
            })
    return rows


def save_csv(path: Path, data: list[dict]) -> None:
    if not data:
        raise ValueError(f'No rows to save at {path}')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def summarize(matrix: list[dict], ks: list[int]) -> list[dict]:
    rows = []
    for k in ks:
        group = [r for r in matrix if r['k_calibration_origins'] == k]
        full = [r for r in group if r['full_cohort_verified']]
        can = [r for r in full if r['n_common_feasible_contexts'] > 0]
        family_hints = {r['family_hint_provisional'] for r in can}
        rows.append({
            'k_calibration_origins': k,
            'audited_tasks': len(group),
            'full_cohort_tasks': len(full),
            'full_cohort_tasks_with_context': len(can),
            'provisional_family_hints_with_context': len(family_hints),
            'interpretation': 'ZERO_PROBES_PRIOR_ONLY' if k == 0 else 'HISTORICAL_PROBES_POSSIBLE',
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--ks', nargs='+', type=int, default=[0,1,2,3])
    args = ap.parse_args()
    profile_path = args.input / 'task_history_profiles.csv'
    manifest_path = args.input / 'audit_manifest.json'
    if not profile_path.exists() or not manifest_path.exists():
        ap.error(f'Missing Phase 2.7 history CSV / manifest under {args.input}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    contexts = sorted(set(int(x) for x in manifest['contexts_steps']))
    with profile_path.open(newline='', encoding='utf-8') as f:
        profiles = list(csv.DictReader(f))
    matrix = build_matrix(profiles, contexts, sorted(set(args.ks)))
    summary = summarize(matrix, sorted(set(args.ks)))
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / 'task_calibration_budget_matrix.csv', matrix)
    save_csv(args.output / 'calibration_budget_summary.csv', summary)
    payload = {
        'schema': 'pertsfm_pre_official_calibration_budget_v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'input_history_profiles_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        'source_audit_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'official_registry_sha256': manifest.get('registry_sha256'),
        'ks': sorted(set(args.ks)),
        'contexts_steps': contexts,
        'tasks_analyzed': len(profiles),
        'accesses_any_dataset_or_ground_truth': False,
        'limitations': [
            'No actual calibration examples are constructed. Only history-length arithmetic is audited.',
            'K=0 means no validation probes; availability of a model context is not a validation result.',
            'Common-context feasibility is based on the shortest inspected history, not per-entity coverage curves.',
            'Only FULL cohort rows support whole-task capacity claims.',
            'Simple horizon spacing is necessary but not sufficient for a correct timestamp and label separation implementation.',
            'Provisional family hints are not proven independent underlying sources.',
            'Official benchmark task horizons, covariates, SQL metrics, and dataset revisions need separate verification.',
        ],
    }
    (args.output / 'manifest.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print('='*89)
    print('PersTSFM PHASE 2.8 — ADAPTIVE CALIBRATION CAPACITY | OFFICIAL INPUT HISTORIES ONLY')
    print('='*89)
    print(f'Profiles: {len(profiles)} | full cohort: {sum(p["coverage"] == "FULL" for p in profiles)}')
    print('K=0 denotes ZERO calibration probes. Family counts use PROVISIONAL hints, not validated splits.')
    print(' K   Full-cohort tasks with feasible context   Provisional source family hints')
    for r in summary:
        print(f' {r["k_calibration_origins"]:<3} {r["full_cohort_tasks_with_context"]:>8}/{r["full_cohort_tasks"]:<8} '
              f'{r["provisional_family_hints_with_context"]:>13}')
    print('\nTASK CAPACITY BY K (minimum history; full-cohort or sampled as marked)')
    headers = 'dataset_config'.ljust(27) + ' ' + ' '.join(f'K{k}'.rjust(7) for k in sorted(set(args.ks))) + '   cohort'
    print(headers)
    bytask = {}
    for r in matrix:
        bytask.setdefault(r['task_id'], {})[r['k_calibration_origins']] = r
    for t in sorted(bytask):
        group = bytask[t]
        anyrow = next(iter(group.values()))
        caps = []
        for k in sorted(set(args.ks)):
            v = group[k]['max_common_feasible_context']
            caps.append(f'{v if v is not None else "NONE":>7}')
        print(f'{anyrow["dataset_config"]:<27} ' + ' '.join(caps) +
              ('   FULL' if anyrow['full_cohort_verified'] else '   SAMPLE'))
    print('\nPASS: no FEV datasets, windows, inference or future ground truth accessed.')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
