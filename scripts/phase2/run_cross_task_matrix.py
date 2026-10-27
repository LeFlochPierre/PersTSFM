#!/usr/bin/env python3
"""Phase 2.13: resume-safe cross-task historical calibration experiment scheduler.

Reads only frozen *preofficial calibration INPUTS* when planning. No FEV import,
no official labels, no scoring in planning process. Delegates forecasts to the
existing Phase 2.10 runner in the model Conda environment and historical-label
scoring to the existing dedicated scorer in the data environment.

Safeguards: per-task matched full cohorts; uniform contexts per task; strict
input/provenance hashes; write-once outputs; no silent dropping; dry-run by
DEFAULT; quantile policy compatibility checks; no official test access.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.phase2.run_preofficial_calibration import (
    INPUT_ROOT, OUTPUT_ROOT, choose_cases, load_frozen_inputs,
    make_forecast_input, sha256,
)

KNOWN_MODELS = {
    'chronos2': 'PERTSFM_ENV',
    'timesfm3': 'PERTSFM_ENV',
    'tirex2': 'PERTSFM_ENV',
    'toto2_22m': 'PERTSFM_TOTO_ENV',
    'toto2_313m': 'PERTSFM_TOTO_ENV',
    'moirai2_small': 'PERTSFM_MOIRAI_ENV',
}
DEFAULT_CONTEXTS = (96, 384, 768, 1536)
ROOT = Path('results/phase2/cross_task_matrix')


@dataclass
class Job:
    task_id: str
    dataset_config: str
    family_id_exact: str
    model: str
    conda_env: str
    context_steps: int
    k: int
    n_cases: int
    n_entities: int
    status: str
    reason: str
    output_dir: str


def safe_run_manifest(out: Path, job: Job, source_manifest: dict, quantile_policy: str) -> dict:
    """Validate existing forecasts; never trust filename alone for resume."""
    run_path = out / 'run_manifest.json'
    pred_path = out / 'predictions.parquet'
    if not run_path.is_file() or not pred_path.is_file():
        raise RuntimeError(f'Partial output, missing run manifest or predictions: {out}')
    run = json.loads(run_path.read_text(encoding='utf-8'))
    expected = {
        'schema': 'pertsfm_preofficial_calibration_predictions_v1',
        'role': 'PREOFFICIAL_HISTORICAL_CALIBRATION',
        'task_id': job.task_id,
        'model': job.model,
        'context_steps': job.context_steps,
        'k': job.k,
        'n_cases': job.n_cases,
        'n_entities': job.n_entities,
        'cohort': 'FULL',
        'input_sha256': source_manifest['input_sha256'],
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise RuntimeError(f'Existing run {out}: {key} mismatch ({run.get(key)!r} != {value!r})')
    existing_policy = run.get('quantile_policy', 'strict')
    if existing_policy != quantile_policy:
        raise RuntimeError(f'Existing run {out}: quantile policy is {existing_policy}; asked for {quantile_policy}')
    input_manifest_sha = sha256(Path(INPUT_ROOT) / job.task_id / 'manifest.json')
    if run.get('input_manifest_sha256') != input_manifest_sha:
        raise RuntimeError(f'Existing run {out}: input manifest hash mismatch')
    if run.get('predictions_sha256') != sha256(pred_path):
        raise RuntimeError(f'Existing run {out}: predictions hash mismatch')
    return run


def classify_output(out: Path, job: Job, source_manifest: dict, quantile_policy: str) -> tuple[str, str]:
    if not out.exists():
        return 'PENDING', ''
    try:
        safe_run_manifest(out, job, source_manifest, quantile_policy)
    except Exception as exc:
        return 'BLOCKED', str(exc)
    score_path = out / 'calibration_score.json'
    case_csv = out / 'calibration_case_metrics.csv'
    if score_path.exists() != case_csv.exists():
        return 'BLOCKED', 'One score output exists but its paired case CSV is missing'
    if not score_path.exists():
        return 'NEEDS_SCORE', 'Existing verified forecasts'
    try:
        s = json.loads(score_path.read_text(encoding='utf-8'))
        for key, value in {
            'schema': 'pertsfm_preofficial_calibration_score_v1',
            'role': 'HISTORICAL_CALIBRATION_ONLY',
            'task_id': job.task_id,
            'model': job.model,
            'context_steps': job.context_steps,
            'k': job.k,
            'n_cases': job.n_cases,
            'n_entities': job.n_entities,
            'cohort': 'FULL',
        }.items():
            if s.get(key) != value:
                raise ValueError(f'{key} mismatch')
    except Exception as exc:
        return 'BLOCKED', f'Corrupt/incompatible score: {exc}'
    return 'DONE', 'Existing predictions and score verified'


def discover_exports(root: Path, task_names: list[str] | None) -> list[tuple[dict, Path]]:
    manifests: list[tuple[dict, Path]] = []
    for p in sorted(root.glob('*/manifest.json')):
        m = json.loads(p.read_text(encoding='utf-8'))
        if (m.get('schema') != 'pertsfm_preofficial_calibration_v1' or
            m.get('role') != 'HISTORICAL_CALIBRATION_ONLY_NOT_OFFICIAL_EVALUATION' or
            m.get('official_ground_truth_accessed') is not False or
            m.get('coverage') != 'FULL'):
            raise RuntimeError(f'Unrecognised or non-full input export: {p}')
        if m.get('task_id') != p.parent.name:
            raise RuntimeError(f'Manifest task ID inconsistent with directory: {p}')
        manifests.append((m, p.parent))
    if not manifests:
        raise RuntimeError(f'No full historical calibration exports found under {root}')
    if task_names:
        selection = set(task_names)
        ordered = []
        for name in task_names:
            matches = [(m,p) for m,p in manifests
                       if m['task_id'] == name or m['dataset_config'] == name]
            if len(matches) != 1:
                raise ValueError(f'Expected one full-cohort export for {name!r}, found {len(matches)}')
            if matches[0][0]['task_id'] not in {m['task_id'] for m,_ in ordered}:
                ordered.extend(matches)
        return ordered
    return manifests


def preflight_task_context(manifest: dict, selected_df, context: int, k: int) -> tuple[int, int]:
    if int(context) not in set(map(int, manifest.get('contexts_steps', []))):
        raise ValueError('Context not in frozen task context grid')
    chosen = choose_cases(selected_df, k=k, context=context)
    for row in chosen.to_dict(orient='records'):
        # The same validation as inference, including regular cadence, no missing
        # observations and *exact* case-specific future timestamp alignment.
        make_forecast_input(row, context)
    return len(chosen), int(chosen.entity_id.nunique())


def build_plan(
    manifests: list[tuple[dict, Path]], models: list[str], contexts: list[int],
    k: int, quantile_policy: str, input_root: Path = INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> list[Job]:
    result = []
    # Ordered by task, then model, then context, preserving the user-supplied order.
    for m, folder in manifests:
        verified_m, df = load_frozen_inputs(folder)
        if verified_m['task_id'] != m['task_id'] or int(k) > int(m['max_origins']):
            raise RuntimeError(f'Invalid frozen export for task {m["task_id"]}')
        feasibility = {}
        for context in contexts:
            try:
                feasibility[context] = (*preflight_task_context(m, df, context, k), '')
            except (ValueError, RuntimeError, IndexError) as exc:
                feasibility[context] = (0, 0, str(exc))
        for model in models:
            for context in contexts:
                out = output_root / m['task_id'] / model / f'context_{context}' / f'k{k}' / 'full'
                n, n_entities, reason = feasibility[context]
                if model == 'tirex2' and context > 2048:
                    status, reason = 'SKIP_INFEASIBLE', 'TiRex-2 max context is 2048'
                elif reason:
                    status = 'SKIP_INFEASIBLE'
                else:
                    job = Job(m['task_id'],m['dataset_config'],m.get('family_id_exact',''),
                              model, KNOWN_MODELS[model],context,k,n,n_entities,'','',str(out))
                    status, reason = classify_output(out, job, m, quantile_policy)
                result.append(Job(m['task_id'],m['dataset_config'],m.get('family_id_exact',''),
                                  model,KNOWN_MODELS[model],context,k,n,n_entities,status,reason,str(out)))
    return result


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('Cannot write empty CSV')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)


def summary_rows(plan: list[Job]) -> list[dict]:
    rows = []
    for j in plan:
        if j.status != 'DONE':
            continue
        s = json.loads((Path(j.output_dir) / 'calibration_score.json').read_text(encoding='utf-8'))
        rows.append({**asdict(j),
                     'calibration_mean_case_mae': s['calibration_mean_case_mae'],
                     'calibration_mean_case_pinball_3': s['calibration_mean_case_pinball_3'],
                     'inference_seconds_exploratory': s['inference_seconds'],
                     'model_load_seconds_exploratory': s['model_load_seconds']})
    return rows


def build_case_catalog(plan: list[Job], output_path: Path) -> int:
    """Consolidate historical *scored diagnostics* into a case-level meta-learning table.

    Never touches historical labels or official evaluation data. Validate that
    the stored scorer output agrees with the frozen full-cohort case count.
    """
    keys = ['task_id','dataset_config','family_id_exact','model','context_steps','k',
            'case_id','entity_id','lag_before_first_official',
            'mae','mean_pinball_3','pinball_q10','pinball_q50','pinball_q90']
    total = 0
    with output_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for j in plan:
            if j.status != 'DONE':
                continue
            path = Path(j.output_dir) / 'calibration_case_metrics.csv'
            with path.open(newline='', encoding='utf-8') as src:
                reader = csv.DictReader(src)
                required = {'case_id','entity_id','lag_before_first_official',
                            'mae','mean_pinball_3','pinball_q10','pinball_q50','pinball_q90'}
                if not required.issubset(reader.fieldnames or []):
                    raise RuntimeError(f'Missing diagnostic columns in {path}')
                count = 0
                case_ids: set[str] = set()
                for r in reader:
                    if not r['case_id'] or r['case_id'] in case_ids:
                        raise RuntimeError(f'Duplicate or blank case ID in {path}')
                    case_ids.add(r['case_id'])
                    writer.writerow({**{col:getattr(j,col) for col in keys[:6]},
                                     **{col:r[col] for col in keys[6:]}})
                    count += 1
                if count != j.n_cases:
                    raise RuntimeError(f'Case count mismatch in {path}: {count} vs {j.n_cases}')
                total += count
    return total


def run_logged(command: list[str], path: Path, *, project_root: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    print('COMMAND:', ' '.join(command), flush=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('Command: ' + ' '.join(command) + '\n')
        f.flush()
        process = subprocess.Popen(command, cwd=project_root,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            f.write(line)
        return int(process.wait())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Dataset configs or frozen fev task IDs; default ALL full-cohort exports')
    parser.add_argument('--models', nargs='+', default=['chronos2','moirai2_small'],
                        help='Valid: ' + ', '.join(KNOWN_MODELS))
    parser.add_argument('--contexts', type=int, nargs='+', default=list(DEFAULT_CONTEXTS))
    parser.add_argument('--k', type=int, default=3, choices=(1,2,3))
    parser.add_argument('--quantile-policy', choices=('strict','rearrange'),default='rearrange')
    parser.add_argument('--execute', action='store_true', help='Opt-in to model inference and scoring; default DRY RUN')
    parser.add_argument('--score-only', action='store_true', help='Score existing forecasts only, no inference')
    parser.add_argument('--max-jobs', type=int, default=0,
                        help='Max actionable jobs per launch (0=no limit); useful for small pilots')
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--conda', default=None,help='Conda executable, usually autodetected')
    args = parser.parse_args(argv)
    if args.max_jobs < 0 or args.k < 1 or any(c <= 1 for c in args.contexts):
        parser.error('Invalid K, context or max-jobs')
    if len(set(args.contexts)) != len(args.contexts) or len(set(args.models)) != len(args.models):
        parser.error('Duplicate context/model arguments')
    unknown = sorted(set(args.models) - set(KNOWN_MODELS))
    if unknown:
        parser.error(f'Unknown model checkpoints: {unknown}')
    if args.score_only and not args.execute:
        parser.error('--score-only requires --execute')
    manifests = discover_exports(args.input_root,args.tasks)
    # Source-root override is used only for explicitly orchestrating non-default
    # roots; resume verification uses the standard root in the current version.
    if args.input_root != INPUT_ROOT:
        parser.error('--input-root override not supported for verified resume; use default frozen root')
    plan = build_plan(manifests,args.models,args.contexts,args.k,args.quantile_policy,
                      args.input_root,args.output_root)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = args.output_dir / f'{stamp}_{"execute" if args.execute else "plan"}'
    run_dir.mkdir(parents=True, exist_ok=False)
    write_csv(run_dir/'plan.csv',[asdict(j) for j in plan])
    manifest = {'schema':'pertsfm_cross_task_matrix_v1','created_utc':stamp,
                'execution_enabled':bool(args.execute),'k':args.k,
                'quantile_policy':args.quantile_policy,'task_ids':[m['task_id'] for m,_ in manifests],
                'models':args.models,'contexts':args.contexts,
                'task_input_provenance':[{
                    'task_id':m['task_id'],
                    'dataset_config':m['dataset_config'],
                    'family_id_exact_provisional':m.get('family_id_exact'),
                    'input_sha256':m['input_sha256'],
                    'historical_labels_sha256':m.get('labels_sha256'),
                    'official_registry_source_sha256':m.get('official_registry_source_sha256'),
                    'coverage':m['coverage']} for m,_ in manifests],
                'n_jobs':len(plan),'official_evaluation_accessed':False,
                'note':'Historical calibration only; not native FEV SQL; family_id_exact not independent family.'}
    (run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('='*92)
    print('PersTSFM PHASE 2.13 — CROSS-TASK HISTORICAL CALIBRATION MATRIX')
    print('='*92)
    print('Mode:', 'EXECUTE' if args.execute else 'DRY RUN (no inference/scoring)')
    print('K:',args.k,'| models:',args.models,'| contexts:',args.contexts)
    print('Tasks:',[(m['dataset_config'],m['task_id']) for m,_ in manifests])
    counts={s:sum(j.status==s for j in plan) for s in ['DONE','NEEDS_SCORE','PENDING','SKIP_INFEASIBLE','BLOCKED']}
    print('Plan statuses:',counts)
    for j in plan:
        print(f'{j.dataset_config:25} {j.model:15} c={j.context_steps:<5} '
              f'K={j.k} cases={j.n_cases:<5} {j.status:16} {j.reason[:90]}')
    print('Saved plan:',run_dir/'plan.csv',flush=True)
    blocked=[j for j in plan if j.status=='BLOCKED']
    if blocked:
        print('ERROR: BLOCKED output(s) detected; inspect before executing. No files changed.')
        return 2
    if not args.execute:
        existing_scores = summary_rows(plan)
        if existing_scores:
            write_csv(run_dir/'score_catalog.csv', existing_scores)
            n_cases = build_case_catalog(plan, run_dir/'case_catalog.csv')
            print(f'Already scored: {len(existing_scores)} configs, {n_cases} case rows | {run_dir/"score_catalog.csv"}')
        print('DRY RUN completed. Add --execute --max-jobs N to run new experiments.')
        return 0
    conda = args.conda or os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not conda:
        print('ERROR: Cannot locate conda executable (set --conda /path/to/conda).')
        return 2
    project_root = Path.cwd().resolve()
    log_dir = run_dir/'logs'
    actionable = [j for j in plan if j.status in ('NEEDS_SCORE','PENDING') and
                  (j.status=='NEEDS_SCORE' or not args.score_only)]
    if args.max_jobs:
        actionable=actionable[:args.max_jobs]
    print(f'Actionable jobs this invocation: {len(actionable)}',flush=True)
    results=[]
    for i, j in enumerate(actionable,1):
        out=Path(j.output_dir)
        pred=out/'predictions.parquet'
        print(f'\n[{i}/{len(actionable)}] {j.dataset_config} | {j.model} | c={j.context_steps} | K={j.k}',flush=True)
        if j.status=='PENDING':
            command=[conda,'run','--no-capture-output','-n',j.conda_env,'python',
                     '-m','scripts.phase2.run_preofficial_calibration',
                     '--task-id',j.task_id,'--model',j.model,
                     '--context',str(j.context_steps),'--k',str(j.k),
                     '--quantile-policy',args.quantile_policy]
            code=run_logged(command,log_dir/f'{i:03d}_inference.log',project_root=project_root)
            if code:
                print('Inference failed; output files are preserved.',flush=True)
                results.append({**asdict(j),'outcome':'FAILED_INFERENCE','return_code':code})
                break
        command=[conda,'run','--no-capture-output','-n','PERTSFM_DATA_ENV','python',
                 '-m','scripts.phase2.score_preofficial_calibration',
                 '--predictions',str(pred)]
        code=run_logged(command,log_dir/f'{i:03d}_score.log',project_root=project_root)
        if code:
            print('Scoring failed; outputs preserved.',flush=True)
            results.append({**asdict(j),'outcome':'FAILED_SCORE','return_code':code})
            break
        j.status='DONE';j.reason='completed this invocation'
        results.append({**asdict(j),'outcome':'PASS','return_code':0})
    if results:
        write_csv(run_dir/'executed_jobs.csv',results)
    # Recompute run states without rerunning any models.
    final_plan=build_plan(manifests,args.models,args.contexts,args.k,args.quantile_policy,
                          args.input_root,args.output_root)
    write_csv(run_dir/'final_plan.csv',[asdict(j) for j in final_plan])
    scores=summary_rows(final_plan)
    if scores:
        write_csv(run_dir/'score_catalog.csv',scores)
        n_cases = build_case_catalog(final_plan, run_dir/'case_catalog.csv')
        print(f'Consolidated scores: {len(scores)} configurations, {n_cases} case rows | {run_dir}')
    passed=sum(r['outcome']=='PASS' for r in results)
    print(f'PASS jobs this invocation: {passed}/{len(actionable)}')
    if results and results[-1]['outcome']!='PASS':
        return 1
    if any(j.status=='BLOCKED' for j in final_plan):
        return 2
    print('No official FEV evaluation targets accessed. Diagnostic MAE and 3-quantile pinball only.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
