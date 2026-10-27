#!/usr/bin/env python3
"""PersTSFM Phase 3.2: GPU cost benchmark on FROZEN pre-official input only.

Driver dry-runs by default; --execute runs one fresh Conda process per job.
Never reads labels, scores, official test data or previous predictions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

MODEL_ENVS = {
    'chronos2': 'PERTSFM_ENV',
    'timesfm3': 'PERTSFM_ENV',
    'tirex2': 'PERTSFM_ENV',
    'toto2_22m': 'PERTSFM_TOTO_ENV',
    'toto2_313m': 'PERTSFM_TOTO_ENV',
    'moirai2_small': 'PERTSFM_MOIRAI_ENV',
}
INPUT_ROOT = Path('results/phase2/preofficial_calibration/fev_bench_v0_10_0')
OUT_ROOT = Path('results/phase3/gpu_cost')


def task_id_by_name(input_root: Path, selector: str):
    """Return unique exported task ID; only trusts frozen historical manifests."""
    from scripts.phase2.run_preofficial_calibration import load_frozen_inputs
    matches = []
    for p in sorted(input_root.glob('*/manifest.json')):
        m, _ = load_frozen_inputs(p.parent)
        if selector in (m['dataset_config'], m['task_id']):
            matches.append(m['task_id'])
    if len(matches) != 1:
        raise ValueError(f'Expected one frozen historical task for {selector!r}; found {matches}')
    return matches[0]


def prepare_cases(task_id, context, entities, *, input_root=INPUT_ROOT):
    from scripts.phase2.run_preofficial_calibration import (
        load_frozen_inputs, choose_cases, make_forecast_input,
    )
    manifest, frozen = load_frozen_inputs(input_root / task_id)
    if context not in manifest['contexts_steps']:
        raise ValueError('Context absent from frozen input manifest')
    # Use lag 1 only to benchmark a single historical forecast per selected entity.
    # Time costs only; this is not a model-selection validation score.
    cases = choose_cases(frozen, k=1, context=context, max_entities=entities)
    for row in cases.to_dict(orient='records'):
        make_forecast_input(row, context)
    return manifest, cases


def validate_job(model, context, manifest, cases, *, expected_case_ids=None):
    if model not in MODEL_ENVS:
        raise ValueError(f'Unknown model {model}')
    if model == 'tirex2' and context > 2048:
        raise ValueError('TiRex-2 adapter max context is 2048')
    ids = tuple(cases['case_id'].astype(str))
    if expected_case_ids is not None and ids != expected_case_ids:
        raise ValueError('Unequal forecasting cases between contexts; cannot compare costs fairly')
    if not ids:
        raise ValueError('No input cases')
    return ids


def sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def predict_once(model, inputs, horizon, cadence, context, batch_size, torch):
    """Time model calls, synchronised; excludes data loading/preparation."""
    sync(torch)
    start = time.perf_counter()
    count = 0
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        results = model.predict_batch(
            inputs=batch, horizon=horizon, resolution_minutes=cadence,
            context_steps=context, batch_size=batch_size,
        )
        if len(results) != len(batch):
            raise RuntimeError('Model output count mismatch')
        # As in the Phase 2 runner, model adapters may return cases in
        # a different order; check identity as a set, not list position.
        if (len({str(x.case_id) for x in results}) != len(results) or
            {str(x.case_id) for x in results} != {str(x.case_id) for x in batch}):
            raise RuntimeError('Model output case IDs mismatch')
        for out in results:
            import numpy as np
            arr = np.asarray(out.prediction, dtype=float)
            if arr.shape != (horizon,) or not np.isfinite(arr).all():
                raise RuntimeError('Invalid model point predictions')
        count += len(results)
    sync(torch)
    return time.perf_counter() - start, count


def worker(args):
    import pandas as pd
    from scripts.phase2.run_preofficial_calibration import make_forecast_input
    import torch

    manifest, cases = prepare_cases(args.task_id, args.context, args.entities, input_root=args.input_root)
    validate_job(args.model, args.context, manifest, cases)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: do not mistake CPU timings for GPU cost')
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_name(0)
    # All GPU test cases must share resolution and horizon.
    horizon = int(manifest['horizon'])
    prepared = []
    cadences = set()
    # Data loading and timestamp construction occur before load and timing.
    for row in cases.to_dict(orient='records'):
        hist, times, _, cadence = make_forecast_input(row, args.context)
        prepared.append((str(row['case_id']), pd.DatetimeIndex(
            pd.to_datetime(times, unit='ns', utc=True)).tz_localize(None), hist))
        cadences.add(cadence)
    if len(cadences) != 1:
        raise RuntimeError('Mixed sampling cadences in cost cohort')
    cadence = cadences.pop()

    before = time.perf_counter()
    from src.models.adapters import build_model, ForecastInput
    inputs = [ForecastInput(case_id=i, timestamps=t, values=v)
              for i, t, v in prepared]
    model = build_model(args.model, batch_size=args.batch_size)
    sync(torch)
    load_seconds = time.perf_counter() - before

    for _ in range(args.warmup):
        _, n = predict_once(model, inputs, horizon, cadence, args.context,
                            args.batch_size, torch)
        assert n == len(inputs)
    sync(torch)
    torch.cuda.reset_peak_memory_stats(0)
    times = []
    for _ in range(args.repeats):
        sec, n = predict_once(model, inputs, horizon, cadence, args.context,
                              args.batch_size, torch)
        if n != len(inputs):
            raise RuntimeError('Output count inconsistent')
        times.append(float(sec))
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))

    record = {
        'schema': 'pertsfm_phase3_2_gpu_cost_v1',
        'role': 'PREOFFICIAL_HISTORICAL_INPUT_ONLY',
        'task_id': args.task_id,
        'dataset_config': manifest['dataset_config'],
        'model': args.model,
        'context_steps': args.context,
        'horizon': horizon,
        'batch_size': args.batch_size,
        'n_entities': len(cases),
        'warmup': args.warmup,
        'repeats': args.repeats,
        'case_ids': [str(x) for x in cases['case_id']],
        'input_sha256': manifest['input_sha256'],
        'gpu_name': device,
        'torch_version': torch.__version__,
        'load_seconds_import_and_build': load_seconds,
        'inference_repeat_seconds': times,
        'inference_median_seconds': statistics.median(times),
        'inference_min_seconds': min(times),
        'inference_max_seconds': max(times),
        'throughput_cases_per_second': len(cases) / statistics.median(times),
        'peak_allocated_bytes': peak_allocated,
        'peak_reserved_bytes': peak_reserved,
        'measurement_limits': (
            'Model loading excludes Conda process startup and prior history preparation. '
            'GPU memory allocated/reserved is PyTorch-specific, not total device or non-PyTorch usage. '
            'Warm inference repeats identical cases; other GPU activity may affect timings. '
            'No model accuracy, official test, or GPU energy was measured.'
        ),
    }
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Will not overwrite {output}')
    output.write_text(json.dumps(record, indent=2, default=str) + '\n', encoding='utf-8')
    print('PASS:', json.dumps({k:record[k] for k in [
        'model','dataset_config','context_steps','n_entities','batch_size',
        'load_seconds_import_and_build','inference_median_seconds',
        'throughput_cases_per_second','peak_allocated_bytes']}, sort_keys=True), flush=True)
    return 0


def plan_jobs(args):
    tasks = [task_id_by_name(args.input_root, t) for t in args.tasks]
    if len(set(tasks)) != len(tasks):
        raise ValueError('Duplicate task selection')
    jobs = []
    for task_id in tasks:
        common_ids = None
        for context in args.contexts:
            try:
                manifest, cases = prepare_cases(task_id, context, args.entities, input_root=args.input_root)
                ids = tuple(cases['case_id'].astype(str))
                if common_ids is None:
                    common_ids = ids
                if ids != common_ids:
                    raise ValueError('Unmatched entity cohort across requested contexts')
                base_status = 'READY'
                reason = ''
            except (ValueError, RuntimeError) as exc:
                base_status, reason = 'SKIP_INFEASIBLE', str(exc)
                cases = None
            for model in args.models:
                status = base_status
                note = reason
                if model == 'tirex2' and context > 2048:
                    status, note = 'SKIP_INFEASIBLE', 'TiRex-2 max context is 2048'
                jobs.append({
                    'task_id': task_id, 'model':model, 'environment':MODEL_ENVS[model],
                    'context_steps':context, 'batch_size': args.batch_size,
                    'n_entities': 0 if cases is None else len(cases),
                    'status':status, 'reason':note,
                })
    return jobs


def driver(args):
    unknown = sorted(set(args.models) - MODEL_ENVS.keys())
    if unknown:
        raise ValueError(f'Unknown models: {unknown}')
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.contexts)) != len(args.contexts):
        raise ValueError('Duplicate tasks or contexts')
    jobs = plan_jobs(args)
    print('=' * 84)
    print('PERTSFM PHASE 3.2 — GPU COST BENCHMARK | HISTORICAL INPUTS ONLY')
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'} | entities={args.entities} | batch={args.batch_size} "
          f'| warmup={args.warmup} | repeats={args.repeats}', flush=True)
    print('=' * 84)
    for j in jobs:
        print(f"{j['task_id']} {j['model']:14s} c={j['context_steps']:<5} "
              f"n={j['n_entities']:<3} {j['status']} {j['reason']}")
    print(f"Feasible: {sum(j['status']=='READY' for j in jobs)} / {len(jobs)}", flush=True)
    if not args.execute:
        print('Dry run only. Add --execute --max-jobs 2 for pilot.')
        return 0
    conda = args.conda or shutil.which('conda')
    if not conda:
        raise RuntimeError('Conda not found; activate PERTSFM_DATA_ENV and check PATH')
    project_root = Path.cwd()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output_root = args.output_dir / stamp
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
    success = []
    failures = []
    for j in jobs:
        if j['status'] != 'READY' or (args.max_jobs and len(success) >= args.max_jobs):
            continue
        label = f"{j['task_id']}_{j['model']}_c{j['context_steps']}_b{j['batch_size']}"
        dst = output_root / 'measurements' / f'{label}.json'
        cmd = [conda,'run','--no-capture-output','-n',j['environment'],
               'python', str(Path(__file__).resolve()),
               '--worker','--task-id',j['task_id'],'--model',j['model'],
               '--context',str(j['context_steps']),'--batch-size',str(j['batch_size']),
               '--entities',str(args.entities),'--warmup',str(args.warmup),
               '--repeats',str(args.repeats),'--worker-output',str(dst.resolve())]
        print('\nJOB:',label,flush=True)
        start = time.perf_counter()
        completed = subprocess.run(cmd,cwd=project_root, text=True,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        log = output_root / 'logs' / f'{label}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(' '.join(cmd)+'\n'+completed.stdout)
        print(completed.stdout[-6000:] if completed.stdout else '',flush=True)
        if completed.returncode or not dst.is_file():
            failures.append({'job':label,'returncode':completed.returncode,'log':str(log)})
            print('FAILED:', log, flush=True)
            break
        result = json.loads(dst.read_text())
        result['cold_process_wall_seconds'] = elapsed
        result['log_path'] = str(log)
        success.append(result)
    import pandas as pd
    if success:
        rows = [{k:v for k,v in r.items() if not isinstance(v,(list,dict))}
                for r in success]
        df = pd.DataFrame(rows)
        df.to_csv(output_root / 'measurements.csv',index=False)
        print('\nMEASURED GPU COSTS:',flush=True)
        cols = ['dataset_config','model','context_steps','batch_size','n_entities',
                'cold_process_wall_seconds','load_seconds_import_and_build',
                'inference_median_seconds','throughput_cases_per_second',
                'peak_allocated_bytes']
        print(df[cols].to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
    (output_root / 'manifest.json').write_text(json.dumps({
        'n_planned':len(jobs), 'n_measured':len(success),'failures':failures,
        'timestamp':stamp,'input_source':'preofficial historical frozen inputs',
        'no_official_test_access':True,
        'notes':'Replay benchmark; no simultaneous workloads; no energy or end-to-end deployment testing.'
    },indent=2)+'\n')
    print('Saved:',output_root,flush=True)
    if failures:
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks',nargs='+',default=['M_DENSE_1H'])
    ap.add_argument('--models',nargs='+',default=['chronos2','moirai2_small'])
    ap.add_argument('--contexts',type=int,nargs='+',default=[96,768,3072])
    ap.add_argument('--entities',type=int,default=16)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--warmup',type=int,default=2)
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--max-jobs',type=int,default=0)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--conda',default=None)
    ap.add_argument('--input-root',type=Path,default=INPUT_ROOT)
    ap.add_argument('--output-dir',type=Path,default=OUT_ROOT)
    ap.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--task-id',default=None,help=argparse.SUPPRESS)
    ap.add_argument('--model',default=None,help=argparse.SUPPRESS)
    ap.add_argument('--context',type=int,default=None,help=argparse.SUPPRESS)
    ap.add_argument('--worker-output',default=None,help=argparse.SUPPRESS)
    args=ap.parse_args(argv)
    if args.entities < 1 or args.batch_size < 1 or args.warmup < 1 or args.repeats < 2 or args.max_jobs < 0:
        ap.error('Need entities,batch-size,warmup>=1; repeats>=2; max-jobs>=0')
    if not args.worker and (not args.tasks or not args.models or not args.contexts):
        ap.error('Need tasks/models/contexts')
    if args.worker and (not args.task_id or not args.model or not args.context or not args.worker_output):
        ap.error('Worker needs task-id,model,context,worker-output')
    return worker(args) if args.worker else driver(args)


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
