"""Leakage-controlled rolling origins, built on the unchanged Phase 3.4 runner.

Each fold is a *prefix* of one FEV PREOFFICIAL input snapshot. Its final 2H
observations become (lag2 validation, lag1 later evaluation). Histories available
in newer folds include older folds' outcomes, as they would in prospective use.
No FEV test labels are ever read here. Roll evaluations must be scored only after
all fold selections have been locked.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .history import Snapshot, Series, load_snapshot, save_snapshot, file_hash, timestamp_ns
from .design import build_plan, normalise
from .artifacts import write_json

SCHEMA = 'pertsfm_phase35_rolling_v1'


@dataclass(frozen=True)
class RollingPlan:
    source: Path
    source_sha256: str
    experiment: dict
    rolling: dict
    folds: tuple[Snapshot, ...]
    plans: tuple
    entities: tuple[str, ...]
    excluded: tuple[str, ...]


def load_rolling_config(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    if raw.get('schema') != SCHEMA or not isinstance(raw.get('experiment'), dict):
        raise ValueError(f'Expected {SCHEMA} with experiment object')
    return raw


def _rolling_options(options: dict) -> dict:
    cfg = dict(options)
    folds = cfg.get('folds', 4)
    stride = cfg.get('stride_horizons', 1)
    if type(folds) is not int or not 2 <= folds <= 100:
        raise ValueError('rolling.folds must be an integer in [2,100]')
    if type(stride) is not int or not 1 <= stride <= 100:
        raise ValueError('rolling.stride_horizons must be integer in [1,100]')
    if type(cfg.get('require_aligned_ends', True)) is not bool:
        raise ValueError('require_aligned_ends must be bool')
    return {'folds':folds,'stride_horizons':stride,
            'require_aligned_ends':cfg.get('require_aligned_ends',True)}


def _fold_snapshots(source: Snapshot, options: dict, records):
    """Oldest to newest; no target observations after each fold's cutoff."""
    n, stride, h = options['folds'], options['stride_horizons'], source.horizon
    out=[]
    for index in range(n):
        trim=(n-1-index)*stride*h
        fold_records=[]
        for r in records:
            end=len(r.y)-trim
            if end < 2*h+10:
                raise ValueError(f'{r.id}: insufficient data for rolling fold {index}')
            fold_records.append(Series(r.id, np.asarray(r.y[:end],dtype=np.float64).copy(),r.time[:end]))
        out.append(Snapshot(source.task,h,tuple(fold_records)))
    return tuple(out)


def make_rolling_plan(config: dict) -> RollingPlan:
    if config.get('schema') != SCHEMA:
        raise ValueError(f'Expected schema {SCHEMA}')
    exp = normalise(config['experiment'])
    options = _rolling_options(config.get('rolling',{}))
    source_path=Path(exp['snapshot']).resolve()
    source=load_snapshot(source_path)
    if source.task != exp['task']:
        raise ValueError('Snapshot task mismatch')
    records=source.records[:exp['max_entities']] if exp['max_entities'] else source.records
    if not records:
        raise ValueError('Empty cohort')
    if options['require_aligned_ends']:
        ends={int(timestamp_ns(r.time[-1:])[0]) for r in records}
        if len(ends)>1:
            raise ValueError('Unaligned entity end timestamps: use a pre-aligned cohort (not automatic target alignment)')
    # Prospective feasibility is checked fold-by-fold. Global fixed matched cohort
    # is obtained by intersection; no filtering by forecasting scores is permitted.
    originals=_fold_snapshots(source,options,records)
    allowed=set(r.id for r in records)
    reasons={}
    for i, snap in enumerate(originals):
        candidate_cfg={**exp,'max_entities':0,'invalid_policy':'drop'}
        trial_plan=build_plan(candidate_cfg,snap,source_path)
        ids={r.id for r in trial_plan.records}
        for r in records:
            if r.id not in ids:
                reasons.setdefault(r.id,[]).append(i)
        allowed &= ids
    if reasons and exp['invalid_policy']=='error':
        raise ValueError(f'{len(reasons)} entities invalid in one or more folds; set invalid_policy=drop to use fixed intersection; examples={list(reasons)[:5]}')
    if not allowed:
        raise ValueError('No entities feasible on ALL rolling folds')
    matched=tuple(r.id for r in records if r.id in allowed)
    folds=[]
    plans=[]
    for snap in originals:
        cohort=Snapshot(snap.task,snap.horizon,tuple(r for r in snap.records if r.id in allowed))
        fold_cfg={**exp,'max_entities':0,'invalid_policy':'error'}
        p=build_plan(fold_cfg,cohort,source_path)
        if tuple(r.id for r in p.records)!=matched:
            raise RuntimeError('Rolling cohort changed across folds')
        folds.append(cohort)
        plans.append(p)
    # Ensure evaluation windows do not overlap chronologically across folds.
    for r in zip(*(snap.records for snap in folds)):
        for older,newer in zip(r[:-1],r[1:]):
            older_eval_end=timestamp_ns(older.time[-1:])[0]
            newer_eval_start=timestamp_ns(newer.time[-source.horizon:-source.horizon+1])[0]
            if newer_eval_start <= older_eval_end:
                raise RuntimeError('Overlapping lag1 evaluation windows')
    return RollingPlan(source_path,file_hash(source_path),exp,options,tuple(folds),tuple(plans),
                       matched,tuple(r.id for r in records if r.id not in allowed))


def describe_rolling(plan: RollingPlan):
    print('='*100)
    print('PERTSFM 3.5 | CAUSAL ROLLING ORIGIN PLAN | PREOFFICIAL HISTORICAL DATA')
    print('='*100)
    print(f"Task={plan.experiment['task']} H={plan.folds[0].horizon} folds={len(plan.folds)} "
          f"stride={plan.rolling['stride_horizons']}*H entities={len(plan.entities)} "
          f'excluded={len(plan.excluded)}')
    for i,(snapshot, p) in enumerate(zip(plan.folds,plan.plans)):
        first=snapshot.records[0]
        h=snapshot.horizon
        print(f' fold={i:02d} end={first.time[-1]} train_end={first.time[-2*h-1]} '
              f'validation=[{first.time[-2*h]}..{first.time[-h-1]}] '
              f'later=[{first.time[-h]}..{first.time[-1]}] '
              f'pre-validation history min={min(len(r.y)-2*h for r in snapshot.records)} '
              f'trials={len(p.trials)}')
    print('TRAIN each fold from only its own prefix; SELECT lag2; SCORE lag1 only after ALL folds lock.')
    print('The later labels are historical PREOFFICIAL data, NOT official FEV test labels.')


def initialize_suite(plan: RollingPlan, output: Path):
    """Materialize independently hashed per-fold snapshots; never overwrite."""
    out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    fold_infos=[]
    for i,fold in enumerate(plan.folds):
        folder=out/'folds'/f'fold_{i:02d}'
        folder.mkdir(parents=True)
        snap_path=folder/'history.npz'
        save_snapshot(fold,snap_path)
        exp={**plan.experiment,'snapshot':str(snap_path),'max_entities':0,'invalid_policy':'error'}
        cfg_path=folder/'experiment.json'
        write_json(cfg_path,{'schema':'pertsfm_phase34_batch_v1','experiments':[exp]})
        fold_infos.append({'index':i,'config':str(cfg_path),'snapshot':str(snap_path),
                           'snapshot_sha256':file_hash(snap_path),'config_sha256':file_hash(cfg_path),
                           'end_timestamp_ns':int(timestamp_ns(fold.records[0].time[-1:])[0]),
                           'pre_lag2_min':min(len(r.y)-2*fold.horizon for r in fold.records)})
    write_json(out/'suite_manifest.json',{
        'schema':SCHEMA,'source_snapshot':str(plan.source),'source_sha256':plan.source_sha256,
        'task':plan.experiment['task'],'horizon':plan.folds[0].horizon,
        'rolling':plan.rolling,'entity_ids':list(plan.entities),'excluded_ids':list(plan.excluded),
        'folds':fold_infos,
        'protocol':'per-origin fresh fitting; lag2 selection; all folds locked before lag1 scoring; FEV input-only',
    })
    print(f'SUITE INITIALIZED: {out}')
    print(f'NEXT: python -m scripts.phase3.rolling_adaptation run --suite {out}')
    return out


def load_suite(root: Path):
    root=Path(root).resolve()
    m=json.loads((root/'suite_manifest.json').read_text())
    if m.get('schema')!=SCHEMA or not m.get('folds'):
        raise RuntimeError('Invalid rolling suite manifest')
    for item in m['folds']:
        for key,sha in [('snapshot','snapshot_sha256'),('config','config_sha256')]:
            if file_hash(Path(item[key]))!=item[sha]:
                raise RuntimeError(f'Frozen fold input changed: {item[key]}')
        snap=load_snapshot(Path(item['snapshot']))
        if snap.task!=m['task'] or snap.horizon!=m['horizon'] or \
           [r.id for r in snap.records]!=m['entity_ids']:
            raise RuntimeError('Fold snapshot no longer matches fixed cohort')
    return root,m


def _fold_run(root, item, *, trace_sampler=False):
    from .experiment import run_one
    index=item['index']
    folder=root/'runs'/f'fold_{index:02d}'
    ref=root/'folds'/f'fold_{index:02d}'/'run_ref.json'
    if ref.exists():
        info=json.loads(ref.read_text())
        run=Path(info['run'])
        if not (run/'selection.json').is_file():
            raise RuntimeError('Run reference exists but selection lock missing')
        return run
    existing=sorted(p for p in folder.iterdir() if p.is_dir()) if folder.exists() else []
    if len(existing)>1:
        raise RuntimeError('Multiple run attempts for a fold: manual reconciliation needed')
    backend = None
    if trace_sampler and (not existing or not (existing[0]/'selection.json').exists()):
        from .sampler_audit import TracedChronos2Backend
        backend=TracedChronos2Backend()
    if existing:
        run=existing[0]
        if not (run/'selection.json').exists():
            print(f'RESUMING FOLD {index} from {run}',flush=True)
            cfg=json.loads(Path(item['config']).read_text())['experiments'][0]
            run_one(cfg,output_root=folder,resume=run,backend=backend)
    else:
        cfg=json.loads(Path(item['config']).read_text())['experiments'][0]
        run=run_one(cfg,output_root=folder,backend=backend)
    write_json(ref,{'run':str(run),'selection_sha256':file_hash(run/'selection.json')})
    return run


def run_suite(root: Path, only_fold: int | None=None, *, trace_sampler=False):
    root,m=load_suite(root)
    if (root/'suite_freeze.json').exists():
        raise RuntimeError('Suite already frozen; no new training permitted')
    if only_fold is not None and only_fold not in range(len(m['folds'])):
        raise ValueError('Fold outside suite range')
    for item in m['folds']:
        if only_fold is not None and item['index'] != only_fold:
            continue
        run=_fold_run(root,item,trace_sampler=trace_sampler)
        print(f"PASS: FOLD {item['index']:02d} locked: {run}",flush=True)
    print(f'NEXT (when all folds are locked): python -m scripts.phase3.rolling_adaptation score --suite {root}')


def _verify_locks(root,m):
    refs=[]
    for item in m['folds']:
        ref_path=root/'folds'/f"fold_{item['index']:02d}"/'run_ref.json'
        if not ref_path.exists():
            raise RuntimeError(f"Fold {item['index']} not locked. Run ALL folds before scoring any.")
        ref=json.loads(ref_path.read_text())
        run=Path(ref['run'])
        if file_hash(run/'selection.json')!=ref['selection_sha256']:
            raise RuntimeError('Selection lock altered after run_ref recorded')
        cfg=json.loads(Path(item['config']).read_text())['experiments'][0]
        manifest=json.loads((run/'manifest.json').read_text())
        expected_config={**cfg,'snapshot':str(Path(cfg['snapshot']).resolve())}
        if (manifest['snapshot_sha256']!=item['snapshot_sha256'] or
            manifest['ids']!=m['entity_ids'] or manifest['config']!=expected_config):
            raise RuntimeError('Fold run config/cohort does not correspond to frozen suite')
        refs.append({'index':item['index'],'run':str(run),'selection_sha256':ref['selection_sha256']})
    return refs


def score_suite(root: Path):
    from .experiment import score_later_run
    from .reporting import report_suite
    root,m=load_suite(root)
    refs=_verify_locks(root,m)
    freeze=root/'suite_freeze.json'
    expected={'schema':'pertsfm_phase35_suite_lock_v1',
              'suite_manifest_sha256':file_hash(root/'suite_manifest.json'),'runs':refs}
    if freeze.exists():
        if json.loads(freeze.read_text())!=expected:
            raise RuntimeError('Suite lock changed after first later scoring')
    else:
        write_json(freeze,expected)
    for r in refs:
        print(f"SCORING FOLD {r['index']:02d} (all decisions already frozen)",flush=True)
        score_later_run(Path(r['run']))
    return report_suite(root)
