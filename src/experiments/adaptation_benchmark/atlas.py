"""Phase 3.7A: preofficial, validation-only adaptation-opportunity atlas.

This module deliberately reuses the audited Phase 3.6 runner. No FEV test labels,
GPU code, or later-period labels are accessed for validation ranking or cohort
selection. Snapshot export is a separate explicit command in the data environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from .history import (Snapshot, cadence_minutes, case_window, file_hash, history_hash,
                      load_snapshot, save_snapshot)

SCHEMA = 'pertsfm_phase37a_atlas_v1'
PLAN_SCHEMA = 'pertsfm_phase37a_dataset_plan_v1'
CONTEXTS = (24, 48, 96, 192, 384, 768, 1536, 3072, 4096, 6144, 8192)


@dataclass(frozen=True)
class ScreeningPlan:
    key: str
    family: str
    task: str
    source: Path
    source_hash: str
    snapshot: Snapshot
    records: tuple
    lora_context: int
    zero_contexts: tuple[int, ...]
    n_eligible: int
    n_total: int
    excluded_count: int
    notes: str


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get('schema') != SCHEMA:
        raise ValueError(f'Atlas schema must be {SCHEMA}')
    entries = cfg.get('datasets')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Expected nonempty datasets list')
    keys = [e.get('key') for e in entries]
    if len(keys) != len(set(keys)) or any(not isinstance(k, str) or
        not re.fullmatch(r'[a-z][a-z0-9_]*', k) for k in keys):
        raise ValueError('Invalid or duplicated dataset keys')
    for entry in entries:
        if not all(entry.get(k) for k in ('family','task','snapshot','key')):
            raise ValueError('Each dataset needs key, family, task and snapshot')
        if entry.get('role') != 'development':
            raise ValueError('Phase 3.7A only supports explicitly marked development datasets')
        target = int(entry.get('preferred_context', 768))
        if target not in CONTEXTS:
            raise ValueError(f"{entry['key']}: preferred_context outside the supported grid")
        if int(entry.get('minimum_entities', 8)) < 2:
            raise ValueError('minimum_entities must be >= 2')
    screening = cfg.get('screening', {})
    for key in ('steps', 'learning_rates', 'seeds'):
        if not isinstance(screening.get(key), list) or not screening[key]:
            raise ValueError(f'screening.{key} must be a nonempty list')
    steps = screening['steps']
    seeds = screening['seeds']
    lrs = screening['learning_rates']
    if (any(type(s) is not int or s <= 0 for s in steps) or
        any(type(s) is not int or s <= 0 for s in seeds) or
        any(not isinstance(r, (float,int)) or not 0 < r < 1 for r in lrs)):
        raise ValueError('Invalid steps, seeds or learning rates')
    if (len(set(steps)) != len(steps) or len(set(seeds)) != len(seeds) or
        len(set(lrs)) != len(lrs)):
        raise ValueError('Duplicated screening values')
    if type(screening.get('train_batch_size', 32)) is not int or screening.get('train_batch_size', 32) <= 0:
        raise ValueError('train_batch_size must be a positive integer')
    if type(screening.get('inference_batch_size', 8)) is not int or screening.get('inference_batch_size', 8) <= 0:
        raise ValueError('inference_batch_size must be a positive integer')
    if int(screening.get('sample_entities', 128)) < 2:
        raise ValueError('sample_entities must be >= 2')
    if int(screening.get('sample_seed', 371)) < 0:
        raise ValueError('sample_seed must be >= 0')
    return cfg


def _valid_record(r, horizon, lora_context, zero_contexts):
    # Training pools must end before the lag2 labels. Same entity must be valid
    # for every candidate to avoid different denominators across strategies.
    try:
        if len(r.y) - 2*horizon < lora_context + 2*horizon:
            return False
        pool = r.y[:-2*horizon]
        if not np.isfinite(pool).all():
            return False
        for c in zero_contexts:
            case_window(r, horizon, c, 2)
            case_window(r, horizon, c, 1)   # does NOT reveal later labels
        cadence_minutes((r,))
        return True
    except (ValueError, RuntimeError):
        return False


def _select_hashed(records, n, seed):
    # Stable, label-free sampling. Avoid systematic sorted-prefix sensor bias.
    order = sorted(records, key=lambda r: (hashlib.sha256(f'{seed}|{r.id}'.encode()).hexdigest(), r.id))
    return tuple(sorted(order[:n], key=lambda r: r.id))


def plan_dataset(entry, cfg, *, cwd=None):
    base = Path.cwd() if cwd is None else Path(cwd)
    source = Path(entry['snapshot'])
    if not source.is_absolute():
        source = base / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{entry['key']}: snapshot missing at {source}")
    snap = load_snapshot(source)
    if snap.task != entry['task']:
        raise RuntimeError(f"{entry['key']}: task mismatch: snapshot={snap.task}, expected={entry['task']}")
    h = snap.horizon
    target = int(entry.get('preferred_context', 768))
    minimum = int(entry.get('minimum_entities', 8))
    ncap = int(cfg['screening'].get('sample_entities', 128))
    if ncap < minimum:
        raise ValueError('sample_entities cannot be less than minimum_entities')
    # The context rule is fixed in advance, uses feasibility only, not outcomes.
    # Try requested context first, then progressively shorter contexts.
    chosen = None
    for c in reversed([x for x in CONTEXTS if x <= target]):
        candidates = [r for r in snap.records if _valid_record(r, h, c, (c,))]
        if len(candidates) >= minimum:
            chosen = (c, candidates)
            break
    if chosen is None:
        raise ValueError(f"{entry['key']}: fewer than {minimum} compatible entities even at C=24")
    c, _ = chosen
    # Context-optimised zero-shot baseline: short, matched and long if they
    # retain at least 80% of the eligible matched-context cohort.
    short = max((v for v in CONTEXTS if v < c), default=None)
    extras = [] if short is None else [short]
    max_long = min(8192, int(entry.get('max_zero_context', 4096)))
    longs = [x for x in CONTEXTS if c < x <= max_long]
    if longs:
        extras.append(max(longs))
    baseline = [c]
    initial = sum(_valid_record(r, h, c, (c,)) for r in snap.records)
    for extra in extras:
        proposed = tuple(sorted(set(baseline + [extra])))
        eligible = [r for r in snap.records if _valid_record(r, h, c, proposed)]
        if len(eligible) >= minimum and len(eligible) >= math.ceil(.8 * initial):
            baseline.append(extra)
    zero = tuple(sorted(set(baseline)))
    eligible = [r for r in snap.records if _valid_record(r, h, c, zero)]
    # Avoid mixed-frequency cohorts, which downstream core build_plan rejects.
    by_cadence = {}
    for r in eligible:
        freq = cadence_minutes((r,))
        by_cadence.setdefault(freq, []).append(r)
    if not by_cadence:
        raise ValueError(f"{entry['key']}: no valid fixed-cadence entities")
    freq, eligible = sorted(by_cadence.items(), key=lambda it: (-len(it[1]), it[0]))[0]
    if len(eligible) < minimum:
        raise ValueError(f"{entry['key']}: fewer than {minimum} compatible entities after cadence filtering")
    records = _select_hashed(eligible, min(ncap,len(eligible)), int(cfg['screening'].get('sample_seed',371)))
    return ScreeningPlan(entry['key'], entry['family'], snap.task, source, file_hash(source),
                         snap, records, c, zero, len(eligible), len(snap.records),
                         len(snap.records)-len(eligible),
                         f"predeclared feasibility-based context fallback; cadence={freq}min; hash-sampled")


def capability_config(plan, cfg, snapshot_path):
    s = cfg['screening']
    variants=[]
    for steps in s['steps']:
        for lr in s['learning_rates']:
            for seed in s['seeds']:
                lr_code = format(float(lr),'.0e').replace('-','m').replace('+','p').replace('e0','e')
                variants.append({'name':f'lora_{steps}_{lr_code}_b{s.get("train_batch_size",32)}_s{seed}',
                                 'mode':'lora','steps':steps,'learning_rate':float(lr),
                                 'batch_size':int(s.get('train_batch_size',32)),'seed':seed})
    return {'schema':'pertsfm_phase36_capability_v1',
            'task':plan.task,'snapshot':str(Path(snapshot_path).resolve()),
            'max_entities':0,'context':plan.lora_context,'zero_contexts':list(plan.zero_contexts),
            'history':'full','batch_size':int(s.get('inference_batch_size',8)),
            'trace_sampler':True,'reload_check':True,'invalid_policy':'error',
            'selection_metric':'validation_mae','variants':variants}


def compile_plan(plan, cfg, root):
    root = Path(root)
    dest = root/'derived_snapshots'/f'{plan.key}.npz'
    manifest = root/'derived_snapshots'/f'{plan.key}.json'
    selected = Snapshot(plan.task,plan.snapshot.horizon,plan.records)
    meta = {'schema':PLAN_SCHEMA,'key':plan.key,'task':plan.task,'family':plan.family,
            'source':str(plan.source),'source_sha256':plan.source_hash,
            'source_history_sha256':history_hash(plan.snapshot.records),
            'selected_history_sha256':history_hash(plan.records),
            'sample_seed':int(cfg['screening'].get('sample_seed',371)),
            'selected_ids':[r.id for r in plan.records],
            'zero_contexts':list(plan.zero_contexts),'lora_context':plan.lora_context,
            'n_total':plan.n_total,'n_eligible':plan.n_eligible,'n_selected':len(plan.records),
            'note':plan.notes}
    if dest.exists() or manifest.exists():
        if not dest.exists() or not manifest.exists():
            raise RuntimeError(f'Incomplete derived snapshot/manifest at {dest}')
        if json.loads(manifest.read_text()) != meta:
            raise RuntimeError(f'Derived snapshot provenance changed: {manifest}; use new output root')
        loaded=load_snapshot(dest)
        if history_hash(loaded.records) != meta['selected_history_sha256']:
            raise RuntimeError(f'Derived snapshot changed since manifest: {dest}')
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        save_snapshot(selected,dest)
        manifest.write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    return dest, meta


def _bootstrap_gain(a, b, seed, reps=1000):
    # Paired entity bootstrap: exploratory CI, not selection-adjusted or family-level.
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    if a.shape != b.shape or a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Cannot compute paired entity bootstrap')
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(a),size=(reps,len(a)))
    av=a[idx].mean(axis=1)
    bv=b[idx].mean(axis=1)
    gains=np.where(av>0,100*(av-bv)/av,np.nan)
    finite=gains[np.isfinite(gains)]
    if not len(finite):return (float('nan'),float('nan'))
    return tuple(float(x) for x in np.percentile(finite,[2.5,97.5]))



def naive_validation_references(snapshot):
    """Evaluate predeclared last-value and calendar-seasonal-naive validation.

    Uses lag2 labels ONLY. The season is fixed from cadence, not tuned on
    validation. Forecasts do not access lag1 labels or official FEV targets.
    """
    h=snapshot.horizon
    freq=cadence_minutes(snapshot.records)
    if freq==1440:
        period=7  # weekly seasonality for daily series
    elif freq>0 and 1440 % freq==0:
        period=1440//freq # one day for regularly sampled intraday series
    else:
        period=1 # no supported calendar season
    by_model={'last_value':[],'seasonal_naive':[]}
    for r in snapshot.records:
        end=len(r.y)-2*h
        truth=np.asarray(r.y[end:end+h],dtype=np.float64)
        prefix=np.asarray(r.y[:end],dtype=np.float64)
        if (len(prefix)<period or not np.isfinite(truth).all() or
            not np.isfinite(prefix).all()):
            raise ValueError(f'{r.id}: cannot score simple references')
        last=np.repeat(prefix[-1],h)
        season=np.resize(prefix[-period:],h)
        by_model['last_value'].append(float(np.mean(np.abs(last-truth))))
        by_model['seasonal_naive'].append(float(np.mean(np.abs(season-truth))))
    return pd.DataFrame([
        {'model':name,'validation_mae':float(np.mean(values)),
         'n_entities':len(values),'seasonal_lag':period if name=='seasonal_naive' else 1}
        for name,values in by_model.items()])


def summarise_validation(run_dir, plan_meta=None, seed=371):
    root=Path(run_dir)
    if not (root/'selection.json').is_file():
        raise RuntimeError(f'Unfinished or unlocked capability run: {root}')
    table=pd.read_csv(root/'validation.csv')
    entities=pd.read_csv(root/'validation_entity.csv')
    lock=json.loads((root/'selection.json').read_text())
    grouped=table.groupby('strategy',sort=False,as_index=False).agg(
        mode=('mode','first'),context=('context','first'),steps=('steps','first'),
        validation_mae=('validation_mae','mean'),replicates=('trial_key','count'),
        train_seconds=('train_seconds','mean'))
    zero=grouped[grouped['mode']=='zero'].sort_values('validation_mae')
    lora=grouped[grouped['mode']=='lora'].sort_values('validation_mae')
    if zero.empty or lora.empty:raise RuntimeError('Both zero-shot and LoRA required')
    z=zero.iloc[0]; l=lora.iloc[0]
    matched=zero.loc[zero.context==int(l.context)]
    if len(matched)!=1:raise RuntimeError('No context-matched zero-shot baseline')
    m=matched.iloc[0]
    def per_entity(strategy):
        keys=table.loc[table.strategy==strategy,'trial_key'].tolist()
        frame=entities[entities.trial_key.isin(keys)]
        x=frame.groupby('entity_id').agg(mae=('validation_mae','mean'),n=('trial_key','count'))
        if len(x)!=len(entities.entity_id.unique()) or not (x.n==len(keys)).all():
            raise RuntimeError(f'Unmatched per-entity cohort for {strategy}')
        return x.mae
    ze=per_entity(z.strategy).sort_index()
    le=per_entity(l.strategy).reindex(ze.index)
    if ze.isna().any() or le.isna().any():raise RuntimeError('Missing matched entity')
    gain=100*(z.validation_mae-l.validation_mae)/max(z.validation_mae,1e-12)
    matched_gain=100*(m.validation_mae-l.validation_mae)/max(m.validation_mae,1e-12)
    low,high=_bootstrap_gain(ze.to_numpy(),le.to_numpy(),seed)
    # Historical validation references are descriptive, not candidate strategies
    # included in the existing Phase 3.6 selection lock.
    manifest=json.loads((root/'manifest.json').read_text())
    naive_best=float('nan')
    naive_name='unavailable'
    if isinstance(manifest.get('config'),dict) and manifest['config'].get('snapshot'):
        historical=load_snapshot(manifest['config']['snapshot'])
        naive=naive_validation_references(historical)
        naive.to_csv(root/'atlas_naive_validation.csv',index=False)
        naive_row=naive.sort_values('validation_mae').iloc[0]
        naive_best=float(naive_row.validation_mae)
        naive_name=str(naive_row.model)
    best_available=min(float(z.validation_mae),naive_best) if np.isfinite(naive_best) else float(z.validation_mae)
    gain_available=100*(best_available-float(l.validation_mae))/max(best_available,1e-12)
    res={'dataset':root.parent.name,'run_dir':str(root.resolve()),
         'n_entities':len(ze),'horizon':json.loads((root/'manifest.json').read_text())['horizon'],
         'best_zero_strategy':str(z.strategy),'best_zero_mae':float(z.validation_mae),
         'best_lora_strategy':str(l.strategy),'best_lora_mae':float(l.validation_mae),
         'matched_zero_mae':float(m.validation_mae),'lora_gain_best_zero_pct':float(gain),
         'best_naive_reference':naive_name,'best_naive_mae':naive_best,
         'best_available_simple_or_zero_mae':best_available,
         'lora_gain_best_available_pct':float(gain_available),
         'lora_gain_matched_zero_pct':float(matched_gain),
         'entity_win_fraction':float((le<ze).mean()),'paired_bootstrap_ci_low_pct':low,
         'paired_bootstrap_ci_high_pct':high,'lora_train_seconds':float(l.train_seconds),
         'selected_strategy':lock['winner'],'selected_is_lora':bool(str(lock['winner']).startswith('lora_')),
         'screening_threshold_met':bool(gain>=5 and gain_available>=5 and len(ze)>=16),
         'ci_is_selection_adjusted':False,
         'evaluation':'development_validation_only'}
    if plan_meta:
        res.update(family=plan_meta['family'],source_sha256=plan_meta['source_sha256'],
                   context_lora=plan_meta['lora_context'])
    return res


def summarize_root(root):
    root=Path(root)
    registry=root/'registry.json'
    if not registry.exists():
        print('No completed atlas datasets yet.');return pd.DataFrame()
    entries=json.loads(registry.read_text())
    rows=[]
    for key,data in sorted(entries.items()):
        if not Path(data['run_dir']).joinpath('selection.json').is_file():
            continue
        result=summarise_validation(data['run_dir'],data.get('plan'),seed=data.get('sample_seed',371))
        result['dataset']=key
        rows.append(result)
    if not rows:
        print('No locked validation results yet.');return pd.DataFrame()
    df=pd.DataFrame(rows).sort_values('lora_gain_best_zero_pct',ascending=False)
    df.to_csv(root/'atlas_validation_summary.csv',index=False)
    print('\nPHASE 3.7A | ADAPTATION OPPORTUNITY ATLAS | VALIDATION ONLY')
    cols=['dataset','family','n_entities','context_lora','best_zero_mae','best_lora_mae',
          'lora_gain_best_zero_pct','lora_gain_matched_zero_pct',
          'lora_gain_best_available_pct','entity_win_fraction','lora_train_seconds','screening_threshold_met']
    print(df[[x for x in cols if x in df]].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('WARNING: exploratory in-sample hyperparameter selection; paired bootstrap CIs are not selection-adjusted.')
    print('5% flag requires gains over best zero-shot AND best simple reference; it is not significance.')
    print('No official FEV targets or later labels were used to rank screening strategies.')
    print(f'SAVED: {root/"atlas_validation_summary.csv"}')
    return df
