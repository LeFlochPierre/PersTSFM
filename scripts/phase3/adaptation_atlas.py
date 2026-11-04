"""Run Phase 3.7A without duplicating audited Phase 3.6 training code.

Example:
  python -m scripts.phase3.adaptation_atlas preflight --config configs/phase3/atlas_development_v1.json
  python -m scripts.phase3.adaptation_atlas run --config configs/phase3/atlas_development_v1.json --dataset loop_seattle
  python -m scripts.phase3.adaptation_atlas summarize --output-root results/phase37a/atlas_dev_v1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.experiments.adaptation_benchmark.atlas import (
    read_config, plan_dataset, capability_config, compile_plan,
    summarize_root, summarise_validation)


def _atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    os.replace(temp,path)


def _dataset(cfg,key=None):
    entries=cfg['datasets']
    if key is not None:
        entries=[x for x in entries if x['key']==key]
        if not entries:raise ValueError(f'Unknown dataset key {key}')
    return entries


def _entries(cfg,key=None,strict=False):
    ready=[];missing=[]
    print('\nPHASE 3.7A | PREOFFICIAL ADAPTATION OPPORTUNITY PREFLIGHT')
    for e in _dataset(cfg,key):
        try:
            plan=plan_dataset(e,cfg)
        except FileNotFoundError as exc:
            print(f"MISSING {e['key']:18s} {e['family']:12s} {exc}")
            missing.append(e['key'])
            continue
        except Exception as exc:
            print(f"INELIGIBLE {e['key']:15s} {type(exc).__name__}: {exc}")
            missing.append(e['key'])
            continue
        print(f"READY   {e['key']:18s} family={plan.family:12s} n={len(plan.records):4d}/{plan.n_total:<4d} "
              f"H={plan.snapshot.horizon:4d} C_lora={plan.lora_context:4d} zero_C={list(plan.zero_contexts)} "
              f"excluded={plan.excluded_count}")
        ready.append((e,plan))
    print(f"READY {len(ready)} / {len(ready)+len(missing)} dataset entries; no lag1 or official targets scored")
    if strict and missing:raise RuntimeError(f'Missing or infeasible datasets: {missing}')
    return ready


def _registry(root):
    reg=Path(root)/'registry.json'
    return json.loads(reg.read_text()) if reg.exists() else {}


def run_screen(config, root, dataset=None):
    from src.experiments.adaptation_benchmark.capability import run as run_capability, make_plan
    from src.experiments.adaptation_benchmark.features import compute_features
    import pandas as pd
    cfg=read_config(config)
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    ready=_entries(cfg,dataset)
    if not ready:raise RuntimeError('No datasets ready. Export or provide historical input snapshots first.')
    reg=_registry(root)
    for e,plan in ready:
        source,meta=compile_plan(plan,cfg,root)
        work_cfg=capability_config(plan,cfg,source)
        conf_dir=root/'screen_configs'
        conf_dir.mkdir(parents=True,exist_ok=True)
        conf_path=conf_dir/f'{plan.key}.json'
        config_text=json.dumps(work_cfg,indent=2,sort_keys=True)+'\n'
        if conf_path.exists() and conf_path.read_text()!=config_text:
            raise RuntimeError(f"Dataset {plan.key} config changed; create a fresh --output-root")
        if not conf_path.exists():conf_path.write_text(config_text)
        # Validate against the existing Phase3.6 implementation before touching GPU.
        check,_,_=make_plan(work_cfg)
        if [r.id for r in check.records]!=[r.id for r in plan.records]:
            raise RuntimeError(f'{plan.key}: cohort mismatch after Phase3.6 feasibility check')
        feats,med=compute_features(plan.records,plan.snapshot.horizon,check.cadence)
        feats.to_csv(root/f'{plan.key}_fingerprint_entities.csv',index=False)
        med.insert(0,'dataset',plan.key)
        med.to_csv(root/f'{plan.key}_fingerprint.csv',index=False)
        if plan.key in reg:
            old=reg[plan.key]
            if old['plan']!=meta or old['screen_config']!=work_cfg:
                raise RuntimeError('Previously completed atlas dataset differs; use a new output root')
            print(f'ALREADY LOCKED: {plan.key} -> {old["run_dir"]}')
            continue
        run_parent=root/'runs'/plan.key
        partials=sorted(p for p in run_parent.glob('*/manifest.json') if not (p.parent/'selection.json').exists()) if run_parent.exists() else []
        if len(partials)>1:
            raise RuntimeError(f'{plan.key}: multiple incomplete runs; inspect before resuming')
        resume=partials[0].parent if partials else None
        print(f'\n=== DATASET {plan.key} | {plan.family} ===',flush=True)
        dest=run_capability(work_cfg,output_root=run_parent,resume=resume)
        result=summarise_validation(dest,meta,seed=cfg['screening'].get('sample_seed',371))
        print(f"OPPORTUNITY {plan.key}: selected={result['selected_strategy']}, "
              f"gain_vs_best_zero={result['lora_gain_best_zero_pct']:+.2f}% "
              f"gain_vs_matched_zero={result['lora_gain_matched_zero_pct']:+.2f}% "
              f"entity_win={result['entity_win_fraction']:.1%}",flush=True)
        reg[plan.key]={'run_dir':str(dest.resolve()),'plan':meta,'screen_config':work_cfg,
                       'sample_seed':cfg['screening'].get('sample_seed',371)}
        _atomic_json(root/'registry.json',reg)
        summarize_root(root)
    print('\nValidation screening complete for ready datasets. Later scoring is a separate explicit command.')


def export_fev(cfg,dataset):
    from src.experiments.adaptation_benchmark.history import export_fev_snapshot
    e=_dataset(cfg,dataset)[0]
    path=Path(e['snapshot'])
    if path.exists():raise FileExistsError(f'Refusing to replace existing snapshot {path}')
    print(f'FEV preofficial export only: task={e["task"]} -> {path}')
    print('DO NOT run this in PERTSFM_ENV if fev is only available in PERTSFM_DATA_ENV.')
    snap=export_fev_snapshot(e['task'],path)
    print(f'PASS {snap.task}: n={len(snap.records)} horizon={snap.horizon} (historical inputs only)')


def fev_list(filter_text=''):
    import fev
    from src.experiments.adaptation_benchmark.history import FEV_YAML
    bench=fev.Benchmark.from_yaml(FEV_YAML)
    names=sorted(set(t.dataset_config for t in bench.tasks))
    print(f'FEV v0.10.0 dataset_config registry ({len(names)} unique):')
    for name in names:
        if filter_text.lower() in name.lower():print(' ',name)
    print('Only exact registry dataset_config names can be exported with export-fev.')


def score_later(root,dataset):
    from src.experiments.adaptation_benchmark.capability import score_later as phase36_score_later
    reg=_registry(root)
    if dataset not in reg:raise ValueError(f'{dataset} does not have a completed locked validation run')
    print('Later-period results are development diagnostics ONLY, not independent test evidence.')
    return phase36_score_later(reg[dataset]['run_dir'])


def main():
    ap=argparse.ArgumentParser(description='Phase 3.7A adaptation opportunity atlas, preofficial development')
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('preflight');p.add_argument('--config',required=True);p.add_argument('--dataset');p.add_argument('--strict',action='store_true')
    p=sub.add_parser('run');p.add_argument('--config',required=True);p.add_argument('--dataset');p.add_argument('--output-root',default='results/phase37a/atlas_dev_v1')
    p=sub.add_parser('summarize');p.add_argument('--output-root',default='results/phase37a/atlas_dev_v1')
    p=sub.add_parser('score-later');p.add_argument('--dataset',required=True);p.add_argument('--output-root',default='results/phase37a/atlas_dev_v1')
    p=sub.add_parser('export-fev');p.add_argument('--config',required=True);p.add_argument('--dataset',required=True)
    p=sub.add_parser('list-fev');p.add_argument('--filter',default='')
    a=ap.parse_args()
    if a.command=='preflight':_entries(read_config(a.config),a.dataset,a.strict)
    elif a.command=='run':run_screen(a.config,a.output_root,a.dataset)
    elif a.command=='summarize':summarize_root(a.output_root)
    elif a.command=='score-later':score_later(a.output_root,a.dataset)
    elif a.command=='export-fev':export_fev(read_config(a.config),a.dataset)
    elif a.command=='list-fev':fev_list(a.filter)


if __name__=='__main__':main()
