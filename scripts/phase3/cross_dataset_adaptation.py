#!/usr/bin/env python3
"""PersTSFM Phase 3.4: reproducible cross-dataset Chronos-2 adaptation benchmark.

Use `python -m scripts.phase3.cross_dataset_adaptation --help` from repo root.
Dependencies: data-env for export; model-env for GPU run; neither for plan/score.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.experiments.adaptation_benchmark.history import export_fev_snapshot, file_hash
from src.experiments.adaptation_benchmark.design import load_batch
from src.experiments.adaptation_benchmark.experiment import (
    prepare,describe,run_one,score_later_run,
)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    exp=sub.add_parser('export',help='Export preofficial FEV input snapshot (DATA environment)')
    exp.add_argument('--task',required=True)
    exp.add_argument('--out',type=Path,required=True)
    plan=sub.add_parser('plan',help='Validate all datasets and full grids, no GPU/FEV required')
    plan.add_argument('--config',required=True,type=Path)
    run=sub.add_parser('run',help='GPU runs, cached independent LoRA restarts')
    run.add_argument('--config',required=True,type=Path)
    run.add_argument('--output-root',default='results/phase3/cross_dataset_adaptation',type=Path)
    run.add_argument('--resume',type=Path,help='Resume exactly one matching unfinished run')
    later=sub.add_parser('score-later',help='Score later forecasts AFTER lag2 decision locked')
    later.add_argument('--run',required=True,type=Path)
    show=sub.add_parser('report',help='Print existing locked validation/later results')
    show.add_argument('--run',required=True,type=Path)
    args=p.parse_args(argv)
    if args.command=='export':
        snap=export_fev_snapshot(args.task,args.out)
        print(f'PASS: exported PREOFFICIAL inputs ONLY to {args.out}',flush=True)
        print(f'Entities={len(snap.records)} H={snap.horizon} SHA256={file_hash(args.out)}',flush=True)
        print('No official FEV evaluation/test labels accessed.',flush=True)
    elif args.command in ('plan','run'):
        specs=load_batch(args.config)
        if args.command=='run' and args.resume and len(specs)!=1:
            p.error('--resume requires a config with exactly one experiment')
        for i,spec in enumerate(specs,1):
            print(f'\n[{i}/{len(specs)}] {spec.get("task","? ")}',flush=True)
            if args.command=='plan':
                describe(prepare(spec))
                print('PASS: FEASIBILITY PLAN; no GPU loaded or lag1 truth scored',flush=True)
            else:
                run_one(spec,output_root=args.output_root,resume=args.resume)
    elif args.command=='score-later':
        score_later_run(args.run)
    elif args.command=='report':
        import json
        import pandas as pd
        directory=args.run
        lock=json.loads((directory/'selection.json').read_text())
        valid=pd.read_csv(directory/'validation_strategies.csv')
        print(f"LOCKED FROM LAG2: {lock['strategy']}")
        print('VALIDATION:')
        print(valid.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
        if (directory/'later.csv').exists():
            print('LATER HISTORICAL EVALUATION:')
            print(pd.read_csv(directory/'later.csv').to_string(index=False,float_format=lambda x:f'{x:.5f}'))
        else:
            print('LATER SCORES: not yet revealed (run score-later).')

if __name__=='__main__':
    main()
