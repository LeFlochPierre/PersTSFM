#!/usr/bin/env python3
"""PersTSFM Phase 3.5: rolling-origin evaluation and LoRA input-pool provenance.

Run from repository root: python -m scripts.phase3.rolling_adaptation --help
Only 'run' loads the GPU; the other commands work with CPU and saved snapshots.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.experiments.adaptation_benchmark.rolling import (
    load_rolling_config,make_rolling_plan,describe_rolling,initialize_suite,
    load_suite,run_suite,score_suite,
)
from src.experiments.adaptation_benchmark.experiment import prepare
from src.experiments.adaptation_benchmark.reporting import report_suite
from src.experiments.adaptation_benchmark.sampler_audit import print_sampler_audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','init'):
        cmd=sub.add_parser(name,help='Check rolling fold feasibility' if name=='plan' else 'Freeze a rolling suite with per-fold snapshots')
        cmd.add_argument('--config',required=True,type=Path)
        if name=='init': cmd.add_argument('--out',required=True,type=Path)
    run=sub.add_parser('run',help='GPU train/validate selected or all folds, with safe resume')
    run.add_argument('--suite',required=True,type=Path)
    run.add_argument('--fold',type=int,default=None,help='Optional zero-indexed fold to run')
    run.add_argument('--trace-sampler',action='store_true',
                     help='Opt-in, version-guarded real Chronos crop tracing (single-process only)')
    score=sub.add_parser('score',help='Require ALL folds locked before revealing later-period metrics')
    score.add_argument('--suite',required=True,type=Path)
    report=sub.add_parser('report',help='Read already-computed rolling summary; cannot select strategies')
    report.add_argument('--suite',required=True,type=Path)
    audit=sub.add_parser('audit-pools',help='Audit eligible C+H training windows, not realized sampler draws')
    audit.add_argument('--suite',required=True,type=Path)
    audit.add_argument('--fold',type=int,default=0)
    audit.add_argument('--context',type=int,required=True)
    audit.add_argument('--history',default='full',help='integer or full')
    args=parser.parse_args(argv)
    if args.command in ('plan','init'):
        p=make_rolling_plan(load_rolling_config(args.config))
        describe_rolling(p)
        if args.command=='init': initialize_suite(p,args.out)
    elif args.command=='run':
        run_suite(args.suite,only_fold=args.fold,trace_sampler=args.trace_sampler)
    elif args.command=='score':
        score_suite(args.suite)
    elif args.command=='report':
        _,m=load_suite(args.suite)
        if not (args.suite/'suite_freeze.json').exists():
            parser.error('All folds must be frozen/scored before reading later summary')
        report_suite(args.suite)
    elif args.command=='audit-pools':
        _,m=load_suite(args.suite)
        if args.fold not in range(len(m['folds'])):
            parser.error('Invalid fold')
        spec=json.loads(Path(m['folds'][args.fold]['config']).read_text())['experiments'][0]
        p=prepare(spec)
        history=(min(len(r.y)-2*p.snapshot.horizon for r in p.records)
                 if args.history=='full' else int(args.history))
        print_sampler_audit(p,context=args.context,history=history)

if __name__=='__main__':
    main()
