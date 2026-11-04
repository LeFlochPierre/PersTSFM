"""Phase 3.6 explicit Chronos-2 LoRA/full fine-tuning capability audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.experiments.adaptation_benchmark.capability import make_plan,print_plan,run,score_later


def main():
    p=argparse.ArgumentParser(description='Phase 3.6, explicit audited adaptation schedules, historical development only')
    sub=p.add_subparsers(dest='action',required=True)
    for name in ('plan','run'):
        s=sub.add_parser(name)
        s.add_argument('--config',type=Path,required=True)
        if name=='run':
            s.add_argument('--output-root',type=Path,default=Path('results/phase3/capability'))
            s.add_argument('--resume',type=Path)
    s=sub.add_parser('score-later')
    s.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    if a.action=='score-later':
        score_later(a.run)
        return
    cfg=json.loads(a.config.read_text())
    if a.action=='plan':
        plan,cfg,history=make_plan(cfg)
        print_plan(plan,cfg,history)
    else:
        run(cfg,output_root=a.output_root,resume=a.resume)

if __name__=='__main__':
    main()
