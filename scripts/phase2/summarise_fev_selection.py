#!/usr/bin/env python3
"""Print full-selection FEv development results without opening test/evaluation data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/phase2/external_forecasts')
    args=p.parse_args()
    records=[]
    for path in sorted(Path(args.root).glob('*/selection/*/context_*/all/selection_score.json')):
        rec=json.loads(path.read_text())
        records.append(rec)
    if not records:
        print('No complete scored selection runs found; run score_fev_selection first.')
        return
    df=pd.DataFrame(records).sort_values(['task','selection_mean_case_mase','inference_seconds'])
    for task,group in df.groupby('task',sort=True):
        print('='*88)
        print(f'FEV development SELECTION only: {task} | {len(group)} configurations')
        print('='*88)
        show=group[['model','context_steps','n_entities','n_cases','selection_mean_case_mae','selection_mean_case_mase','model_load_seconds','inference_seconds']]
        print(show.round(5).to_string(index=False))
        print('Leader (selection MASE):',group.iloc[0]['model'],group.iloc[0]['context_steps'])
        if len(group)>1:
            print('NOTE: leaderboard uses matched selection cases, not later evaluation windows.')
        print()
    print('DEV ONLY. No evaluation-window outcomes were opened by this summary.')

if __name__=='__main__':
    main()
