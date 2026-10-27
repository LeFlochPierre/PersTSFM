import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.phase3.PersTSFM_phase3_0_probe_value import task_matrix, simulated_probes, personalization


def synth(k=3, n_entities=8):
    out=[]
    for m in ['chronos2','toto2_313m']:
        for ctx in [96, 384]:
            for e in range(n_entities):
                for lag in range(1, k+1):
                    val=(5+e*.1 + (ctx==384)*(.5 if lag==1 else -.2)
                         +(m=='toto2_313m')*.25 + lag*.1)
                    out.append(dict(task_id='t',dataset_config='example',model=m,
                                    context_steps=ctx,k=k,case_id=f'e{e}_lag{lag}',
                                    entity_id=f'e{e}',lag_before_first_official=lag,
                                    mae=val,mean_pinball_3=val*.35))
    return pd.DataFrame(out)


def test_select_not_leaking_future():
    df=synth(3)
    pivot,configs,k=task_matrix(df,'mae')
    p=simulated_probes(pivot,configs,k,'mae',3,42,[2,4])
    later_rows=df.lag_before_first_official == 1
    changed=df.copy()
    changed.loc[later_rows & changed.model.eq('chronos2'), 'mae']+=1_000_000
    pivot2,configs2,k2=task_matrix(changed,'mae')
    p2=simulated_probes(pivot2,configs2,k2,'mae',3,42,[2,4])
    z=p.loc[~p.strategy.str.startswith('fixed_'),'selected'].tolist()
    z2=p2.loc[~p2.strategy.str.startswith('fixed_'),'selected'].tolist()
    assert z==z2, 'Selection must never depend on lag1 target outcomes'


def test_unmatched_cases_refused():
    df=synth()
    df=df.drop(df.index[-1])
    with pytest.raises(ValueError,match='Unmatched'):
        task_matrix(df,'mae')


def test_k2_support_and_personalization_alpha0_is_global():
    p,configs,k=task_matrix(synth(k=2),'mae')
    trials=simulated_probes(p,configs,k,'mae',5,7,[2,4])
    assert set(trials.sampled_older_lags)=={0,1}
    assert trials[trials.strategy.eq('full_grid_all_older')].probe_case_forecasts.iloc[0] == 8*4
    r=personalization(p,configs,k,'mae')
    assert len(r)==5
    assert abs(r.loc[r.alpha.eq(0),'improvement_over_global_pct'].iloc[0]) < 1e-8


def test_no_na_or_negative():
    df=synth()
    df.iloc[0,df.columns.get_loc('mae')] = np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        task_matrix(df,'mae')


def test_equal_cost_all_cases():
    pivot,configs,k=task_matrix(synth(k=3,n_entities=8),'mae')
    trials=simulated_probes(pivot,configs,k,'mae',5,7,[2,8])
    full=trials[trials.strategy.eq('full_grid_all_older')]
    assert full.probe_case_forecasts.iloc[0]==8*2*4
    short=trials[(trials.strategy.eq('full_grid')) &
                 (trials.sampled_entities.eq(2)) &
                 (trials.sampled_older_lags.eq(1))]
    assert len(short)==5
    assert (short.probe_case_forecasts==8).all()
