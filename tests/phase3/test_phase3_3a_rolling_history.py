"""Regression cases for causal rolling windows used by Phase 2 exporter."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('phase33_fixed',BASE/'scripts/phase3/PersTSFM_phase3_3a_chronos_lora.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def create_cases(*, rolling, h=3, length=12):
    # Source series is length 15; old cutoff=12, new cutoff=15.
    t_old=np.arange(100,100+length,dtype=np.int64)
    t_new=np.arange(100+h if rolling else 100,100+length+h,dtype=np.int64)
    if not rolling:
        t_new=np.arange(100,100+length+h,dtype=np.int64)
    older=dict(entity_id='E',case_id='old',lag_before_first_official=2,
        history_timestamps_ns=t_old,history=t_old.astype(float),
        future_timestamps_ns=np.arange(100+length,100+length+h,dtype=np.int64))
    newer=dict(entity_id='E',case_id='new',lag_before_first_official=1,
        history_timestamps_ns=t_new,history=t_new.astype(float),
        future_timestamps_ns=np.arange(100+length+h,100+length+2*h,dtype=np.int64))
    return pd.DataFrame([older,newer])


def test_capped_rolling_windows_accept():
    old,new = m.task_pair_checks(create_cases(rolling=True), {'horizon':3})
    assert len(old)==len(new)==1


def test_uncapped_prefix_windows_accept():
    old,new = m.task_pair_checks(create_cases(rolling=False),{'horizon':3})
    assert len(old)==len(new)==1


def test_reject_shifted_observation():
    df=create_cases(rolling=True)
    ys=df.at[1,'history'].copy()
    ys[1]+=0.2
    df.at[1,'history']=ys
    with pytest.raises(RuntimeError,match='Unsafe historical chronology'):
        m.task_pair_checks(df,{'horizon':3})


def test_reject_missing_lag2_target_in_newer_history():
    df=create_cases(rolling=True)
    ts=df.at[1,'history_timestamps_ns'].copy()
    ts[-1]+=1
    df.at[1,'history_timestamps_ns']=ts
    with pytest.raises(RuntimeError,match='Unsafe historical chronology'):
        m.task_pair_checks(df,{'horizon':3})


def test_reject_overlap_with_changed_time_range():
    df=create_cases(rolling=True)
    ts=df.at[1,'history_timestamps_ns'].copy()+1
    df.at[1,'history_timestamps_ns']=ts
    with pytest.raises(RuntimeError,match='Unsafe historical chronology'):
        m.task_pair_checks(df,{'horizon':3})


def test_training_history_more_than_export_refused():
    df=create_cases(rolling=True)
    older=df.iloc[:1]
    with pytest.raises(RuntimeError,match='only 12 saved'):
        m.training_series(older,train_history_steps=20,context=4,horizon=3)


def test_train_data_ends_before_validation():
    df=create_cases(rolling=True)
    data=m.training_series(df.iloc[:1],train_history_steps=12,context=4,horizon=3)
    assert data[0][-1]==111
    assert 112 not in data[0]
