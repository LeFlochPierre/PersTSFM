import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('phase33', ROOT/'scripts/phase3/PersTSFM_phase3_3a_chronos_lora.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def cases():
    first = np.arange(100, 110, dtype=np.int64)
    older = dict(entity_id='E',case_id='E_lag2',lag_before_first_official=2,
                 history_timestamps_ns=first,history=np.arange(10.),
                 future_timestamps_ns=np.arange(110, 112),horizon=2)
    later = dict(entity_id='E',case_id='E_lag1',lag_before_first_official=1,
                 history_timestamps_ns=np.arange(100, 112),history=np.arange(12.),
                 future_timestamps_ns=np.arange(112, 114),horizon=2)
    return pd.DataFrame([older,later])


def test_chronological_entity_pairs_pass():
    old,new=m.task_pair_checks(cases(),{'horizon':2})
    assert old.case_id.tolist()==['E_lag2']
    assert new.case_id.tolist()==['E_lag1']


def test_cannot_train_on_validation_or_later_labels_as_prefix():
    rows=cases()
    rows.at[1,'history']=np.arange(12.)+3
    with pytest.raises(RuntimeError,match='Unsafe historical chronology'):
        m.task_pair_checks(rows, {'horizon':2})


def test_no_mismatched_prediction_set():
    expected=cases().query('lag_before_first_official==2')
    with pytest.raises(ValueError,match='case sets'):
        m.score_rows([], {'E_lag2':np.array([1.,2.])}, expected=expected)


def test_selection_uses_validation_only_and_tie_prefers_zero_shot():
    df=pd.DataFrame([dict(model='zero_shot',mae=2.0),dict(model='lora_20',mae=2.0),
                     dict(model='lora_100',mae=2.3)])
    assert m.best_budget(df)=='zero_shot'
    df.loc[df.model=='lora_20','mae']=1.9
    assert m.best_budget(df)=='lora_20'


def test_training_prefix_does_not_see_next_horizon():
    old=cases().query('lag_before_first_official==2')
    out=m.training_series(old,train_history_steps=8,context=3,horizon=2)
    assert len(out)==1
    assert list(out[0])==list(np.arange(2.,10.))


def test_pinball_and_mae_scoring():
    expected=cases().query('lag_before_first_official==2')
    pred=[dict(model='zero_shot',case_id='E_lag2',entity_id='E',lag=2,
               future_timestamps_ns=[110,111],prediction=[2.,2.],q10=[1.,1.],
               q50=[2.,2.],q90=[3.,3.],crossing_elements=0)]
    df=m.score_rows(pred,{'E_lag2':np.array([1.,3.])},expected=expected)
    assert df.mae.iloc[0]==1.0 and np.isfinite(df.mean_pinball_3.iloc[0])


def test_labels_only_loaded_by_lag(monkeypatch,tmp_path):
    # Reject wrong lag before any filesystem access.
    with pytest.raises(ValueError,match='Only lag'):
        m.load_truths(tmp_path,{'task_id':'x'},3,pd.DataFrame())
