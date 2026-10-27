"""Context-local missing-data tests: strict finite model inputs, no imputation."""
import numpy as np
import pandas as pd
import pytest
from scripts.phase2.run_preofficial_calibration import choose_cases, make_forecast_input

HOUR = 3_600_000_000_000

def case(nan_index):
    n, context = 64, 16
    t = 1_700_000_000_000_000_000 + np.arange(n, dtype=np.int64) * HOUR
    y = np.arange(n, dtype=float)
    y[nan_index] = np.nan
    row = dict(task_id='synthetic', entity_id='e', case_id='synthetic__e__lag01',
               lag_before_first_official=1, feasible_contexts=[16,32],
               history=y.tolist(), observed_steps_saved=n, history_missing_saved=1,
               history_timestamps_ns=t.tolist(),
               future_timestamps_ns=(t[-1] + np.arange(1,5)*HOUR).tolist(),
               horizon=4)
    return row

def test_missing_far_outside_context_is_allowed_without_filling():
    row = case(0)
    selected = choose_cases(pd.DataFrame([row]), k=1, context=16)
    x, stamps, future, cadence = make_forecast_input(selected.iloc[0].to_dict(),16)
    assert len(selected) == 1
    assert len(x) == 16 and x[0] == 48 and np.isfinite(x).all()
    assert cadence == 60
    assert np.isnan(row['history'][0])

def test_missing_in_actual_context_still_rejected_in_both_paths():
    row = case(63)
    with pytest.raises(ValueError,match='Missing input observations inside'):
        choose_cases(pd.DataFrame([row]),k=1,context=16)
    with pytest.raises(ValueError,match='nonfinite history'):
        make_forecast_input(row,16)

def test_context_dependence_is_strict():
    row = case(40)  # 24 points back: outside 16 but inside 32
    assert len(choose_cases(pd.DataFrame([row]),k=1,context=16)) == 1
    with pytest.raises(ValueError,match='Missing input observations inside'):
        choose_cases(pd.DataFrame([row]),k=1,context=32)
