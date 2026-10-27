"""Regression tests: FEV datetime precision must never silently change stored units."""
import numpy as np
import pandas as pd
import pytest

from scripts.phase2.freeze_fev_protocol import timestamps_ns
from scripts.phase2.run_fev_selection import resolution_minutes
from scripts.phase2.score_fev_selection import truth_timestamps_ns


@pytest.mark.parametrize('source_unit', ['s', 'ms', 'us', 'ns'])
def test_datetime_precision_is_normalized_to_ns(source_unit):
    dates = pd.date_range('2018-06-18 20:00:00', periods=4, freq='h', tz='UTC').as_unit(source_unit)
    history = timestamps_ns(dates[:2], 'history')
    targets = truth_timestamps_ns(dates[2:])
    hour_ns = 3_600_000_000_000
    assert np.all(np.diff(history) == hour_ns)
    assert targets[0] - history[-1] == hour_ns
    assert np.all(np.diff(targets) == hour_ns)
    record = {'history_timestamps_ns': history.tolist()}
    assert resolution_minutes(record) == 60


def test_different_source_precisions_are_identical():
    dt = pd.date_range('2021-08-24', periods=3, freq='h', tz='UTC')
    forms = [timestamps_ns(dt.as_unit(u), 'test') for u in ['s', 'ms', 'us', 'ns']]
    for arr in forms[1:]:
        np.testing.assert_array_equal(arr, forms[0])
