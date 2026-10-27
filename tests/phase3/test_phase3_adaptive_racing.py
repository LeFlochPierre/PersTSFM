import numpy as np
import pandas as pd
from scripts.phase3.PersTSFM_phase3_1_adaptive_racing import analyze_one


def fixture():
    names = [(m, c) for m in ('chronos2', 'moirai2_small', 'toto2_313m') for c in (96, 384)]
    index = pd.MultiIndex.from_product(
        [[str(i) for i in range(10)], [1, 2, 3]],
        names=['entity_id', 'lag_before_first_official'],
    )
    cols = pd.MultiIndex.from_tuples(names, names=['model', 'context_steps'])
    data = [[8 + j + 0.25 * int(e) + lag for j in range(6)] for e, lag in index]
    return pd.DataFrame(data, index=index, columns=cols), names


def test_cost_accounting():
    pivot, names = fixture()
    result = analyze_one(pivot, names, 'mae', 'dummy', 4, 5, 42, 2)
    assert result.groupby('repeat').size().eq(8).all()
    assert result[result.strategy == 'adaptive_top2'].probe_case_forecasts.eq(18).all()
    assert result[result.strategy == 'full_grid'].probe_case_forecasts.eq(24).all()
    assert result[result.strategy == 'longest_only'].probe_case_forecasts.eq(12).all()
    assert (result.regret_pct >= -1e-7).all()
    for rep, frame in result.groupby('repeat'):
        cost = frame.set_index('strategy')['probe_case_forecasts']
        assert cost['longest_cost_matched_top2'] <= cost['adaptive_top2']
        assert cost['full_cost_matched_top2'] <= cost['adaptive_top2']


def test_future_labels_cannot_change_selection():
    pivot, names = fixture()
    other = pivot.copy()
    other.loc[pd.IndexSlice[:, 1], :] = other.loc[pd.IndexSlice[:, 1], :].to_numpy()[:, ::-1] + 15
    x = analyze_one(pivot, names, 'mae', 'dummy', 4, 5, 42, 1)
    y = analyze_one(other, names, 'mae', 'dummy', 4, 5, 42, 1)
    cols = ['repeat', 'strategy', 'selected', 'probe_case_forecasts']
    pd.testing.assert_frame_equal(x[cols].reset_index(drop=True), y[cols].reset_index(drop=True))


def test_deterministic_seeding_and_later_nonnegative():
    pivot, names = fixture()
    x = analyze_one(pivot, names, 'mean_pinball_3', 'dummy', 8, 7, 123, 2)
    y = analyze_one(pivot, names, 'mean_pinball_3', 'dummy', 8, 7, 123, 2)
    pd.testing.assert_frame_equal(x, y)
    assert np.isfinite(x['later_loss']).all()
    assert (x['regret_pct'] >= -1e-7).all()
