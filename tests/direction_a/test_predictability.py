from datetime import datetime, timedelta, timezone
from dataclasses import replace

import numpy as np
import pytest

from src.budgeted_forecasting.cache import Candidate, Case
from src.budgeted_forecasting.predictability import (
    FEATURE_NAMES, HISTORY_STEPS, _cluster_ci, _fit_ridge, _predict_ridge,
    evaluate, history_features, load_electricity_histories,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_cases():
    cases = []
    features = {}
    # Six calibration and six evaluation cases over three entities.
    for j in range(3):
        for i in range(4):
            is_cal = i < 2
            origin = BASE + timedelta(days=10 if is_cal else 30, hours=i)
            a, b = ((2., 4.) if (j + i) % 2 == 0 else (4., 2.))
            case_id = f"e{j}_o{i}"
            cases.append(Case(
                "electricity", f"e{j}", case_id, "calibration" if is_cal else "evaluation",
                origin, origin + timedelta(hours=24), (0.0,),
                {"a": Candidate("a", (a,), 1.0),
                 "b": Candidate("b", (b,), 5.0)},
            ))
            vec = np.full(len(FEATURE_NAMES), float(j) / 3)
            vec[0] += float(i) / 10
            features[case_id] = (vec, 2.)
    return cases, features


def test_history_feature_size_and_scale():
    vec, scale = history_features(np.linspace(0.0, 5.0, HISTORY_STEPS))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vec).all()
    assert scale >= 1.0


def test_only_pre_origin_input_matters():
    original = np.arange(HISTORY_STEPS + 96, dtype=float)
    a = history_features(original[:HISTORY_STEPS])
    original[HISTORY_STEPS:] = -1000
    b = history_features(original[:HISTORY_STEPS])
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]


@pytest.mark.parametrize("bad", [np.ones(HISTORY_STEPS - 1),
                                      np.r_[np.full(HISTORY_STEPS - 1, 2), np.nan],
                                      np.r_[np.full(HISTORY_STEPS - 1, 2), np.inf]])
def test_bad_history_rejected(bad):
    with pytest.raises(ValueError):
        history_features(bad)


def test_flat_history_is_finite():
    x, scale = history_features(np.zeros(HISTORY_STEPS))
    assert np.isfinite(x).all()
    assert scale == 1.0


def test_ridge_works_and_is_deterministic():
    x = np.array([[0., 1.], [1., 2.], [2., 2.], [3., 0.]])
    y = np.array([[1., 2.], [2., 1.], [3., 0.], [4., 2.]])
    m = _fit_ridge(x, y)
    assert _predict_ridge(m, x[0]).shape == (2,)
    assert np.array_equal(_predict_ridge(m, x[1]), _predict_ridge(m, x[1]))


def test_ridge_alpha_must_be_positive():
    with pytest.raises(ValueError):
        _fit_ridge(np.ones((3, 2)), np.ones((3, 2)), 0.)


def test_evaluate_compares_fixed_series_ridge_oracle():
    cases, features = make_cases()
    report, rows = evaluate(cases, features, [1., 5.], bootstrap=30)
    assert report["calibration_cases"] == 6
    assert report["evaluation_entities"] == 3
    assert len(rows) == 12
    assert report["budgets"][0]["eligible_actions"] == ["a"]
    for budget in report["budgets"]:
        assert budget["hindsight_oracle_mae"] <= budget["fixed_mae"] + 1e-9
        assert len(budget["ridge_action_counts"]) <= 2


def test_evaluation_truth_never_changes_action_selection():
    cases, features = make_cases()
    _, original = evaluate(cases, features, [5.], bootstrap=5)
    mutated = [replace(c, truth=(9999.,)) if c.split == "evaluation" else c for c in cases]
    _, after = evaluate(mutated, features, [5.], bootstrap=5)
    assert [(r["ridge_action"], r["entity_action"], r["fixed_action"]) for r in original] == [
        (r["ridge_action"], r["entity_action"], r["fixed_action"]) for r in after
    ]


def test_calibration_labels_must_be_mature():
    cases, features = make_cases()
    changed = [replace(c, label_available_at=BASE + timedelta(days=50))
               if c.split == "calibration" else c for c in cases]
    with pytest.raises(ValueError, match="unavailable"):
        evaluate(changed, features, [5.], bootstrap=5)


def test_feature_keys_exact_and_finite():
    cases, features = make_cases()
    bad = dict(features)
    bad.pop("e0_o0")
    with pytest.raises(ValueError, match="keys"):
        evaluate(cases, bad, [5.], bootstrap=5)
    bad = dict(features)
    bad["e0_o0"] = (np.full(len(FEATURE_NAMES), np.nan), 1.)
    with pytest.raises(ValueError, match="Invalid history"):
        evaluate(cases, bad, [5.], bootstrap=5)


def test_ci_deterministic():
    rows = [{"series_id": f"e{i}", "fixed_mae": 3., "ridge_mae": 2.} for i in range(4)]
    assert _cluster_ci(rows, bootstrap=40, seed=17) == _cluster_ci(rows, bootstrap=40, seed=17)


def test_ci_requires_three_entities():
    rows = [{"series_id": "e1", "fixed_mae": 2., "ridge_mae": 1.}]
    assert _cluster_ci(rows, bootstrap=10, seed=42) is None


def test_reject_unsupported_dataset_before_io():
    cases, _ = make_cases()
    with pytest.raises(ValueError, match="electricity"):
        load_electricity_histories([replace(cases[0], dataset_id="solar")])


def test_electricity_history_reconstruction_excludes_future(monkeypatch):
    import sys
    import types
    import pandas as pd

    origin_naive = pd.Timestamp("2026-01-20 00:00:00")
    index = pd.date_range(end=origin_naive + pd.Timedelta(hours=24),
                          periods=HISTORY_STEPS + 96, freq="15min")
    # The last 96 points represent the future target and must NOT be read.
    # Here, the pre-origin history is precisely the first 1344 points.
    values = np.r_[np.linspace(2, 5, HISTORY_STEPS), np.full(96, 999.0)]
    series = pd.Series(values, index=index)
    assert series.index[HISTORY_STEPS - 1] == origin_naive

    fake_registry = types.ModuleType("src.data.registry")
    fake_registry.get_dataset_spec = lambda key: types.SimpleNamespace(
        key=key, resolution_minutes=15, forecast_steps=96
    )
    fake_registry._load_electricity = lambda spec, ids: {"e0": series}
    monkeypatch.setitem(sys.modules, "src.data.registry", fake_registry)
    case = Case(
        "electricity", "e0", "case_0", "evaluation",
        origin_naive.tz_localize("UTC").to_pydatetime(),
        (origin_naive + pd.Timedelta(hours=24, minutes=15)).tz_localize("UTC").to_pydatetime(),
        tuple(np.zeros(96)), {"a": Candidate("a", tuple(np.ones(96)), 1.)}
    )
    observed = load_electricity_histories([case])["case_0"]
    expected = history_features(values[:HISTORY_STEPS])
    assert np.array_equal(observed[0], expected[0])
    assert observed[1] == expected[1]
    # Mutating future observations must not affect reconstructed features.
    series.iloc[HISTORY_STEPS:] = -9999
    after = load_electricity_histories([case])["case_0"]
    assert np.array_equal(observed[0], after[0])
