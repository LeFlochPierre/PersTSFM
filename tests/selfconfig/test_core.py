import numpy as np

from src.selfconfig.fingerprint import compute_fingerprint
from src.selfconfig.planner import build_plan
from src.selfconfig.search_space import build_search_space


def _seasonal_data(n_series=12, n_steps=1000, period=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    rows = []
    for i in range(n_series):
        phase = rng.uniform(0, 2 * np.pi)
        y = np.sin(2 * np.pi * t / period + phase) + 0.15 * rng.normal(size=n_steps)
        rows.append(y)
    return np.asarray(rows)


def test_fingerprint_detects_periodicity():
    x = _seasonal_data(period=24)
    fp = compute_fingerprint(
        x,
        dataset_name="toy",
        sampling_interval_seconds=3600,
        forecast_horizon_steps=6,
    )
    assert fp.n_series == 12
    assert fp.dominant_period_steps is not None
    assert 20 <= fp.dominant_period_steps <= 28
    assert fp.seasonality_strength > 0.5


def test_search_space_respects_tirex_limit():
    grid = build_search_space(horizon_steps=384)
    tirex = [c.context_steps for c in grid if c.model == "tirex2"]
    assert max(tirex) <= 2048
    assert 3072 not in tirex


def test_plan_prunes_grid():
    x = _seasonal_data(period=24)
    plan = build_plan(
        x,
        dataset_name="toy",
        sampling_interval_seconds=3600,
        forecast_horizon_steps=6,
    )
    assert len(plan.proposed_candidates) > 0
    assert len(plan.proposed_candidates) < len(plan.full_candidates)
    assert 0 < plan.search_reduction < 1
