import numpy as np

from src.selfconfig.capacity import (
    compute_history_capacity,
)


def test_history_capacity_respects_quantile_and_budget():
    histories = [
        np.zeros(100),
        np.zeros(200),
        np.zeros(300),
        np.zeros(400),
    ]

    cap = compute_history_capacity(
        histories,
        horizon_steps=10,
        support_quantile=0.25,
        resource_limit_steps=4096,
    )

    assert cap.p25_steps == 175
    assert cap.robust_history_limit_steps == 175
    assert cap.effective_limit_steps == 175
    assert cap.effective_limit_horizons == 17.5


def test_history_capacity_respects_resource_limit():
    histories = [
        np.zeros(10000)
        for _ in range(5)
    ]

    cap = compute_history_capacity(
        histories,
        horizon_steps=100,
        support_quantile=0.25,
        resource_limit_steps=4096,
    )

    assert cap.robust_history_limit_steps == 10000
    assert cap.effective_limit_steps == 4096
