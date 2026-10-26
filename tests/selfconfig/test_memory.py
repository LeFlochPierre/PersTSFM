import numpy as np

from src.selfconfig.memory import (
    compute_temporal_memory_fingerprint,
)


def test_temporal_memory_detects_persistence():
    rng = np.random.default_rng(7)

    histories = []

    for _ in range(10):
        x = np.zeros(2000)
        noise = rng.normal(
            scale=0.2,
            size=x.size,
        )

        for t in range(1, x.size):
            x[t] = (
                0.95 * x[t - 1]
                + noise[t]
            )

        histories.append(x)

    memory = (
        compute_temporal_memory_fingerprint(
            histories,
            horizon_steps=6,
            sampling_interval_seconds=60,
        )
    )

    assert memory.acf_h > 0
    assert memory.memory_lag_02_steps is not None
    assert memory.memory_lag_02_steps >= 6
