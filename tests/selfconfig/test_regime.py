import numpy as np

from src.selfconfig.regime import (
    compute_regime_fingerprint,
)


def test_regime_fingerprint_detects_shift():
    rng = np.random.default_rng(9)

    histories = []

    for _ in range(8):
        x = rng.normal(
            0.0,
            1.0,
            size=1000,
        )
        x[-100:] += 3.0
        histories.append(x)

    fp = compute_regime_fingerprint(
        histories,
        horizon_steps=12,
        sampling_interval_seconds=300,
    )

    assert fp.recent_vs_prior_mean_shift > 0.5
    assert fp.usable_series_fraction > 0.9
