import numpy as np

from src.selfconfig.fingerprint import (
    compute_fingerprint_ragged,
)


def test_ragged_fingerprint():
    x = np.linspace(
        0.0,
        10.0,
        400,
    )
    histories = [
        np.sin(x),
        np.sin(x[:300]),
        np.sin(x[:250]),
    ]

    fp = compute_fingerprint_ragged(
        histories,
        dataset_name="ragged",
        sampling_interval_seconds=60,
        forecast_horizon_steps=12,
    )

    assert fp.n_series == 3
    assert fp.min_series_length == 250
    assert fp.max_series_length == 400
    assert np.isfinite(fp.acf_lag1)
