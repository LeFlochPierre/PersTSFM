from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class TemporalMemoryFingerprint:
    """Multi-scale temporal dependence summary.

    `memory_lag_02_steps` and `memory_lag_01_steps` are the largest tested
    horizon-relative lags at which the median absolute ACF across series is
    still at least 0.20 or 0.10, respectively.
    """

    acf_h: float
    acf_2h: float
    acf_4h: float
    acf_8h: float
    acf_16h: float
    acf_32h: float

    acf_period: float | None
    acf_2period: float | None

    memory_lag_02_steps: int | None
    memory_lag_01_steps: int | None

    memory_lag_02_seconds: float | None
    memory_lag_01_seconds: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _pairwise_acf(values: np.ndarray, lag: int) -> float:
    lag = int(lag)

    if lag <= 0:
        raise ValueError("lag must be positive")

    x = np.asarray(values, dtype=float).reshape(-1)

    if x.size <= lag + 2:
        return float("nan")

    a = x[:-lag]
    b = x[lag:]

    mask = np.isfinite(a) & np.isfinite(b)

    if int(mask.sum()) < 8:
        return float("nan")

    a = a[mask]
    b = b[mask]

    if np.std(a) <= _EPS or np.std(b) <= _EPS:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def _median_acf(histories: list[np.ndarray], lag: int) -> float:
    values = np.asarray(
        [_pairwise_acf(x, lag) for x in histories],
        dtype=float,
    )
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.median(values))


def _safe_abs(value: float | None) -> float:
    if value is None:
        return float("nan")

    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(value):
        return float("nan")

    return abs(value)


def compute_temporal_memory_fingerprint(
    histories: Any,
    *,
    horizon_steps: int,
    sampling_interval_seconds: float,
    dominant_period_steps: float | None = None,
    max_series: int = 256,
    max_points_per_series: int = 32768,
    seed: int = 42,
) -> TemporalMemoryFingerprint:
    """Compute multi-scale temporal dependence from leakage-safe histories."""

    H = int(horizon_steps)

    if H <= 0:
        raise ValueError("horizon_steps must be positive")

    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be positive")

    series: list[np.ndarray] = []

    for item in histories:
        if hasattr(item, "to_numpy"):
            item = item.to_numpy()

        arr = np.asarray(item, dtype=float).reshape(-1)

        if arr.size >= 8:
            series.append(arr)

    if not series:
        raise ValueError("No usable histories supplied.")

    rng = np.random.default_rng(seed)

    if len(series) > max_series:
        idx = np.sort(
            rng.choice(
                len(series),
                size=max_series,
                replace=False,
            )
        )
        series = [series[int(i)] for i in idx]

    trimmed = [
        x[-max_points_per_series:]
        if x.size > max_points_per_series
        else x
        for x in series
    ]

    multipliers = (1, 2, 4, 8, 16, 32)

    acf_by_multiplier: dict[int, float] = {}

    for multiplier in multipliers:
        lag = H * multiplier
        acf_by_multiplier[multiplier] = _median_acf(
            trimmed,
            lag,
        )

    acf_period = None
    acf_2period = None

    if (
        dominant_period_steps is not None
        and np.isfinite(dominant_period_steps)
        and dominant_period_steps >= 2
    ):
        P = int(round(float(dominant_period_steps)))

        acf_period_value = _median_acf(trimmed, P)
        acf_2period_value = _median_acf(trimmed, 2 * P)

        if np.isfinite(acf_period_value):
            acf_period = float(acf_period_value)

        if np.isfinite(acf_2period_value):
            acf_2period = float(acf_2period_value)

    def persistence_lag(threshold: float) -> int | None:
        eligible: list[int] = []

        for multiplier in multipliers:
            value = _safe_abs(acf_by_multiplier[multiplier])

            if np.isfinite(value) and value >= threshold:
                eligible.append(H * multiplier)

        if not eligible:
            return None

        return int(max(eligible))

    lag02 = persistence_lag(0.20)
    lag01 = persistence_lag(0.10)

    return TemporalMemoryFingerprint(
        acf_h=float(acf_by_multiplier[1]),
        acf_2h=float(acf_by_multiplier[2]),
        acf_4h=float(acf_by_multiplier[4]),
        acf_8h=float(acf_by_multiplier[8]),
        acf_16h=float(acf_by_multiplier[16]),
        acf_32h=float(acf_by_multiplier[32]),
        acf_period=acf_period,
        acf_2period=acf_2period,
        memory_lag_02_steps=lag02,
        memory_lag_01_steps=lag01,
        memory_lag_02_seconds=(
            float(lag02 * sampling_interval_seconds)
            if lag02 is not None
            else None
        ),
        memory_lag_01_seconds=(
            float(lag01 * sampling_interval_seconds)
            if lag01 is not None
            else None
        ),
    )
