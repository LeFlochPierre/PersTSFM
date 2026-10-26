from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True)
class RegimeFingerprint:
    recent_window_steps: int
    profile_window_steps: int

    recent_vs_prior_mean_shift: float
    recent_vs_prior_scale_shift: float
    recent_vs_prior_iqr_shift: float
    recent_vs_prior_quantile_shift: float

    rolling_mean_variability: float
    rolling_scale_variability: float
    adjacent_mean_shift: float
    window_mean_persistence: float

    usable_series_fraction: float
    median_windows_per_series: float

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _iqr(x: np.ndarray) -> float:
    x = _finite(x)
    if x.size == 0:
        return float("nan")
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def _safe_std(x: np.ndarray) -> float:
    x = _finite(x)
    if x.size < 2:
        return float("nan")
    return float(np.std(x))


def _safe_median(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _safe_corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")

    a = a[mask]
    b = b[mask]

    if np.std(a) <= _EPS or np.std(b) <= _EPS:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def _normalised_mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    a = _finite(a)
    b = _finite(b)

    if a.size < 4 or b.size < 4:
        return float("nan")

    pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)

    return float(
        abs(np.mean(a) - np.mean(b))
        / max(pooled, _EPS)
    )


def _scale_shift(a: np.ndarray, b: np.ndarray) -> float:
    sa = _safe_std(a)
    sb = _safe_std(b)

    if not np.isfinite(sa) or not np.isfinite(sb):
        return float("nan")

    return float(
        abs(
            np.log(
                (sa + _EPS)
                / (sb + _EPS)
            )
        )
    )


def _iqr_shift(a: np.ndarray, b: np.ndarray) -> float:
    ia = _iqr(a)
    ib = _iqr(b)

    if not np.isfinite(ia) or not np.isfinite(ib):
        return float("nan")

    return float(
        abs(
            np.log(
                (ia + _EPS)
                / (ib + _EPS)
            )
        )
    )


def _quantile_shift(a: np.ndarray, b: np.ndarray) -> float:
    a = _finite(a)
    b = _finite(b)

    if a.size < 8 or b.size < 8:
        return float("nan")

    qa = np.percentile(a, [10, 50, 90])
    qb = np.percentile(b, [10, 50, 90])

    scale = max(_iqr(b), _EPS)

    return float(
        np.mean(np.abs(qa - qb))
        / scale
    )


def compute_regime_fingerprint(
    histories: Any,
    *,
    horizon_steps: int,
    sampling_interval_seconds: float,
    max_series: int = 256,
    seed: int = 42,
) -> RegimeFingerprint:
    """Compute slow-state/distributional features from leakage-safe histories."""

    H = int(horizon_steps)

    if H <= 0:
        raise ValueError("horizon_steps must be positive")

    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be positive")

    recent_steps = max(8, 4 * H)
    profile_steps = recent_steps
    max_profile_span = 64 * H

    series = []

    for item in histories:
        if hasattr(item, "to_numpy"):
            item = item.to_numpy()

        arr = np.asarray(item, dtype=float).reshape(-1)

        if arr.size >= 3 * recent_steps:
            series.append(arr)

    if not series:
        raise ValueError(
            "No sufficiently long histories for regime fingerprint."
        )

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

    mean_shifts = []
    scale_shifts = []
    iqr_shifts = []
    quantile_shifts = []

    rolling_mean_vars = []
    rolling_scale_vars = []
    adjacent_mean_shifts = []
    window_mean_persistences = []
    n_windows_all = []

    usable = 0

    for x in series:
        recent = x[-recent_steps:]

        prior_end = x.size - recent_steps
        prior_start = max(0, prior_end - 28 * H)
        prior = x[prior_start:prior_end]

        if (
            np.isfinite(recent).sum() >= 8
            and np.isfinite(prior).sum() >= 8
        ):
            usable += 1

            mean_shifts.append(
                _normalised_mean_shift(recent, prior)
            )
            scale_shifts.append(
                _scale_shift(recent, prior)
            )
            iqr_shifts.append(
                _iqr_shift(recent, prior)
            )
            quantile_shifts.append(
                _quantile_shift(recent, prior)
            )

        tail = x[-min(x.size, max_profile_span):]

        means = []
        scales = []

        end = tail.size

        while end - profile_steps >= 0:
            w = tail[end - profile_steps:end]
            finite = _finite(w)

            if finite.size >= 8:
                means.append(float(np.mean(finite)))
                scales.append(float(np.std(finite)))

            end -= profile_steps

        means = means[::-1]
        scales = scales[::-1]

        n_windows_all.append(len(means))

        if len(means) >= 3:
            means_arr = np.asarray(means, dtype=float)
            scales_arr = np.asarray(scales, dtype=float)

            overall = _finite(tail)
            overall_scale = max(
                _safe_std(overall),
                _EPS,
            )

            rolling_mean_vars.append(
                float(
                    np.std(means_arr)
                    / overall_scale
                )
            )

            median_scale = max(
                float(
                    np.median(
                        np.abs(scales_arr)
                    )
                ),
                _EPS,
            )

            rolling_scale_vars.append(
                float(
                    np.std(scales_arr)
                    / median_scale
                )
            )

            adjacent_mean_shifts.append(
                float(
                    np.median(
                        np.abs(
                            np.diff(means_arr)
                        )
                    )
                    / overall_scale
                )
            )

            window_mean_persistences.append(
                _safe_corr(
                    means_arr[:-1],
                    means_arr[1:],
                )
            )

    return RegimeFingerprint(
        recent_window_steps=int(recent_steps),
        profile_window_steps=int(profile_steps),
        recent_vs_prior_mean_shift=_safe_median(mean_shifts),
        recent_vs_prior_scale_shift=_safe_median(scale_shifts),
        recent_vs_prior_iqr_shift=_safe_median(iqr_shifts),
        recent_vs_prior_quantile_shift=_safe_median(quantile_shifts),
        rolling_mean_variability=_safe_median(rolling_mean_vars),
        rolling_scale_variability=_safe_median(rolling_scale_vars),
        adjacent_mean_shift=_safe_median(adjacent_mean_shifts),
        window_mean_persistence=_safe_median(window_mean_persistences),
        usable_series_fraction=float(
            usable / max(len(series), 1)
        ),
        median_windows_per_series=_safe_median(n_windows_all),
    )
