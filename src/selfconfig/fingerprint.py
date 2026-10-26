from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class FingerprintConfig:
    max_series: int = 256
    max_points_per_series: int = 8192
    seed: int = 42


@dataclass(frozen=True)
class DatasetFingerprint:
    dataset_name: str
    n_series: int
    n_variables: int
    sampling_interval_seconds: float
    forecast_horizon_steps: int
    forecast_horizon_seconds: float
    median_series_length: float
    min_series_length: int
    max_series_length: int
    missing_fraction: float
    zero_fraction: float
    constant_series_fraction: float
    value_median: float
    value_iqr: float
    skewness: float
    excess_kurtosis: float
    coefficient_of_variation: float
    acf_lag1: float
    acf_horizon: float
    trend_strength: float
    dominant_period_steps: float | None
    dominant_period_seconds: float | None
    seasonality_strength: float
    spectral_entropy: float
    between_series_scale_variability: float
    between_series_mean_variability: float
    between_series_acf_variability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _as_2d(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected a 1D/2D array-like object, got shape={x.shape}.")
    return x


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _safe_mean(values: list[float] | np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def _safe_median(values: list[float] | np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _moment_skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    x = _finite(x)
    if x.size < 4:
        return float("nan"), float("nan")
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std <= _EPS:
        return 0.0, 0.0
    z = (x - mean) / std
    skew = float(np.mean(z**3))
    excess_kurtosis = float(np.mean(z**4) - 3.0)
    return skew, excess_kurtosis


def _interpolate_missing(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return None
    idx = np.arange(x.size)
    out = x.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return out


def _acf(x: np.ndarray, lag: int) -> float:
    if lag <= 0:
        return float("nan")
    y = _interpolate_missing(x)
    if y is None or y.size <= lag + 2:
        return float("nan")
    a = y[:-lag]
    b = y[lag:]
    if np.std(a) <= _EPS or np.std(b) <= _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _trend_strength(x: np.ndarray) -> float:
    y = _interpolate_missing(x)
    if y is None or y.size < 4:
        return float("nan")
    t = np.arange(y.size, dtype=float)
    t -= t.mean()
    y0 = y - y.mean()
    denom = float(np.dot(t, t))
    if denom <= _EPS:
        return 0.0
    slope = float(np.dot(t, y0) / denom)
    fitted = slope * t
    sst = float(np.dot(y0, y0))
    if sst <= _EPS:
        return 0.0
    r2 = float(np.dot(fitted, fitted) / sst)
    return float(np.clip(r2, 0.0, 1.0))


def _spectral_features(x: np.ndarray) -> tuple[float | None, float]:
    y = _interpolate_missing(x)
    if y is None or y.size < 8:
        return None, float("nan")

    # Remove mean and linear trend so DC/slow drift do not dominate the period estimate.
    t = np.arange(y.size, dtype=float)
    coef = np.polyfit(t, y, deg=1)
    y = y - np.polyval(coef, t)
    if np.std(y) <= _EPS:
        return None, 0.0

    spectrum = np.fft.rfft(y)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(y.size, d=1.0)

    # Remove DC and periods shorter than 2 samples.
    valid = (freqs > 0.0) & (freqs <= 0.5)
    power = power[valid]
    freqs = freqs[valid]
    if power.size == 0 or float(power.sum()) <= _EPS:
        return None, 0.0

    p = power / power.sum()
    entropy = -float(np.sum(p * np.log(p + _EPS))) / math.log(max(len(p), 2))

    peak = int(np.argmax(power))
    freq = float(freqs[peak])
    period = (1.0 / freq) if freq > 0 else None
    return period, float(np.clip(entropy, 0.0, 1.0))


def _sample_rows(x: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    rows = np.arange(x.shape[0])
    if rows.size > cfg.max_series:
        rows = np.sort(rng.choice(rows, size=cfg.max_series, replace=False))
    return x[rows]


def _truncate_series(row: np.ndarray, max_points: int) -> np.ndarray:
    if row.size <= max_points:
        return row
    # Preserve recent history; this also bounds FFT/autocorrelation cost.
    return row[-max_points:]


def compute_fingerprint(
    values: Any,
    *,
    dataset_name: str,
    sampling_interval_seconds: float,
    forecast_horizon_steps: int,
    n_variables: int = 1,
    config: FingerprintConfig | None = None,
) -> DatasetFingerprint:
    """Compute an interpretable dataset-level fingerprint from training data only.

    Parameters
    ----------
    values:
        Array-like with shape (n_series, n_timepoints). For a single series,
        a 1D array is accepted. Missing values should be NaN.
    dataset_name:
        Human-readable task/dataset name.
    sampling_interval_seconds:
        Sampling interval after any task-specific regularisation/resampling.
    forecast_horizon_steps:
        Forecast horizon in samples.
    n_variables:
        Number of input/target variables in the forecasting task. V1 computes
        temporal statistics on the supplied target matrix but stores this count.
    config:
        Controls deterministic subsampling for large datasets.
    """
    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be > 0")
    if forecast_horizon_steps <= 0:
        raise ValueError("forecast_horizon_steps must be > 0")

    cfg = config or FingerprintConfig()
    x = _as_2d(values)
    n_series = int(x.shape[0])

    finite_counts = np.sum(np.isfinite(x), axis=1)
    valid_lengths = finite_counts[finite_counts > 0]
    if valid_lengths.size == 0:
        raise ValueError("No finite observations found in values.")

    flattened = _finite(x)
    missing_fraction = float(1.0 - np.isfinite(x).mean())
    zero_fraction = float(np.mean(flattened == 0.0)) if flattened.size else float("nan")

    per_series_std = np.nanstd(x, axis=1)
    constant_mask = np.isfinite(per_series_std) & (per_series_std <= _EPS)
    constant_series_fraction = float(np.mean(constant_mask))

    q25, q75 = np.percentile(flattened, [25, 75])
    value_iqr = float(q75 - q25)
    skewness, excess_kurtosis = _moment_skew_kurtosis(flattened)
    mean_abs = abs(float(np.mean(flattened)))
    coefficient_of_variation = float(np.std(flattened) / max(mean_abs, _EPS))

    sampled = _sample_rows(x, cfg)
    acf1_values: list[float] = []
    acfh_values: list[float] = []
    trend_values: list[float] = []
    period_values: list[float] = []
    entropy_values: list[float] = []
    seasonal_acf_values: list[float] = []
    series_means: list[float] = []
    series_iqrs: list[float] = []

    for row in sampled:
        row = _truncate_series(row, cfg.max_points_per_series)
        finite = _finite(row)
        if finite.size < 3:
            continue

        series_means.append(float(np.mean(finite)))
        q1, q3 = np.percentile(finite, [25, 75])
        series_iqrs.append(float(q3 - q1))
        acf1_values.append(_acf(row, 1))
        acfh_values.append(_acf(row, forecast_horizon_steps))
        trend_values.append(_trend_strength(row))

        period, entropy = _spectral_features(row)
        entropy_values.append(entropy)
        if period is not None and np.isfinite(period):
            # Do not treat periods longer than half the observed series as reliable.
            if 2.0 <= period <= max(2.0, finite.size / 2.0):
                period_values.append(float(period))
                seasonal_acf_values.append(abs(_acf(row, max(1, int(round(period))))))

    dominant_period_steps = _safe_median(period_values) if period_values else None
    dominant_period_seconds = (
        float(dominant_period_steps * sampling_interval_seconds)
        if dominant_period_steps is not None
        else None
    )

    scale_arr = np.asarray(series_iqrs, dtype=float)
    scale_arr = scale_arr[np.isfinite(scale_arr)]
    positive_scales = scale_arr[scale_arr > _EPS]
    if positive_scales.size >= 2:
        between_scale = float(np.std(np.log(positive_scales)))
    else:
        between_scale = 0.0

    means_arr = np.asarray(series_means, dtype=float)
    means_arr = means_arr[np.isfinite(means_arr)]
    between_mean = float(np.std(means_arr) / max(np.median(np.abs(means_arr)), _EPS)) if means_arr.size else float("nan")

    acf1_arr = np.asarray(acf1_values, dtype=float)
    acf1_arr = acf1_arr[np.isfinite(acf1_arr)]
    between_acf = float(np.std(acf1_arr)) if acf1_arr.size else float("nan")

    return DatasetFingerprint(
        dataset_name=str(dataset_name),
        n_series=n_series,
        n_variables=int(n_variables),
        sampling_interval_seconds=float(sampling_interval_seconds),
        forecast_horizon_steps=int(forecast_horizon_steps),
        forecast_horizon_seconds=float(forecast_horizon_steps * sampling_interval_seconds),
        median_series_length=float(np.median(valid_lengths)),
        min_series_length=int(np.min(valid_lengths)),
        max_series_length=int(np.max(valid_lengths)),
        missing_fraction=missing_fraction,
        zero_fraction=zero_fraction,
        constant_series_fraction=constant_series_fraction,
        value_median=float(np.median(flattened)),
        value_iqr=value_iqr,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        coefficient_of_variation=coefficient_of_variation,
        acf_lag1=_safe_median(acf1_values),
        acf_horizon=_safe_median(acfh_values),
        trend_strength=_safe_median(trend_values),
        dominant_period_steps=dominant_period_steps,
        dominant_period_seconds=dominant_period_seconds,
        seasonality_strength=_safe_median(seasonal_acf_values),
        spectral_entropy=_safe_median(entropy_values),
        between_series_scale_variability=between_scale,
        between_series_mean_variability=between_mean,
        between_series_acf_variability=between_acf,
    )


def compute_fingerprint_ragged(
    values: Any,
    *,
    dataset_name: str,
    sampling_interval_seconds: float,
    forecast_horizon_steps: int,
    n_variables: int = 1,
    config: FingerprintConfig | None = None,
) -> DatasetFingerprint:
    """Compute a fingerprint from variable-length training histories.

    This is the preferred entry point for real datasets where entities have
    different available history lengths. Dataset-level length statistics use
    all supplied histories. Distributional and temporal features are computed
    on a deterministic bounded sample to control memory/runtime.
    """
    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be > 0")
    if forecast_horizon_steps <= 0:
        raise ValueError("forecast_horizon_steps must be > 0")

    cfg = config or FingerprintConfig()

    series: list[np.ndarray] = []
    for item in values:
        if hasattr(item, "to_numpy"):
            item = item.to_numpy()
        arr = np.asarray(item, dtype=float).reshape(-1)
        if arr.size:
            series.append(arr)

    if not series:
        raise ValueError("No series supplied.")

    n_series = len(series)

    finite_counts = np.asarray(
        [int(np.isfinite(row).sum()) for row in series],
        dtype=int,
    )
    valid_lengths = finite_counts[finite_counts > 0]
    if valid_lengths.size == 0:
        raise ValueError("No finite observations found in values.")

    rng = np.random.default_rng(cfg.seed)
    sample_idx = np.arange(n_series)
    if sample_idx.size > cfg.max_series:
        sample_idx = np.sort(
            rng.choice(
                sample_idx,
                size=cfg.max_series,
                replace=False,
            )
        )

    sampled = [
        _truncate_series(
            series[int(i)],
            cfg.max_points_per_series,
        )
        for i in sample_idx
    ]

    total_sampled_points = int(
        sum(row.size for row in sampled)
    )
    if total_sampled_points == 0:
        raise ValueError("Sampled histories are empty.")

    total_finite = int(
        sum(np.isfinite(row).sum() for row in sampled)
    )
    missing_fraction = float(
        1.0 - total_finite / total_sampled_points
    )

    finite_chunks = [
        _finite(row)
        for row in sampled
        if np.isfinite(row).any()
    ]
    if not finite_chunks:
        raise ValueError(
            "No finite observations found in sampled histories."
        )

    flattened = np.concatenate(finite_chunks)
    zero_fraction = float(
        np.mean(flattened == 0.0)
    )

    constant_flags: list[bool] = []
    for row in sampled:
        finite = _finite(row)
        if finite.size:
            constant_flags.append(
                bool(np.std(finite) <= _EPS)
            )
    constant_series_fraction = (
        float(np.mean(constant_flags))
        if constant_flags
        else float("nan")
    )

    q25, q75 = np.percentile(
        flattened,
        [25, 75],
    )
    value_iqr = float(q75 - q25)
    skewness, excess_kurtosis = (
        _moment_skew_kurtosis(flattened)
    )
    mean_abs = abs(float(np.mean(flattened)))
    coefficient_of_variation = float(
        np.std(flattened)
        / max(mean_abs, _EPS)
    )

    acf1_values: list[float] = []
    acfh_values: list[float] = []
    trend_values: list[float] = []
    period_values: list[float] = []
    entropy_values: list[float] = []
    seasonal_acf_values: list[float] = []
    series_means: list[float] = []
    series_iqrs: list[float] = []

    for row in sampled:
        finite = _finite(row)
        if finite.size < 3:
            continue

        series_means.append(
            float(np.mean(finite))
        )
        q1, q3 = np.percentile(
            finite,
            [25, 75],
        )
        series_iqrs.append(
            float(q3 - q1)
        )

        acf1_values.append(
            _acf(row, 1)
        )
        acfh_values.append(
            _acf(
                row,
                forecast_horizon_steps,
            )
        )
        trend_values.append(
            _trend_strength(row)
        )

        period, entropy = (
            _spectral_features(row)
        )
        entropy_values.append(entropy)

        if (
            period is not None
            and np.isfinite(period)
            and 2.0
            <= period
            <= max(
                2.0,
                finite.size / 2.0,
            )
        ):
            period_values.append(
                float(period)
            )
            seasonal_acf_values.append(
                abs(
                    _acf(
                        row,
                        max(
                            1,
                            int(round(period)),
                        ),
                    )
                )
            )

    dominant_period_steps = (
        _safe_median(period_values)
        if period_values
        else None
    )
    dominant_period_seconds = (
        float(
            dominant_period_steps
            * sampling_interval_seconds
        )
        if dominant_period_steps
        is not None
        else None
    )

    scale_arr = np.asarray(
        series_iqrs,
        dtype=float,
    )
    scale_arr = scale_arr[
        np.isfinite(scale_arr)
    ]
    positive_scales = scale_arr[
        scale_arr > _EPS
    ]
    if positive_scales.size >= 2:
        between_scale = float(
            np.std(
                np.log(
                    positive_scales
                )
            )
        )
    else:
        between_scale = 0.0

    means_arr = np.asarray(
        series_means,
        dtype=float,
    )
    means_arr = means_arr[
        np.isfinite(means_arr)
    ]
    between_mean = (
        float(
            np.std(means_arr)
            / max(
                np.median(
                    np.abs(means_arr)
                ),
                _EPS,
            )
        )
        if means_arr.size
        else float("nan")
    )

    acf1_arr = np.asarray(
        acf1_values,
        dtype=float,
    )
    acf1_arr = acf1_arr[
        np.isfinite(acf1_arr)
    ]
    between_acf = (
        float(np.std(acf1_arr))
        if acf1_arr.size
        else float("nan")
    )

    return DatasetFingerprint(
        dataset_name=str(
            dataset_name
        ),
        n_series=n_series,
        n_variables=int(
            n_variables
        ),
        sampling_interval_seconds=float(
            sampling_interval_seconds
        ),
        forecast_horizon_steps=int(
            forecast_horizon_steps
        ),
        forecast_horizon_seconds=float(
            forecast_horizon_steps
            * sampling_interval_seconds
        ),
        median_series_length=float(
            np.median(valid_lengths)
        ),
        min_series_length=int(
            np.min(valid_lengths)
        ),
        max_series_length=int(
            np.max(valid_lengths)
        ),
        missing_fraction=(
            missing_fraction
        ),
        zero_fraction=zero_fraction,
        constant_series_fraction=(
            constant_series_fraction
        ),
        value_median=float(
            np.median(flattened)
        ),
        value_iqr=value_iqr,
        skewness=skewness,
        excess_kurtosis=(
            excess_kurtosis
        ),
        coefficient_of_variation=(
            coefficient_of_variation
        ),
        acf_lag1=_safe_median(
            acf1_values
        ),
        acf_horizon=_safe_median(
            acfh_values
        ),
        trend_strength=_safe_median(
            trend_values
        ),
        dominant_period_steps=(
            dominant_period_steps
        ),
        dominant_period_seconds=(
            dominant_period_seconds
        ),
        seasonality_strength=(
            _safe_median(
                seasonal_acf_values
            )
        ),
        spectral_entropy=(
            _safe_median(
                entropy_values
            )
        ),
        between_series_scale_variability=(
            between_scale
        ),
        between_series_mean_variability=(
            between_mean
        ),
        between_series_acf_variability=(
            between_acf
        ),
    )
