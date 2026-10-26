from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.registry import load_dataset


@dataclass(frozen=True)
class FingerprintInput:
    dataset_key: str
    split: str
    histories: tuple[np.ndarray, ...]
    resolution_minutes: int
    forecast_steps: int
    cutoff_method: str
    n_protocol_entities: int


def _normalise_id(value: Any) -> str:
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def _lookup_series(series_map: dict, entity_id: Any):
    if entity_id in series_map:
        return series_map[entity_id]

    target = _normalise_id(entity_id)

    for key, value in series_map.items():
        if _normalise_id(key) == target:
            return value

    raise KeyError(
        f"Entity {entity_id!r} not found in loaded series."
    )


def _series_to_array_and_index(obj):
    if isinstance(obj, pd.Series):
        return (
            obj.to_numpy(dtype=float),
            obj.index,
        )

    if isinstance(obj, pd.DataFrame):
        numeric = obj.select_dtypes(
            include=[np.number]
        )

        if numeric.shape[1] != 1:
            raise ValueError(
                "Expected one numeric target column, "
                f"found {list(numeric.columns)}"
            )

        col = numeric.columns[0]

        return (
            numeric[col].to_numpy(dtype=float),
            obj.index,
        )

    arr = np.asarray(
        obj,
        dtype=float,
    )

    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D series, got {arr.shape}"
        )

    return arr, None


def _cutoff_from_eval_origin(
    rows,
    index,
    n_points,
):
    if (
        "eval_origin" not in rows.columns
        or index is None
    ):
        return None

    origins = pd.to_datetime(
        rows["eval_origin"],
        errors="coerce",
    ).dropna()

    if origins.empty:
        return None

    # Do not interpret integer RangeIndex values
    # as timestamps around 1970.
    if isinstance(index, pd.RangeIndex):
        return None

    try:
        dt_index = pd.DatetimeIndex(
            pd.to_datetime(
                index,
                errors="coerce",
            )
        )
    except Exception:
        return None

    if len(dt_index) != n_points:
        return None

    if dt_index.isna().all():
        return None

    origin = origins.min()

    try:
        cutoff = int(
            dt_index.searchsorted(
                origin,
                side="left",
            )
        )
    except Exception:
        return None

    if cutoff >= 3 and cutoff <= n_points:
        return cutoff

    return None


def _cutoff_from_eval_index(
    rows,
    n_points,
):
    if "eval_index" not in rows.columns:
        return None

    values = pd.to_numeric(
        rows["eval_index"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    cutoff = int(
        np.floor(values.min())
    )

    # Only accept this if it really looks like
    # an absolute position in the source series.
    if cutoff >= 3 and cutoff <= n_points:
        return cutoff

    return None


def _cutoff_position(
    rows,
    *,
    index,
    n_points,
):
    # IMPORTANT:
    # eval_origin is preferred because eval_index
    # may simply be an evaluation-case number.
    cutoff = _cutoff_from_eval_origin(
        rows,
        index,
        n_points,
    )

    if cutoff is not None:
        return (
            cutoff,
            "earliest eval_origin",
        )

    cutoff = _cutoff_from_eval_index(
        rows,
        n_points,
    )

    if cutoff is not None:
        return (
            cutoff,
            "earliest eval_index fallback",
        )

    raise RuntimeError(
        "Could not determine pre-evaluation cutoff. "
        f"Series length={n_points}; "
        f"protocol columns={list(rows.columns)}"
    )


def load_fingerprint_input(
    dataset_key: str,
    *,
    split: str | None = None,
) -> FingerprintInput:

    dataset = load_dataset(
        dataset_key,
        split=split,
    )

    spec = dataset.spec
    protocol = dataset.protocol.copy()

    split_name = (
        split
        if split is not None
        else spec.default_split
    )

    histories = []
    methods = set()

    if protocol.empty:
        raise RuntimeError(
            f"No protocol rows for {dataset_key}"
        )

    for entity_id, rows in protocol.groupby(
        spec.id_col,
        sort=False,
    ):
        series_obj = _lookup_series(
            dataset.series,
            entity_id,
        )

        values, index = (
            _series_to_array_and_index(
                series_obj
            )
        )

        cutoff, method = (
            _cutoff_position(
                rows,
                index=index,
                n_points=len(values),
            )
        )

        history = np.asarray(
            values[:cutoff],
            dtype=float,
        )

        n_finite = int(
            np.isfinite(history).sum()
        )

        if n_finite < 3:
            continue

        histories.append(history)
        methods.add(method)

    if not histories:
        raise RuntimeError(
            "No usable pre-evaluation histories "
            f"were extracted for {dataset_key}."
        )

    return FingerprintInput(
        dataset_key=dataset_key,
        split=split_name,
        histories=tuple(histories),
        resolution_minutes=int(
            spec.resolution_minutes
        ),
        forecast_steps=int(
            spec.forecast_steps
        ),
        cutoff_method=", ".join(
            sorted(methods)
        ),
        n_protocol_entities=int(
            protocol[
                spec.id_col
            ].nunique()
        ),
    )
