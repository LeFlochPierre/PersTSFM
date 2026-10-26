from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HistoryCapacityFingerprint:
    min_steps: int
    p10_steps: int
    p25_steps: int
    median_steps: int
    p75_steps: int
    p90_steps: int
    max_steps: int

    horizon_steps: int

    robust_history_limit_steps: int
    resource_limit_steps: int
    effective_limit_steps: int

    effective_limit_horizons: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_history_capacity(
    histories: Any,
    *,
    horizon_steps: int,
    support_quantile: float = 0.25,
    resource_limit_steps: int = 4096,
) -> HistoryCapacityFingerprint:
    """Summarise leakage-safe history availability.

    The robust history limit is the selected lower quantile of available
    pre-evaluation history. The effective limit is additionally bounded by a
    fixed computational study budget. This is a feasibility constraint, not
    a claim about the optimal context length.
    """

    H = int(horizon_steps)

    if H <= 0:
        raise ValueError(
            "horizon_steps must be positive"
        )

    if not 0.0 <= support_quantile <= 1.0:
        raise ValueError(
            "support_quantile must be in [0, 1]"
        )

    resource_limit_steps = int(
        resource_limit_steps
    )

    if resource_limit_steps <= 0:
        raise ValueError(
            "resource_limit_steps must be positive"
        )

    lengths = np.asarray(
        [
            len(h)
            for h in histories
            if len(h) > 0
        ],
        dtype=float,
    )

    if lengths.size == 0:
        raise ValueError(
            "No non-empty histories supplied."
        )

    q = np.percentile(
        lengths,
        [0, 10, 25, 50, 75, 90, 100],
    )

    robust = int(
        np.floor(
            np.quantile(
                lengths,
                support_quantile,
            )
        )
    )

    effective = min(
        robust,
        resource_limit_steps,
    )

    if effective < H:
        effective = int(
            max(
                1,
                effective,
            )
        )

    return HistoryCapacityFingerprint(
        min_steps=int(np.floor(q[0])),
        p10_steps=int(np.floor(q[1])),
        p25_steps=int(np.floor(q[2])),
        median_steps=int(np.floor(q[3])),
        p75_steps=int(np.floor(q[4])),
        p90_steps=int(np.floor(q[5])),
        max_steps=int(np.floor(q[6])),
        horizon_steps=H,
        robust_history_limit_steps=robust,
        resource_limit_steps=resource_limit_steps,
        effective_limit_steps=effective,
        effective_limit_horizons=(
            float(effective) / H
        ),
    )
