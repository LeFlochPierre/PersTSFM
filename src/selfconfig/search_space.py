from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .fingerprint import DatasetFingerprint
from .memory import TemporalMemoryFingerprint


@dataclass(frozen=True)
class CandidateConfig:
    model: str
    context_steps: int
    context_multiplier: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "context_steps": int(
                self.context_steps
            ),
            "context_multiplier": float(
                self.context_multiplier
            ),
        }


MODELS = (
    "chronos2",
    "timesfm3",
    "tirex2",
)

MODEL_MAX_CONTEXT = {
    "chronos2": None,
    "timesfm3": None,
    "tirex2": 2048,
}


def _add_context(
    values: set[int],
    value: float | int | None,
    *,
    limit: int | None = None,
) -> None:
    if value is None:
        return

    value = int(
        round(
            float(value)
        )
    )

    if limit is not None:
        value = min(
            value,
            int(limit),
        )

    if value >= 2:
        values.add(value)


def _period_is_credible(
    fingerprint: DatasetFingerprint,
    memory: TemporalMemoryFingerprint | None,
) -> bool:
    period = (
        fingerprint.dominant_period_steps
    )

    if period is None:
        return False

    seasonal = float(
        fingerprint.seasonality_strength
    )

    if seasonal >= 0.50:
        return True

    if (
        memory is not None
        and memory.acf_period is not None
        and seasonal >= 0.30
        and abs(
            float(
                memory.acf_period
            )
        ) >= 0.25
    ):
        return True

    return False


def build_search_space(
    horizon_steps: int,
    *,
    fingerprint: DatasetFingerprint | None = None,
    memory: TemporalMemoryFingerprint | None = None,
    history_limit_steps: int | None = None,
    models: Iterable[str] = MODELS,
) -> list[CandidateConfig]:
    """Build the dataset-specific feasible context search space.

    Horizon-relative candidates are powers of two:
        H, 2H, 4H, 8H, ...

    They stop at the dataset's leakage-safe history/resource limit. Credible
    period-relative candidates and the history boundary are added as extra
    feasible configurations.
    """

    H = int(horizon_steps)

    if H <= 0:
        raise ValueError(
            "horizon_steps must be positive."
        )

    limit = (
        int(history_limit_steps)
        if history_limit_steps is not None
        else 16 * H
    )

    if limit <= 0:
        raise ValueError(
            "history_limit_steps must be positive."
        )

    context_steps: set[int] = set()

    # ---------------------------------------------------------
    # Horizon-relative powers of two
    # ---------------------------------------------------------

    multiplier = 1

    while (
        H * multiplier
        <= limit
    ):
        _add_context(
            context_steps,
            H * multiplier,
            limit=limit,
        )

        multiplier *= 2

        if multiplier > 4096:
            break

    # Always expose the feasibility boundary itself.
    _add_context(
        context_steps,
        limit,
        limit=limit,
    )

    # ---------------------------------------------------------
    # Period-relative candidates
    # ---------------------------------------------------------

    if (
        fingerprint is not None
        and _period_is_credible(
            fingerprint,
            memory,
        )
    ):
        P = int(
            round(
                float(
                    fingerprint
                    .dominant_period_steps
                )
            )
        )

        seasonal_multiplier = 1

        while (
            P * seasonal_multiplier
            <= limit
        ):
            _add_context(
                context_steps,
                P * seasonal_multiplier,
                limit=limit,
            )

            seasonal_multiplier *= 2

            if seasonal_multiplier > 64:
                break

    # ---------------------------------------------------------
    # Memory-scale candidates
    # ---------------------------------------------------------

    if memory is not None:
        _add_context(
            context_steps,
            memory.memory_lag_02_steps,
            limit=limit,
        )

        _add_context(
            context_steps,
            memory.memory_lag_01_steps,
            limit=limit,
        )

    # ---------------------------------------------------------
    # Apply model constraints
    # ---------------------------------------------------------

    candidates: list[
        CandidateConfig
    ] = []

    for model in models:
        if model not in MODEL_MAX_CONTEXT:
            raise ValueError(
                f"Unknown model: {model}"
            )

        model_limit = (
            MODEL_MAX_CONTEXT[
                model
            ]
        )

        for steps in sorted(
            context_steps
        ):
            if (
                model_limit is not None
                and steps > model_limit
            ):
                continue

            candidates.append(
                CandidateConfig(
                    model=model,
                    context_steps=int(
                        steps
                    ),
                    context_multiplier=(
                        float(steps)
                        / H
                    ),
                )
            )

    return candidates
