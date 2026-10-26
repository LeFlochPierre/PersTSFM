from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Iterable

from .fingerprint import DatasetFingerprint
from .memory import TemporalMemoryFingerprint
from .regime import RegimeFingerprint
from .search_space import CandidateConfig


@dataclass(frozen=True)
class RuleConfig:
    strong_seasonality_threshold: float = 0.50
    moderate_seasonality_threshold: float = 0.30
    periodic_acf_threshold: float = 0.25

    high_horizon_acf_threshold: float = 0.30

    regime_mean_shift_threshold: float = 0.50
    regime_min_capacity_horizons: float = 32.0
    regime_long_anchor_max_horizons: float = 128.0

    candidates_per_model: int = 3


@dataclass(frozen=True)
class RuleResult:
    selected: tuple[CandidateConfig, ...]
    rationale: tuple[str, ...]


def _nearest(
    candidates: list[CandidateConfig],
    target: float,
) -> CandidateConfig:
    return min(
        candidates,
        key=lambda c: abs(
            log(
                max(
                    c.context_steps,
                    1,
                )
                / max(
                    target,
                    1.0,
                )
            )
        ),
    )


def _safe_abs(
    value: float | None,
) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if value != value:
        return 0.0

    return abs(value)


def _period_is_credible(
    fingerprint: DatasetFingerprint,
    memory: TemporalMemoryFingerprint | None,
    cfg: RuleConfig,
) -> bool:
    period = (
        fingerprint.dominant_period_steps
    )

    if period is None:
        return False

    seasonal = _safe_abs(
        fingerprint.seasonality_strength
    )

    if (
        seasonal
        >= cfg.strong_seasonality_threshold
    ):
        return True

    if (
        memory is not None
        and memory.acf_period is not None
        and seasonal
        >= cfg.moderate_seasonality_threshold
        and _safe_abs(
            memory.acf_period
        )
        >= cfg.periodic_acf_threshold
    ):
        return True

    return False


def apply_rules(
    fingerprint: DatasetFingerprint,
    candidates: Iterable[CandidateConfig],
    *,
    memory: TemporalMemoryFingerprint | None = None,
    regime: RegimeFingerprint | None = None,
    history_limit_steps: int | None = None,
    config: RuleConfig | None = None,
) -> RuleResult:
    """Deterministic nnU-Net-style context planner."""

    cfg = config or RuleConfig()

    candidates = list(
        candidates
    )

    if not candidates:
        raise ValueError(
            "No candidates supplied."
        )

    by_model: dict[
        str,
        list[CandidateConfig],
    ] = {}

    for candidate in candidates:
        by_model.setdefault(
            candidate.model,
            [],
        ).append(
            candidate
        )

    for model in by_model:
        by_model[model].sort(
            key=lambda c: c.context_steps
        )

    H = float(
        fingerprint.forecast_horizon_steps
    )

    seasonal = _safe_abs(
        fingerprint.seasonality_strength
    )

    acfh = _safe_abs(
        fingerprint.acf_horizon
    )

    period = (
        float(
            fingerprint.dominant_period_steps
        )
        if (
            fingerprint.dominant_period_steps
            is not None
        )
        else None
    )

    credible_period = (
        _period_is_credible(
            fingerprint,
            memory,
            cfg,
        )
    )

    if history_limit_steps is None:
        history_limit_steps = max(
            c.context_steps
            for c in candidates
        )

    capacity_h = (
        float(
            history_limit_steps
        )
        / H
    )

    regime_shift = (
        float(
            regime.recent_vs_prior_mean_shift
        )
        if (
            regime is not None
            and regime.recent_vs_prior_mean_shift
            == regime.recent_vs_prior_mean_shift
        )
        else 0.0
    )

    regime_rich = (
        not credible_period
        and regime_shift
        >= cfg.regime_mean_shift_threshold
        and capacity_h
        >= cfg.regime_min_capacity_horizons
    )

    # ---------------------------------------------------------
    # Human-readable rationale
    # ---------------------------------------------------------

    rationale: list[str] = [
        (
            "Selected at most "
            f"{cfg.candidates_per_model} "
            "interpretable context configurations "
            "per model."
        ),
        (
            "Feasible context is bounded by "
            f"{history_limit_steps} steps "
            f"({capacity_h:.1f}×H) from leakage-safe "
            "history/resource capacity."
        ),
    ]

    if credible_period:
        periodic_acf_text = ""

        if (
            memory is not None
            and memory.acf_period
            is not None
        ):
            periodic_acf_text = (
                f", ACF(P)="
                f"{memory.acf_period:.3f}"
            )

        rationale.append(
            (
                "Credible repeating structure "
                f"detected (spectral strength="
                f"{seasonal:.3f}, dominant period≈"
                f"{period:.1f} steps"
                f"{periodic_acf_text}): "
                "retaining a complete-cycle context."
            )
        )

    elif regime_rich:
        rationale.append(
            (
                "No credible repeating period was "
                "found, but recent-vs-prior mean "
                f"shift is substantial "
                f"({regime_shift:.3f}) and long "
                "history is available: using a "
                "log-spaced long-context regime scan."
            )
        )

    elif (
        acfh
        >= cfg.high_horizon_acf_threshold
    ):
        rationale.append(
            (
                "Strong dependence remains at the "
                f"forecast horizon (|ACF(H)|="
                f"{acfh:.3f}), but no strong "
                "long-period/regime trigger was met."
            )
        )

    # ---------------------------------------------------------
    # Select configurations
    # ---------------------------------------------------------

    selected: list[
        CandidateConfig
    ] = []

    for model, options in by_model.items():
        chosen: dict[
            int,
            CandidateConfig,
        ] = {}

        def add_target(
            target: float | int | None,
        ) -> None:
            if target is None:
                return

            candidate = _nearest(
                options,
                float(
                    target
                ),
            )

            chosen[
                candidate.context_steps
            ] = candidate

        # Slot 1: forecast-scale baseline.
        add_target(H)

        if credible_period:
            # Slot 2: one complete cycle.
            add_target(
                period
            )

            # Slot 3: several cycles, constrained by capacity.
            add_target(
                min(
                    float(
                        history_limit_steps
                    ),
                    max(
                        8.0 * H,
                        4.0 * period,
                    ),
                )
            )

        elif regime_rich:
            # Long anchor: at most 128 forecast horizons and never beyond
            # leakage-safe/resource capacity.
            long_target = min(
                float(
                    history_limit_steps
                ),
                cfg.regime_long_anchor_max_horizons
                * H,
            )

            # Three log-spaced scales: short, geometric midpoint, long.
            middle_target = sqrt(
                H
                * long_target
            )

            add_target(
                middle_target
            )

            add_target(
                long_target
            )

        else:
            # Ordinary short/medium-memory regime.
            add_target(
                min(
                    4.0 * H,
                    float(
                        history_limit_steps
                    ),
                )
            )

            persistence = (
                memory.memory_lag_02_steps
                if (
                    memory is not None
                    and memory.memory_lag_02_steps
                    is not None
                )
                else None
            )

            long_target = max(
                8.0 * H,
                float(
                    persistence
                )
                if persistence
                is not None
                else 8.0 * H,
            )

            add_target(
                min(
                    long_target,
                    float(
                        history_limit_steps
                    ),
                )
            )

        # Deterministic fallback if two targets collapse to one feasible point.
        fallback_targets = (
            2.0 * H,
            4.0 * H,
            8.0 * H,
            16.0 * H,
            32.0 * H,
            64.0 * H,
            128.0 * H,
        )

        for target in fallback_targets:
            if (
                len(chosen)
                >= cfg.candidates_per_model
            ):
                break

            if target <= history_limit_steps:
                add_target(
                    target
                )

        # If still short, include the largest feasible candidate.
        if (
            len(chosen)
            < cfg.candidates_per_model
        ):
            chosen[
                options[-1].context_steps
            ] = options[-1]

        model_selected = sorted(
            chosen.values(),
            key=lambda c: c.context_steps,
        )[
            : cfg.candidates_per_model
        ]

        selected.extend(
            model_selected
        )

    selected = sorted(
        selected,
        key=lambda c: (
            c.model,
            c.context_steps,
        ),
    )

    return RuleResult(
        selected=tuple(
            selected
        ),
        rationale=tuple(
            rationale
        ),
    )
