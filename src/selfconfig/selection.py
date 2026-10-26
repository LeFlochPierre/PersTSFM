from __future__ import annotations

from collections.abc import Iterable
import pandas as pd

from .search_space import CandidateConfig


REQUIRED_SCORE_COLUMNS = {"dataset", "model", "context_steps", "split", "mae"}


def _validate_scores(scores: pd.DataFrame) -> None:
    missing = REQUIRED_SCORE_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(f"Score table is missing required columns: {sorted(missing)}")


def candidate_keys(candidates: Iterable[CandidateConfig]) -> set[tuple[str, int]]:
    return {(c.model, int(c.context_steps)) for c in candidates}


def select_best(
    scores: pd.DataFrame,
    *,
    dataset: str,
    candidates: Iterable[CandidateConfig],
    split: str = "val",
    metric: str = "mae",
) -> pd.Series:
    _validate_scores(scores)
    if metric not in scores.columns:
        raise ValueError(f"Metric {metric!r} not present in score table.")

    keys = candidate_keys(candidates)
    frame = scores[(scores["dataset"] == dataset) & (scores["split"] == split)].copy()
    frame = frame[frame.apply(lambda r: (r["model"], int(r["context_steps"])) in keys, axis=1)]
    frame = frame[pd.to_numeric(frame[metric], errors="coerce").notna()]
    if frame.empty:
        raise ValueError(f"No {split!r} scores found for dataset={dataset!r} and supplied candidates.")

    idx = pd.to_numeric(frame[metric], errors="coerce").idxmin()
    return frame.loc[idx]
