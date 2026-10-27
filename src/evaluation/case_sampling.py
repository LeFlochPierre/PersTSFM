"""Deterministic, multi-entity sampling for PersTSFM development pilots.

Uses SHA-256 entity ranking so the cohort does not depend on NumPy/Python RNG
versions across the separate TSFM conda environments. Samples temporally
spread forecast origins within each entity. This is *development* sampling;
final scientific results must also use full evaluation protocols.
"""
from __future__ import annotations

import hashlib

import pandas as pd


def _priority(seed: int, entity_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{entity_id}".encode()).digest()


def balanced_subset(
    protocol: pd.DataFrame,
    id_col: str,
    n_entities: int | None = None,
    origins_per_entity: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if id_col not in protocol.columns or "eval_origin" not in protocol.columns:
        raise ValueError("protocol needs id column and eval_origin")
    if n_entities is not None and n_entities < 1:
        raise ValueError("n_entities must be >= 1")
    if origins_per_entity is not None and origins_per_entity < 1:
        raise ValueError("origins_per_entity must be >= 1")
    if protocol.empty:
        raise ValueError("protocol is empty")

    frame = protocol.copy()
    frame[id_col] = frame[id_col].astype(str)
    entity_ids = sorted(frame[id_col].unique())
    if n_entities is not None:
        entity_ids = sorted(
            entity_ids,
            key=lambda entity: (_priority(seed, entity), entity),
        )[:n_entities]
    entity_ids = sorted(entity_ids)
    selected = []
    for entity in entity_ids:
        rows = frame[frame[id_col] == entity].sort_values(
            ["eval_origin", "eval_index"] if "eval_index" in frame else ["eval_origin"],
            kind="mergesort",
        )
        if origins_per_entity is not None and len(rows) > origins_per_entity:
            k, count = origins_per_entity, len(rows)
            if k == 1:
                positions = [count - 1]
            else:
                positions = [round(i * (count - 1) / (k - 1)) for i in range(k)]
            rows = rows.iloc[positions]
        selected.append(rows)
    result = pd.concat(selected, ignore_index=True)
    sort_cols = [id_col] + (["eval_index"] if "eval_index" in result else ["eval_origin"])
    return result.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
