"""Coverage-aware, metadata-only cohort selection for B3 forecasting experiments.

Chooses the latest globally shared calibration/evaluation time blocks with at
least `entities` represented in *both*. Does not inspect feature/target values,
model predictions, or model errors. All target-end maturity is enforced.
"""
from __future__ import annotations

import hashlib

import pandas as pd


def select_coverage_aware(protocol: pd.DataFrame, id_col: str, *, entities: int,
                          calibration_origins: int, evaluation_origins: int,
                          seed: int) -> tuple[pd.DataFrame, dict]:
    if min(entities, calibration_origins, evaluation_origins) < 1:
        raise ValueError("entities and origin counts must all be >=1")
    required = {id_col, "eval_origin", "target_end", "eval_index"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Protocol missing columns: {sorted(missing)}")
    f = protocol.copy()
    f[id_col] = f[id_col].astype(str)
    for column in ("eval_origin", "target_end"):
        f[column] = pd.to_datetime(f[column], errors="raise")
    if f.empty or f[["eval_origin", "target_end"]].isna().any().any():
        raise ValueError("Protocol is empty or contains missing timestamps")
    if (f["target_end"] <= f["eval_origin"]).any():
        raise ValueError("Invalid protocol: target_end must follow eval_origin")
    if f.duplicated([id_col, "eval_origin"]).any():
        raise ValueError("Duplicate entity/origin in original protocol")
    times = sorted(f["eval_origin"].unique())
    if len(times) < calibration_origins + evaluation_origins:
        raise ValueError("Not enough distinct time slots for both partitions")

    best_count = 0
    for end_idx in range(len(times) - 1, evaluation_origins - 1, -1):
        eval_times = times[end_idx - evaluation_origins + 1:end_idx + 1]
        eval_start = eval_times[0]
        mature = f[f["target_end"] < eval_start]
        cal_times = sorted(mature["eval_origin"].unique())[-calibration_origins:]
        if len(cal_times) != calibration_origins:
            continue
        cal = mature[mature["eval_origin"].isin(cal_times)].copy()
        ev = f[f["eval_origin"].isin(eval_times)].copy()
        common_ids = sorted(set(cal[id_col]) & set(ev[id_col]))
        best_count = max(best_count, len(common_ids))
        if len(common_ids) < entities:
            continue
        # Deterministic entity selection uses only ids; no outcomes or features.
        ranked = sorted(common_ids, key=lambda x: (
            hashlib.sha256(f"{seed}|{x}".encode()).digest(), x))
        chosen = ranked[:entities]
        cal = cal[cal[id_col].isin(chosen)].copy()
        ev = ev[ev[id_col].isin(chosen)].copy()
        cal["split"] = "calibration"
        ev["split"] = "evaluation"
        result = pd.concat([cal, ev], ignore_index=True).sort_values(
            ["eval_origin", id_col, "eval_index"], kind="mergesort").reset_index(drop=True)
        cover = result.groupby([id_col, "split"]).size().unstack(fill_value=0)
        assert len(cover) == entities
        assert (cover["calibration"] >= 1).all() and (cover["evaluation"] >= 1).all()
        assert cal["target_end"].max() < ev["eval_origin"].min()
        details = dict(
            selection="coverage_aware_latest_complete_metadata_block",
            eligible_entity_count=len(common_ids), selected_entities=sorted(chosen),
            calibration_slots=[pd.Timestamp(x).isoformat() for x in cal_times],
            evaluation_slots=[pd.Timestamp(x).isoformat() for x in eval_times],
            rejected_later_endpoints=int(len(times) - 1 - end_idx),
            calibration_rows=int(len(cal)), evaluation_rows=int(len(ev)),
        )
        return result, details
    raise ValueError(
        f"Cannot find {entities} entities with both calibration/evaluation origins "
        f"in a global chronological window; maximum eligible together: {best_count}. "
        "Try fewer --entities or adjust origin counts; do not use future labels."
    )
