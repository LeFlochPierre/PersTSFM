"""Metadata-only per-entity origin selection with one pooled chronological boundary.

Designed for datasets whose evaluation origins are not globally shared across
entities. Exactly N calibration and M evaluation origins per selected entity;
all calibration outcomes mature before *any* selected evaluation origin.
Never inspect target values, forecast errors, predictions or model features.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib

import pandas as pd


def select_per_entity_origins(protocol: pd.DataFrame, id_col: str, *, entities: int,
                              calibration_origins: int, evaluation_origins: int,
                              seed: int, resolution_minutes: int) -> tuple[pd.DataFrame, dict]:
    if min(entities, calibration_origins, evaluation_origins, resolution_minutes) < 1:
        raise ValueError("entities, origin counts and resolution_minutes must be positive")
    required = {id_col, "eval_origin", "target_end", "eval_index"}
    absent = required - set(protocol.columns)
    if absent:
        raise ValueError(f"Protocol missing columns: {sorted(absent)}")
    f = protocol.copy()
    if f.empty or f[id_col].isna().any():
        raise ValueError("Protocol is empty or has missing entity IDs")
    f[id_col] = f[id_col].astype(str)
    for key in ("eval_origin", "target_end"):
        f[key] = pd.to_datetime(f[key], errors="raise")
    if f[["eval_origin", "target_end"]].isna().any().any():
        raise ValueError("Protocol has missing timestamps")
    if (f.target_end <= f.eval_origin).any():
        raise ValueError("Protocol has invalid target_end <= eval_origin")
    if f.duplicated([id_col, "eval_origin"]).any():
        raise ValueError("Duplicate entity/origin in source protocol")
    if f.duplicated([id_col, "eval_index"]).any():
        raise ValueError("Duplicate entity/eval_index in source protocol")
    maturity_delta = pd.Timedelta(minutes=resolution_minutes)
    # Exact same label-availability convention as B3 freeze (_extract).
    f["_available_at"] = f["target_end"] + maturity_delta

    by_id = {}
    for identifier, group in f.groupby(id_col, sort=True):
        g = group.sort_values(["eval_origin", "eval_index"], kind="mergesort").reset_index(drop=True)
        origins = list(g.eval_origin)
        maturity = list(g._available_at)
        # A fixed-horizon forecasting protocol should have chronological target
        # availability; this permits fast bisection instead of nested scans.
        if any(a > b for a, b in zip(maturity, maturity[1:])):
            raise ValueError(f"Non-monotonic target availability for entity {identifier}")
        if len(g) >= calibration_origins + evaluation_origins:
            by_id[str(identifier)] = (g, origins, maturity)

    candidates = sorted(f.eval_origin.unique(), reverse=True)
    if not by_id or not candidates:
        raise ValueError("No entities with enough eligible protocol rows")
    best_count = 0
    best_cutoff = None
    for raw_cutoff in candidates:
        cutoff = pd.Timestamp(raw_cutoff)
        available = []
        for identifier, (_, origins, maturity) in by_id.items():
            n_cal = bisect_right(maturity, cutoff)
            first_eval = bisect_left(origins, cutoff)
            if n_cal >= calibration_origins and (len(origins) - first_eval) >= evaluation_origins:
                # The two sets cannot overlap: target_end > origin and
                # _available_at <= cutoff for calibration, origin >= cutoff for eval.
                available.append((identifier, n_cal, first_eval))
        if len(available) > best_count:
            best_count, best_cutoff = len(available), cutoff
        if len(available) < entities:
            continue
        ranked = sorted(available, key=lambda t: (
            hashlib.sha256(f"{seed}|{t[0]}".encode()).digest(), t[0]))[:entities]
        chosen_rows = []
        per_entity = {}
        for identifier, n_cal, first_eval in ranked:
            g, _, _ = by_id[identifier]
            # Choose the most recent mature calibration and the first eligible
            # evaluation origins, both relative to the *same* pooled cutoff.
            cal = g.iloc[n_cal - calibration_origins:n_cal].copy()
            ev = g.iloc[first_eval:first_eval + evaluation_origins].copy()
            cal["split"] = "calibration"
            ev["split"] = "evaluation"
            chosen_rows.extend([cal, ev])
            per_entity[identifier] = dict(
                calibration_origins=[v.isoformat() for v in cal.eval_origin],
                evaluation_origins=[v.isoformat() for v in ev.eval_origin],
            )
        result = pd.concat(chosen_rows, ignore_index=True).drop(columns=["_available_at"])
        result = result.sort_values(["eval_origin", id_col, "eval_index"], kind="mergesort").reset_index(drop=True)
        cal = result[result.split == "calibration"]
        ev = result[result.split == "evaluation"]
        selected_mature = cal.target_end + maturity_delta
        if selected_mature.max() > ev.eval_origin.min():
            raise AssertionError("Pooled target maturity violated")
        counts = result.groupby([id_col, "split"]).size().unstack(fill_value=0)
        if (len(counts) != entities or not (counts.calibration == calibration_origins).all()
                or not (counts.evaluation == evaluation_origins).all()):
            raise AssertionError("Unbalanced per-entity origin counts")
        if result.duplicated([id_col, "eval_origin"]).any():
            raise AssertionError("Repeated entity/origin in selected cohort")
        details = dict(
            selection="latest_feasible_global_cutoff_entity_specific_origins",
            boundary=cutoff.isoformat(),
            eligible_entity_count=len(available),
            selected_entities=sorted(per_entity),
            calibration_rows=int(len(cal)),
            evaluation_rows=int(len(ev)),
            earliest_calibration=cal.eval_origin.min().isoformat(),
            latest_calibration_label_available=selected_mature.max().isoformat(),
            earliest_evaluation=ev.eval_origin.min().isoformat(),
            latest_evaluation=ev.eval_origin.max().isoformat(),
            evaluation_span_days=float((ev.eval_origin.max() - ev.eval_origin.min()) / pd.Timedelta(days=1)),
            per_entity=per_entity,
            caveat=("Forecast origins are entity-specific rather than common timestamps; "
                    "entity eligibility depends on data availability. This is a development cohort."),
        )
        return result, details

    raise ValueError(
        f"Cannot find {entities} entities with {calibration_origins} mature calibration "
        f"and {evaluation_origins} evaluation origins under a pooled time split; "
        f"maximum feasible: {best_count} at/near cutoff {best_cutoff}. "
        "Reduce --entities / origin counts or inspect the source protocol; "
        "do not select evaluation outcomes to construct the cohort."
    )
