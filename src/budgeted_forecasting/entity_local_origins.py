"""Leakage-safe, metadata-only, entity-local forecasting origin selection.

No global chronological comparison across unrelated people or recordings.
No target values or model predictions are used in cohort selection.
"""

import hashlib
import pandas as pd


def select_entity_local_origins(
    protocol, id_col, *,
    entities, calibration_origins, evaluation_origins,
    seed, resolution_minutes,
):
    if min(entities, calibration_origins, evaluation_origins, resolution_minutes) < 1:
        raise ValueError("Selection counts and resolution must be positive")

    required = {id_col, "eval_origin", "target_end", "eval_index"}
    if not required.issubset(protocol.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(protocol.columns))}")

    frame = protocol.copy()
    if frame.empty or frame[id_col].isna().any():
        raise ValueError("Empty protocol or missing entity IDs")

    frame[id_col] = frame[id_col].astype(str)

    for col in ("eval_origin", "target_end"):
        frame[col] = pd.to_datetime(frame[col], errors="raise")

    if frame[["eval_origin", "target_end"]].isna().any().any():
        raise ValueError("Missing timestamps")

    if (frame["target_end"] <= frame["eval_origin"]).any():
        raise ValueError("Invalid target interval")

    if frame.duplicated([id_col, "eval_origin"]).any():
        raise ValueError("Duplicate entity/origin")

    if frame.duplicated([id_col, "eval_index"]).any():
        raise ValueError("Duplicate entity/eval_index")

    cadence = pd.Timedelta(minutes=resolution_minutes)
    frame["_available_at"] = frame["target_end"] + cadence

    eligible = {}

    for entity, group in frame.groupby(id_col, sort=True):
        group = group.sort_values(
            ["eval_origin", "eval_index"], kind="mergesort"
        ).reset_index(drop=True)

        if len(group) < calibration_origins + evaluation_origins:
            continue

        # Reserve the last N origins for evaluation.
        ev = group.iloc[-evaluation_origins:].copy()
        first_eval = ev["eval_origin"].min()

        # Calibration labels must be available before evaluation.
        earlier = group.iloc[:-evaluation_origins]
        mature = earlier[earlier["_available_at"] <= first_eval]

        if len(mature) < calibration_origins:
            continue

        cal = mature.iloc[-calibration_origins:].copy()

        if cal["_available_at"].max() > first_eval:
            raise AssertionError(f"Temporal leakage: {entity}")

        cal["split"] = "calibration"
        ev["split"] = "evaluation"

        eligible[entity] = (cal, ev)

    ranked = sorted(
        eligible,
        key=lambda x: (
            hashlib.sha256(f"{seed}|{x}".encode()).digest(),
            x,
        ),
    )

    if len(ranked) < entities:
        raise ValueError(
            f"Requested {entities} entities but only "
            f"{len(ranked)} have eligible local origins"
        )

    chosen = ranked[:entities]
    rows = []
    details = {}

    for entity in chosen:
        cal, ev = eligible[entity]
        rows.extend((cal, ev))

        details[entity] = {
            "calibration_origins": [
                x.isoformat() for x in cal["eval_origin"]
            ],
            "evaluation_origins": [
                x.isoformat() for x in ev["eval_origin"]
            ],
            "latest_label_available": cal["_available_at"].max().isoformat(),
        }

    selected = (
        pd.concat(rows, ignore_index=True)
        .drop(columns=["_available_at"])
        .sort_values(["eval_origin", id_col, "eval_index"], kind="mergesort")
        .reset_index(drop=True)
    )

    counts = (
        selected.groupby([id_col, "split"])
        .size()
        .unstack(fill_value=0)
    )

    assert len(counts) == entities
    assert (counts["calibration"] == calibration_origins).all()
    assert (counts["evaluation"] == evaluation_origins).all()
    assert not selected.duplicated([id_col, "eval_origin"]).any()

    info = {
        "selection": "entity_local_chronology",
        "eligible_entity_count": len(eligible),
        "selected_entities": chosen,
        "calibration_rows": int(counts["calibration"].sum()),
        "evaluation_rows": int(counts["evaluation"].sum()),
        "per_entity": details,
        "caveat": (
            "Chronology enforced independently within each entity. "
            "No pooled calendar boundary or unseen-entity generalisation claimed."
        ),
    }

    return selected, info
