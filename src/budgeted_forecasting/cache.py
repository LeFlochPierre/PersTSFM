"""Load and validate a long-form forecast cache with strict temporal contracts.

A row is ONE (forecast origin, expert/action, target horizon step). The latency
value is the measured *per origin and action* execution time, repeated per step;
it is never summed over horizon steps. No model call is made here.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

FIELDS = (
    "dataset_id", "series_id", "case_id", "split", "origin_time",
    "label_available_at", "action_id", "step", "truth", "prediction",
    "latency_ms",
)
SPLITS = {"calibration", "evaluation"}


@dataclass(frozen=True)
class Candidate:
    action_id: str
    prediction: tuple[float, ...]
    latency_ms: float


@dataclass(frozen=True)
class Case:
    dataset_id: str
    series_id: str
    case_id: str
    split: str
    origin_time: datetime
    label_available_at: datetime
    truth: tuple[float, ...]
    actions: Mapping[str, Candidate]


def parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 datetime: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timestamp must include timezone offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def _float(value: str, label: str, line: int) -> float:
    try:
        parsed = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Line {line}: invalid {label}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Line {line}: nonfinite {label}")
    return parsed


def load_cache(path: str | Path, *, chronology: str = "pooled") -> list[Case]:
    if chronology not in ("pooled", "entity_local"):
        raise ValueError(f"Unsupported chronology policy: {chronology}")
    """Read a cache, rejecting missing experts, incomplete horizons and leakage.

    All cases must have the same set of action IDs for this first feasibility
    implementation. Series may have different forecast horizons across datasets,
    but every action for a given case must predict exactly the same steps.
    """
    path = Path(path)
    groups: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            missing = sorted(set(FIELDS) - set(reader.fieldnames or []))
            raise ValueError(f"Missing required columns: {missing}")
        for line, row in enumerate(reader, start=2):
            case_id = row["case_id"].strip()
            action_id = row["action_id"].strip()
            dataset = row["dataset_id"].strip()
            series = row["series_id"].strip()
            split = row["split"].strip()
            if not all((case_id, action_id, dataset, series)):
                raise ValueError(f"Line {line}: empty case/action/dataset/series identifier")
            if split not in SPLITS:
                raise ValueError(f"Line {line}: split must be calibration or evaluation")
            origin = parse_time(row["origin_time"])
            available = parse_time(row["label_available_at"])
            if available <= origin:
                raise ValueError(f"Line {line}: label availability must be after forecast origin")
            try:
                step = int(row["step"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Line {line}: invalid forecast step") from exc
            if step < 1:
                raise ValueError(f"Line {line}: forecast step must be positive")
            truth = _float(row["truth"], "truth", line)
            prediction = _float(row["prediction"], "prediction", line)
            latency = _float(row["latency_ms"], "latency_ms", line)
            if latency < 0:
                raise ValueError(f"Line {line}: latency_ms must be nonnegative")
            meta = (dataset, series, split, origin, available)
            group = groups.setdefault(case_id, {"meta": meta, "truth": {}, "actions": {}, "latencies": {}})
            if group["meta"] != meta:
                raise ValueError(f"Line {line}: inconsistent metadata for case_id={case_id}")
            if step in group["truth"] and group["truth"][step] != truth:
                raise ValueError(f"Line {line}: inconsistent truth across actions for case_id={case_id}, step={step}")
            group["truth"][step] = truth
            action_steps = group["actions"].setdefault(action_id, {})
            if step in action_steps:
                raise ValueError(f"Line {line}: duplicate case/action/step: {case_id}/{action_id}/{step}")
            action_steps[step] = prediction
            if action_id in group["latencies"] and group["latencies"][action_id] != latency:
                raise ValueError(f"Line {line}: inconsistent latency for case/action: {case_id}/{action_id}")
            group["latencies"][action_id] = latency

    if not groups:
        raise ValueError("Forecast cache is empty")
    cases: list[Case] = []
    all_actions: set[str] | None = None
    for case_id, group in groups.items():
        steps = sorted(group["truth"])
        if steps != list(range(1, len(steps) + 1)):
            raise ValueError(f"Case {case_id}: missing forecast steps")
        ids = set(group["actions"])
        if all_actions is None:
            all_actions = ids
        elif ids != all_actions:
            raise ValueError(f"Case {case_id}: missing or extra actions relative to cache action set")
        predictions = {}
        for action_id, values in group["actions"].items():
            if sorted(values) != steps:
                raise ValueError(f"Case {case_id}, action {action_id}: incomplete horizon")
            predictions[action_id] = Candidate(
                action_id, tuple(values[s] for s in steps), group["latencies"][action_id]
            )
        dataset, series, split, origin, available = group["meta"]
        cases.append(Case(
            dataset, series, case_id, split, origin, available,
            tuple(group["truth"][s] for s in steps), predictions,
        ))

    # Distinct cases at the same origin would provide duplicated evaluation mass.
    seen_origins: set[tuple[str, str, datetime]] = set()
    series_cases: dict[tuple[str, str], list[Case]] = {}
    for case in cases:
        key = (case.dataset_id, case.series_id, case.origin_time)
        if key in seen_origins:
            raise ValueError(f"Duplicate forecast origin for series: {key}")
        seen_origins.add(key)
        series_cases.setdefault(key[:2], []).append(case)
    for key, items in series_cases.items():
        cal = [c for c in items if c.split == "calibration"]
        evaluation = [c for c in items if c.split == "evaluation"]
        if chronology == "entity_local" and (not cal or not evaluation):
            raise ValueError(
                f"Entity-local cache requires calibration and evaluation for {key}"
            )
        if cal and evaluation and max(c.label_available_at for c in cal) > min(c.origin_time for c in evaluation):
            raise ValueError(f"Temporal leakage: calibration labels overlap evaluation origins for {key}")
    calibration = [c for c in cases if c.split == "calibration"]
    evaluation = [c for c in cases if c.split == "evaluation"]
    if not calibration:
        raise ValueError("Missing calibration cases")
    if not evaluation:
        raise ValueError("Missing evaluation cases")
    # A pooled calibration policy is fitted ONCE before evaluation. Therefore
    # every calibration outcome must already be known at the first evaluation
    # forecast origin, even when origins belong to different series/datasets.
    if chronology == "pooled" and max(c.label_available_at for c in calibration) > min(c.origin_time for c in evaluation):
        raise ValueError("Temporal leakage: pooled calibration labels are not all available before evaluation begins")
    return sorted(cases, key=lambda c: (c.dataset_id, c.series_id, c.origin_time, c.case_id))
