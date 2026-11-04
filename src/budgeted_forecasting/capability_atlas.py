"""PersTSFM Direction A B3.0: zero-shot model/context capability atlas.

Extends existing PersTSFM dataset/model APIs without modifying them.
No fine-tuning, no policy selection and no new dataset protocols.
Freeze -- one fixed cohort; run-action -- one independent model/context;
report -- verify shards and evaluate calibration/evaluation accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import FIELDS, load_cache
from .real_atlas import _timed_predict, _timestamp, select_origins, write_csv_atomic

MODEL_KEYS = (
    "last_value", "seasonal_naive", "chronos2", "tirex2", "moirai2_small",
    "toto2_22m", "toto2_313m", "timesfm3",
)
COST_FIELDS = ("case_id", "split", "action_id", "latency_ms", "peak_gpu_allocated_mb")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(arr) -> str:
    values = np.asarray(arr, dtype="<f8")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite observations are not permitted in B3")
    return hashlib.sha256(values.tobytes()).hexdigest()


def context_steps(hours: float, minutes: int) -> int:
    exact = float(hours) * 60 / minutes
    n = round(exact)
    if hours <= 0 or not np.isclose(n, exact):
        raise ValueError(f"Context {hours}h not divisible by {minutes}m resolution")
    return int(n)


def context_label(hours: float) -> str:
    hours = float(hours)
    if hours > 0 and hours % 24 == 0:
        return f"{int(hours / 24)}d"
    return f"{hours:g}".replace(".", "p") + "h"


def action_name(model: str, hours: float | None = None) -> str:
    if model not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model}; supported: {MODEL_KEYS}")
    if model in ("last_value", "seasonal_naive"):
        return model
    if hours is None:
        raise ValueError("TSFM actions require --context-hours")
    return f"{model}_{context_label(hours)}"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, timeout=5, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_json_new(path: Path, value: dict):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    if temp.exists():
        raise FileExistsError(f"Refusing to overwrite partial output {temp}")
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _registry():
    from src.data import registry
    return registry


def _load_selected(spec, selected: pd.DataFrame):
    """Use existing loaders, reading only the IDs needed for the frozen pilot."""
    r = _registry()
    loader = getattr(r, "_load_" + spec.loader_kind, None)
    if loader is None:
        raise ValueError(f"No existing dataset loader for {spec.loader_kind}")
    ids = sorted(selected[spec.id_col].astype(str).unique())
    if not spec.raw_path.exists():
        raise FileNotFoundError(spec.raw_path)
    return r.LoadedDataset(spec=spec, protocol=selected, series=loader(spec, ids))


def _extract(loaded, row, max_context_hours: float):
    """Independently validate the complete context/target grid and their hashes."""
    r = _registry()
    spec = loaded.spec
    item = r.extract_case(loaded, row, max_context_hours)
    ctx, y = item["context"], item["truth"]
    origin = pd.Timestamp(item["eval_origin"])
    delta = pd.Timedelta(minutes=int(spec.resolution_minutes))
    n = context_steps(max_context_hours, int(spec.resolution_minutes))
    expected_ctx = pd.date_range(end=origin, periods=n, freq=delta)
    expected_y = pd.date_range(start=origin + delta, periods=int(spec.forecast_steps), freq=delta)
    if not ctx.index.equals(expected_ctx) or not y.index.equals(expected_y):
        raise ValueError(f"Irregular timestamps, missing history or future overlap: {item['case_id']}")
    if ctx.index[-1] != origin or y.index[0] <= origin:
        raise ValueError(f"Context/target leakage: {item['case_id']}")
    arr = np.asarray(ctx, dtype=np.float64)
    truth = np.asarray(y, dtype=np.float64)
    return {
        "case_id": f"{spec.key}::{item['entity_id']}::{origin.isoformat()}",
        "series_id": str(item["entity_id"]),
        "origin_time": _timestamp(origin).isoformat(),
        "label_available_at": _timestamp(y.index[-1] + delta).isoformat(),
        "split": str(row["split"]),
        "eval_index": int(row["eval_index"]),
        "truth_sha256": array_hash(truth),
        "context_sha256": array_hash(arr),
    }, (ctx, truth)


def _protocol_index(spec, frame: pd.DataFrame):
    entries = {}
    for _, row in frame.iterrows():
        key = (str(row[spec.id_col]), _timestamp(row["eval_origin"]).isoformat())
        if key in entries:
            raise ValueError(f"Duplicate entity/origin in source protocol: {key}")
        entries[key] = row
    return entries


def freeze_cohort(dataset: str, protocol_split: str, entities: int, cal_origins: int,
                  eval_origins: int, max_context_hours: float, seed: int, out_dir: Path,
                  coverage_aware: bool = False, per_entity_origins: bool = False,
                  entity_local_origins: bool = False):
    r = _registry()
    spec = r.get_dataset_spec(dataset)
    if max_context_hours not in spec.context_hours:
        raise ValueError(f"Context {max_context_hours}h not registered for {dataset}; available={spec.context_hours}")
    protocol = r.load_protocol(spec, split=protocol_split)
    if sum(map(bool, (coverage_aware, per_entity_origins, entity_local_origins))) > 1:
        raise ValueError("Choose either --coverage-aware or --per-entity-origins, not both")
    selection_details = None
    if entity_local_origins:
        from .entity_local_origins import select_entity_local_origins
        selection, selection_details = select_entity_local_origins(
            protocol, spec.id_col, entities=entities,
            calibration_origins=cal_origins,
            evaluation_origins=eval_origins,
            seed=seed, resolution_minutes=int(spec.resolution_minutes))
    elif per_entity_origins:
        from .per_entity_origins import select_per_entity_origins
        selection, selection_details = select_per_entity_origins(
            protocol, spec.id_col, entities=entities,
            calibration_origins=cal_origins, evaluation_origins=eval_origins,
            seed=seed, resolution_minutes=int(spec.resolution_minutes))
    elif coverage_aware:
        from .coverage import select_coverage_aware
        selection, selection_details = select_coverage_aware(
            protocol, spec.id_col, entities=entities,
            calibration_origins=cal_origins, evaluation_origins=eval_origins, seed=seed)
    else:
        selection = select_origins(protocol, spec.id_col, entities=entities,
                                   calibration_origins=cal_origins,
                                   evaluation_origins=eval_origins, seed=seed)
    loaded = _load_selected(spec, selection)
    records = []
    for _, row in selection.iterrows():
        rec, _ = _extract(loaded, row, max_context_hours)
        records.append(rec)
    if len({x["case_id"] for x in records}) != len(records):
        raise ValueError("Duplicate frozen cases")
    cal = [x for x in records if x["split"] == "calibration"]
    ev = [x for x in records if x["split"] == "evaluation"]
    if not cal or not ev:
        raise ValueError("Both calibration and evaluation required")
    if not entity_local_origins and max(x["label_available_at"] for x in cal) > min(x["origin_time"] for x in ev):
        raise ValueError("Calibration labels must mature before first evaluation origin")

    if entity_local_origins:
        by_entity = {}
        for rec in records:
            by_entity.setdefault(rec["series_id"], {}).setdefault(
                rec["split"], []
            ).append(rec)

        if len(by_entity) != entities:
            raise AssertionError("Frozen entity count differs from request")

        for entity, parts in by_entity.items():
            entity_cal = parts.get("calibration", [])
            entity_ev = parts.get("evaluation", [])

            if len(entity_cal) != cal_origins or len(entity_ev) != eval_origins:
                raise AssertionError(f"Unbalanced cases: {entity}")

            if max(x["label_available_at"] for x in entity_cal) > min(
                x["origin_time"] for x in entity_ev
            ):
                raise AssertionError(f"Entity-local leakage: {entity}")

    payload = dict(schema_version="pertsfm_b3_cohort_v1", dataset_id=dataset,
                   protocol_split=protocol_split, seed=int(seed), max_context_hours=float(max_context_hours),
                   resolution_minutes=int(spec.resolution_minutes), horizon_steps=int(spec.forecast_steps),
                   protocol_sha256=sha256(spec.protocol_path), protocol_path=str(spec.protocol_path),
                   data_path=str(spec.raw_path), git_sha=_git_sha(),
                   selection=("entity_local_chronology" if entity_local_origins else
                     "per_entity_origins_pooled_cutoff" if per_entity_origins else
                              "coverage_aware_global_times" if coverage_aware else
                              "shared_origin_times; pooled calibration label maturity enforced"),
                   selection_details=selection_details,
                   timestamp_note=("Series-relative timestamps: chronology enforced within each entity only"
                                    if entity_local_origins else
                                    "Naive research timestamps UTC-tagged for ordering; no local timezone inferred"),
                   cases=records)
    path = Path(out_dir) / "cohort.json"
    _write_json_new(path, payload)
    print(f"PASS B3 freeze: {dataset}: {len(records)} cases, {len(set(x['series_id'] for x in records))} entities"
          f"; calibration={len(cal)} evaluation={len(ev)}; horizon={spec.forecast_steps}")
    print(f"Frozen cohort: {path} | sha256={sha256(path)}")
    return payload


def load_frozen_cases(path: Path):
    """Rebuild exact frozen case set, never reselect origins. Assert ground-truth and history hashes."""
    path = Path(path)
    cohort = json.loads(path.read_text())
    if cohort.get("schema_version") != "pertsfm_b3_cohort_v1":
        raise ValueError("Invalid B3 cohort schema")
    r = _registry()
    spec = r.get_dataset_spec(cohort["dataset_id"])
    if int(spec.forecast_steps) != cohort["horizon_steps"] or int(spec.resolution_minutes) != cohort["resolution_minutes"]:
        raise ValueError("Dataset registry changed since freeze")
    if sha256(spec.protocol_path) != cohort["protocol_sha256"]:
        raise ValueError("Protocol changed since cohort freeze")
    protocol = r.load_protocol(spec, split=cohort["protocol_split"])
    index = _protocol_index(spec, protocol)
    selected_rows = []
    for rec in cohort["cases"]:
        key = (rec["series_id"], rec["origin_time"])
        if key not in index:
            raise ValueError(f"Frozen origin absent from protocol: {key}")
        row = index[key].copy()
        if int(row["eval_index"]) != int(rec["eval_index"]):
            raise ValueError(f"eval_index changed for {key}")
        row["split"] = rec["split"]
        selected_rows.append(row)
    if len({r["case_id"] for r in cohort["cases"]}) != len(cohort["cases"]):
        raise ValueError("Duplicate case in frozen cohort")
    if not selected_rows:
        raise ValueError("Empty cohort")
    selected = pd.DataFrame(selected_rows)
    loaded = _load_selected(spec, selected)
    rebuilt = []
    for (_, row), rec in zip(selected.iterrows(), cohort["cases"]):
        actual, data = _extract(loaded, row, float(cohort["max_context_hours"]))
        if actual != rec:
            mismatched = [k for k in actual if actual[k] != rec.get(k)]
            raise ValueError(f"Frozen case changed {rec['case_id']}: {mismatched}")
        rebuilt.append((rec, *data))
    return cohort, rebuilt


def _simple_predict(model: str, context: np.ndarray, horizon: int, minutes: int):
    if model == "last_value":
        return np.repeat(context[-1], horizon)
    if model == "seasonal_naive":
        period = 1440 // minutes if (1440 % minutes) == 0 else None
        if period is None or len(context) < period:
            raise ValueError("Daily seasonal naive requires >= one complete day of history and daily-compatible cadence")
        return np.resize(context[-period:], horizon)
    raise ValueError(f"Unexpected simple model {model}")


def _load_cohort_cache(path, cohort):
    """Validate forecasts using the frozen cohort's chronology policy."""
    selection = cohort.get("selection", "")
    chronology = (
        "entity_local"
        if selection == "entity_local_chronology"
        else "pooled"
    )
    return load_cache(path, chronology=chronology)


def run_single_action(cohort_file: Path, model_key: str, hours: float | None,
                      output: Path, *, dry_run: bool = False):
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model_key}")
    cohort, cases = load_frozen_cases(cohort_file)
    spec = _registry().get_dataset_spec(cohort["dataset_id"])
    if model_key not in ("last_value", "seasonal_naive"):
        if hours is None or float(hours) not in spec.context_hours:
            raise ValueError(f"Model context must be registered for {spec.key}: {spec.context_hours}")
        if hours > cohort["max_context_hours"]:
            raise ValueError("Requested context exceeds frozen maximum")
        n = context_steps(hours, spec.resolution_minutes)
    else:
        n = context_steps(cohort["max_context_hours"], spec.resolution_minutes)
    action = action_name(model_key, hours)
    output = Path(output)
    costs_path = output.with_name(output.stem + "_costs.csv")
    manifest_path = output.with_name(output.stem + "_manifest.json")
    if dry_run:
        print(f"DRY RUN PASS: {len(cases)} exact {spec.key} cases, action={action}; model not loaded")
        return
    for p in (output, costs_path, manifest_path):
        if p.exists() or p.with_name(p.name + ".part").exists():
            raise FileExistsError(f"Refusing to overwrite existing or partial output: {p}")
    model = None
    load_ms = warmup_ms = 0.0
    sync = lambda: None
    reset_peak = read_peak = None
    if model_key not in ("last_value", "seasonal_naive"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Real TSFM profiling requires CUDA; refusing silent CPU fallback")
        from src.models.adapters import ForecastInput, build_model
        sync = torch.cuda.synchronize
        sync()
        start = time.perf_counter_ns()
        model = build_model(model_key, batch_size=1)
        sync()
        load_ms = (time.perf_counter_ns() - start) / 1e6
        reset_peak = torch.cuda.reset_peak_memory_stats
        read_peak = lambda: torch.cuda.max_memory_allocated() / 1e6
        rec, ctx, y = next(item for item in cases if item[0]["split"] == "calibration")
        values = np.asarray(ctx, dtype=np.float64)[-n:].copy()
        request = ForecastInput(case_id=rec["case_id"], timestamps=pd.DatetimeIndex(ctx.index[-n:]), values=values)
        with torch.inference_mode():
            _, warmup_ms = _timed_predict(lambda: _call_model(model, request, rec["case_id"],
                                                                   len(y), spec.resolution_minutes, n), sync)
    rows, costs = [], []
    for rec, ctx, truth in cases:
        arr = np.asarray(ctx, dtype=np.float64)
        if model_key in ("last_value", "seasonal_naive"):
            pred, ms = _timed_predict(lambda: _simple_predict(model_key, arr, len(truth), spec.resolution_minutes), sync)
            peak = None
        else:
            import torch
            request = ForecastInput(case_id=rec["case_id"], timestamps=pd.DatetimeIndex(ctx.index[-n:]),
                                    values=arr[-n:].copy())
            reset_peak()
            with torch.inference_mode():
                pred, ms = _timed_predict(lambda: _call_model(model, request, rec["case_id"],
                                                               len(truth), spec.resolution_minutes, n), sync)
            peak = float(read_peak())
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid forecast for {rec['case_id']} and {action}")
        costs.append(dict(case_id=rec["case_id"], split=rec["split"], action_id=action,
                          latency_ms=ms, peak_gpu_allocated_mb=peak))
        for step, (y, p) in enumerate(zip(truth, pred), start=1):
            rows.append(dict(dataset_id=cohort["dataset_id"], series_id=rec["series_id"],
                             case_id=rec["case_id"], split=rec["split"],
                             origin_time=rec["origin_time"], label_available_at=rec["label_available_at"],
                             action_id=action, step=step, truth=float(y), prediction=float(p), latency_ms=ms))
    # All computations and validations finished before producing persistent artifacts.
    write_csv_atomic(rows, output, FIELDS)
    validated = _load_cohort_cache(output, cohort)
    if len(validated) != len(cohort["cases"]):
        raise RuntimeError("Unexpected shard case count")
    write_csv_atomic(costs, costs_path, COST_FIELDS)
    _write_json_new(manifest_path, dict(schema_version="pertsfm_b3_shard_v1",
                                       cohort_sha256=sha256(cohort_file), action_id=action,
                                       shard_sha256=sha256(output), model_key=model_key,
                                       model_id=getattr(model, "model_id", None), context_hours=hours,
                                       dataset=cohort["dataset_id"], cases=len(validated),
                                       model_load_ms=load_ms, warmup_ms=warmup_ms,
                                       timing="Warm one-case inference, batch=1; CUDA-synchronised for TSFMs",
                                       peak_gpu_note="Allocated GPU memory includes resident model; not incremental",
                                       hardware=(torch.cuda.get_device_name() if model is not None else "CPU baseline"),
                                       python=platform.python_version(), git_sha=_git_sha()))
    print(f"PASS B3 action: {action}, {len(validated)} matched cases, {output}")


def _call_model(model, request, expected_id: str, horizon: int, minutes: int, n: int):
    outputs = model.predict_batch([request], horizon=horizon, resolution_minutes=minutes,
                                  context_steps=n, batch_size=1)
    if len(outputs) != 1 or outputs[0].case_id != expected_id:
        raise RuntimeError(f"Model response identity mismatch: {expected_id}")
    return np.asarray(outputs[0].prediction, dtype=np.float64).reshape(-1)


def analyze_shards(cohort_path: Path, shard_paths: list[Path], out_dir: Path):
    """Only complete, exactly matched shards are compared; no missing-case dropping."""
    cohort = json.loads(Path(cohort_path).read_text())
    records = {x["case_id"]: x for x in cohort["cases"]}
    if not shard_paths or not records:
        raise ValueError("At least one shard and one frozen case required")
    action_files = {}
    all_rows = []
    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        meta_path = shard_path.with_name(shard_path.stem + "_manifest.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing action manifest: {meta_path}")
        meta = json.loads(meta_path.read_text())
        if meta.get("schema_version") != "pertsfm_b3_shard_v1" or meta.get("cohort_sha256") != sha256(cohort_path):
            raise ValueError(f"Shard cohort identity mismatch: {shard_path}")
        if meta.get("shard_sha256") != sha256(shard_path):
            raise ValueError(f"Shard content hash mismatch: {shard_path}")
        cases = _load_cohort_cache(shard_path, cohort)
        if set(x.case_id for x in cases) != set(records):
            raise ValueError(f"Missing or extra cases in shard {shard_path}")
        actions = {a for c in cases for a in c.actions}
        if len(actions) != 1 or next(iter(actions)) != meta["action_id"]:
            raise ValueError("Shard action identity mismatch")
        action = next(iter(actions))
        if action in action_files:
            raise ValueError(f"Duplicate model/context action {action}")
        action_files[action] = str(shard_path)
        for case in cases:
            rec = records[case.case_id]
            if (case.dataset_id != cohort["dataset_id"] or case.series_id != rec["series_id"]
                or case.split != rec["split"] or case.origin_time.isoformat() != _timestamp(rec["origin_time"]).isoformat()
                or case.label_available_at.isoformat() != _timestamp(rec["label_available_at"]).isoformat()
                or array_hash(case.truth) != rec["truth_sha256"]):
                raise ValueError(f"Unmatched case metadata/target: {case.case_id}")
        with shard_path.open(newline="", encoding="utf-8") as file:
            all_rows.extend({k: row[k] for k in FIELDS} for row in csv.DictReader(file))
    # Full cache loader is the final pooled temporal and action-completeness validator.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        tmp_cache = Path(tmp) / "full.csv"
        write_csv_atomic(all_rows, tmp_cache, FIELDS)
        cases = _load_cohort_cache(tmp_cache, cohort)
    if len(cases) != len(records):
        raise RuntimeError("Merged case count changed")
    action_ids = sorted(action_files)
    per_case_rows = []
    per_series = defaultdict(list)
    summary = {}
    for case in cases:
        for action in action_ids:
            prediction = np.asarray(case.actions[action].prediction)
            truth = np.asarray(case.truth)
            err = prediction - truth
            entry = dict(case_id=case.case_id, series_id=case.series_id, split=case.split,
                         action_id=action, mae=float(np.mean(np.abs(err))),
                         rmse=float(np.sqrt(np.mean(err**2))), latency_ms=case.actions[action].latency_ms)
            per_case_rows.append(entry)
            per_series[(case.split, case.series_id, action)].append(entry["mae"])
    for split in ("calibration", "evaluation"):
        subset = [row for row in per_case_rows if row["split"] == split]
        summary[split] = {}
        for action in action_ids:
            group = [x for x in subset if x["action_id"] == action]
            if not group:
                raise ValueError(f"Missing {split} for {action}")
            by_series = [statistics.fmean(v) for (s, _, a), v in per_series.items() if s == split and a == action]
            times = sorted(x["latency_ms"] for x in group)
            summary[split][action] = dict(cases=len(group), entities=len(by_series),
                origin_macro_mae=statistics.fmean(x["mae"] for x in group),
                series_macro_mae=statistics.fmean(by_series),
                origin_macro_rmse=statistics.fmean(x["rmse"] for x in group),
                mean_latency_ms=statistics.fmean(times), median_latency_ms=statistics.median(times),
                p90_latency_ms=float(np.quantile(times, 0.9)))
    evaluation = [case for case in cases if case.split == "evaluation"]
    wins = {a: 0.0 for a in action_ids}
    for case in evaluation:
        per_a = {a: float(np.mean(np.abs(np.asarray(case.actions[a].prediction) - case.truth))) for a in action_ids}
        minimum = min(per_a.values())
        tied = [a for a, err in per_a.items() if np.isclose(err, minimum, atol=1e-9, rtol=0)]
        for action in tied:
            wins[action] += 1 / len(tied)
    best_cal = min(action_ids, key=lambda a: (summary["calibration"][a]["series_macro_mae"], a))
    report = dict(schema_version="pertsfm_b3_report_v1", dataset=cohort["dataset_id"],
                  cohort_sha256=sha256(cohort_path), case_count=len(cases), actions=action_ids,
                  calibration_selected_action=best_cal, calibration=summary["calibration"],
                  evaluation=summary["evaluation"],
                  hindsight_win_counts=wins, shard_paths=action_files,
                  note=("Zero-shot capability atlas. The evaluation winner per origin uses future truth and is "
                        "only a diagnostic. Context length may alter model quality. Warm batch-1 latency "
                        "excludes model loading and is not a complete compute cost."))
    out = Path(out_dir)
    for path in (out / "forecast_cache.csv", out / "report.json", out / "per_case.csv"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    write_csv_atomic(all_rows, out / "forecast_cache.csv", FIELDS)
    write_csv_atomic(per_case_rows, out / "per_case.csv",
                     ("case_id", "series_id", "split", "action_id", "mae", "rmse", "latency_ms"))
    _write_json_new(out / "report.json", report)
    print(f"PASS B3 report: dataset={cohort['dataset_id']}, cases={len(cases)}, actions={len(action_ids)}")
    print(f"Calibration-selected: {best_cal}")
    for action in sorted(action_ids, key=lambda a: summary['evaluation'][a]['series_macro_mae']):
        cal = summary['calibration'][action]
        ev = summary['evaluation'][action]
        print(f"  {action:24} | cal={cal['series_macro_mae']:.4f} | eval={ev['series_macro_mae']:.4f} "
              f"| win={wins[action]:.1f}/{len(evaluation)} | warm_mean={ev['mean_latency_ms']:.2f}ms")
    return report



def aggregate_reports(report_paths: list[Path], out_dir: Path):
    """Cross-dataset summary; normalise within each dataset, never pool raw MAEs."""
    if not report_paths:
        raise ValueError("At least one B3 dataset report required")
    rows, seen = [], set()
    for path in report_paths:
        report = json.loads(Path(path).read_text())
        if report.get("schema_version") != "pertsfm_b3_report_v1":
            raise ValueError(f"Unrecognised report format: {path}")
        dataset = report["dataset"]
        if dataset in seen:
            raise ValueError(f"Dataset {dataset} appears in multiple reports")
        seen.add(dataset)
        evaluation = report["evaluation"]
        reference = "seasonal_naive" if "seasonal_naive" in evaluation else "last_value"
        if reference not in evaluation:
            raise ValueError(f"Dataset {dataset} lacks seasonal_naive or last_value reference")
        baseline = evaluation[reference]["series_macro_mae"]
        if baseline <= 0:
            raise ValueError(f"Dataset {dataset}: nonpositive reference MAE")
        for action, result in evaluation.items():
            rows.append(dict(dataset=dataset, action_id=action, reference_action=reference,
                             calibration_selected=action == report["calibration_selected_action"],
                             evaluation_series_macro_mae=result["series_macro_mae"],
                             reference_series_macro_mae=baseline,
                             relative_mae_to_reference=result["series_macro_mae"] / baseline,
                             entity_count=result["entities"], evaluation_cases=result["cases"]))
    rows.sort(key=lambda x: (x["dataset"], x["relative_mae_to_reference"], x["action_id"]))
    out = Path(out_dir)
    write_csv_atomic(rows, out / "cross_dataset.csv",
                     ("dataset", "action_id", "reference_action", "calibration_selected", "evaluation_series_macro_mae",
                      "reference_series_macro_mae", "relative_mae_to_reference",
                      "entity_count", "evaluation_cases"))
    _write_json_new(out / "aggregate.json", dict(schema_version="pertsfm_b3_aggregate_v1",
        datasets=sorted(seen), reports=[str(x) for x in report_paths],
        note="Raw MAE is not comparable across datasets. Ratios are within-dataset exploratory comparisons. No final ranking is asserted."))
    print(f"PASS B3 aggregate: {len(seen)} datasets; {len(rows)} model/context rows -> {out}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze = subs.add_parser("freeze", help="Freeze a matched development cohort; no GPU")
    freeze.add_argument("--dataset", required=True)
    freeze.add_argument("--protocol-split", default="validation")
    freeze.add_argument("--entities", type=int, default=8)
    freeze.add_argument("--calibration-origins", type=int, default=3)
    freeze.add_argument("--evaluation-origins", type=int, default=3)
    freeze.add_argument("--seed", type=int, default=20261102)
    freeze.add_argument("--max-context-hours", type=float, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--coverage-aware", action="store_true",
                        help="Select latest metadata-only time block with enough eligible entities")
    freeze.add_argument("--per-entity-origins", action="store_true",
                        help="Use entity-specific forecast origins across one pooled calibration/evaluation cutoff")
    run = subs.add_parser("run-action", help="Run exactly one action on frozen cases")
    run.add_argument("--cohort", required=True, type=Path)
    run.add_argument("--model", required=True, choices=MODEL_KEYS)
    run.add_argument("--context-hours", type=float)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--dry-run", action="store_true")
    report = subs.add_parser("report", help="Strictly join shards and compare matched performance")
    report.add_argument("--cohort", required=True, type=Path)
    report.add_argument("--shard", required=True, action="append", type=Path)
    report.add_argument("--output-dir", required=True, type=Path)
    collect = subs.add_parser("aggregate", help="Cross-dataset table normalised to seasonal naive / last value")
    collect.add_argument("--report", required=True, type=Path, action="append")
    collect.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_cohort(args.dataset, args.protocol_split, args.entities, args.calibration_origins,
                      args.evaluation_origins, args.max_context_hours, args.seed, args.output_dir,
                      coverage_aware=args.coverage_aware,
                      per_entity_origins=args.per_entity_origins)
    elif args.command == "run-action":
        run_single_action(args.cohort, args.model, args.context_hours, args.output, dry_run=args.dry_run)
    elif args.command == "report":
        analyze_shards(args.cohort, args.shard, args.output_dir)
    else:
        aggregate_reports(args.report, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
