"""Direction A B1.1: small real electricity accuracy/cost pilot.

Reuses PersTSFM's frozen electricity validation origins, extraction and Chronos-2
adapter; never changes their code. Produces a cache readable by B0. Never uses
future targets as model inputs. Forecast errors are for offline feasibility only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import importlib.metadata
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .cache import FIELDS, load_cache
from .feasibility import run_feasibility

CONTEXT_HOURS = (96, 336)  # 4 days, 14 days
ACTIONS = ("seasonal_naive", "chronos2_4d", "chronos2_14d")


@dataclass(frozen=True)
class OriginCase:
    case_id: str
    entity_id: str
    split: str
    origin: pd.Timestamp
    label_available_at: pd.Timestamp
    context: pd.Series  # maximum history, ends at origin
    truth: np.ndarray


def select_origins(protocol: pd.DataFrame, id_col: str, *, entities: int,
                   calibration_origins: int, evaluation_origins: int,
                   seed: int) -> pd.DataFrame:
    """Choose disjoint global time blocks, with complete horizons before evaluation.

    'origins' counts distinct forecast-origin timestamps, not necessarily rows.
    If pooled chronology or shared-series coverage cannot be established, fail.
    """
    if min(entities, calibration_origins, evaluation_origins) < 1:
        raise ValueError("entities and origin counts must be >=1")
    required = {id_col, "eval_origin", "target_end", "eval_index"}
    if not required.issubset(protocol):
        raise ValueError(f"Protocol missing columns: {sorted(required - set(protocol))}")
    frame = protocol.copy()
    frame[id_col] = frame[id_col].astype(str)
    for col in ("eval_origin", "target_end"):
        frame[col] = pd.to_datetime(frame[col], errors="raise")
    if frame[["eval_origin", "target_end"]].isna().any().any():
        raise ValueError("Protocol contains missing timestamps")
    if (frame["target_end"] <= frame["eval_origin"]).any():
        raise ValueError("Protocol contains invalid forecast horizons")
    ids = sorted(frame[id_col].unique(), key=lambda e: (hashlib.sha256(f"{seed}|{e}".encode()).digest(), e))[:entities]
    if len(ids) != entities:
        raise ValueError(f"Only {len(ids)} entities available; requested {entities}")
    frame = frame[frame[id_col].isin(ids)].copy()
    if frame.duplicated([id_col, "eval_origin"]).any():
        raise ValueError("Repeated evaluation origin within an entity")
    origin_times = sorted(frame["eval_origin"].drop_duplicates().tolist())
    if len(origin_times) < calibration_origins + evaluation_origins:
        raise ValueError(f"Need >= {calibration_origins + evaluation_origins} distinct origin times; found {len(origin_times)}")
    evaluation_times = origin_times[-evaluation_origins:]
    eval_start = evaluation_times[0]
    mature = frame[frame["target_end"] < eval_start]
    eligible_times = sorted(mature["eval_origin"].drop_duplicates().tolist())
    if len(eligible_times) < calibration_origins:
        raise ValueError("Not enough calibration origins with all targets observed before first evaluation")
    calibration_times = eligible_times[-calibration_origins:]
    cal = frame[frame["eval_origin"].isin(calibration_times)].copy()
    cal["split"] = "calibration"
    ev = frame[frame["eval_origin"].isin(evaluation_times)].copy()
    ev["split"] = "evaluation"
    selected = pd.concat((cal, ev), ignore_index=True)
    if (cal["target_end"].max() >= ev["eval_origin"].min()):
        raise AssertionError("Temporal leakage in pooled split")
    cover = selected.groupby([id_col, "split"]).size().unstack(fill_value=0)
    if (cover.get("calibration", 0) == 0).any() or (cover.get("evaluation", 0) == 0).any():
        missing = cover[(cover.get("calibration", 0) == 0) | (cover.get("evaluation", 0) == 0)].index.tolist()
        raise ValueError(f"Selected entities missing calibration/evaluation origins: {missing}. Increase origin counts or change seed.")
    if selected.duplicated([id_col, "eval_origin"]).any():
        raise ValueError("Repeated evaluation origin within an entity")
    return selected.sort_values(["eval_origin", id_col, "eval_index"], kind="mergesort").reset_index(drop=True)


def _timestamp(value: object) -> pd.Timestamp:
    """Timestamp for B0's timezone-aware representation.

    Native electricity timestamps are timezone-naive. This assigns UTC purely
    as a monotonic *research index*, not a geographic timezone assertion.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def prepare_cases(loaded, selected: pd.DataFrame) -> list[OriginCase]:
    """Extract using existing PersTSFM code and add strictly regular-grid checks."""
    from src.data.registry import context_steps, extract_case

    spec = loaded.spec
    if spec.key != "electricity" or spec.resolution_minutes != 15 or spec.forecast_steps != 96:
        raise ValueError("B1.1 currently supports electricity 15-minute / horizon=96 only")
    cadence = pd.Timedelta(minutes=spec.resolution_minutes)
    n = context_steps(max(CONTEXT_HOURS), spec.resolution_minutes)
    cases: list[OriginCase] = []
    for _, row in selected.iterrows():
        item = extract_case(loaded, row, max(CONTEXT_HOURS))
        origin = pd.Timestamp(item["eval_origin"])
        ctx = item["context"]
        truth_series = item["truth"]
        expected_context = pd.date_range(end=origin, periods=n, freq=cadence)
        expected_target = pd.date_range(start=origin + cadence, periods=spec.forecast_steps, freq=cadence)
        if not ctx.index.equals(expected_context):
            raise ValueError(f"Irregular or misaligned context for {item['case_id']}")
        if not truth_series.index.equals(expected_target):
            raise ValueError(f"Irregular or misaligned truth for {item['case_id']}")
        truth = truth_series.to_numpy(dtype=np.float64)
        history = ctx.to_numpy(dtype=np.float64)
        if not (np.isfinite(truth).all() and np.isfinite(history).all()):
            raise ValueError(f"Nonfinite context or target for {item['case_id']}")
        label_at = _timestamp(truth_series.index[-1] + cadence)
        # Namespaced ID, stable across runs and distinct per entity/time.
        case_id = f"electricity::{item['entity_id']}::{origin.isoformat()}"
        cases.append(OriginCase(case_id=case_id, entity_id=item["entity_id"],
                                split=str(row["split"]), origin=_timestamp(origin),
                                label_available_at=label_at, context=ctx,
                                truth=truth))
    if not cases:
        raise ValueError("No usable cases")
    cal = [c for c in cases if c.split == "calibration"]
    ev = [c for c in cases if c.split == "evaluation"]
    if not cal or not ev or max(c.label_available_at for c in cal) > min(c.origin for c in ev):
        raise ValueError("Pooled calibration labels unavailable when evaluation starts")
    return cases


def load_electricity_pilot(*, entities: int = 4, calibration_origins: int = 4,
                           evaluation_origins: int = 4, seed: int = 20261102,
                           protocol_split: str = "validation") -> tuple[list[OriginCase], dict]:
    from src.data.registry import (LoadedDataset, _load_electricity,
                                   get_dataset_spec, load_protocol)
    spec = get_dataset_spec("electricity")
    if not spec.protocol_path.is_file() or not spec.raw_path.is_file():
        raise FileNotFoundError(f"Missing electricity data. Protocol: {spec.protocol_path}; raw: {spec.raw_path}")
    protocol = load_protocol(spec, split=protocol_split)
    selection = select_origins(protocol, spec.id_col, entities=entities,
                               calibration_origins=calibration_origins,
                               evaluation_origins=evaluation_origins, seed=seed)
    selected_ids = sorted(selection[spec.id_col].astype(str).unique())
    # Selectively loads only pilot series; avoids loading all ~351 eligible entities.
    series = _load_electricity(spec, selected_ids)
    loaded = LoadedDataset(spec=spec, protocol=selection, series=series)
    cases = prepare_cases(loaded, selection)
    return cases, {
        "dataset": "electricity", "protocol_split": protocol_split,
        "entities": selected_ids, "seed": seed,
        "calibration_cases": sum(c.split == "calibration" for c in cases),
        "evaluation_cases": sum(c.split == "evaluation" for c in cases),
        "calibration_last_label_available_at": max(c.label_available_at.isoformat() for c in cases if c.split == "calibration"),
        "evaluation_first_origin": min(c.origin.isoformat() for c in cases if c.split == "evaluation"),
        "timestamp_note": "Source naive electricity timestamps are UTC-tagged for ordering only; no local timezone inferred",
        "context_hours": list(CONTEXT_HOURS), "horizon_steps": 96,
        "resolution_minutes": 15,
    }


def seasonal_naive(context: np.ndarray, horizon: int, period: int = 96) -> np.ndarray:
    values = np.asarray(context, dtype=np.float64).reshape(-1)
    if len(values) < period or period < 1 or horizon < 1 or not np.isfinite(values).all():
        raise ValueError("Invalid seasonal naive context/horizon")
    return np.resize(values[-period:], horizon)


def _timed_predict(predict: Callable[[], np.ndarray], sync: Callable[[], None]) -> tuple[np.ndarray, float]:
    sync()
    start = time.perf_counter_ns()
    result = np.asarray(predict(), dtype=np.float64).reshape(-1)
    sync()
    elapsed_ms = max((time.perf_counter_ns() - start) / 1_000_000.0, 0.000001)
    return result, elapsed_ms


def build_cache_rows(cases: list[OriginCase], model, *, sync: Callable[[], None] = lambda: None,
                     forecast_input_factory=None, require_model: bool = True,
                     reset_gpu_peak: Callable[[], None] | None = None,
                     read_gpu_peak_mb: Callable[[], float] | None = None) -> tuple[list[dict], list[dict]]:
    """Run one origin/action per inference call (warm, batch=1); return cache+cost log.

    Pass a fake model and ForecastInput factory for CPU-only integration tests.
    """
    if forecast_input_factory is None and model is not None:
        from src.models.adapters import ForecastInput
        forecast_input_factory = ForecastInput
    if require_model and model is None:
        raise ValueError("Chronos-2 model is required for the real three-action atlas")
    all_rows: list[dict] = []
    action_costs: list[dict] = []
    for case in cases:
        ctx = case.context.to_numpy(dtype=np.float64)
        for action in ACTIONS:
            if reset_gpu_peak is not None:
                reset_gpu_peak()
            if action == "seasonal_naive":
                result, elapsed = _timed_predict(lambda: seasonal_naive(ctx, len(case.truth)), sync)
            else:
                hours = 96 if action == "chronos2_4d" else 336
                n = hours * 4  # 15-minute electricity observations
                input_item = forecast_input_factory(
                    case_id=case.case_id,
                    timestamps=pd.DatetimeIndex(case.context.index[-n:]),
                    values=ctx[-n:].copy(),
                )
                def forecast():
                    output = model.predict_batch([input_item], horizon=len(case.truth),
                                                 resolution_minutes=15, context_steps=n,
                                                 batch_size=1)
                    if len(output) != 1 or output[0].case_id != case.case_id:
                        raise RuntimeError(f"Adapter response mismatch: {case.case_id}")
                    return output[0].prediction
                result, elapsed = _timed_predict(forecast, sync)
            if len(result) != len(case.truth) or not np.isfinite(result).all():
                raise ValueError(f"Invalid {action} forecast for {case.case_id}: {result.shape}")
            peak_mb = read_gpu_peak_mb() if read_gpu_peak_mb is not None else None
            action_costs.append({"case_id": case.case_id, "split": case.split,
                                 "action_id": action, "latency_ms": elapsed,
                                 "peak_gpu_allocated_mb": peak_mb})
            for step, (y, pred) in enumerate(zip(case.truth, result), 1):
                all_rows.append({"dataset_id": "electricity", "series_id": case.entity_id,
                                 "case_id": case.case_id, "split": case.split,
                                 "origin_time": case.origin.isoformat(),
                                 "label_available_at": case.label_available_at.isoformat(),
                                 "action_id": action, "step": step, "truth": float(y),
                                 "prediction": float(pred), "latency_ms": elapsed})
    return all_rows, action_costs


def write_csv_atomic(rows: list[dict], path: Path, fields: tuple[str, ...]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    if temp.exists():
        raise FileExistsError(f"Partial file exists: {temp}")
    try:
        with temp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
        if path.exists():
            raise FileExistsError(f"Output appeared during write: {path}")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def auto_budgets(cases) -> list[float]:
    cal = [case for case in cases if case.split == "calibration"]
    ids = sorted(cal[0].actions)
    medians = [statistics.median(c.actions[action].latency_ms for c in cal) for action in ids]
    # Exact observed median thresholds, suitable for a pilot (not independent profiling).
    return sorted(set(float(cost) for cost in medians))


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_sha() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=True, timeout=4)
        return p.stdout.strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Direction A B1.1 real electricity pilot (existing PersTSFM adapters)")
    p.add_argument("command", choices=("preflight", "run"))
    p.add_argument("--entities", type=int, default=4)
    p.add_argument("--calibration-origins", type=int, default=4,
                   help="Number of distinct origin timestamps for calibration")
    p.add_argument("--evaluation-origins", type=int, default=4,
                   help="Number of distinct origin timestamps for evaluation")
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--protocol-split", choices=("development", "validation"), default="validation")
    p.add_argument("--output-dir", type=Path, default=Path("results/direction_a/b1_electricity"))
    args = p.parse_args(argv)
    cases, details = load_electricity_pilot(entities=args.entities,
                                            calibration_origins=args.calibration_origins,
                                            evaluation_origins=args.evaluation_origins,
                                            seed=args.seed, protocol_split=args.protocol_split)
    print("B1.1 ELECTRICITY PILOT")
    print(json.dumps(details, indent=2))
    if args.command == "preflight":
        print("PREFLIGHT PASS: real input/target windows valid; no GPU inference executed")
        return 0
    paths = [args.output_dir / name for name in ("forecast_cache.csv", "measured_costs.csv", "manifest.json", "feasibility.json", "decisions.csv")]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Existing B1.1 outputs; use a new output-dir: {existing}")
    import torch
    from src.models.adapters import build_model
    if not torch.cuda.is_available():
        raise RuntimeError("B1.1 TSFM profiling requires CUDA on RTX6 (no silent CPU fallback)")
    sync = torch.cuda.synchronize
    sync()
    t0 = time.perf_counter_ns()
    model = build_model("chronos2", batch_size=1)
    sync()
    model_load_ms = (time.perf_counter_ns() - t0) / 1_000_000
    # Warm model with a calibration-origin input. Warmup cost recorded separately.
    warm = next(c for c in cases if c.split == "calibration")
    from src.models.adapters import ForecastInput
    warmup_by_context_ms = {}
    for hours in CONTEXT_HOURS:
        ctx = warm.context.iloc[-(hours * 4):]
        sync()
        t1 = time.perf_counter_ns()
        with torch.inference_mode():
            _ = model.predict_batch([ForecastInput(warm.case_id, pd.DatetimeIndex(ctx.index),
                                                  ctx.to_numpy(dtype=np.float64))],
                                    horizon=96, resolution_minutes=15,
                                    context_steps=hours * 4, batch_size=1)
        sync()
        warmup_by_context_ms[str(hours)] = (time.perf_counter_ns() - t1) / 1_000_000
    with torch.inference_mode():
        rows, costs = build_cache_rows(cases, model, sync=sync, forecast_input_factory=ForecastInput,
                                       reset_gpu_peak=torch.cuda.reset_peak_memory_stats,
                                       read_gpu_peak_mb=lambda: torch.cuda.max_memory_allocated() / 1e6)
    cache_path, cost_path, manifest_path, report_path, decision_path = paths
    write_csv_atomic(rows, cache_path, FIELDS)
    validated = load_cache(cache_path)
    budgets = auto_budgets(validated)
    report, decisions = run_feasibility(validated, budgets)
    write_csv_atomic(costs, cost_path,
                     ("case_id", "split", "action_id", "latency_ms", "peak_gpu_allocated_mb"))
    from dataclasses import asdict
    write_csv_atomic([asdict(d) for d in decisions], decision_path, tuple(asdict(decisions[0])))
    model_config = getattr(getattr(model, "pipeline", None), "model", None)
    model_config = getattr(model_config, "config", None)
    checkpoint_revision = getattr(model_config, "_commit_hash", None)
    git_status = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, check=False)
    manifest = {**details, "status": "real_small_pilot", "git_head": git_sha(),
                "git_dirty": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
                "python_version": platform.python_version(), "numpy_version": np.__version__,
                "pandas_version": pd.__version__, "torch_version": torch.__version__,
                "chronos_package_version": package_version("chronos-forecasting"),
                "checkpoint_revision_if_available": checkpoint_revision,
                "cuda_device": torch.cuda.get_device_name(),
                "model": "amazon/chronos-2", "model_loading_ms": model_load_ms,
                "warmup_by_context_ms": warmup_by_context_ms, "batch_size": 1,
                "timing": "per-origin warm synchronized wall-clock call time; no model loading",
                "peak_gpu_memory_note": "Absolute max CUDA allocated bytes including resident model, reset for each action; not incremental memory",
                "actions": list(ACTIONS), "budget_ms": budgets,
                "note": "Pilot-specific measured latencies; do not generalise to GPU throughput/batching."}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(validated)} cases, {len(ACTIONS)} actions; reports at {args.output_dir}")
    for b in report["budgets"]:
        print(f"  {b['budget_ms']:.3f} ms: fixed MAE={b['strategies']['validation_selected']['macro_mae']:.4f}; "
              f"hindsight oracle MAE={b['strategies']['hindsight_oracle']['macro_mae']:.4f}; "
              f"gap={b['oracle_gap_percent_of_fixed']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
