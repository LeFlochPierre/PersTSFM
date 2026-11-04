"""Direction A B1.2: independent TSFM actions, matched to an existing B1.1 atlas.

This extension never modifies the original real_atlas, B0 benchmark, or any
existing forecast caches. One model is loaded per process to reduce memory
pressure and permit distinct existing Conda environments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .cache import FIELDS, Case, load_cache
from .feasibility import run_feasibility
from .real_atlas import (
    OriginCase, _timed_predict, auto_budgets, git_sha,
    prepare_cases, write_csv_atomic,
)

SUPPORTED_MODELS = {
    "chronos2", "timesfm3", "tirex2", "toto2_22m", "toto2_313m", "moirai2_small",
}
CONTEXT_HOURS = (96, 336)


def action_name(model_key: str, context_hours: int) -> str:
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_key!r}. Supported: {sorted(SUPPORTED_MODELS)}")
    if context_hours not in CONTEXT_HOURS:
        raise ValueError(f"Unsupported context {context_hours}h; use 96 or 336")
    return f"{model_key}_{context_hours // 24}d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _case_map(cases: list[Case]):
    return {case.case_id: case for case in cases}


def _assert_matched_cases(base: list[Case], real: list[OriginCase]) -> None:
    """Check that a shard uses the *exact* B1.1 origins, targets and splits."""
    a, b = _case_map(base), {case.case_id: case for case in real}
    if set(a) != set(b):
        raise ValueError(f"Case mismatch: missing={sorted(set(a)-set(b))[:5]}, extra={sorted(set(b)-set(a))[:5]}")
    for cid, ref in a.items():
        got = b[cid]
        if (ref.dataset_id != "electricity" or ref.series_id != got.entity_id
            or ref.split != got.split or ref.origin_time != got.origin.to_pydatetime()
            or ref.label_available_at != got.label_available_at.to_pydatetime()):
            raise ValueError(f"Case metadata mismatch for {cid}")
        if len(ref.truth) != len(got.truth) or not np.allclose(ref.truth, got.truth, rtol=0, atol=1e-8):
            raise ValueError(f"Ground-truth mismatch for {cid}")


def load_matched_dataset(base_cache: Path, base_manifest: Path):
    """Rebuild EXACT baseline cache origins from the frozen validation protocol.

    A B1.1 global-origin selection can give different entities different numbers
    of cases. Resampling by counts or seed is not an exact inverse operation.
    Cache IDs/metadata identify the required cases; raw protocol and observed
    targets independently verify them. All reconstruction remains read-only.
    """
    import pandas as pd
    from src.data.registry import LoadedDataset, _load_electricity, get_dataset_spec, load_protocol

    base = load_cache(base_cache)
    info = json.loads(base_manifest.read_text(encoding="utf-8"))
    if info.get("dataset") != "electricity" or info.get("horizon_steps") != 96:
        raise ValueError("Expected B1.1 electricity baseline manifest (horizon=96)")
    entities = sorted({case.series_id for case in base})
    if entities != sorted(str(x) for x in info["entities"]):
        raise ValueError("Manifest entity list does not match the base cache")
    spec = get_dataset_spec("electricity")
    if spec.key != "electricity" or spec.resolution_minutes != 15 or spec.forecast_steps != 96:
        raise ValueError("Unexpected electricity registry configuration")
    protocol = load_protocol(spec, split=info["protocol_split"]).copy()
    protocol[spec.id_col] = protocol[spec.id_col].astype(str)
    protocol["eval_origin"] = pd.to_datetime(protocol["eval_origin"], errors="raise")
    if protocol["eval_origin"].isna().any():
        raise ValueError("Protocol contains missing evaluation origins")
    if protocol["eval_origin"].dt.tz is not None:
        raise ValueError("Expected timezone-naive electricity protocol origins")
    if protocol.duplicated([spec.id_col, "eval_origin"]).any():
        raise ValueError("Duplicate entity/origin in source protocol")

    # Electricity uses a naive chronological index; B0 cache timestamps are
    # UTC-tagged solely for stable chronology, not as a geographic assertion.
    required = {}
    for case in base:
        if case.dataset_id != "electricity":
            raise ValueError(f"Unexpected dataset in baseline cache: {case.dataset_id}")
        origin = pd.Timestamp(case.origin_time).tz_convert("UTC").tz_localize(None)
        key = (case.series_id, origin)
        if key in required:
            raise ValueError(f"Duplicate baseline origin: {key}")
        required[key] = case

    selected_rows = []
    seen = set()
    for _, row in protocol.iterrows():
        key = (str(row[spec.id_col]), pd.Timestamp(row["eval_origin"]))
        if key in required:
            case = required[key]
            part = row.copy()
            part["split"] = case.split
            selected_rows.append(part)
            seen.add(key)
    missing = sorted(set(required) - seen)
    if missing:
        raise ValueError(f"Baseline origins missing from source protocol: {missing[:5]}")
    selected = pd.DataFrame(selected_rows)
    if len(selected) != len(base):
        raise ValueError("Reconstructed origin count does not match baseline cache")
    selected = selected.sort_values(["eval_origin", spec.id_col, "eval_index"], kind="mergesort")
    series = _load_electricity(spec, entities)
    loaded = LoadedDataset(spec=spec, protocol=selected, series=series)
    real = prepare_cases(loaded, selected)
    _assert_matched_cases(base, real)
    details = {
        "selection": "exact_origin_keys_from_base_cache_checked_against_protocol_and_truth",
        "dataset": "electricity", "protocol_split": info["protocol_split"],
        "entities": entities, "seed_original": info.get("seed"),
        "cases": len(real),
        "calibration_cases": sum(c.split == "calibration" for c in real),
        "evaluation_cases": sum(c.split == "evaluation" for c in real),
    }
    return base, real, info, details


def _predict(model, case: OriginCase, context_hours: int, factory, sync):
    n = context_hours * 4  # electricity observations at 15-minute frequency
    from pandas import DatetimeIndex
    index = DatetimeIndex(case.context.index[-n:])
    values = case.context.to_numpy(dtype=np.float64)[-n:]
    if len(values) != n or not np.isfinite(values).all():
        raise ValueError(f"Invalid context for {case.case_id}")
    request = factory(case_id=case.case_id, timestamps=index, values=values)

    def run():
        outputs = model.predict_batch([request], horizon=len(case.truth),
                                      resolution_minutes=15, context_steps=n, batch_size=1)
        if len(outputs) != 1 or outputs[0].case_id != case.case_id:
            raise RuntimeError(f"Adapter output case mismatch: {case.case_id}")
        return outputs[0].prediction

    prediction, latency_ms = _timed_predict(run, sync)
    if prediction.shape != case.truth.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Bad prediction for {case.case_id}: shape={prediction.shape}")
    return prediction, latency_ms


def make_shard_rows(real: list[OriginCase], model, context_hours: int, model_key: str,
                    factory, sync, reset_peak=None, read_peak=None):
    action_id = action_name(model_key, context_hours)
    rows, costs = [], []
    for case in real:
        if reset_peak is not None:
            reset_peak()
        pred, ms = _predict(model, case, context_hours, factory, sync)
        peak = read_peak() if read_peak is not None else None
        costs.append(dict(case_id=case.case_id, split=case.split,
                          action_id=action_id, latency_ms=ms, peak_gpu_allocated_mb=peak))
        for step, (y, p) in enumerate(zip(case.truth, pred), 1):
            rows.append(dict(dataset_id="electricity", series_id=case.entity_id,
                             case_id=case.case_id, split=case.split,
                             origin_time=case.origin.isoformat(),
                             label_available_at=case.label_available_at.isoformat(),
                             action_id=action_id, step=step, truth=float(y),
                             prediction=float(p), latency_ms=ms))
    return rows, costs


def run_action(args) -> int:
    base_path, manifest_path = args.base_cache, args.base_manifest
    if not base_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Baseline cache/manifest missing: run B1.1 first")
    action_id = action_name(args.model, args.context_hours)
    base, real, _, details = load_matched_dataset(base_path, manifest_path)
    if action_id in base[0].actions:
        raise ValueError(f"Action already present in B1.1 baseline: {action_id}")
    print(f"Matched {len(real)} electricity forecast origins; action={action_id}")
    if args.dry_run:
        print("DRY RUN PASS: targets, origins, splits, and context verified; no model loaded")
        return 0
    output = args.output
    costs_path = output.with_name(output.stem + "_costs.csv")
    meta_path = output.with_name(output.stem + "_manifest.json")
    for path in (output, costs_path, meta_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output {path}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmarking requires CUDA; no silent CPU timing fallback")
    from src.models.adapters import ForecastInput, build_model
    sync = torch.cuda.synchronize
    sync()
    start = time.perf_counter_ns()
    model = build_model(args.model, batch_size=1)
    sync()
    load_ms = (time.perf_counter_ns() - start) / 1e6
    warm = next(c for c in real if c.split == "calibration")
    with torch.inference_mode():
        _, warmup_ms = _predict(model, warm, args.context_hours, ForecastInput, sync)
        rows, costs = make_shard_rows(real, model, args.context_hours, args.model,
                                     ForecastInput, sync,
                                     reset_peak=torch.cuda.reset_peak_memory_stats,
                                     read_peak=lambda: torch.cuda.max_memory_allocated() / 1e6)
    write_csv_atomic(rows, output, FIELDS)
    checked = load_cache(output)
    if len(checked) != len(base):
        raise RuntimeError("Unexpected shard count after writing")
    write_csv_atomic(costs, costs_path,
                     ("case_id", "split", "action_id", "latency_ms", "peak_gpu_allocated_mb"))
    git = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    obj = dict(schema_version="direction_a_shard_v1", action_id=action_id,
               model_key=args.model, model_id=getattr(model, "model_id", None),
               context_hours=args.context_hours, base_cache_sha256=sha256(base_path),
               base_manifest_sha256=sha256(manifest_path), shard_sha256=sha256(output),
               cases=len(checked), origin_selection=details, git_sha=git_sha(),
               git_dirty=bool(git.stdout.strip()) if git.returncode == 0 else None,
               model_load_ms=load_ms, warmup_ms=warmup_ms,
               timing="single-case warm, CUDA-synchronised wall clock; batch size 1",
               gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
               python_version=platform.python_version(),
               note="Absolute peak GPU memory includes resident model; not incremental cost."
               " Model initialization and warmup are excluded from warm per-forecast latency.")
    meta_path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"PASS: {len(checked)} cases for {action_id}; shard={output}; costs={costs_path}")
    return 0


def _assert_matched_cache(base: list[Case], shard: list[Case]) -> str:
    a, b = _case_map(base), _case_map(shard)
    if set(a) != set(b):
        raise ValueError("Shard and base case IDs do not match exactly")
    actions = {x for case in shard for x in case.actions}
    if len(actions) != 1:
        raise ValueError("Each shard must contain exactly one action")
    action = next(iter(actions))
    if action in base[0].actions:
        raise ValueError(f"Duplicate action {action}")
    for cid in a:
        x, y = a[cid], b[cid]
        if ((x.dataset_id, x.series_id, x.split, x.origin_time, x.label_available_at)
            != (y.dataset_id, y.series_id, y.split, y.origin_time, y.label_available_at)):
            raise ValueError(f"Shard metadata mismatch at {cid}")
        if x.truth != y.truth:
            raise ValueError(f"Shard truth mismatch at {cid}")
    return action


def _read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [{k: row[k] for k in FIELDS} for row in reader]


def merge(base_path: Path, shard_paths: list[Path], out_dir: Path):
    if not shard_paths:
        raise ValueError("At least one shard required")
    base = load_cache(base_path)
    existing = set(base[0].actions)
    all_rows = _read_rows(base_path)
    detail = {"base": str(base_path), "base_sha256": sha256(base_path), "shards": []}
    for path in shard_paths:
        meta_path = path.with_name(path.stem + "_manifest.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing shard manifest: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != "direction_a_shard_v1":
            raise ValueError(f"Unrecognised shard manifest {meta_path}")
        if (meta.get("base_cache_sha256") != detail["base_sha256"]
            or meta.get("shard_sha256") != sha256(path)):
            raise ValueError(f"Shard provenance or checksum mismatch: {path}")
        shard = load_cache(path)
        action_id = _assert_matched_cache(base, shard)
        if action_id != meta.get("action_id") or action_id in existing:
            raise ValueError(f"Duplicate or mismatched shard action: {action_id}")
        existing.add(action_id)
        all_rows.extend(_read_rows(path))
        detail["shards"].append(dict(path=str(path), sha256=sha256(path), action=action_id))
    out_cache = out_dir / "forecast_cache.csv"
    targets = [out_cache, out_dir / "feasibility.json", out_dir / "decisions.csv", out_dir / "manifest.json"]
    for p in targets:
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")
    # Validate the full merged result before touching any final destination.
    with tempfile.TemporaryDirectory(prefix="pertsfm_b12_merge_") as scratch:
        candidate = Path(scratch) / "forecast_cache.csv"
        write_csv_atomic(all_rows, candidate, FIELDS)
        cases = load_cache(candidate)
        budgets = auto_budgets(cases)
        report, decisions = run_feasibility(cases, budgets)
    write_csv_atomic(all_rows, out_cache, FIELDS)
    write_csv_atomic([asdict(d) for d in decisions], out_dir / "decisions.csv",
                     tuple(asdict(decisions[0])))
    (out_dir / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
    detail.update(dict(actions=sorted(existing), cases=len(cases),
                       budget_ms=budgets, status="offline_single-action_oracle_only",
                       warning="Hindsight oracle sees outcomes; not a sequential deployable policy."
                       " The cost for each action is its own inference latency;"
                       " acquisition chains are not modelled."))
    (out_dir / "manifest.json").write_text(json.dumps(detail, indent=2) + "\n")
    print(f"PASS: merged {len(cases)} matched cases, {len(existing)} actions -> {out_dir}")
    for b in report["budgets"]:
        print(f"  {b['budget_ms']:.3f} ms | selected={b['validation_selected_action']} "
              f"| fixed={b['strategies']['validation_selected']['macro_mae']:.4f} "
              f"| oracle={b['strategies']['hindsight_oracle']['macro_mae']:.4f} "
              f"| gap={b['oracle_gap_percent_of_fixed']:.2f}%")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-action", help="Run one model/context on B1.1's exact cases")
    run.add_argument("--base-cache", type=Path, required=True)
    run.add_argument("--base-manifest", type=Path, required=True)
    run.add_argument("--model", choices=sorted(SUPPORTED_MODELS), required=True)
    run.add_argument("--context-hours", type=int, choices=CONTEXT_HOURS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    join = sub.add_parser("merge", help="Join fully matched single-action caches")
    join.add_argument("--base-cache", type=Path, required=True)
    join.add_argument("--shard", action="append", required=True, type=Path)
    join.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run-action":
        return run_action(args)
    merge(args.base_cache, args.shard, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
