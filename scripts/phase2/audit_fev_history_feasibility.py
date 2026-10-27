#!/usr/bin/env python3
"""Phase 2.7: conservative input-only audit of selected official fev-bench tasks.

Loads ONLY the first OFFICIAL window's prediction-time historical inputs.
NEVER calls get_ground_truth(), task.evaluation_summary(), load_full_dataset(), or later windows.

Context feasibility is measured at the earliest of K synthetic, strictly pre-official
calibration origins: history_length - K * forecast_horizon (disjoint horizons).
This script DOES NOT build actual calibration forecasts or claim official benchmark scores.

Examples:
  python -m scripts.phase2.audit_fev_history_feasibility --list-candidates
  python -m scripts.phase2.audit_fev_history_feasibility --preset starter
  python -m scripts.phase2.audit_fev_history_feasibility --configs hospital ecdc_ili
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REGISTRY = Path("results/phase2/benchmark_registry/fev_bench_v0_10_0")
DEFAULT_OUT = Path("results/phase2/benchmark_feasibility/fev_bench_v0_10_0")
DEFAULT_CONTEXTS = (8, 16, 32, 64, 96, 192, 384, 768, 1536, 3072)
# Deliberately include both low-frequency and high-frequency, modest candidates.
# Dataset download sizes may still be significant. Run --list-candidates first.
STARTER_CONFIGS = (
    "australian_tourism", "world_tourism", "world_co2_emissions",
    "hospital", "ecdc_ili", "us_consumption_1M",
)
# Heuristic only: needs curator review before assigning train/test folds.
FREQ_SUFFIX = re.compile(r"_(?:\d+(?:T|S|H|D|W|M|Q|Y))$", re.IGNORECASE)
PILOT_ALIASES = {"solar", "ercot", "hospital", "m4_hourly", "ett1"}


def underlying_family_hint(dataset_config: str) -> str:
    base = FREQ_SUFFIX.sub("", str(dataset_config))
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")


def calibration_steps_available(history_length: int, horizon: int, calibration_origins: int) -> int:
    """History visible at oldest calibration cutoff; all labels end before official test."""
    if min(history_length, horizon, calibration_origins) < 0 or horizon == 0 or calibration_origins == 0:
        raise ValueError("History length must be >=0, horizon/origins must be positive")
    return max(0, history_length - horizon * calibration_origins)


def feasibility_counts(lengths: list[int], contexts: tuple[int, ...], horizon: int, origins: int) -> list[dict]:
    return [
        {"context_steps": c,
         "entities_feasible_first_official_window": sum(length >= c for length in lengths),
         "entities_feasible_all_calibration_origins": sum(
             calibration_steps_available(length, horizon, origins) >= c for length in lengths
         ),
         "total_entities_inspected": len(lengths)}
        for c in contexts
    ]


def read_catalog(path: Path) -> tuple[list[dict], str]:
    catalog = path / "task_catalog.csv"
    manifest_path = path / "benchmark_manifest.json"
    yaml_path = path / "official_tasks.yaml"
    if not catalog.exists() or not manifest_path.exists() or not yaml_path.exists():
        raise FileNotFoundError(f"Missing Phase 2.6 official benchmark registry files at {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    if actual_sha != manifest["source_sha256"]:
        raise RuntimeError("Official YAML SHA256 disagrees with Phase 2.6 manifest")
    rows = list(csv.DictReader(catalog.open(newline="", encoding="utf-8")))
    if len(rows) != int(manifest["task_count"]):
        raise RuntimeError("Task catalog count disagrees with Phase 2.6 manifest")
    return rows, actual_sha


def list_candidates(rows: list[dict]) -> None:
    print("OFFICIAL TARGET-ONLY CANDIDATES (metadata screen; history still unknown)")
    print(f"{'ID':<19} {'DATASET_CONFIG':<33} {'H':>5} {'W':>3} {'FAMILY_HYPOTHESIS'}")
    for r in rows:
        if r["target_only_candidate"].lower() == "true":
            print(f"{r['task_id']:<19} {r['dataset_config']:<33} {r['horizon']:>5} "
                  f"{r['num_windows']:>3} {underlying_family_hint(r['dataset_config'])}")


def select_rows(rows: list[dict], task_ids: list[str], configs: list[str], preset: str) -> list[dict]:
    if preset == "starter" and (task_ids or configs):
        raise ValueError("Use either --preset starter or explicit --task-ids/--configs")
    if preset == "starter":
        configs = list(STARTER_CONFIGS)
    if not task_ids and not configs:
        raise ValueError("Choose --preset starter, --configs, or --task-ids (use --list-candidates to inspect)")
    known_ids = {r["task_id"] for r in rows}
    known_configs = {r["dataset_config"] for r in rows}
    unknown_ids = set(task_ids) - known_ids
    unknown_configs = set(configs) - known_configs
    if unknown_ids or unknown_configs:
        raise ValueError(f"Unknown task IDs={sorted(unknown_ids)}, configs={sorted(unknown_configs)}")
    selected = [r for r in rows if r["task_id"] in task_ids or r["dataset_config"] in configs]
    incompatible = [r["task_id"] for r in selected if r["target_only_candidate"].lower() != "true"]
    if incompatible:
        raise ValueError(f"Not compatible with target-only audit: {incompatible}")
    return selected


def summarize_lengths(lengths: list[int]) -> dict:
    a = sorted(lengths)
    if not a:
        raise ValueError("No entities survived the official input window")
    return {"min": a[0], "median": (a[(len(a)-1)//2] + a[len(a)//2]) / 2,
            "max": a[-1]}


def inspect_first_window(row: dict, contexts: tuple[int, ...], calibration_origins: int,
                         entity_limit: int) -> dict:
    """Load ONLY get_window(0).get_input_data() for this official task."""
    import fev
    import numpy as np

    spec = json.loads(row["official_task_json"])
    task = fev.Task(**spec)
    if len(task.target_columns) != 1:
        raise ValueError(f"Expected one target column, got {task.target_columns}")
    window = task.get_window(0)
    past, _known_future = window.get_input_data()  # no future target, no scoring
    n_total = len(past)
    if n_total == 0:
        raise ValueError("No entities available at first official window")
    target = task.target_columns[0]
    n_inspect = n_total if not entity_limit else min(n_total, entity_limit)
    lengths, missing, nonfinite = [], 0, 0
    for index in range(n_inspect):
        item = past[index]
        values = np.asarray(item[target], dtype=np.float64)
        timestamps = item[task.timestamp_column]
        if values.ndim != 1 or len(values) == 0 or len(values) != len(timestamps):
            raise ValueError(f"Bad target/time shape at series #{index}")
        lengths.append(int(len(values)))
        missing += int(np.isnan(values).sum())
        nonfinite += int((~np.isfinite(values)).sum())
    # Return actual *data*-based feasibility. If sampled, mark explicitly not full census.
    h = int(task.horizon)
    summary = summarize_lengths(lengths)
    context_counts = feasibility_counts(lengths, contexts, h, calibration_origins)
    print(f"  entities: {n_inspect}/{n_total} | history min/median/max="
          f"{summary['min']}/{summary['median']:g}/{summary['max']} | "
          f"missing={missing}, nonfinite={nonfinite}", flush=True)
    for ct in context_counts:
        print(f"    context {ct['context_steps']:>5}: official {ct['entities_feasible_first_official_window']}/{n_inspect}, "
              f"pre-official calibration {ct['entities_feasible_all_calibration_origins']}/{n_inspect}", flush=True)
    return {
        "task_id": row["task_id"], "task_name": row["task_name"],
        "dataset_config": row["dataset_config"], "family_id_exact": row["family_id_exact"],
        "underlying_family_hint": underlying_family_hint(row["dataset_config"]),
        "source_family_review_required": True,
        "near_existing_pilot_name": underlying_family_hint(row["dataset_config"]) in PILOT_ALIASES,
        "official_horizon": h, "official_num_windows": int(task.num_windows),
        "official_metric": str(task.eval_metric), "first_official_window_index": 0,
        "entities_official_total": n_total, "entities_inspected": n_inspect,
        "coverage": "FULL" if n_inspect == n_total else "SAMPLED_NOT_FULL_CENSUS",
        "history_min": summary["min"], "history_median": summary["median"],
        "history_max": summary["max"], "nonfinite_history_values_inspected": nonfinite,
        "nan_history_values_inspected": missing,
        "calibration_origins": calibration_origins,
        "calibration_first_cutoff_formula": "history_length - calibration_origins*horizon",
        "calibration_windows_strictly_pre_official": True,
        "context_counts": context_counts,
        "dataset_revision": "NOT_PINNED_check_HF_revision_before_publication",
        "future_truth_read": False,
    }


def csv_dump(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def rebuild_summaries(out: Path, registry_sha: str, selected: list[dict], contexts: tuple[int, ...],
                      calibration_origins: int, fev_version: str) -> None:
    profiles = []
    # Consolidate every previously completed compatible profile, not only this invocation.
    # This preserves a cumulative feasibility CSV when the user audits extra tasks later.
    for path in sorted((out / "task_profiles").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("registry_sha256") != registry_sha or
                data.get("calibration_origins") != calibration_origins or
                tuple(x["context_steps"] for x in data.get("context_counts", [])) != contexts):
            raise RuntimeError(f"Stale profile incompatible with requested audit: {path}")
        profiles.append(data)
    summaries = []
    contexts_flat = []
    for p in profiles:
        summaries.append({k: p[k] for k in (
            "task_id", "task_name", "dataset_config", "underlying_family_hint", "near_existing_pilot_name",
            "official_horizon", "official_num_windows", "official_metric", "entities_official_total",
            "entities_inspected", "coverage", "history_min", "history_median", "history_max",
            "nan_history_values_inspected", "calibration_origins")})
        for c in p["context_counts"]:
            contexts_flat.append({"task_id": p["task_id"], "dataset_config": p["dataset_config"],
                                  "family_hint": p["underlying_family_hint"], "coverage": p["coverage"], **c})
    if summaries:
        csv_dump(out / "task_history_profiles.csv", summaries, list(summaries[0]))
    if contexts_flat:
        csv_dump(out / "calibration_context_feasibility.csv", contexts_flat, list(contexts_flat[0]))
    hints: dict[str,list[dict]] = defaultdict(list)
    for p in profiles:
        hints[p["underlying_family_hint"]].append(p)
    family_rows = [{"provisional_family": name, "n_audited_tasks": len(ps),
                    "dataset_configs": json.dumps(sorted(x["dataset_config"] for x in ps)),
                    "overlaps_current_pilot_by_name": name in PILOT_ALIASES,
                    "split_status": "BLOCKED_PENDING_SOURCE_LINEAGE_REVIEW"}
                   for name, ps in sorted(hints.items())]
    if family_rows:
        csv_dump(out / "provisional_family_groups.csv", family_rows, list(family_rows[0]))
    manifest = {
        "schema": "pertsfm_fev_bench_first_window_history_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": registry_sha, "fev_version": fev_version,
        "selected_tasks": [r["task_id"] for r in selected],
        "completed_tasks": [p["task_id"] for p in profiles],
        "calibration_origins": calibration_origins,
        "contexts_steps": list(contexts),
        "official_ground_truth_accessed": False,
        "official_window_indices_accessed": [0],
        "limitations": [
            "Only first official window prediction-time history is inspected.",
            "Official SQL/quantile metrics are NOT implemented by this script.",
            "Availability at oldest calibration origin is estimated from length, not verified as feasible model input.",
            "Provisional family name grouping is not proof of source independence.",
            "Dataset revision is not yet pinned. Inspect and freeze HF data revisions before full studies.",
            "Sampled entities, if any, do not establish whole-dataset feasibility.",
            "Official fev evaluation targets are not opened; no model-selection or test scores are computed.",
        ]
    }
    (out / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--list-candidates", action="store_true")
    p.add_argument("--preset", choices=["starter"], default=None)
    p.add_argument("--configs", nargs="*", default=[])
    p.add_argument("--task-ids", nargs="*", default=[])
    p.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    p.add_argument("--calibration-origins", type=int, default=3)
    p.add_argument("--entity-limit", type=int, default=0, help="0 = inspect all entities; positive = sample ONLY, flagged")
    p.add_argument("--force", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()
    if args.calibration_origins < 1 or args.entity_limit < 0 or any(c < 1 for c in args.contexts):
        p.error("calibration origins and contexts must be positive; entity limit must be >= 0")
    contexts = tuple(sorted(set(args.contexts)))
    rows, registry_sha = read_catalog(args.registry)
    if args.list_candidates:
        list_candidates(rows)
        return
    selected = select_rows(rows, args.task_ids, args.configs, args.preset)
    import fev  # type: ignore
    fev_version = importlib.metadata.version("fev")
    if fev_version != "0.10.0":
        raise RuntimeError(f"Requires fev==0.10.0 matching registry; found {fev_version}")
    args.output.mkdir(parents=True, exist_ok=True)
    profile_dir = args.output / "task_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 87)
    print("PersTSFM PHASE 2.7 — OFFICIAL FEV INPUT-HISTORY FEASIBILITY; DEVELOPMENT AUDIT")
    print("=" * 87)
    print(f"FEV: {fev_version} | tasks: {len(selected)} | K calibration origins: {args.calibration_origins}")
    print("ONLY get_window(0).get_input_data(); NO future ground-truth or scoring", flush=True)
    failures = []
    for index, row in enumerate(selected, start=1):
        path = profile_dir / f"{row['task_id']}.json"
        if path.exists() and not args.force:
            prev = json.loads(path.read_text(encoding="utf-8"))
            if (prev.get("registry_sha256") != registry_sha or
                prev.get("calibration_origins") != args.calibration_origins or
                tuple(c["context_steps"] for c in prev.get("context_counts", [])) != contexts or
                (args.entity_limit == 0 and prev.get("coverage") != "FULL") or
                (args.entity_limit != 0 and prev.get("entities_inspected") !=
                 min(args.entity_limit, prev.get("entities_official_total", 0)))):
                raise RuntimeError(f"Existing profile incompatible, rerun with --force: {path}")
            print(f"SKIP {index}/{len(selected)} {row['dataset_config']}: profile exists", flush=True)
            continue
        print(f"AUDIT {index}/{len(selected)} {row['dataset_config']} "
              f"(h={row['horizon']}, official_windows={row['num_windows']})", flush=True)
        try:
            profile = inspect_first_window(row, contexts, args.calibration_origins, args.entity_limit)
            profile["registry_sha256"] = registry_sha
            profile["fev_version"] = fev_version
            path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"FAIL {row['task_id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(limit=4)
            failures.append((row["task_id"], str(e)))
            if args.fail_fast:
                break
    rebuild_summaries(args.output, registry_sha, selected, contexts, args.calibration_origins, fev_version)
    print("=" * 87)
    print(f"Attempted {len(selected)} tasks | failures {len(failures)} | cumulative completed profiles "
          f"{len(list(profile_dir.glob('*.json')))}")
    print(f"Saved: {args.output}")
    print("SOURCE-FAMILY GROUPS PROVISIONAL; DO NOT ASSIGN FINAL TRAIN/TEST SPLITS YET")
    print("OFFICIAL FUTURE TARGETS NOT ACCESSED")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
