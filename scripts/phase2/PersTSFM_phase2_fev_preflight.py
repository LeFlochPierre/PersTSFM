#!/usr/bin/env python3
"""PersTSFM Phase-2.4: inspect external fev task windows, without forecasting.

Development-only task configurations. No model inference or test-target access.
Requires: pip install fev
Usage:
    python PersTSFM_phase2_fev_preflight.py --task m4_hourly --windows 8
    python PersTSFM_phase2_fev_preflight.py --task ett1_ot --windows 8
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


TASK_SPECS = {
    "m4_hourly": {
        "family": "M4",
        "dataset_path": "autogluon/chronos_datasets",
        "dataset_config": "m4_hourly",
        "horizon": 24,
        "window_step_size": 24,
        "time_semantics": "series_relative_not_global_calendar",
    },
    "ett1_ot": {
        "family": "ETT1",
        "dataset_path": "autogluon/fev_datasets",
        "dataset_config": "ETT_1H",
        "generate_univariate_targets_from": ["OT"],
        "horizon": 24,
        "window_step_size": 24,
        "time_semantics": "calendar_check_required",
    },
    "ercot": {
        "family": "ERCOT",
        "dataset_path": "autogluon/chronos_datasets",
        "dataset_config": "ercot",
        "horizon": 24,
        "window_step_size": 24,
        "time_semantics": "calendar_check_required",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--contexts", type=int, nargs="+", default=[96, 192, 384, 768, 1536, 3072])
    parser.add_argument("--out", default="results/phase2/external_preflight")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.windows < 4:
        raise ValueError("Require >=4 chronological windows so selection and evaluation can be separated")
    if not args.contexts or any(x <= 0 for x in args.contexts):
        raise ValueError("Context lengths must be positive integers")


def iso(x: object) -> str | None:
    return str(x) if x is not None else None


def main() -> None:
    args = parse_args()
    validate_args(args)
    try:
        import fev
        from importlib.metadata import version
    except ImportError as exc:
        raise RuntimeError("Install fev in a separate data environment: python -m pip install fev") from exc

    spec = dict(TASK_SPECS[args.task])
    kwargs = {k: v for k, v in spec.items() if k not in {"family", "time_semantics"}}
    kwargs["num_windows"] = args.windows
    task = fev.Task(**kwargs)

    print("=" * 80)
    print("PersTSFM external task preflight — development data only")
    print("=" * 80)
    print(f"Task: {args.task} | family: {spec['family']} | fev: {version('fev')}")
    print(f"Dataset: {spec['dataset_path']}/{spec['dataset_config']}")
    print(f"Horizon: {spec['horizon']} | windows: {args.windows}")
    print(f"Temporal interpretation: {spec['time_semantics']}")
    print("No forecasting or future target retrieval is performed.\n")

    id_col = task.id_column
    target_col = getattr(task, "target_column", None)
    if target_col is None:
        target_col = getattr(task, "target", "target")
    if isinstance(target_col, (list, tuple)):
        if len(target_col) != 1:
            raise RuntimeError(f"Expected one target column, found {target_col}")
        target_col = target_col[0]
    timestamp_col = task.timestamp_column

    audit_windows = []
    for wi, window in enumerate(task.iter_windows()):
        past, future_known = window.get_input_data()
        if len(past) == 0:
            raise RuntimeError(f"Window {wi} has no entities")
        if len(past) != len(future_known):
            raise RuntimeError(f"Window {wi}: past/future-known entity count mismatch")
        if target_col not in past.column_names:
            raise RuntimeError(f"Window {wi}: {target_col!r} missing, columns={past.column_names}")
        if id_col not in past.column_names:
            raise RuntimeError(f"Window {wi}: {id_col!r} missing")

        history_lengths = []
        ids = []
        last_past_times = []
        first_future_times = []
        n_zero_length = 0
        for j in range(len(past)):
            record = past[j]
            known = future_known[j]
            values = record[target_col]
            n = len(values)
            history_lengths.append(n)
            ids.append(str(record[id_col]))
            n_zero_length += n == 0
            if timestamp_col in record and len(record[timestamp_col]) > 0:
                last_past_times.append(iso(record[timestamp_col][-1]))
            if timestamp_col in known and len(known[timestamp_col]) > 0:
                first_future_times.append(iso(known[timestamp_col][0]))
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"Window {wi}: duplicate entity identifiers")

        feasible = {str(c): sum(n >= c for n in history_lengths) for c in args.contexts}
        item = {
            "window_index": wi,
            "cutoff": iso(window.cutoff),
            "n_entities": len(past),
            "history_min": min(history_lengths),
            "history_median": statistics.median(history_lengths),
            "history_max": max(history_lengths),
            "n_empty_histories": n_zero_length,
            "feasible_entities_by_context_steps": feasible,
            "min_last_past_timestamp": min(last_past_times) if last_past_times else None,
            "max_last_past_timestamp": max(last_past_times) if last_past_times else None,
            "min_first_future_timestamp": min(first_future_times) if first_future_times else None,
            "max_first_future_timestamp": max(first_future_times) if first_future_times else None,
            "example_entity_ids": ids[:5],
        }
        audit_windows.append(item)
        print(f"Window {wi:02d} | cutoff={window.cutoff} | entities={len(past)} | "
              f"history min/median/max={item['history_min']}/{item['history_median']}/{item['history_max']}")
        print("  Context feasibility (# entities with enough history): " + ", ".join(
            f"{c}:{feasible[str(c)]}" for c in args.contexts
        ))
        if item["min_first_future_timestamp"] is not None:
            print("  First future timestamp range: " +
                  f"{item['min_first_future_timestamp']} .. {item['max_first_future_timestamp']}")

    if len(audit_windows) != args.windows:
        raise RuntimeError(f"Expected {args.windows} windows, found {len(audit_windows)}")

    manifest = {
        "schema": "pertsfm_external_task_preflight_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DEVELOPMENT_ONLY_NOT_HELD_OUT",
        "task_name": args.task,
        "family": spec["family"],
        "time_semantics": spec["time_semantics"],
        "fev_version": version("fev"),
        "task_kwargs": kwargs,
        "target_column": target_col,
        "id_column": id_col,
        "timestamp_column": timestamp_col,
        "contexts_steps": args.contexts,
        "windows": audit_windows,
        "warning": "Window index order alone does not certify global calendar separation. "
                   "A valid selection/evaluation protocol must check nonoverlapping targets "
                   "and chronological order using actual timestamps before benchmarks.",
    }
    stable_spec = json.dumps({"kwargs": kwargs, "contexts": args.contexts}, sort_keys=True)
    manifest["task_spec_sha256"] = hashlib.sha256(stable_spec.encode()).hexdigest()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.task}_preflight.json"
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print("\nSaved:", out)
    print("NEXT: validate chronology/target separation and freeze task spec before model inference.")


if __name__ == "__main__":
    main()
