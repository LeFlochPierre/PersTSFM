"""Build a metadata-only, version-pinned fev-bench task registry.

Safe to run in PERTSFM_DATA_ENV; does NOT load datasets or future targets.
Output never claims that available-history feasibility has been established.

Usage:
 python -m scripts.phase2.audit_official_benchmarks
 python -m scripts.phase2.audit_official_benchmarks --source path/to/tasks.yaml --expected-tasks 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

DEFAULT_URL = (
    "https://raw.githubusercontent.com/autogluon/fev/"
    "refs/tags/v0.10.0/benchmarks/fev_bench/tasks.yaml"
)
DEFAULT_OUT = Path("results/phase2/benchmark_registry/fev_bench_v0_10_0")
DEFAULT_CONTEXTS = (96, 192, 384, 768, 1536, 3072)
PILOTS = {
    "m4_hourly": ("autogluon/chronos_datasets", "m4_hourly"),
    "ett1_ot": ("autogluon/fev_datasets", "ETT_1H"),
    "ercot": ("autogluon/chronos_datasets", "ercot"),
}


def load_yaml_bytes(source: str, timeout: int = 45) -> bytes:
    if source.startswith(("https://", "http://")):
        req = Request(source, headers={"User-Agent": "PersTSFM-benchmark-metadata-audit/1.0"})
        with urlopen(req, timeout=timeout) as response:
            return response.read()
    return Path(source).read_bytes()


def clean_alias(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
    return s.removeprefix("monash_")


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def dataset_key(config: dict) -> str:
    return f"{config.get('dataset_path','')}/{config.get('dataset_config') or '__default__'}"


def _cols(conf: dict, name: str) -> list:
    v = conf.get(name) or []
    if isinstance(v, list):
        return v
    raise ValueError(f"Expected {name} to be a list; got {v!r}")


def build_tables(configs: list[dict], contexts: tuple[int, ...] = DEFAULT_CONTEXTS):
    catalog, feasibility, overlaps = [], [], []
    pilot_alias = {name: clean_alias(conf) for name, (_, conf) in PILOTS.items()}
    for index, conf in enumerate(configs):
        if not isinstance(conf, dict):
            raise ValueError(f"Task {index}: expected mapping, received {type(conf)}")
        if "dataset_path" not in conf or "horizon" not in conf:
            raise ValueError(f"Task {index}: dataset_path and horizon are required")
        target = conf.get("target", "target")
        if not isinstance(target, (str, list)):
            raise ValueError(f"Task {index}: unsupported target type {type(target)}")
        dynamic_known = _cols(conf, "known_dynamic_columns")
        dynamic_past = _cols(conf, "past_dynamic_columns")
        static = _cols(conf, "static_columns")
        multivariate = isinstance(target, list)
        generation = conf.get("generate_univariate_targets_from")
        has_covars = bool(dynamic_known or dynamic_past or static)
        target_only_candidate = not (multivariate or has_covars or generation)
        config = str(conf.get("dataset_config") or "__default__")
        group = dataset_key(conf)
        alias = clean_alias(config)
        task_label = str(conf.get("task_name") or f"{config}/h{conf['horizon']}/w{conf.get('num_windows',1)}")
        # Task names are NOT guaranteed unique across all horizon/covariate variants.
        entry_id = f"fev_0_10_0_{index:03d}"
        row = {
            "task_index": index,
            "task_id": entry_id,
            "task_name": task_label,
            "dataset_path": str(conf["dataset_path"]),
            "dataset_config": config,
            "family_id_exact": group,
            "family_alias_review_key": alias,
            "horizon": int(conf["horizon"]),
            "num_windows": int(conf.get("num_windows", 1)),
            "initial_cutoff": str(conf.get("initial_cutoff", "auto")),
            "window_step_size": str(conf.get("window_step_size", "auto")),
            "min_context_length": str(conf.get("min_context_length", "default")),
            "max_context_length": str(conf.get("max_context_length", "none")),
            "seasonality": str(conf.get("seasonality", "default")),
            "eval_metric": str(conf.get("eval_metric", "default")),
            "extra_metrics_json": encode(conf.get("extra_metrics", [])),
            "quantile_levels_json": encode(conf.get("quantile_levels", [])),
            "target_json": encode(target),
            "is_multivariate": multivariate,
            "has_covariates": has_covars,
            "known_covariate_count": len(dynamic_known),
            "past_covariate_count": len(dynamic_past),
            "static_covariate_count": len(static),
            "uses_generated_targets": bool(generation),
            "target_only_candidate": target_only_candidate,
            "current_runner_status": (
                "candidate_requires_adapter_and_history_audit"
                if target_only_candidate else "requires_feature_support_or_distinct_protocol"
            ),
            "official_task_json": encode(conf),
        }
        catalog.append(row)
        for context in contexts:
            feasibility.append({
                "task_id": entry_id,
                "family_id_exact": group,
                "context_steps": context,
                "target_only_candidate": target_only_candidate,
                "declared_max_context": row["max_context_length"],
                "history_feasibility": "UNKNOWN_requires_per_entity_history_audit",
                "model_context_feasibility": "UNKNOWN_requires_checkpoint_capability_audit",
                "official_task_compatible": (
                    "UNKNOWN_requires_native_metric_adapter"
                    if target_only_candidate else "NO_not_with_current_target_only_runner"
                ),
            })
        for pilot, (pilot_path, pilot_config) in PILOTS.items():
            match_type = (
                "exact_dataset_identifier"
                if (str(conf["dataset_path"]), config) == (pilot_path, pilot_config)
                else ("normalized_name_candidate" if alias == pilot_alias[pilot] else None)
            )
            if match_type:
                overlaps.append({
                    "task_id": entry_id, "official_task_name": task_label,
                    "family_id_exact": group, "existing_pilot": pilot,
                    "match_type": match_type,
                    "action": "exclude_from_independent_holdout_pending_source_review",
                })

    groups: dict[str, list[dict]] = defaultdict(list)
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in catalog:
        groups[row["family_id_exact"]].append(row)
        aliases[row["family_alias_review_key"]].add(row["family_id_exact"])
    families = []
    for group, rows in sorted(groups.items()):
        alias = rows[0]["family_alias_review_key"]
        families.append({
            "family_id_exact": group,
            "family_alias_review_key": alias,
            "n_tasks": len(rows),
            "task_ids_json": encode([r["task_id"] for r in rows]),
            "horizons_json": encode(sorted({r["horizon"] for r in rows})),
            "n_target_only_candidates": sum(r["target_only_candidate"] for r in rows),
            "alias_collision_across_datasets": len(aliases[alias]) > 1,
            "source_overlap_review_required": True,
        })
    for alias, exact_groups in aliases.items():
        if len(exact_groups) <= 1:
            continue
        for group in sorted(exact_groups):
            overlaps.append({
                "task_id": "", "official_task_name": "",
                "family_id_exact": group, "existing_pilot": "",
                "match_type": f"possible_cross_dataset_alias:{alias}",
                "action": "manual_source_lineage_review_before_family_split",
            })
    return catalog, families, feasibility, overlaps


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def audit(source: str, out: Path, contexts: tuple[int, ...], expected_tasks: int) -> dict:
    payload = load_yaml_bytes(source)
    parsed = yaml.safe_load(payload)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tasks"), list):
        raise ValueError("Benchmark YAML must have a top-level 'tasks' list")
    confs = parsed["tasks"]
    if expected_tasks and len(confs) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} tasks, got {len(confs)}; refusing to silently switch benchmark revision")
    catalog, groups, feasibility, overlaps = build_tables(confs, contexts)
    try:
        fev_version = importlib.metadata.version("fev")
    except importlib.metadata.PackageNotFoundError:
        fev_version = "not_installed"
    sha = hashlib.sha256(payload).hexdigest()
    manifest = {
        "benchmark": "fev-bench",
        "source": source,
        "source_sha256": sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "installed_fev_version": fev_version,
        "task_count": len(catalog),
        "exact_dataset_family_count": len(groups),
        "target_only_candidate_tasks": sum(r["target_only_candidate"] for r in catalog),
        "tasks_with_covariates": sum(r["has_covariates"] for r in catalog),
        "tasks_with_multivariate_targets": sum(r["is_multivariate"] for r in catalog),
        "tasks_with_generated_targets": sum(r["uses_generated_targets"] for r in catalog),
        "tasks_with_single_official_window": sum(r["num_windows"] == 1 for r in catalog),
        "contexts_for_feasibility_audit_only": list(contexts),
        "data_downloaded": False,
        "future_ground_truth_accessed": False,
        "caveats": [
            "Not all official tasks are compatible with univariate target-only models.",
            "Historical context feasibility is UNKNOWN until history lengths are inspected.",
            "Official fev-bench commonly uses probabilistic SQL; present scoring is not native metric parity.",
            "Official evaluation windows must not be used to tune configuration policies.",
            "Exact dataset identifiers can miss source-level overlap; family grouping needs manual review.",
            "The benchmark YAML pin does not freeze Hugging Face dataset revisions by itself.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "official_tasks.yaml").write_bytes(payload)
    write_csv(out / "task_catalog.csv", catalog, list(catalog[0]) if catalog else [])
    write_csv(out / "family_groups.csv", groups, list(groups[0]) if groups else [])
    write_csv(out / "feasibility.csv", feasibility, list(feasibility[0]) if feasibility else [])
    write_csv(out / "overlap_audit.csv", overlaps, [
        "task_id", "official_task_name", "family_id_exact", "existing_pilot", "match_type", "action"
    ])
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("=" * 84)
    print("PersTSFM PHASE 2.6 — OFFICIAL FEV-BENCH METADATA AUDIT (NO DATA DOWNLOAD)")
    print("=" * 84)
    print(f"Source: {source}")
    print(f"Raw YAML SHA256: {sha}")
    print(f"Tasks: {len(catalog)} | Exact dataset IDs: {len(groups)}")
    print(f"Target-only metadata candidates: {manifest['target_only_candidate_tasks']}")
    print(f"Covariate tasks: {manifest['tasks_with_covariates']} | Multivariate targets: {manifest['tasks_with_multivariate_targets']}")
    print(f"Single official window: {manifest['tasks_with_single_official_window']}")
    print(f"Potential pilot / alias overlap flags: {len(overlaps)}")
    print("Official metrics:", dict(Counter(r["eval_metric"] for r in catalog)))
    print("All context feasibility is UNKNOWN pending a series-level data audit.")
    print("All official evaluation targets remain untouched.")
    print(f"Saved: {out}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_URL, help="Pinned YAML URL or local YAML path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-tasks", type=int, default=100, help="0 disables expected count validation")
    parser.add_argument("--contexts", default=",".join(map(str, DEFAULT_CONTEXTS)))
    args = parser.parse_args()
    contexts = tuple(int(n.strip()) for n in args.contexts.split(","))
    if not contexts or min(contexts) < 1 or len(contexts) != len(set(contexts)):
        parser.error("Contexts must be positive, distinct integers")
    try:
        audit(args.source, args.out, contexts, args.expected_tasks)
    except Exception as ex:
        print(f"AUDIT FAILED: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
