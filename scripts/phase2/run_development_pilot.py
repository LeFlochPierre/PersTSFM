"""Orchestrate comparable PersTSFM Phase-2 validation benchmarks.

No neural training, no result tuning, and no modification of frozen V1 policies.
Execute with the core PERTSFM_ENV. Each TSFM runs in its own conda env.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from src.data.registry import extract_case, load_dataset
from src.evaluation.case_sampling import balanced_subset

CONTEXT_HOURS = {
    # Every grid includes the original reference context to maintain
    # the legacy aggregation contract. These are *pilot* grids, NOT full search.
    "electricity": [96, 192, 384],       # 384,768,1536 steps
    "pems_bay": [24, 72, 168],          # 288,864,2016 steps
    "hupa_ucm": [24, 72, 120],          # 288,864,1440 steps
    "vitaldb": [0.5, 2, 3],             # 30,120,180 steps
    "solar": [6, 24, 96],              # 36,144,576 steps
    "physionet2019": [6, 24, 36],       # 6,24,36 steps
}
MODEL_ENVS = {
    "toto2_22m": "PERTSFM_TOTO_ENV",
    "toto2_313m": "PERTSFM_TOTO_ENV",
    "moirai2_small": "PERTSFM_MOIRAI_ENV",
    "chronos2": "PERTSFM_ENV",
    "timesfm3": "PERTSFM_ENV",
    "tirex2": "PERTSFM_ENV",
}
DEFAULT_MODELS = ["toto2_22m", "toto2_313m", "moirai2_small"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=sorted(CONTEXT_HOURS),
                   default=["electricity", "pems_bay"])
    p.add_argument("--models", nargs="+", choices=sorted(MODEL_ENVS),
                   default=DEFAULT_MODELS)
    p.add_argument("--entities", type=int, default=8)
    p.add_argument("--origins-per-entity", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true", help="All protocol cases, not a pilot")
    p.add_argument("--dry-run", action="store_true", help="Show commands without loading data")
    p.add_argument("--check-only", action="store_true", help="Validate selected history windows, no inference")
    p.add_argument("--force", action="store_true", help="Rerun and overwrite existing model output")
    p.add_argument("--output-root", default="results/phase2/development_pilot")
    return p.parse_args()


def get_python(env_name):
    # Resolve directly so torch/HF libraries use the correct conda env.
    candidates = []
    if os.getenv("CONDA_EXE"):
        conda_root = Path(os.environ["CONDA_EXE"]).resolve().parent.parent
        candidates.append(conda_root / "envs" / env_name / "bin" / "python")
    if os.getenv("CONDA_PREFIX"):
        conda_prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
        candidates.append(conda_prefix.parent / env_name / "bin" / "python")
    candidates.append(Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python")
    for path in candidates:
        if path.is_file():
            return str(path)
    raise RuntimeError(f"Cannot find python for {env_name}. Checked {candidates}")


def get_subset(dataset, args):
    protocol = dataset.protocol.copy()
    if not args.full:
        protocol = balanced_subset(
            protocol,
            id_col=dataset.spec.id_col,
            n_entities=args.entities,
            origins_per_entity=args.origins_per_entity,
            seed=args.seed,
        )
    return protocol


def preflight(dataset_key, args):
    print(f"\nPreflight: {dataset_key}", flush=True)
    dataset = load_dataset(dataset_key)
    protocol = get_subset(dataset, args)
    if protocol.empty:
        raise RuntimeError(f"{dataset_key}: empty protocol")
    for hours in CONTEXT_HOURS[dataset_key]:
        n_checked = 0
        for _, row in protocol.iterrows():
            case = extract_case(dataset, row, hours)
            if len(case["context"]) != case["context_steps"]:
                raise RuntimeError("Unexpected context mismatch")
            n_checked += 1
        print(f"  {hours:g} hours: {n_checked} eligible cases", flush=True)
    print(f"  Entities: {protocol[dataset.spec.id_col].nunique()} | "
          f"Cases: {len(protocol)}", flush=True)


def run_job(dataset_key, model_key, args, output_root):
    model_dir = output_root / dataset_key / "context" / model_key
    summary_path = model_dir / "summary.csv"
    run_info = model_dir / "phase2_run.json"
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_file = logs / f"{dataset_key}__{model_key}.log"

    if summary_path.exists() and not (args.force or args.dry_run or args.check_only):
        raise RuntimeError(
            f"Existing output would be overwritten: {summary_path}. "
            "Use --force or a fresh --output-root."
        )
    python = (get_python(MODEL_ENVS[model_key])
              if not args.dry_run and not args.check_only
              else f"<python:{MODEL_ENVS[model_key]}>")
    command = [
        python, "-m", "src.experiments.run_context_curve",
        "--dataset", dataset_key, "--model", model_key,
        "--contexts", *[str(x) for x in CONTEXT_HOURS[dataset_key]],
        "--output-root", str(output_root),
    ]
    if not args.full:
        command.extend([
            "--sample-entities", str(args.entities),
            "--origins-per-entity", str(args.origins_per_entity),
            "--sample-seed", str(args.seed),
        ])
    print(f"\n=== RUN {dataset_key} / {model_key} ===", flush=True)
    print(" ".join(command), flush=True)
    if args.dry_run or args.check_only:
        return
    start = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
        rc = proc.wait()
    elapsed = time.perf_counter() - start
    model_dir.mkdir(parents=True, exist_ok=True)
    run_info.write_text(json.dumps({
        "dataset": dataset_key, "model": model_key,
        "env": MODEL_ENVS[model_key], "command": command,
        "contexts_hours": CONTEXT_HOURS[dataset_key],
        "mode": "full" if args.full else "balanced_pilot",
        "n_entities_requested": None if args.full else args.entities,
        "origins_per_entity_requested": None if args.full else args.origins_per_entity,
        "seed": args.seed, "subprocess_wall_seconds": elapsed,
        "exit_code": rc, "note": "Wall time includes Python startup, dataset IO, loading and inference",
    }, indent=2) + "\n")
    if rc != 0:
        raise RuntimeError(f"{dataset_key}/{model_key}: failed (exit={rc}); see {log_file}")
    if not summary_path.is_file():
        raise RuntimeError(f"{dataset_key}/{model_key}: summary.csv missing")
    print(f"PASS: {dataset_key}/{model_key} ({elapsed:.1f}s total)", flush=True)


def main():
    args = parse_args()
    if not args.full and (args.entities < 1 or args.origins_per_entity < 1):
        raise ValueError("entities/origins-per-entity must be positive")
    output_root = Path(args.output_root).resolve()
    print("PersTSFM Phase-2 development benchmark", flush=True)
    print("Mode:", "full" if args.full else "balanced multi-entity pilot", flush=True)
    print("Datasets:", args.datasets, flush=True)
    print("Models:", args.models, flush=True)
    print("Output:", output_root, flush=True)
    if not args.dry_run:
        for dataset in args.datasets:
            preflight(dataset, args)
    for dataset in args.datasets:
        for model in args.models:
            run_job(dataset, model, args, output_root)
    if args.check_only:
        print("PASS: all selected context windows are leakage-safe and extractable")
    elif not args.dry_run:
        print("\nPASS: all requested model runs completed. Next: summarise_phase2.py")


if __name__ == "__main__":
    main()
