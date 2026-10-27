"""Audit and summarise the Phase-2 multi-model development pilot.

Do not call the best observed configuration an exhaustive oracle. Model-level
MAEs are comparable ONLY when selected forecast cases match across models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_outputs(root: Path):
    results = []
    selection_signatures: dict[str, set[str]] = {}
    bad_runs = []
    for summary_path in sorted(root.glob("*/context/*/summary.csv")):
        model_dir = summary_path.parent
        model = model_dir.name
        dataset = model_dir.parent.parent.name
        run_file = model_dir / "phase2_run.json"
        selection_file = model_dir / "case_selection.csv"
        if not run_file.exists() or not selection_file.exists():
            bad_runs.append(f"{dataset}/{model}: missing run metadata or case selection")
            continue
        run = json.loads(run_file.read_text())
        if run["exit_code"] != 0:
            bad_runs.append(f"{dataset}/{model}: non-zero return code")
            continue
        selection = pd.read_csv(selection_file, dtype=str).fillna("")
        selection = selection.reindex(sorted(selection.columns), axis=1)
        selection = selection.sort_values(list(selection.columns)).reset_index(drop=True)
        sig = hashlib.sha256(selection.to_csv(index=False).encode()).hexdigest()
        selection_signatures.setdefault(dataset, set()).add(sig)

        frame = pd.read_csv(summary_path)
        required = {"context_hours", "context_steps", "mean_entity_mae", "n_cases", "n_entities"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"Unexpected summary format: {summary_path}")
        timings = model_dir / "runtime_batches.csv"
        t = pd.read_csv(timings) if timings.exists() else None
        for _, row in frame.iterrows():
            context_steps = int(row["context_steps"])
            relevant = (t[t["context_steps"] == context_steps]
                        if t is not None and "context_steps" in t else None)
            infer_seconds = (float(relevant["inference_seconds"].sum())
                             if relevant is not None and "inference_seconds" in relevant else np.nan)
            peak_gb = (float(relevant["peak_cuda_allocated_bytes"].max()) / 1e9
                       if relevant is not None and "peak_cuda_allocated_bytes" in relevant
                       and not relevant.empty else np.nan)
            results.append({
                "dataset": dataset, "model": model,
                "context_hours": float(row["context_hours"]),
                "context_steps": context_steps,
                "mean_entity_mae": float(row["mean_entity_mae"]),
                "n_entities": int(row["n_entities"]),
                "n_cases": int(row["n_cases"]),
                "inference_seconds": infer_seconds,
                "peak_cuda_allocated_gb": peak_gb,
                "subprocess_wall_seconds": run["subprocess_wall_seconds"],
                "case_selection_sha256": sig,
            })
    for dataset, signatures in selection_signatures.items():
        if len(signatures) != 1:
            bad_runs.append(f"{dataset}: different case selection across models!")
    if bad_runs:
        raise RuntimeError("Benchmark QC failed:\n" + "\n".join(bad_runs))
    if not results:
        raise RuntimeError(f"No complete benchmark results in {root}")
    return pd.DataFrame(results)


def build_analysis(df: pd.DataFrame):
    # Same context grid required to compare rankings across models.
    for dataset, d in df.groupby("dataset"):
        grids = d.groupby("model")["context_steps"].apply(lambda s: tuple(sorted(s)))
        if grids.nunique() > 1:
            raise RuntimeError(f"{dataset}: model context grids differ: {grids.to_dict()}")
        if d.groupby("model")["n_cases"].nunique().max() != 1:
            raise RuntimeError(f"{dataset}: inconsistent n_cases within a model")

    winners = df.loc[df.groupby("dataset")["mean_entity_mae"].idxmin()].copy()
    winners = winners.sort_values("dataset")
    gains = []
    flips = []
    for (dataset, model), d in df.groupby(["dataset", "model"]):
        d = d.sort_values("context_steps")
        short, long = d.iloc[0], d.iloc[-1]
        gains.append({
            "dataset": dataset, "model": model,
            "short_context_steps": int(short.context_steps),
            "long_context_steps": int(long.context_steps),
            "short_mae": float(short.mean_entity_mae),
            "long_mae": float(long.mean_entity_mae),
            "relative_gain": (
                (short.mean_entity_mae - long.mean_entity_mae) / short.mean_entity_mae
                if short.mean_entity_mae != 0 else np.nan
            ),
            "best_context_steps": int(d.loc[d.mean_entity_mae.idxmin(), "context_steps"]),
        })
    for dataset, d in df.groupby("dataset"):
        contexts = sorted(d.context_steps.unique())
        first = d[d.context_steps == contexts[0]].set_index("model").mean_entity_mae
        last = d[d.context_steps == contexts[-1]].set_index("model").mean_entity_mae
        models = sorted(first.index)
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                delta_first = first[m1] - first[m2]
                delta_last = last[m1] - last[m2]
                flips.append({"dataset": dataset, "model_a": m1, "model_b": m2,
                              "short_context_steps": contexts[0],
                              "long_context_steps": contexts[-1],
                              "rank_reversal": bool(delta_first * delta_last < 0),
                              "short_mae_difference": float(delta_first),
                              "long_mae_difference": float(delta_last)})
    flip_cols = ["dataset", "model_a", "model_b", "short_context_steps",
                 "long_context_steps", "rank_reversal", "short_mae_difference",
                 "long_mae_difference"]
    return winners, pd.DataFrame(gains), pd.DataFrame(flips, columns=flip_cols)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="results/phase2/development_pilot")
    args = p.parse_args()
    root = Path(args.output_root)
    results = read_outputs(root)
    winners, gains, flips = build_analysis(results)
    outdir = root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(outdir / "configurations.csv", index=False)
    winners.to_csv(outdir / "best_observed_per_dataset.csv", index=False)
    gains.to_csv(outdir / "context_gains.csv", index=False)
    flips.to_csv(outdir / "model_rank_reversals.csv", index=False)
    print("=" * 78)
    print("PersTSFM Phase-2 development pilot: QC PASS")
    print("=" * 78)
    print(f"Datasets: {results.dataset.nunique()} | Checkpoints: {results.model.nunique()} | "
          f"Configurations: {len(results)}")
    print("\nBest within evaluated pilot grid (NOT exhaustive oracle):")
    print(winners[["dataset", "model", "context_steps", "mean_entity_mae", "n_entities", "n_cases"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nShort-to-long gains (positive means longer was better):")
    print(gains[["dataset", "model", "relative_gain", "best_context_steps"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3%}"))
    print(f"\nModel rank reversals between shortest/longest contexts: "
          f"{int(flips.rank_reversal.sum())}/{len(flips)} pairs")
    print(f"\nSaved summary tables to {outdir}")
    print("\nWarning: all results are from development data. Do not infer held-out generalisation.")


if __name__ == "__main__":
    main()
