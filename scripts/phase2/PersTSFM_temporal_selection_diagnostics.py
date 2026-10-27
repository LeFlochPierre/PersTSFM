"""PersTSFM Phase-2: paired diagnostics for early-vs-later policy selection.

Development data ONLY. No GPU execution, no forecast rewriting, no policy fitting.
Requires scripts.phase2.temporal_policy_audit from the previous milestone.

Copy to scripts/phase2/temporal_selection_diagnostics.py and run from project root:
python -m scripts.phase2.temporal_selection_diagnostics --bootstrap 2000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase2.temporal_policy_audit import (
    MODELS, KEY, load_dataset, temporal_split, mean_entity_mae,
    pick, policy_candidates,
)

POLICIES = ("long_only", "adaptive_band", "full_search")


def score_one_split(frame: pd.DataFrame, n_early: int, band: float, *,
                    bootstrap: int, seed: int, dataset: str) -> tuple[list[dict], dict]:
    split = temporal_split(frame, n_early=n_early)
    early = split[split.role == "selection"].copy()
    later = split[split.role == "evaluation"].copy()
    early_table = mean_entity_mae(early)
    later_table = mean_entity_mae(later)
    later_lookup = later_table.set_index(KEY).mae
    if len(early_table) != len(later_table):
        raise RuntimeError("Early/later configuration grids differ")
    oracle = float(later_table.mae.min())
    if oracle <= 0:
        raise ValueError("Relative regret undefined for nonpositive later-grid oracle MAE")

    diag = {
        "dataset": dataset,
        "n_early": n_early,
        "n_later": int(later.groupby("entity_id").size().iloc[0] / len(later_table)),
        "n_entities": int(early.entity_id.nunique()),
        "early_to_later_spearman": float(early_table.set_index(KEY).mae.corr(
            later_table.set_index(KEY).mae, method="spearman")),
        "global_origin_order": "strict" if early.eval_origin.max() < later.eval_origin.min() else "overlapping",
        "latest_early_origin": str(early.eval_origin.max()),
        "earliest_later_origin": str(later.eval_origin.min()),
        "later_grid_oracle_mae": oracle,
        "early_grid_winner": str(pick(early_table)["model"]) + "/" + str(int(pick(early_table)["context_steps"])),
        "later_grid_winner": str(pick(later_table)["model"]) + "/" + str(int(pick(later_table)["context_steps"])),
    }
    selections = {}
    outcomes = []
    for policy in POLICIES:
        subset = policy_candidates(early_table, policy, band)
        best = pick(subset)
        name = str(best["model"])
        context = int(best["context_steps"])
        later_mae = float(later_lookup.loc[(name, context)])
        selections[policy] = (name, context)
        outcomes.append({
            "dataset": dataset, "n_early": n_early, "policy": policy,
            "chosen_model": name, "chosen_context_steps": context,
            "selection_mae": float(best.mae), "later_mae": later_mae,
            "later_oracle_mae": oracle, "later_absolute_regret": later_mae-oracle,
            "later_regret_pct": 100*(later_mae-oracle)/oracle,
            "configurations_evaluated": len(subset),
        })
    # Evaluate the paired post-selection comparison on the SAME entities/late origins.
    late_entity = (later.groupby(KEY + ["entity_id"], as_index=False)
                   .agg(mae=("mae", "mean")))
    entities = sorted(late_entity.entity_id.unique())
    if len(entities) < 2:
        raise RuntimeError("At least two entities required for paired bootstrap")
    rng = np.random.default_rng(seed)
    for comparator in ("full_search", "long_only"):
        a = selections["adaptive_band"]
        b = selections[comparator]
        x = (late_entity[(late_entity.model == a[0]) &
                         (late_entity.context_steps == a[1])].set_index("entity_id").mae)
        y = (late_entity[(late_entity.model == b[0]) &
                         (late_entity.context_steps == b[1])].set_index("entity_id").mae)
        differences = (x.reindex(entities)-y.reindex(entities)).to_numpy(dtype=float)
        if not np.isfinite(differences).all():
            raise RuntimeError("Unmatched later entities")
        idx = rng.integers(0, len(entities), size=(bootstrap, len(entities)))
        draws = differences[idx].mean(axis=1)
        diag[f"adaptive_minus_{comparator}_mae"] = float(differences.mean())
        diag[f"adaptive_minus_{comparator}_ci025"] = float(np.percentile(draws, 2.5))
        diag[f"adaptive_minus_{comparator}_ci975"] = float(np.percentile(draws, 97.5))
        diag[f"adaptive_beats_{comparator}_bootstrap_fraction"] = float(np.mean(draws < 0))
    return outcomes, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/phase2/development_pilot"))
    ap.add_argument("--band", type=float, default=0.02)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-origins", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    if args.band < 0 or args.bootstrap < 50 or min(args.early_origins) < 1:
        ap.error("Invalid band/bootstrap/early-origins")
    datasets = [p for p in sorted(args.root.iterdir()) if (p / "context").is_dir()]
    if not datasets:
        raise FileNotFoundError(f"No datasets: {args.root}")
    all_rows, diagnostics = [], []
    for d in datasets:
        frame = load_dataset(d, MODELS)
        n_origins = frame[frame.model == MODELS[0]].groupby("entity_id").eval_origin.nunique()
        if n_origins.nunique() != 1:
            raise ValueError(f"{d.name}: number of origins varies by entity")
        for n_early in args.early_origins:
            if n_early >= int(n_origins.iloc[0]):
                raise ValueError(f"{d.name}: {n_early} early origins leaves no later origin")
            rows, diag = score_one_split(frame, n_early, args.band,
                                         bootstrap=args.bootstrap,
                                         seed=args.seed, dataset=d.name)
            all_rows.extend(rows)
            diagnostics.append(diag)
    results = pd.DataFrame(all_rows)
    diag = pd.DataFrame(diagnostics)
    output = args.root / "analysis" / "temporal_selection_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "policy_by_split.csv", index=False)
    diag.to_csv(output / "paired_diagnostics.csv", index=False)
    macro = (results.groupby(["n_early", "policy"], as_index=False)
             .agg(mean_later_regret_pct=("later_regret_pct", "mean"),
                  worst_later_regret_pct=("later_regret_pct", "max"),
                  mean_absolute_regret=("later_absolute_regret", "mean"),
                  evaluations=("configurations_evaluated", "sum")))
    macro.to_csv(output / "policy_macro_by_split.csv", index=False)
    print("="*85)
    print("PersTSFM: TEMPORAL SELECTION DIAGNOSTICS — DEVELOPMENT ONLY")
    print("="*85)
    print(f"Datasets: {len(datasets)} | bootstrap: {args.bootstrap} | band: {args.band:.1%}")
    print("Exploratory chronological cutoffs, based on per-entity ordering (NOT globally locked dates)")
    print("\nMacro regrets by early-origin count (the cuts are not independent):")
    print(macro.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    principal = diag[diag.n_early == 2]
    if not principal.empty:
        print("\n2+2 split: rank agreement, paired adaptive vs full (negative delta favours adaptive):")
        print(principal[["dataset", "early_to_later_spearman", "global_origin_order",
                         "adaptive_minus_full_search_mae", "adaptive_minus_full_search_ci025",
                         "adaptive_minus_full_search_ci975",
                         "adaptive_beats_full_search_bootstrap_fraction"]].to_string(
                             index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved:", output)
    print("Do not optimise the band on these exploratory diagnostics or claim test generalisation.")


if __name__ == "__main__":
    main()
