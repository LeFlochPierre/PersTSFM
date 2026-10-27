"""PersTSFM Phase-2 development-only temporal policy audit.

Uses existing case_metrics.csv files; NO model inference and NO result modification.
Each entity's earlier origins select a configuration; later origins evaluate it.

Run from the repository root:
    cp PersTSFM_phase2_temporal_policy_audit.py scripts/phase2/temporal_policy_audit.py
    python -m scripts.phase2.temporal_policy_audit --random-draws 1000 --seed 42

This is a *small exploratory split* (typically 8 entities, 2+2 origins).
Do not tune and report it as an unseen-dataset generalisation result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = (
    "chronos2", "timesfm3", "tirex2", "toto2_22m", "toto2_313m", "moirai2_small"
)
IDENTITY = ["entity_id", "eval_index", "eval_origin"]
KEY = ["model", "context_steps"]


def load_dataset(dataset_dir: Path, models: tuple[str, ...]) -> pd.DataFrame:
    pieces = []
    for model in models:
        path = dataset_dir / "context" / model / "case_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected results: {path}")
        df = pd.read_csv(path)
        needed = set(IDENTITY + ["context_steps", "mae"])
        if not needed.issubset(df.columns):
            raise ValueError(f"{path}: missing {sorted(needed - set(df.columns))}")
        df = df.loc[:, IDENTITY + ["context_steps", "mae"]].copy()
        df["model"] = model
        df["entity_id"] = df["entity_id"].astype(str)
        df["eval_index"] = df["eval_index"].astype(int)
        df["context_steps"] = df["context_steps"].astype(int)
        df["eval_origin"] = pd.to_datetime(df["eval_origin"], errors="raise")
        df["mae"] = pd.to_numeric(df["mae"], errors="raise")
        if not np.isfinite(df.mae.to_numpy(dtype=float)).all() or (df.mae < 0).any():
            raise ValueError(f"{path}: invalid MAE")
        if df.duplicated(KEY + IDENTITY).any():
            raise ValueError(f"{path}: duplicate case rows")
        pieces.append(df)
    data = pd.concat(pieces, ignore_index=True)
    grids = data.groupby("model")["context_steps"].apply(lambda s: tuple(sorted(s.unique())))
    if grids.nunique() != 1 or len(grids.iloc[0]) != 3:
        raise ValueError(f"Expected the same THREE contexts for each checkpoint:\n{grids}")
    # Identical case IDs per configuration. No unfair model-to-model case differences.
    reference = None
    for (model, context), chunk in data.groupby(KEY):
        ids = set(zip(chunk.entity_id, chunk.eval_index, chunk.eval_origin))
        if reference is None:
            reference = ids
        elif ids != reference:
            raise ValueError(f"Unmatched cases for {model}/{context}: "
                             f"missing={len(reference - ids)}, extra={len(ids - reference)}")
    return data


def temporal_split(data: pd.DataFrame, n_early: int = 2) -> pd.DataFrame:
    cases = data.loc[:, IDENTITY].drop_duplicates()
    cases = cases.sort_values(["entity_id", "eval_origin", "eval_index"])
    cases["order"] = cases.groupby("entity_id").cumcount()
    counts = cases.groupby("entity_id").size()
    if (counts <= n_early).any():
        raise ValueError("Every entity must have more origins than --early-origins")
    if counts.nunique() != 1:
        raise ValueError("Unequal origin counts across entities; investigate sampling first")
    if cases.duplicated(["entity_id", "eval_origin"]).any():
        raise ValueError("Duplicate forecast origin within entity")
    cases["role"] = np.where(cases.order < n_early, "selection", "evaluation")
    joined = data.merge(cases.loc[:, IDENTITY + ["role"]], on=IDENTITY,
                        how="left", validate="many_to_one")
    if joined.role.isna().any():
        raise RuntimeError("Split join failed")
    return joined


def mean_entity_mae(data: pd.DataFrame) -> pd.DataFrame:
    # The current PersTSFM metric: average cases per entity, then entities equally.
    em = (data.groupby(KEY + ["entity_id"], as_index=False)
          .agg(entity_mae=("mae", "mean")))
    return (em.groupby(KEY, as_index=False)
            .agg(mae=("entity_mae", "mean")))


def pick(table: pd.DataFrame) -> pd.Series:
    return table.sort_values(["mae", "model", "context_steps"]).iloc[0]


def policy_candidates(selection: pd.DataFrame, policy: str, band: float) -> pd.DataFrame:
    contexts = sorted(selection.context_steps.unique())
    if len(contexts) != 3:
        raise ValueError("Expected exactly 3 contexts")
    short, middle, longest = contexts
    if policy == "short_only":
        return selection.loc[selection.context_steps == short]
    if policy == "middle_only":
        return selection.loc[selection.context_steps == middle]
    if policy == "long_only":
        return selection.loc[selection.context_steps == longest]
    if policy == "middle_plus_long":
        return selection.loc[selection.context_steps.isin([middle, longest])]
    if policy == "short_plus_long":
        return selection.loc[selection.context_steps.isin([short, longest])]
    if policy == "full_search":
        return selection
    if policy == "adaptive_band":
        initial = selection.loc[selection.context_steps.isin([short, middle])]
        middle_scores = selection.loc[selection.context_steps == middle]
        threshold = middle_scores.mae.min() * (1.0 + band + 1e-12)
        competitive = set(middle_scores.loc[middle_scores.mae <= threshold, "model"])
        extra = selection.loc[(selection.context_steps == longest) &
                              (selection.model.isin(competitive))]
        return pd.concat([initial, extra], ignore_index=True)
    raise ValueError(f"Unknown policy: {policy}")


def scored_policy(selection: pd.DataFrame, evaluation: pd.DataFrame,
                  dataset: str, policy: str, band: float) -> dict:
    observed = policy_candidates(selection, policy, band)
    chosen = pick(observed)
    scored = evaluation.merge(
        pd.DataFrame([{"model": chosen["model"], "context_steps": int(chosen["context_steps"])}]),
        on=KEY, how="inner", validate="one_to_one")
    if len(scored) != 1:
        raise RuntimeError("Selected configuration missing from held-out origins")
    actual = float(scored.iloc[0].mae)
    oracle = float(evaluation.mae.min())  # ONLY used for retrospective regret.
    regret = 100.0 * (actual - oracle) / oracle if oracle > 0 else 0.0
    return {
        "dataset": dataset, "policy": policy,
        "selected_model": str(chosen.model),
        "selected_context_steps": int(chosen.context_steps),
        "selection_mae": float(chosen.mae),
        "later_mae": actual,
        "later_grid_oracle_mae": oracle,
        "later_regret_pct": max(0.0, regret),
        "evaluations": len(observed), "full_evaluations": len(selection),
        "within_1pct": int(regret <= 1.0 + 1e-10),
        "within_2pct": int(regret <= 2.0 + 1e-10),
        "within_5pct": int(regret <= 5.0 + 1e-10),
    }


def random_matched(selection: pd.DataFrame, evaluation: pd.DataFrame,
                   k: int, draws: int, seed: int, dataset: str) -> dict:
    rg = np.random.default_rng(seed)
    tests = evaluation.set_index(KEY)["mae"]
    oracle = float(evaluation.mae.min())
    regrets = []
    for _ in range(draws):
        chosen_indices = rg.choice(len(selection), size=k, replace=False)
        selected = pick(selection.iloc[chosen_indices])
        actual = float(tests.loc[(selected.model, int(selected.context_steps))])
        regrets.append(100 * (actual - oracle) / oracle if oracle > 0 else 0.0)
    regrets = np.asarray(regrets)
    return {
        "dataset": dataset, "policy": "random_matched",
        "selected_model": "(random)", "selected_context_steps": -1,
        "selection_mae": np.nan,
        "later_mae": np.nan,
        "later_grid_oracle_mae": oracle,
        "later_regret_pct": float(regrets.mean()),
        "later_regret_p90_pct": float(np.quantile(regrets, 0.90)),
        "evaluations": k, "full_evaluations": len(selection),
        "within_1pct": float(np.mean(regrets <= 1.0 + 1e-10)),
        "within_2pct": float(np.mean(regrets <= 2.0 + 1e-10)),
        "within_5pct": float(np.mean(regrets <= 5.0 + 1e-10)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/phase2/development_pilot"))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--early-origins", type=int, default=2)
    p.add_argument("--band", type=float, default=0.02)
    p.add_argument("--random-draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.early_origins < 1 or args.random_draws < 1 or args.band < 0:
        p.error("Invalid split, draws, or band argument")
    root = args.root.resolve()
    out_dir = root / "analysis" / "temporal_policy_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    datasets = [d for d in sorted(root.iterdir())
                if d.is_dir() and (d / "context").is_dir()]
    if not datasets:
        raise FileNotFoundError(f"No datasets under {root}")
    policies = ["short_only", "middle_only", "long_only", "middle_plus_long",
                "short_plus_long", "adaptive_band", "full_search"]
    for d in datasets:
        data = load_dataset(d, tuple(args.models))
        data = temporal_split(data, n_early=args.early_origins)
        selection = mean_entity_mae(data.loc[data.role == "selection"])
        evaluation = mean_entity_mae(data.loc[data.role == "evaluation"])
        if set(map(tuple, selection[KEY].to_numpy())) != set(map(tuple, evaluation[KEY].to_numpy())):
            raise ValueError(f"{d.name}: selection/evaluation config mismatch")
        for policy in policies:
            rows.append(scored_policy(selection, evaluation, d.name, policy, args.band))
        adaptive = [r for r in rows if r["dataset"] == d.name and r["policy"] == "adaptive_band"][0]
        rows.append(random_matched(selection, evaluation, adaptive["evaluations"],
                                   args.random_draws, args.seed, d.name))
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "per_dataset.csv", index=False)
    macro = result.groupby("policy", as_index=False).agg(
        mean_later_regret_pct=("later_regret_pct", "mean"),
        worst_later_regret_pct=("later_regret_pct", "max"),
        total_evaluations=("evaluations", "sum"),
        total_full=("full_evaluations", "sum"),
        fraction_within_1pct=("within_1pct", "mean"),
        fraction_within_2pct=("within_2pct", "mean"),
        fraction_within_5pct=("within_5pct", "mean"),
    )
    macro["evaluation_reduction_pct"] = 100 * (1 - macro.total_evaluations / macro.total_full)
    macro = macro.sort_values(["total_evaluations", "mean_later_regret_pct"])
    macro.to_csv(out_dir / "policy_summary.csv", index=False)
    print("=" * 95)
    print("PersTSFM — TEMPORAL POLICY AUDIT, DEVELOPMENT ONLY")
    print("=" * 95)
    print(f"Datasets: {len(datasets)} | models: {len(args.models)} | "
          f"earlier origins/entity: {args.early_origins} | band: {100 * args.band:g}%")
    print("Selection uses EARLIER origins; regret evaluated on LATER origins.")
    print("Results reuse development data, not unseen dataset families. No GPU-cost assumptions.")
    print("\nPrimary question: does LONG-ONLY beat the adaptive policy with fewer evaluations?")
    shown = result.loc[result.policy.isin(["long_only", "adaptive_band", "full_search"]),
                       ["dataset", "policy", "selected_model", "selected_context_steps",
                        "later_regret_pct", "evaluations"]]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMacro policy comparisons:")
    print(macro[["policy", "mean_later_regret_pct", "worst_later_regret_pct",
                 "total_evaluations", "evaluation_reduction_pct", "fraction_within_2pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved: {out_dir}/per_dataset.csv and policy_summary.csv")
    print("CAUTION: only a 2+2 (by default) temporal split within eight entities, ")
    print("not a held-out dataset-family study. Performance on later origins is NOT used in selection.")


if __name__ == "__main__":
    main()
