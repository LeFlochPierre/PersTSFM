from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATASETS = (
    "electricity",
    "vitaldb",
    "physionet2019",
    "hupa_ucm",
    "solar",
    "pems_bay",
)


def _steps(df: pd.DataFrame, model: str) -> list[int]:
    vals = (
        df.loc[df["model"] == model, "context_steps"]
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not vals:
        raise ValueError(f"No contexts found for model={model!r}")
    return vals


def _row(df: pd.DataFrame, model: str, step: int) -> pd.Series:
    x = df.loc[
        (df["model"] == model)
        & (df["context_steps"].astype(int) == int(step))
    ]
    if len(x) != 1:
        raise ValueError(
            f"Expected exactly one row for {model=} {step=}, found {len(x)}"
        )
    return x.iloc[0]


def three_anchors(
    proposed: pd.DataFrame,
    full: pd.DataFrame,
    model: str,
) -> dict[str, int]:
    """
    V1.1 anchors.

    short:
        Smallest context proposed by the frozen V1 planner.

    structural:
        Largest context proposed by the frozen V1 planner. This is the
        planner's existing long/structural guess.

    capacity:
        Largest model-feasible context present in the frozen exhaustive
        search space.

    If structural == capacity, keep the existing V1 middle context as the
    third distinct context so that the fixed V1.1 budget remains three
    contexts/model whenever possible.
    """
    p = _steps(proposed, model)
    f = _steps(full, model)

    short = p[0]
    structural = p[-1]
    capacity = f[-1]

    distinct = []
    for s in (short, structural, capacity):
        if s not in distinct:
            distinct.append(s)

    # Preserve one existing V1 interior anchor when capacity duplicates
    # structural (e.g. Solar / PhysioNet).
    for s in p:
        if len(distinct) >= min(3, len(f)):
            break
        if s not in distinct:
            distinct.append(s)

    # Defensive fallback: use feasible full-search contexts if a future
    # dataset has fewer distinct V1 contexts.
    if len(distinct) < min(3, len(f)):
        for s in f:
            if len(distinct) >= min(3, len(f)):
                break
            if s not in distinct:
                distinct.append(s)

    return {
        "short": int(short),
        "structural": int(structural),
        "capacity": int(capacity),
        "selected": sorted(int(s) for s in distinct),
    }


def _evaluate_selected(
    full: pd.DataFrame,
    selected: dict[str, Iterable[int]],
) -> tuple[pd.DataFrame, pd.Series]:
    parts = []
    for model, steps in selected.items():
        for step in steps:
            parts.append(_row(full, model, int(step)).to_frame().T)

    evaluated = pd.concat(parts, ignore_index=True)
    evaluated["context_steps"] = evaluated["context_steps"].astype(int)
    evaluated["mean_entity_mae"] = evaluated["mean_entity_mae"].astype(float)
    best = evaluated.sort_values("mean_entity_mae").iloc[0]
    return evaluated, best


def simulate_v11(
    proposed: pd.DataFrame,
    full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, dict[str, int]]]:
    selected: dict[str, list[int]] = {}
    anchors: dict[str, dict[str, int]] = {}

    for model in sorted(proposed["model"].unique()):
        a = three_anchors(proposed, full, model)
        anchors[model] = a
        selected[model] = a["selected"]

    evaluated, best = _evaluate_selected(full, selected)
    return evaluated, best, anchors


def simulate_v12(
    proposed: pd.DataFrame,
    full: pd.DataFrame,
    tie_band: float = 0.02,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    dict[str, dict[str, int]],
    list[str],
]:
    """
    Adaptive V1.2 policy.

    1. Evaluate SHORT + STRUCTURAL for every model.
    2. Compare models at their STRUCTURAL anchor.
    3. Any model whose structural MAE is within `tie_band` of the best
       structural MAE is considered competitive.
    4. Only competitive models receive a CAPACITY evaluation (unless
       structural == capacity already).
    5. Choose the best observed configuration.

    tie_band=0.02 means "within 2% of the best structural MAE".
    """
    anchors: dict[str, dict[str, int]] = {}
    initial: dict[str, list[int]] = {}
    structural_scores: dict[str, float] = {}

    models = sorted(proposed["model"].unique())

    for model in models:
        a = three_anchors(proposed, full, model)
        anchors[model] = a

        init = []
        for s in (a["short"], a["structural"]):
            if s not in init:
                init.append(s)
        initial[model] = init

        structural_scores[model] = float(
            _row(full, model, a["structural"])["mean_entity_mae"]
        )

    best_structural = min(structural_scores.values())
    cutoff = best_structural * (1.0 + float(tie_band))

    competitive = [
        model
        for model in models
        if structural_scores[model] <= cutoff
    ]

    selected = {model: list(steps) for model, steps in initial.items()}

    for model in competitive:
        capacity = anchors[model]["capacity"]
        if capacity not in selected[model]:
            selected[model].append(capacity)

    evaluated, best = _evaluate_selected(full, selected)
    return evaluated, best, anchors, competitive


def _policy_record(
    dataset: str,
    policy: str,
    best: pd.Series,
    oracle: pd.Series,
    n_evaluated: int,
    n_full: int,
    tie_band: float | None = None,
) -> dict[str, object]:
    oracle_mae = float(oracle["mean_entity_mae"])
    best_mae = float(best["mean_entity_mae"])
    regret = best_mae - oracle_mae

    return {
        "dataset": dataset,
        "policy": policy,
        "tie_band": tie_band,
        "best_model": best["model"],
        "best_context_steps": int(best["context_steps"]),
        "best_context_hours": float(best["context_hours"]),
        "best_mae": best_mae,
        "oracle_model": oracle["model"],
        "oracle_context_steps": int(oracle["context_steps"]),
        "oracle_context_hours": float(oracle["context_hours"]),
        "oracle_mae": oracle_mae,
        "absolute_regret": regret,
        "relative_regret_pct": 100.0 * regret / oracle_mae,
        "n_evaluated": int(n_evaluated),
        "n_full": int(n_full),
        "search_reduction_pct": 100.0 * (1.0 - n_evaluated / n_full),
    }


def simulate_dataset(
    dataset: str,
    matrix_root: Path,
    tie_band: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    proposed_path = matrix_root / "proposed" / f"{dataset}.csv"
    full_path = matrix_root / "full" / f"{dataset}.csv"

    proposed = pd.read_csv(proposed_path)
    full = pd.read_csv(full_path)

    oracle = full.sort_values("mean_entity_mae").iloc[0]

    # V1 current frozen policy.
    v1_best = proposed.sort_values("mean_entity_mae").iloc[0]

    # V1.1 fixed three-anchor policy.
    v11_eval, v11_best, anchors = simulate_v11(proposed, full)

    # V1.2 adaptive policy.
    v12_eval, v12_best, _, competitive = simulate_v12(
        proposed,
        full,
        tie_band=tie_band,
    )

    records = [
        _policy_record(
            dataset=dataset,
            policy="v1_current",
            best=v1_best,
            oracle=oracle,
            n_evaluated=len(proposed),
            n_full=len(full),
        ),
        _policy_record(
            dataset=dataset,
            policy="v1_1_three_anchor",
            best=v11_best,
            oracle=oracle,
            n_evaluated=len(v11_eval),
            n_full=len(full),
        ),
        _policy_record(
            dataset=dataset,
            policy="v1_2_adaptive",
            best=v12_best,
            oracle=oracle,
            n_evaluated=len(v12_eval),
            n_full=len(full),
            tie_band=tie_band,
        ),
    ]

    audit = {
        "dataset": dataset,
        "anchors": anchors,
        "v12_competitive_models": competitive,
        "v12_n_evaluated": len(v12_eval),
    }
    return records, audit


def sensitivity(
    datasets: Iterable[str],
    matrix_root: Path,
    tie_bands: Iterable[float],
) -> pd.DataFrame:
    rows = []

    for tie_band in tie_bands:
        for dataset in datasets:
            proposed = pd.read_csv(
                matrix_root / "proposed" / f"{dataset}.csv"
            )
            full = pd.read_csv(
                matrix_root / "full" / f"{dataset}.csv"
            )
            oracle = full.sort_values("mean_entity_mae").iloc[0]

            evaluated, best, _, competitive = simulate_v12(
                proposed,
                full,
                tie_band=tie_band,
            )

            rec = _policy_record(
                dataset=dataset,
                policy="v1_2_adaptive",
                best=best,
                oracle=oracle,
                n_evaluated=len(evaluated),
                n_full=len(full),
                tie_band=tie_band,
            )
            rec["competitive_models"] = ",".join(competitive)
            rows.append(rec)

    return pd.DataFrame(rows)


def _print_policy_table(df: pd.DataFrame) -> None:
    cols = [
        "dataset",
        "policy",
        "best_model",
        "best_context_steps",
        "best_mae",
        "oracle_model",
        "oracle_context_steps",
        "oracle_mae",
        "relative_regret_pct",
        "n_evaluated",
        "n_full",
        "search_reduction_pct",
    ]
    print(
        df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def _print_overall(df: pd.DataFrame) -> None:
    print()
    print("=" * 120)
    print("OVERALL BY POLICY")
    print("=" * 120)

    rows = []
    for policy, g in df.groupby("policy", sort=False):
        total_eval = int(g["n_evaluated"].sum())
        total_full = int(g["n_full"].sum())
        rows.append(
            {
                "policy": policy,
                "mean_rel_regret_pct": g["relative_regret_pct"].mean(),
                "median_rel_regret_pct": g["relative_regret_pct"].median(),
                "max_rel_regret_pct": g["relative_regret_pct"].max(),
                "zero_regret_datasets": int(
                    (g["absolute_regret"].abs() < 1e-12).sum()
                ),
                "datasets": len(g),
                "total_evaluated": total_eval,
                "total_full": total_full,
                "overall_search_reduction_pct": (
                    100.0 * (1.0 - total_eval / total_full)
                ),
            }
        )

    out = pd.DataFrame(rows)
    print(
        out.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay of PersTSFM V1, V1.1 and V1.2 policies "
            "using already-computed exhaustive benchmark matrices."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("results/selfconfig/benchmark_matrix"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/selfconfig/evaluation"),
    )
    parser.add_argument(
        "--tie-band",
        type=float,
        default=0.02,
        help="V1.2 competitive structural-performance band, e.g. 0.02 = 2%%.",
    )
    parser.add_argument(
        "--sensitivity",
        nargs="*",
        type=float,
        default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.10],
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    records = []
    audits = []

    for dataset in args.datasets:
        ds_records, audit = simulate_dataset(
            dataset=dataset,
            matrix_root=args.matrix_root,
            tie_band=args.tie_band,
        )
        records.extend(ds_records)
        audits.append(audit)

    results = pd.DataFrame(records)

    print("=" * 120)
    print("PERS-TSFM OFFLINE POLICY REPLAY")
    print("=" * 120)
    print(
        f"V1.2 tie band: {100.0 * args.tie_band:.2f}%"
    )
    print()

    _print_policy_table(results)
    _print_overall(results)

    print()
    print("=" * 120)
    print("V1.1 / V1.2 ANCHOR AUDIT")
    print("=" * 120)

    for audit in audits:
        print()
        print(audit["dataset"].upper())
        for model, a in audit["anchors"].items():
            print(
                f"  {model:10s} "
                f"short={a['short']:5d}  "
                f"structural={a['structural']:5d}  "
                f"capacity={a['capacity']:5d}  "
                f"v1.1={a['selected']}"
            )
        print(
            "  V1.2 competitive at structural scale: "
            + ", ".join(audit["v12_competitive_models"])
        )
        print(
            f"  V1.2 evaluations: {audit['v12_n_evaluated']}"
        )

    results_path = args.output_root / "v1_v11_v12_policy_replay.csv"
    results.to_csv(results_path, index=False)

    sens = sensitivity(
        datasets=args.datasets,
        matrix_root=args.matrix_root,
        tie_bands=args.sensitivity,
    )
    sens_path = args.output_root / "v12_tie_band_sensitivity.csv"
    sens.to_csv(sens_path, index=False)

    print()
    print("=" * 120)
    print("V1.2 TIE-BAND SENSITIVITY")
    print("=" * 120)

    summary = (
        sens.groupby("tie_band", as_index=False)
        .agg(
            mean_rel_regret_pct=("relative_regret_pct", "mean"),
            median_rel_regret_pct=("relative_regret_pct", "median"),
            max_rel_regret_pct=("relative_regret_pct", "max"),
            total_evaluated=("n_evaluated", "sum"),
            total_full=("n_full", "sum"),
        )
    )
    summary["overall_search_reduction_pct"] = (
        100.0
        * (
            1.0
            - summary["total_evaluated"]
            / summary["total_full"]
        )
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved:")
    print(f"  {results_path}")
    print(f"  {sens_path}")


if __name__ == "__main__":
    main()
