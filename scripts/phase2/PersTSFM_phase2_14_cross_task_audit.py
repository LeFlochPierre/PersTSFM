#!/usr/bin/env python3
"""PersTSFM Phase 2.14: cross-task historical calibration audit.

Uses only existing Phase 2.13 score and case catalogs.
Selects on historical lags 3 and 2; evaluates on lag 1.
Never accesses official FEV evaluation targets.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "mae": "calibration_mean_case_mae",
    "mean_pinball_3": "calibration_mean_case_pinball_3",
}


def find_catalog(root):
    candidates = sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and (p / "score_catalog.csv").is_file()
        and (p / "case_catalog.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No complete catalogs found in {root}")
    return candidates[-1]


def load_catalog(directory):
    scores = pd.read_csv(directory / "score_catalog.csv")
    cases = pd.read_csv(
        directory / "case_catalog.csv",
        dtype={"case_id": str, "entity_id": str},
    )

    score_cols = {
        "task_id", "dataset_config", "model", "context_steps",
        "k", "n_cases", "n_entities",
        "calibration_mean_case_mae",
        "calibration_mean_case_pinball_3",
    }
    case_cols = {
        "task_id", "dataset_config", "model", "context_steps",
        "k", "case_id", "entity_id",
        "lag_before_first_official", "mae", "mean_pinball_3",
    }

    if not score_cols.issubset(scores.columns):
        raise ValueError(
            f"Missing score columns: {score_cols - set(scores.columns)}"
        )
    if not case_cols.issubset(cases.columns):
        raise ValueError(
            f"Missing case columns: {case_cols - set(cases.columns)}"
        )

    if scores.empty or cases.empty:
        raise ValueError("Empty catalog")

    keys = ["task_id", "model", "context_steps", "k"]

    if scores.duplicated(keys).any():
        raise ValueError("Duplicate configuration in score catalog")
    if cases.duplicated(keys + ["case_id"]).any():
        raise ValueError("Duplicate configuration/case ID")

    if set(scores["k"]) != {3} or set(cases["k"]) != {3}:
        raise ValueError("Expected exclusively K=3 experiments")

    if set(cases["lag_before_first_official"]) != {1, 2, 3}:
        raise ValueError("Expected historical lags 1, 2, and 3")

    numeric_score = [
        "n_cases", "n_entities",
        "calibration_mean_case_mae",
        "calibration_mean_case_pinball_3",
    ]
    numeric_case = ["mae", "mean_pinball_3"]

    if not np.isfinite(scores[numeric_score].to_numpy(float)).all():
        raise ValueError("Nonfinite aggregate scores")
    if not np.isfinite(cases[numeric_case].to_numpy(float)).all():
        raise ValueError("Nonfinite case losses")
    if (cases[numeric_case] < 0).any().any():
        raise ValueError("Negative case losses")

    aggregated = (
        cases.groupby(keys)
        .agg(
            actual_cases=("case_id", "size"),
            actual_entities=("entity_id", "nunique"),
            actual_mae=("mae", "mean"),
            actual_pinball=("mean_pinball_3", "mean"),
        )
        .reset_index()
    )

    check = scores.merge(
        aggregated, on=keys, validate="one_to_one",
        how="outer", indicator=True,
    )

    if not check["_merge"].eq("both").all():
        raise ValueError("Score and case configurations do not match")

    if not (check["n_cases"] == check["actual_cases"]).all():
        raise ValueError("Case counts inconsistent")

    if not (check["n_entities"] == check["actual_entities"]).all():
        raise ValueError("Entity counts inconsistent")

    for a, b in [
        ("calibration_mean_case_mae", "actual_mae"),
        ("calibration_mean_case_pinball_3", "actual_pinball"),
    ]:
        if not np.allclose(check[a], check[b], rtol=1e-6, atol=1e-5):
            raise ValueError(f"Aggregate mismatch: {a}")

    for task, group in cases.groupby("task_id"):
        signatures = []
        for _, config in group.groupby(["model", "context_steps"]):
            signatures.append(
                frozenset(zip(
                    config["case_id"],
                    config["entity_id"],
                    config["lag_before_first_official"],
                ))
            )
            lag_counts = (
                config.groupby("entity_id")["lag_before_first_official"]
                .nunique()
            )
            if not lag_counts.eq(3).all():
                raise ValueError(f"Incomplete historical origins: {task}")

        if len(set(signatures)) != 1:
            raise ValueError(f"Unmatched historical cases: {task}")

    return scores, cases


def select_configuration(curves, strategy):
    if strategy == "full":
        candidates = curves

    elif strategy in ("longest", "middle", "shortest"):
        rows = []
        for _, group in curves.groupby("model"):
            group = group.sort_values("context_steps")
            if strategy == "longest":
                row = group.iloc[-1]
            elif strategy == "shortest":
                row = group.iloc[0]
            else:
                row = group.iloc[len(group) // 2]
            rows.append(row)

        candidates = pd.DataFrame(rows)

    else:
        raise ValueError(strategy)

    return candidates.sort_values(
        ["older_loss", "model", "context_steps"],
        kind="stable",
    ).iloc[0]


def analyse_task(task, scores, cases, metric, n_boot, seed):
    c = cases[cases.task_id == task].copy()
    s = scores[scores.task_id == task].copy()

    curves = (
        c.groupby(
            ["model", "context_steps", "lag_before_first_official"]
        )[metric]
        .mean()
        .unstack("lag_before_first_official")
        .reset_index()
        .rename(columns={1: "lag1", 2: "lag2", 3: "lag3"})
    )

    curves["older_loss"] = (curves["lag3"] + curves["lag2"]) / 2
    curves["all_three"] = (
        curves["lag3"] + curves["lag2"] + curves["lag1"]
    ) / 3

    curves = curves.merge(
        s[["model", "context_steps", METRICS[metric]]],
        on=["model", "context_steps"],
        validate="one_to_one",
    )

    if not np.allclose(
        curves["all_three"], curves[METRICS[metric]],
        rtol=1e-6, atol=1e-5,
    ):
        raise ValueError("Temporal aggregate mismatch")

    oracle = curves.sort_values(
        ["lag1", "model", "context_steps"]
    ).iloc[0]

    oracle_loss = float(oracle["lag1"])
    if oracle_loss <= 0:
        raise ValueError("Oracle loss must be positive")

    n_models = curves["model"].nunique()
    n_entities = c["entity_id"].nunique()

    policies = []

    for strategy in ("full", "longest", "middle", "shortest"):
        chosen = select_configuration(curves, strategy)

        n_probed = len(curves) if strategy == "full" else n_models

        policies.append({
            "task_id": task,
            "metric": metric,
            "policy": strategy,
            "selected_model": chosen["model"],
            "selected_context": int(chosen["context_steps"]),
            "older_loss": float(chosen["older_loss"]),
            "later_loss": float(chosen["lag1"]),
            "regret_pct": 100 * (
                float(chosen["lag1"]) / oracle_loss - 1
            ),
            "configurations_probed": n_probed,
            "selection_case_forecasts": n_probed * 2 * n_entities,
            "oracle_model": oracle["model"],
            "oracle_context": int(oracle["context_steps"]),
            "oracle_loss": oracle_loss,
        })

    for model, group in curves.groupby("model"):
        chosen = group.sort_values("context_steps").iloc[-1]

        policies.append({
            "task_id": task,
            "metric": metric,
            "policy": f"fixed_{model}_longest",
            "selected_model": model,
            "selected_context": int(chosen["context_steps"]),
            "older_loss": float(chosen["older_loss"]),
            "later_loss": float(chosen["lag1"]),
            "regret_pct": 100 * (
                float(chosen["lag1"]) / oracle_loss - 1
            ),
            "configurations_probed": 0,
            "selection_case_forecasts": 0,
            "oracle_model": oracle["model"],
            "oracle_context": int(oracle["context_steps"]),
            "oracle_loss": oracle_loss,
        })

    policy_df = pd.DataFrame(policies)

    decisions = {
        p["policy"]: (p["selected_model"], p["selected_context"])
        for p in policies
    }

    recent = c[c["lag_before_first_official"] == 1]
    entities = sorted(recent["entity_id"].unique())

    lookup = {}
    for (model, context), group in recent.groupby(
        ["model", "context_steps"]
    ):
        lookup[(model, int(context))] = (
            group.set_index("entity_id")[metric]
            .reindex(entities)
            .to_numpy(float)
        )

    full_loss = lookup[decisions["full"]]
    longest_loss = lookup[decisions["longest"]]

    differences = full_loss - longest_loss

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(entities), size=(n_boot, len(entities))
    )

    bootstrap_means = differences[sampled_indices].mean(axis=1)

    paired = {
        "task_id": task,
        "metric": metric,
        "n_entities": len(entities),
        "full_minus_longest": float(differences.mean()),
        "ci025": float(np.quantile(bootstrap_means, 0.025)),
        "ci975": float(np.quantile(bootstrap_means, 0.975)),
        "fraction_full_better": float(
            (bootstrap_means < 0).mean()
        ),
    }

    curves.insert(0, "metric", metric)
    curves.insert(0, "task_id", task)

    return policy_df, curves, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("results/phase2/cross_task_matrix"),
    )
    parser.add_argument("--catalog-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error("--bootstrap must be >= 100")

    catalog_dir = args.catalog_dir or find_catalog(args.catalog_root)

    scores, cases = load_catalog(catalog_dir)

    timestamp = dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    output = (
        Path("results/phase2/cross_task_analysis") / timestamp
    )
    output.mkdir(parents=True, exist_ok=False)

    all_policies = []
    all_curves = []
    all_paired = []

    print("=" * 90)
    print("PERTSFM PHASE 2.14 — CROSS-TASK HISTORICAL AUDIT")
    print("NO OFFICIAL FEV EVALUATION TARGETS ACCESSED")
    print("=" * 90)
    print(f"Catalog: {catalog_dir}")
    print(f"Tasks: {scores.task_id.nunique()}")
    print(f"Models: {scores.model.nunique()}")
    print(f"Configurations: {len(scores)}")
    print(f"Case-level results: {len(cases):,}")

    for task in sorted(cases.task_id.unique()):
        task_scores = scores[scores.task_id == task]
        name = task_scores["dataset_config"].iloc[0]

        print("\n" + "=" * 90)
        print(f"DATASET: {name}")
        print(f"Configurations: {len(task_scores)}")
        print(f"Entities: {cases[cases.task_id == task].entity_id.nunique()}")
        print("=" * 90)

        for metric in METRICS:
            policy, curves, paired = analyse_task(
                task, scores, cases, metric,
                args.bootstrap, args.seed,
            )

            all_policies.append(policy)
            all_curves.append(curves)
            all_paired.append(paired)

            print(f"\nMETRIC: {metric}")
            print("\nTOP 10 CONFIGURATIONS — ALL HISTORICAL ORIGINS")
            print(
                curves.nsmallest(10, "all_three")[
                    ["model", "context_steps", "all_three", "lag1"]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )

            print("\nTEMPORAL SELECTION — OLDER LAGS -> LAG 1")
            print(
                policy[policy.policy.isin(
                    ["full", "longest", "middle", "shortest"]
                )][
                    [
                        "policy", "selected_model",
                        "selected_context", "later_loss",
                        "regret_pct", "configurations_probed",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.5f}",
                )
            )

            print(
                "\nPAIRED FULL vs LONGEST:",
                f"delta={paired['full_minus_longest']:+.5f}",
                f"95% CI=[{paired['ci025']:+.5f},",
                f"{paired['ci975']:+.5f}]",
            )

    policies = pd.concat(all_policies, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)
    paired = pd.DataFrame(all_paired)

    policies.to_csv(
        output / "temporal_policy_comparison.csv", index=False
    )
    curves.to_csv(
        output / "response_curves_by_lag.csv", index=False
    )
    paired.to_csv(
        output / "paired_bootstrap.csv", index=False
    )

    manifest = {
        "phase": "2.14",
        "catalog": str(catalog_dir),
        "n_tasks": int(scores.task_id.nunique()),
        "n_models": int(scores.model.nunique()),
        "n_configs": int(len(scores)),
        "n_case_rows": int(len(cases)),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "official_test_labels_accessed": False,
        "note": (
            "Within-task historical replay only. "
            "Not official FEV evaluation or "
            "held-out source-family generalisation."
        ),
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("\n" + "=" * 90)
    print("MACRO-AVERAGED LATER REGRET ACROSS DATASETS")
    print("=" * 90)

    summary = (
        policies.groupby(["metric", "policy"])["regret_pct"]
        .agg(["mean", "max", "count"])
        .reset_index()
    )

    print(
        summary[summary.policy.isin(
            ["full", "longest", "middle", "shortest"]
        )].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print(f"\nPASS: Results saved to {output}")
    print("Historical calibration only — not official FEV results.")


if __name__ == "__main__":
    main()
