#!/usr/bin/env python3
"""Phase 3.3B — pre-official history/data-length audit; no ground truth API calls.

Uses the v0.10.0 FEV benchmark registry. Every feature is calculated using
observations ending BEFORE historical validation lag2. Does not train models.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BENCHMARK_URL = (
    "https://raw.githubusercontent.com/autogluon/fev/refs/tags/v0.10.0/"
    "benchmarks/fev_bench/tasks.yaml"
)
CONTEXTS = [96, 384, 768, 1536, 3072, 4096, 6144, 8192]


def corr_lag(x: np.ndarray, lag: int) -> float:
    if lag < 1 or len(x) <= lag + 2:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) < 1e-10 or np.std(b[ok]) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def characterize(values, *, horizon, seasonality, entity):
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size <= 2 * horizon:
        return None
    # Absolute historical indices. Never use lag2/lag1 targets as training features.
    train = x[:-2 * horizon]
    val_context = x[:-2 * horizon]
    later_context = x[:-horizon]
    finite = train[np.isfinite(train)]
    valid = finite.size > 0
    rec = {
        "entity_id": str(entity),
        "full_preofficial_points": int(x.size),
        "training_pool_points_before_lag2": int(train.size),
        "training_pool_finite_points": int(finite.size),
        "training_pool_missing_fraction": float(1 - finite.size / train.size),
        "train_mean": float(np.mean(finite)) if valid else float("nan"),
        "train_std": float(np.std(finite)) if valid else float("nan"),
        "train_zero_fraction": float(np.mean(finite == 0)) if valid else float("nan"),
        "train_lag1_corr": corr_lag(train, 1),
        "train_season_corr": corr_lag(train, seasonality),
    }
    # A descriptive recent-vs-older shift, from training-only observations.
    block = 4 * horizon
    if len(train) >= 2 * block:
        recent, previous = train[-block:], train[-2*block:-block]
        if np.isfinite(recent).all() and np.isfinite(previous).all():
            recent_mean, previous_mean = np.mean(recent), np.mean(previous)
            scale = max(float(np.std(previous)), 1e-8)
            rec["train_recent_mean_shift_in_sd"] = float((recent_mean - previous_mean) / scale)
        else:
            rec["train_recent_mean_shift_in_sd"] = float("nan")
    else:
        rec["train_recent_mean_shift_in_sd"] = float("nan")
    for c in CONTEXTS:
        # Context feasibility is assessed on the actual last c historical values,
        # not on the whole series (earlier NAs may be outside model context).
        rec[f"lag2_valid_c{c}"] = bool(len(val_context) >= c and np.isfinite(val_context[-c:]).all())
        # Diagnostic only: lag1 has become available by the later origin.
        # Never use these lag1 flags as features for adaptation-budget selection.
        rec[f"lag1_valid_c{c}"] = bool(len(later_context) >= c and np.isfinite(later_context[-c:]).all())
    return rec


def run(names, output, limit_entities):
    import fev
    print("=" * 100)
    print("PERTSFM 3.3B | FULL PRE-OFFICIAL HISTORY AUDIT | NO TRAINING, NO TEST LABELS")
    print("FEV registry: v0.10.0; lag2 features from prefix ending before lag2 targets")
    print("=" * 100)
    benchmark = fev.Benchmark.from_yaml(BENCHMARK_URL)
    tasks = {t.dataset_config: t for t in benchmark.tasks}
    all_rows = []
    for name in names:
        if name not in tasks:
            raise ValueError(f"Unknown dataset_config: {name}; available: {sorted(tasks)}")
        task = tasks[name]
        # This accessor supplies input histories only; it DOES NOT open official test targets.
        past, _future_known_only = task.get_window(0).get_input_data()
        rows = []
        target = task.target
        if not isinstance(target, str):
            raise TypeError(f"{name}: expected a univariate target name, got {target!r}")
        for i, item in enumerate(past):
            if limit_entities and i >= limit_entities:
                break
            rec = characterize(item[target], horizon=int(task.horizon),
                               seasonality=int(task.seasonality), entity=item[task.id_column])
            if rec is None:
                continue
            rec["dataset_config"] = name
            rec["horizon"] = int(task.horizon)
            rec["seasonality"] = int(task.seasonality)
            rows.append(rec)
        if not rows:
            print(f"{name}: no entities with >2 horizons of history; SKIPPING")
            continue
        df = pd.DataFrame(rows)
        all_rows.extend(rows)
        lengths = df["training_pool_points_before_lag2"]
        print(f"\n{name} | entities={len(df)} | H={task.horizon} | seasonal_period={task.seasonality}")
        print(f"  PRE-LAG2 TRAINING POOL lengths: min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}")
        print(f"  ENTRIES WITH >3072 PRE-LAG2 TRAINING POINTS: {int((lengths > 3072).sum())}/{len(df)}")
        print(f"  TRAIN-ONLY FEATURES: mean lag1 acf={df.train_lag1_corr.mean():.3f}; "
              f"mean seasonal acf={df.train_season_corr.mean():.3f}; "
              f"mean missing fraction={df.training_pool_missing_fraction.mean():.5f}")
        print("  CONTEXT (lag2 valid / lag1 valid):")
        for c in CONTEXTS:
            print(f"    c={c:5d}: {int(df[f'lag2_valid_c{c}'].sum()):4d}/{len(df)}  / "
                  f"{int(df[f'lag1_valid_c{c}'].sum()):4d}/{len(df)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output, index=False)
    print(f"\nPASS: audit saved to {output}")
    print("CAUTION: Lag1 feasibility flags are diagnostics, not training or selection features.")
    print("CAUTION: Chronos-2 8192 is a model-context limit, not a cap on the LoRA training-data pool.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=["M_DENSE_1H"])
    p.add_argument("--limit-entities", type=int, default=0, help="0 = all entities")
    p.add_argument("--output", default="results/phase3/history_audit/preofficial_history_features.csv")
    args = p.parse_args()
    run(args.tasks, args.output, args.limit_entities)

if __name__ == "__main__":
    main()
