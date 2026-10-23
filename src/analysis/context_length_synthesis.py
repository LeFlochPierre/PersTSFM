from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.registry import get_dataset_spec


DATASETS = [
    "electricity",
    "pems_bay",
    "hupa_ucm",
    "vitaldb",
    "solar",
    "physionet2019",
]

MODELS = {
    "chronos2": "Chronos-2",
    "timesfm3": "TimesFM 3",
    "tirex2": "TiRex-2",
}

OUT = Path("results/context_length")
FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def context_label(hours):
    hours = float(hours)

    if hours < 24:
        return f"{hours:g}h"

    if hours % 24 == 0:
        return f"{hours / 24:g}d"

    return f"{hours:g}h"


# ---------------------------------------------------------------------
# Read all 18 settings
# ---------------------------------------------------------------------

all_rows = []
curve_rows = []

for dataset in DATASETS:
    spec = get_dataset_spec(dataset)

    for model_key, model_label in MODELS.items():

        summary_path = Path(
            f"results/{dataset}/context/"
            f"{model_key}/summary.csv"
        )

        case_path = Path(
            f"results/{dataset}/context/"
            f"{model_key}/case_metrics.csv"
        )

        if not summary_path.exists():
            print(f"MISSING: {summary_path}")
            continue

        df = pd.read_csv(summary_path)
        df = df.sort_values("context_hours").reset_index(drop=True)

        # -------------------------------------------------------------
        # Reference / best / longest
        # -------------------------------------------------------------

        ref_matches = df[
            np.isclose(
                df["context_hours"].astype(float),
                float(spec.reference_context_hours),
            )
        ]

        if len(ref_matches) != 1:
            raise RuntimeError(
                f"{dataset} | {model_label}: "
                "reference context not uniquely found"
            )

        ref = ref_matches.iloc[0]
        best = df.loc[df["mean_entity_mae"].idxmin()]
        longest = df.loc[df["context_hours"].idxmax()]

        reference_mae = float(ref["mean_entity_mae"])
        best_mae = float(best["mean_entity_mae"])
        max_mae = float(longest["mean_entity_mae"])

        best_context = float(best["context_hours"])
        max_context = float(longest["context_hours"])

        improvement_pct = (
            100
            * (reference_mae - best_mae)
            / reference_mae
        )

        degradation_after_best_pct = (
            100
            * (max_mae - best_mae)
            / best_mae
        )

        right_censored = np.isclose(
            best_context,
            max_context,
        )

        # -------------------------------------------------------------
        # Optional case-level MSE
        # -------------------------------------------------------------

        mse_by_context = {}

        if case_path.exists():
            cases = pd.read_csv(case_path)

            if "rmse" in cases.columns:
                cases["mse"] = cases["rmse"] ** 2

                mse_by_context = (
                    cases
                    .groupby("context_hours")["mse"]
                    .mean()
                    .to_dict()
                )

        # -------------------------------------------------------------
        # Store context curve
        # -------------------------------------------------------------

        for _, row in df.iterrows():
            h = float(row["context_hours"])
            mae = float(row["mean_entity_mae"])

            curve_rows.append({
                "dataset": dataset,
                "model": model_key,
                "model_label": model_label,
                "context_hours": h,
                "context_label": context_label(h),
                "mean_entity_mae": mae,
                "mae_relative_to_reference": (
                    mae / reference_mae
                ),
                "improvement_vs_reference_pct": (
                    100
                    * (reference_mae - mae)
                    / reference_mae
                ),
                "mean_case_mse": (
                    mse_by_context.get(h, np.nan)
                ),
            })

        # -------------------------------------------------------------
        # Store summary
        # -------------------------------------------------------------

        all_rows.append({
            "dataset": dataset,
            "model": model_key,
            "model_label": model_label,

            "reference_context_hours":
                float(spec.reference_context_hours),

            "reference_mae": reference_mae,

            "best_context_hours": best_context,
            "best_mae": best_mae,

            "max_context_hours": max_context,
            "max_context_mae": max_mae,

            "improvement_pct":
                improvement_pct,

            "degradation_after_best_pct":
                degradation_after_best_pct,

            "right_censored":
                right_censored,
        })


summary = pd.DataFrame(all_rows)
curves = pd.DataFrame(curve_rows)

summary.to_csv(
    OUT / "context_regime_summary_18.csv",
    index=False,
)

curves.to_csv(
    OUT / "context_curves_18.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("FULL CONTEXT-LENGTH SYNTHESIS")
print("=" * 120)

display_cols = [
    "dataset",
    "model_label",
    "reference_context_hours",
    "best_context_hours",
    "max_context_hours",
    "reference_mae",
    "best_mae",
    "improvement_pct",
    "degradation_after_best_pct",
    "right_censored",
]

print(
    summary[display_cols]
    .sort_values(["dataset", "model_label"])
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("=" * 120)
print("KEY COUNTS")
print("=" * 120)

n = len(summary)
n_not_max = int((~summary["right_censored"]).sum())
n_max = int(summary["right_censored"].sum())

print(f"Settings available:               {n}/18")
print(
    "Best context shorter than max:    "
    f"{n_not_max}/{n}"
)
print(
    "Best context = maximum tested:    "
    f"{n_max}/{n}"
)

if n:
    print(
        "Mean best-vs-reference gain:     "
        f"{summary['improvement_pct'].mean():.2f}%"
    )

    non_censored = summary[
        ~summary["right_censored"]
    ]

    if len(non_censored):
        print(
            "Mean deterioration after peak: "
            f"{non_censored['degradation_after_best_pct'].mean():.2f}%"
        )


# ---------------------------------------------------------------------
# Figures: raw MAE
# ---------------------------------------------------------------------

for dataset in DATASETS:

    d = curves[curves["dataset"] == dataset]

    if d.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_key, model_label in MODELS.items():
        m = d[d["model"] == model_key].sort_values(
            "context_hours"
        )

        if m.empty:
            continue

        ax.plot(
            m["context_hours"],
            m["mean_entity_mae"],
            marker="o",
            label=model_label,
        )

        best = m.loc[
            m["mean_entity_mae"].idxmin()
        ]

        ax.scatter(
            [best["context_hours"]],
            [best["mean_entity_mae"]],
            s=90,
            zorder=5,
        )

    ax.set_xscale("log", base=2)

    ax.set_xlabel("Historical context length (hours)")
    ax.set_ylabel("Mean entity MAE")
    ax.set_title(
        f"{dataset}: forecasting error vs context length"
    )

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIG / f"{dataset}_context_mae.png",
        dpi=220,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Figures: normalized MAE improvement
# ---------------------------------------------------------------------

for dataset in DATASETS:

    d = curves[curves["dataset"] == dataset]

    if d.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_key, model_label in MODELS.items():
        m = d[d["model"] == model_key].sort_values(
            "context_hours"
        )

        if m.empty:
            continue

        ax.plot(
            m["context_hours"],
            m["improvement_vs_reference_pct"],
            marker="o",
            label=model_label,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xscale("log", base=2)

    ax.set_xlabel("Historical context length (hours)")
    ax.set_ylabel(
        "MAE reduction vs reference context (%)"
    )

    ax.set_title(
        f"{dataset}: benefit of additional context"
    )

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIG / f"{dataset}_context_normalized.png",
        dpi=220,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Figures: MSE if available
# ---------------------------------------------------------------------

for dataset in DATASETS:

    d = curves[
        (curves["dataset"] == dataset)
        & curves["mean_case_mse"].notna()
    ]

    if d.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_key, model_label in MODELS.items():
        m = d[d["model"] == model_key].sort_values(
            "context_hours"
        )

        if m.empty:
            continue

        ax.plot(
            m["context_hours"],
            m["mean_case_mse"],
            marker="o",
            label=model_label,
        )

    ax.set_xscale("log", base=2)

    ax.set_xlabel("Historical context length (hours)")
    ax.set_ylabel("Mean case MSE")
    ax.set_title(
        f"{dataset}: MSE vs context length"
    )

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIG / f"{dataset}_context_mse.png",
        dpi=220,
    )

    plt.close(fig)


print()
print("=" * 120)
print("SAVED")
print("=" * 120)

print(OUT / "context_regime_summary_18.csv")
print(OUT / "context_curves_18.csv")
print(FIG)
