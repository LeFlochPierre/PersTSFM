from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FIND THE SAVED AUROC RESULTS
# ============================================================

root = Path("results")

dataset_names = ["dataset", "domain", "dataset_name"]
model_names = ["model", "model_label", "tsfm", "foundation_model"]

candidates = []

for path in root.rglob("*.csv"):
    try:
        df = pd.read_csv(path)
    except Exception:
        continue

    # Find dataset/domain and model columns
    dataset_col = next(
        (c for c in df.columns if c.lower() in dataset_names),
        None
    )
    model_col = next(
        (c for c in df.columns if c.lower() in model_names),
        None
    )

    # Find AUROC/AUC-like columns
    auc_cols = [
        c for c in df.columns
        if "auroc" in c.lower()
        or "roc_auc" in c.lower()
        or c.lower().endswith("_auc")
    ]

    if dataset_col is None or model_col is None or not auc_cols:
        continue

    for auc_col in auc_cols:
        tmp = df[[dataset_col, model_col, auc_col]].copy()

        tmp[auc_col] = pd.to_numeric(
            tmp[auc_col],
            errors="coerce"
        )
        tmp = tmp.dropna()

        # Collapse duplicates only for identifying candidate files.
        grouped = (
            tmp.groupby([dataset_col, model_col], as_index=False)[auc_col]
            .mean()
        )

        n = len(grouped)

        if n == 0:
            continue

        mean_auc = grouped[auc_col].mean()
        above = int((grouped[auc_col] > 0.50).sum())

        # Score against the known Router-1 summary:
        # 18 settings, mean ~0.552, 16/18 above chance
        score = (
            abs(n - 18) * 2
            + abs(mean_auc - 0.552) * 20
            + abs(above - 16) * 0.5
        )

        candidates.append(
            {
                "score": score,
                "path": path,
                "dataset_col": dataset_col,
                "model_col": model_col,
                "auc_col": auc_col,
                "data": grouped,
                "n": n,
                "mean": mean_auc,
                "above": above,
            }
        )

if not candidates:
    raise RuntimeError(
        "Could not find a CSV containing dataset/model/AUROC columns."
    )

candidates.sort(key=lambda x: x["score"])

print("\n=== AUROC CANDIDATES ===")
for c in candidates[:10]:
    print(
        f"{c['path']} | "
        f"column={c['auc_col']} | "
        f"n={c['n']} | "
        f"mean={c['mean']:.3f} | "
        f">0.5={c['above']}/{c['n']}"
    )

best = candidates[0]

print("\n=== USING ===")
print("File:       ", best["path"])
print("AUROC col:  ", best["auc_col"])
print("Dataset col:", best["dataset_col"])
print("Model col:  ", best["model_col"])

df = best["data"].copy()

dataset_col = best["dataset_col"]
model_col = best["model_col"]
auc_col = best["auc_col"]

# ============================================================
# SANITY CHECK
# ============================================================

mean_auc = df[auc_col].mean()
median_auc = df[auc_col].median()
n_above = int((df[auc_col] > 0.50).sum())

print("\n=== SUMMARY ===")
print(f"Settings       : {len(df)}")
print(f"Mean AUROC     : {mean_auc:.3f}")
print(f"Median AUROC   : {median_auc:.3f}")
print(f"Above chance   : {n_above}/{len(df)}")
print(f"Minimum AUROC  : {df[auc_col].min():.3f}")
print(f"Maximum AUROC  : {df[auc_col].max():.3f}")

print("\n=== VALUES ===")
print(
    df.sort_values([dataset_col, model_col])
      .to_string(index=False)
)

# Warn if this does not look like the expected result
if not (
    len(df) == 18
    and abs(mean_auc - 0.552) < 0.03
    and n_above >= 15
):
    print(
        "\nWARNING: Automatically selected results do not perfectly "
        "match the expected Router-1 summary."
    )
    print(
        "Check the candidate list printed above before using the figure."
    )

# ============================================================
# CLEAN LABELS
# ============================================================

dataset_map = {
    "pems": "PEMS-BAY",
    "pems-bay": "PEMS-BAY",
    "pems_bay": "PEMS-BAY",
    "electricity": "Electricity",
    "hupa": "HUPA-UCM",
    "hupa-ucm": "HUPA-UCM",
    "hupa_ucm": "HUPA-UCM",
    "vital": "VitalDB",
    "vitaldb": "VitalDB",
    "solar": "Solar",
    "physio": "PhysioNet 2019",
    "physionet": "PhysioNet 2019",
    "physionet2019": "PhysioNet 2019",
}

model_map = {
    "chronos": "Chronos-2",
    "chronos-2": "Chronos-2",
    "chronos2": "Chronos-2",
    "timesfm": "TimesFM3",
    "timesfm3": "TimesFM3",
    "timesfm-3": "TimesFM3",
    "tirex": "TiRex-2",
    "tirex2": "TiRex-2",
    "tirex-2": "TiRex-2",
}

def clean_dataset(x):
    s = str(x)
    return dataset_map.get(s.lower(), s)

def clean_model(x):
    s = str(x)
    return model_map.get(s.lower(), s)

df["dataset_clean"] = df[dataset_col].map(clean_dataset)
df["model_clean"] = df[model_col].map(clean_model)

# Sort by AUROC so the figure is immediately readable
df = df.sort_values(auc_col).reset_index(drop=True)

df["label"] = (
    df["dataset_clean"]
    + "  |  "
    + df["model_clean"]
)

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7.5))

y = np.arange(len(df))

# Thin lines from chance to observed AUROC
for i, auc in enumerate(df[auc_col]):
    ax.plot(
        [0.50, auc],
        [i, i],
        linewidth=1.2,
        alpha=0.45,
    )

# Actual AUROC values
ax.scatter(
    df[auc_col],
    y,
    s=65,
    zorder=3,
)

# Chance
ax.axvline(
    0.50,
    linestyle="--",
    linewidth=1.5,
    label="Chance (0.50)",
)

# Mean
ax.axvline(
    mean_auc,
    linestyle=":",
    linewidth=1.8,
    label=f"Mean AUROC = {mean_auc:.2f}",
)

ax.set_yticks(y)
ax.set_yticklabels(df["label"], fontsize=9.5)

ax.set_xlabel(
    "AUROC",
    fontsize=11,
)

ax.set_title(
    "Can we predict when additional personal history will help?",
    fontsize=14,
    fontweight="bold",
    pad=14,
)

# Keep scale focused around the actual result
xmin = min(0.44, df[auc_col].min() - 0.025)
xmax = max(0.66, df[auc_col].max() + 0.025)

ax.set_xlim(xmin, xmax)

ax.grid(
    axis="x",
    alpha=0.18,
)

ax.legend(
    frameon=False,
    loc="lower right",
)

# Main takeaway
ax.text(
    0.98,
    0.98,
    f"{n_above}/{len(df)} settings above chance",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    fontweight="bold",
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

png = out_dir / "adaptive_personalisation_auroc.png"
pdf = out_dir / "adaptive_personalisation_auroc.pdf"

plt.savefig(
    png,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf,
    bbox_inches="tight",
)

print("\n=== SAVED ===")
print(png)
print(pdf)

plt.show()
