import numpy as np
import matplotlib.pyplot as plt

# Rows = datasets
datasets = [
    "PEMS-BAY",
    "Electricity",
    "HUPA-UCM",
    "VitalDB",
    "Solar",
    "PhysioNet 2019",
]

# Columns = model pairs
pairs = [
    "Chronos-2 vs TimesFM3",
    "Chronos-2 vs TiRex-2",
    "TimesFM3 vs TiRex-2",
]

# Adjusted within-entity correlations
rho = np.array([
    [0.550, 0.515, 0.669],  # PEMS-BAY
    [0.544, 0.523, 0.641],  # Electricity
    [0.478, 0.273, 0.427],  # HUPA-UCM
    [0.558, 0.594, 0.744],  # VitalDB
    [0.253, 0.244, 0.497],  # Solar
    [0.657, 0.612, 0.750],  # PhysioNet 2019
])

fig, ax = plt.subplots(figsize=(9, 5.5))

im = ax.imshow(
    rho,
    vmin=0,
    vmax=0.8,
    aspect="auto",
)

# Labels
ax.set_xticks(np.arange(len(pairs)))
ax.set_xticklabels(pairs, fontsize=10)

ax.set_yticks(np.arange(len(datasets)))
ax.set_yticklabels(datasets, fontsize=11)

# Numbers inside cells
for i in range(rho.shape[0]):
    for j in range(rho.shape[1]):
        ax.text(
            j,
            i,
            f"{rho[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

# Colour bar
cbar = fig.colorbar(im, ax=ax, pad=0.03)
cbar.set_label(
    "Adjusted Spearman correlation (ρ)",
    fontsize=11,
)

ax.set_title(
    "Different TSFMs agree on when personal history is useful",
    fontsize=14,
    fontweight="bold",
    pad=16,
)

ax.text(
    0.5,
    -0.18,
    "18/18 model-pair comparisons were positive and statistically significant",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
)

plt.tight_layout()

plt.savefig(
    "state_dependent_context_agreement.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "state_dependent_context_agreement.pdf",
    bbox_inches="tight",
)

print("Saved:")
print("  state_dependent_context_agreement.png")
print("  state_dependent_context_agreement.pdf")

print(f"\nMean rho   = {rho.mean():.3f}")
print(f"Median rho = {np.median(rho):.3f}")
print(f"Min rho    = {rho.min():.3f}")
print(f"Max rho    = {rho.max():.3f}")
