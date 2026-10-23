import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Mean results across the 18 dataset × model settings
# ------------------------------------------------------------

budget = np.array([0, 10, 25, 50, 75, 100])

# Reduction in forecasting error relative to always-short context (%)
random_selection = np.array([
    0.00,
    1.84,
    4.61,
    9.22,
    13.83,
    18.44,
])

adaptive = np.array([
    0.00,
    8.12,
    11.83,
    15.17,
    17.17,
    18.44,
])

perfect_selection = np.array([
    0.00,
    14.65,
    20.07,
    23.18,
    23.69,
    18.44,
])

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.plot(
    budget,
    random_selection,
    marker="o",
    linewidth=2,
    markersize=7,
    label="Random selection",
)

ax.plot(
    budget,
    adaptive,
    marker="o",
    linewidth=2.7,
    markersize=7,
    label="Adaptive personalisation",
)

ax.plot(
    budget,
    perfect_selection,
    marker="o",
    linewidth=2,
    markersize=7,
    linestyle="--",
    label="Perfect selection",
)

# Highlight the important 25% operating point
x = 25

ax.scatter(
    [x, x, x],
    [
        random_selection[2],
        adaptive[2],
        perfect_selection[2],
    ],
    s=70,
    zorder=5,
)

ax.annotate(
    "4.61%",
    (25, 4.61),
    xytext=(8, -18),
    textcoords="offset points",
    fontsize=9,
)

ax.annotate(
    "11.83%",
    (25, 11.83),
    xytext=(8, 7),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
)

ax.annotate(
    "20.07%",
    (25, 20.07),
    xytext=(8, 7),
    textcoords="offset points",
    fontsize=9,
)

ax.axvline(
    25,
    linestyle=":",
    linewidth=1,
    alpha=0.6,
)

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_title(
    "Adaptive use of personal history improves forecasting",
    fontsize=15,
    fontweight="bold",
    pad=14,
)

ax.set_xlabel(
    "Predictions allowed to use longer personal history (%)",
    fontsize=11,
)

ax.set_ylabel(
    "Reduction in forecasting error vs short context (%)",
    fontsize=11,
)

ax.set_xticks(budget)
ax.set_ylim(0, 26)

ax.grid(
    axis="y",
    alpha=0.2,
)

ax.legend(
    frameon=False,
    fontsize=10,
)

# Simple takeaway
ax.text(
    0.5,
    -0.20,
    "At a 25% history budget: adaptive selection gives 11.83% error reduction "
    "vs 4.61% for random selection (~2.6× larger).",
    transform=ax.transAxes,
    ha="center",
    fontsize=9.5,
)

plt.tight_layout()

plt.savefig(
    "adaptive_personalisation_budget.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "adaptive_personalisation_budget.pdf",
    bbox_inches="tight",
)

print("Saved:")
print("  adaptive_personalisation_budget.png")
print("  adaptive_personalisation_budget.pdf")

print("\n25% budget:")
print(f"Random   : {random_selection[2]:.2f}%")
print(f"Adaptive : {adaptive[2]:.2f}%")
print(f"Perfect  : {perfect_selection[2]:.2f}%")
print(
    f"Adaptive / random improvement = "
    f"{adaptive[2] / random_selection[2]:.2f}x"
)
