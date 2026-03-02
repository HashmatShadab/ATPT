import numpy as np
import matplotlib.pyplot as plt

# Consistent, publication-friendly styling
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})
plt.style.use("seaborn-v0_8-whitegrid")

# Make hatch patterns render crisply in vector/print outputs
plt.rcParams.update({
    "hatch.linewidth": 1.0,
})

# ---------------------------
# Data
# ---------------------------

methods = [
    "RN",
    "RN (Ours)",
    "R-TPT",
    "TTC",
    "TTC (Ours)",
    "AOM",
    "AOM (Ours)"
]

clean_scores = [65.40, 73.70, 75.74, 61.70, 73.20, 62.10, 73.00]
adv_scores   = [60.50, 59.50, 61.92, 69.70, 69.60, 74.70, 73.40]

# ---------------------------
# Plot Setup
# ---------------------------

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5.6))

# Colorblind-friendly palette (Okabe–Ito)
clean_color = "#0072B2"   # blue
adv_color = "#D55E00"     # vermillion

# Patterns help in grayscale print / for accessibility
clean_hatch = "///"
adv_hatch = "\\\\"

bar_edgecolor = "#1a1a1a"
bar_linewidth = 0.8

bars_clean = ax.bar(
    x - width / 2,
    clean_scores,
    width,
    label="Clean",
    color=clean_color,
    edgecolor=bar_edgecolor,
    linewidth=bar_linewidth,
    hatch=clean_hatch,
    alpha=0.95,
)
bars_adv = ax.bar(
    x + width / 2,
    adv_scores,
    width,
    label="Adversarial",
    color=adv_color,
    edgecolor=bar_edgecolor,
    linewidth=bar_linewidth,
    hatch=adv_hatch,
    alpha=0.95,
)

# ---------------------------
# Formatting
# ---------------------------

ax.set_ylabel("Accuracy (%)", fontweight="semibold")
ax.set_title(
    "Clean vs. Adversarial Performance Across Methods",
    fontweight="semibold",
    pad=10,
)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=0)
ax.set_ylim(55, 80)
ax.margins(x=0.02)

# Subtle grid + cleaner frame
ax.grid(axis="y", linestyle="-", alpha=0.25)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(frameon=True, ncols=2, loc="upper left")
leg.get_frame().set_alpha(0.95)
leg.get_frame().set_linewidth(0.0)

# Add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
        )

add_labels(bars_clean)
add_labels(bars_adv)

# Lightly emphasize "(Ours)" methods for quick scanning
for tick in ax.get_xticklabels():
    if "(Ours)" in tick.get_text():
        tick.set_fontweight("semibold")

fig.tight_layout()
plt.show()