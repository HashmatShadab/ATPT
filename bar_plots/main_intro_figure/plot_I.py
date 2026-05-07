import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Consistent, publication-friendly styling
# (bumped font sizes for readability in papers/slides)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 32,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.linewidth": 0.9,
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
    # "RN",
    # "RN (Ours)",
    "TTC",
    "TTC (Ours)",
    "AOM",
    "AOM (Ours)"
]
#
# clean_scores = [59.80, 71.90,  50.10, 70.50, 56.9, 71.90]
# adv_scores   = [54.10, 53.10,  62, 61.80, 67.2, 63.3]
clean_scores = [61.7, 73.2, 58.6, 72.9]
adv_scores   = [59.6, 58.8, 60.3, 60.8]



# ---------------------------
# Plot Setup
# ---------------------------

x = np.arange(len(methods))
width = 0.36

fig, ax = plt.subplots(figsize=(11.5, 5.2))

# Colorblind-friendly palette (Okabe–Ito)
clean_color = "#0072B2"   # blue
adv_color = "#D55E00"     # vermillion

# Patterns help in grayscale print / for accessibility
clean_hatch = "///"
adv_hatch = "\\\\"

bar_edgecolor = "#1a1a1a"
bar_linewidth = 0.9

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
    label="PGD-100 (ε=8/255)",
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
# ax.set_title(
#     "Clean vs. PGD-100 (ε=4/255) Performance",
#     fontweight="semibold",
#     pad=10,
# )
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=0)
ax.set_ylim(45, 80)
ax.margins(x=0.02)

# Subtle grid + cleaner frame
ax.grid(axis="y", linestyle="-", alpha=0.25)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Improve readability of y-axis
ax.tick_params(axis="y", which="major", length=3.5, width=0.8)
ax.tick_params(axis="x", which="major", pad=6)

leg = ax.legend(
    frameon=True,
    ncols=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    borderaxespad=0.2,
    handlelength=1.6,
    columnspacing=1.4,
)
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
            fontsize=21,
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

fig.tight_layout(pad=1.0)

# Save figure
project_root = Path(__file__).resolve().parents[3]
fig.savefig("main_intro_figure_eps_8.png", bbox_inches="tight")
# plt.show()