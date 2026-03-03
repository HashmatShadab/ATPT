"""Feature-drift visualization utilities.

This script loads precomputed image feature vectors for three conditions:

1) `clean`   : features for the original (non-attacked) images
2) `adv`     : features after an adversarial perturbation
3) `counter` : features after a countermeasure / recovery procedure applied to `adv`

It then computes drift vectors and a small set of summary metrics, and generates
several plots to help interpret how the attack moved features and whether the
countermeasure reverses/mitigates that movement.

Notes:
- All computations are done per-sample. Shapes are expected to be `[N, D]`.
- The script aligns array lengths defensively and then takes a shared random
  subsample for faster, clearer plots.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from cycler import cycler


# ---------------------------
# CONFIG
# ---------------------------

# Base directory that contains subfolders like `ADV_Generation_eps_.../image_features/...`.
BASE_DIR = "tsne/vit_l_14_datacomp_1b/Caltech101"

# Feature folders (relative to `BASE_DIR`).
CLEAN_FOLDER = "ADV_Generation_eps_0.0_steps_0"
ADV_FOLDER = "ADV_Generation_eps_4.0_steps_100"
# One or more countermeasure variants (relative to `BASE_DIR`).
#
# Example (multiple):
# COUNTER_FOLDERS = (
#     "..._Added_Noise_uniform_Eps_48.0_...",
#     "..._Added_Noise_uniform_Eps_24.0_...",
# )
COUNTER_FOLDERS = (
    "ADV_Generation_eps_4.0_steps_100_Added_Noise_uniform_Eps_48.0_Tau_Type_normal_num_anchors_1",
    # "ADV_Generation_eps_4.0_steps_100_Added_Noise_uniform_Eps_4.0_Tau_Type_normal_num_anchors_1",
    # "ADV_Generation_eps_4.0_steps_100_Added_Noise_uniform_Eps_8.0_Tau_Type_normal_num_anchors_1",
    # "ADV_Generation_eps_4.0_steps_100_Added_Noise_uniform_Eps_16.0_Tau_Type_normal_num_anchors_1",
    # "ADV_Generation_eps_4.0_steps_100_Added_Noise_gaussian_Sigma_0.06_Tau_Type_noisy_num_anchors_10",
    "ADV_Generation_eps_0.0_steps_0_Added_Noise_uniform_Eps_48.0_Tau_Type_normal_num_anchors_1",

)

# Feature filenames stored under `<folder>/image_features/`.
CLEAN_FILE = "image_features_clean.npy"
ADV_FILE = "image_features_adv.npy"
COUNTER_FILE = "image_features_adv_counter.npy"

# Metadata used to obtain labels. This script currently uses labels only for the optional
# per-class boxplot at the end.
META_PATH = os.path.join(
    BASE_DIR,
    "ADV_Generation_eps_4.0_steps_100_Added_Noise_gaussian_Sigma_0.005_Tau_Type_noisy_num_anchors_10",
    "image_features",
    "image_features_metadata.json",
)

RANDOM_SEED = 42

# Subsample size to keep plots readable and fast. Set to `None` to use all samples.
MAX_SAMPLES = 5000


# ---------------------------
# OUTPUT (saving plots)
# ---------------------------

# Set to `True` to write all figures to disk under `PLOTS_ROOT_DIR/<run_folder>/...`.
SAVE_PLOTS = True

# Set to `True` if you also want interactive windows (`plt.show()`).
# For non-interactive / headless runs, keep this `False`.
SHOW_PLOTS = True

# Root directory (relative to project root) for generated figures.
PLOTS_ROOT_DIR = "plots_output"


def _safe_name(s: str, *, max_len: int = 180) -> str:
    """Convert an arbitrary string into a filesystem-safe filename fragment."""

    s = (s or "").strip()
    s = re.sub(r"[^0-9a-zA-Z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.-")
    if not s:
        s = "plot"
    return s[:max_len]


def _make_plots_dir() -> str:
    """Create (if needed) and return the per-run plots directory."""

    # Keep the run folder informative but not excessively long.
    dataset = _safe_name(os.path.basename(BASE_DIR))
    adv = _safe_name(ADV_FOLDER)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = _safe_name(f"tsne_vis__{dataset}__{adv}__{ts}")
    out_dir = os.path.join(PLOTS_ROOT_DIR, run_folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _finalize_figure(fig: plt.Figure, *, save_path: str | None) -> None:
    """Save/show/close a Matplotlib figure based on global flags."""

    if save_path is not None and SAVE_PLOTS:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------
# PLOTTING STYLE
# ---------------------------

# A small, colorblind-friendly palette (Okabe–Ito inspired; explicit hex for consistency).
# These tend to print well and remain distinguishable under common color-vision deficiencies.
COLORS = {
    "attack": "#0072B2",  # blue
    "recovery": "#E69F00",  # orange
    "counter": "#009E73",  # green
    "neutral": "#666666",  # gray
}

# Default cycle for multi-series plots (hist overlays, etc.).
# Kept distinct from the semantic COLORS above (attack/counter/neutral).
COLOR_CYCLE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def _scatter_drift(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str | tuple[float, float, float, float],
    marker: str,
    n: int,
) -> None:
    """Consistent scatter styling for the drift PCA plot."""

    # Visual tuning for readability across different subsample sizes.
    # Slightly larger markers + a bit less transparency improves perceived dot quality.
    s = 22 if n <= 2500 else (16 if n <= 6000 else 12)
    a = 0.65 if n <= 2500 else (0.50 if n <= 6000 else 0.40)

    # White edge strokes look good for small-to-medium n, but can get noisy for large n.
    # Also avoid Matplotlib warnings for unfilled markers.
    use_edge = (marker in ("o", "s", "D", "^")) and n <= 9000
    edgecolors = "white" if use_edge else "none"
    linewidths = 0.35 if use_edge else 0.0

    ax.scatter(
        x,
        y,
        s=s,
        alpha=a,
        color=color,
        marker=marker,
        edgecolors=edgecolors,
        linewidths=linewidths,
        label=label,
        antialiased=True,
        # Rasterize only for very large clouds to keep vector outputs responsive.
        rasterized=n > 8000,
    )


def apply_plot_style() -> None:
    """Apply a consistent, publication-friendly Matplotlib style.

    This is intentionally self-contained (no seaborn dependency) and focuses on:
    - consistent font sizes
    - subtle grid
    - clean axes
    - higher DPI for sharper output
    """

    # Use a clean base style if available.
    # (Name changed in newer Matplotlib; fall back gracefully.)
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style_name in plt.style.available:
            plt.style.use(style_name)
            break

    mpl.rcParams.update(
        {
            # Higher DPI gives sharper markers/lines in saved PNGs.
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 20,
            "axes.titlesize": 20,
            # Axis titles (x/y labels) and tick labels.
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.titleweight": "semibold",
            "axes.prop_cycle": cycler(color=COLOR_CYCLE),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            # Use a full, solid border (all spines visible) for a clean, bounded look.
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.1,
            "legend.frameon": False,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#dddddd",
            "legend.fontsize": 16,
            # Slightly smoother output for lines/markers.
            "lines.antialiased": True,
        }
    )

def load_npy(folder: str, fname: str) -> np.ndarray:
    """Load a `.npy` feature matrix from `<BASE_DIR>/<folder>/image_features/<fname>`."""

    path = os.path.join(BASE_DIR, folder, "image_features", fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path)


def load_labels(meta_path: str) -> np.ndarray:
    """Load `true_labels` from the given metadata JSON file."""

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "true_labels" not in meta:
        raise KeyError(f"'true_labels' not found in {meta_path}. Keys={list(meta.keys())}")
    return np.asarray(meta["true_labels"])


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between two matrices `a` and `b`.

    Args:
        a: Array of shape `[N, D]`.
        b: Array of shape `[N, D]`.
        eps: Small value to avoid division by zero.

    Returns:
        Array of shape `[N]` where entry i is `cos(a[i], b[i])`.
    """

    a = np.asarray(a)
    b = np.asarray(b)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + eps
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + eps
    return np.sum((a / a_norm) * (b / b_norm), axis=1)


def l2_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise L2 distance between two matrices `a` and `b` (shape `[N, D]`)."""

    return np.linalg.norm(a - b, axis=1)


def counter_legend_label(counter_folder: str) -> str:
    """Build a short legend label for a counter folder.

    Requirement (per issue): legend must include the noise type and its main parameter:
    - uniform noise uses ε (epsilon)
    - gaussian noise uses σ (sigma)
    """

    def _pretty_noise_type(noise_type: str) -> str:
        """Normalize noise type tokens from folder names for legend display."""

        noise_type = (noise_type or "").strip()
        if noise_type.lower() == "uniform":
            return "Uniform"
        if noise_type.lower() == "gaussian":
            return "Gaussian"
        # Fallback: capitalize first letter to keep legend readable.
        return noise_type[:1].upper() + noise_type[1:] if noise_type else "?"

    # Folder fragments we need to handle (examples):
    # - `..._Added_Noise_uniform_Eps_48.0_...`
    # - `..._Added_Noise_gaussian_Sigma_0.03_...`
    m_eps = re.search(r"Added_Noise_([^_]+)_Eps_([0-9]*\.?[0-9]+)", counter_folder)
    if m_eps is not None:
        noise_type, eps = m_eps.group(1), m_eps.group(2)
        return f"Noise ({_pretty_noise_type(noise_type)}, ε={int(float(eps))}/255)"

    m_sigma = re.search(r"Added_Noise_([^_]+)_Sigma_([0-9]*\.?[0-9]+)", counter_folder)
    if m_sigma is not None:
        noise_type, sigma = m_sigma.group(1), m_sigma.group(2)
        # Keep sigma as-is (already in natural units in folder names, e.g. 0.03).
        return f"Noise ({_pretty_noise_type(noise_type)}, σ={sigma})"

    # Fallback: still show the noise type if possible.
    m_type = re.search(r"Added_Noise_([^_]+)", counter_folder)
    if m_type is not None:
        return f"Noise ({_pretty_noise_type(m_type.group(1))}, ε=?/σ=?)"

    return "Noise (ε=?/σ=?)"


def counter_source_label(counter_folder: str) -> str:
    """Return the source condition name for a counter folder.

    Requirement (per issue): if the counter folder name contains `eps_0.0_steps_0`,
    the legend should refer to the `Clean` condition (no adversarial attack).
    Otherwise, treat it as starting from `Adversarial`.
    """

    return "Clean" if "eps_0.0_steps_0" in counter_folder else "Adv."


def plot_histogram_multi_overlay(
    *,
    xs: list[np.ndarray],
    labels: list[str],
    title: str,
    xlabel: str,
    bins: int = 30,
    figsize: tuple[int, int] = (10, 7.4),
    colors: Iterable[str | tuple[float, float, float, float]] | None = None,
    show_mean_lines: bool = True,
    show_legend: bool = True,
    save_path: str | None = None,
) -> None:
    """Overlay multiple histograms to compare distributions."""

    if len(xs) != len(labels):
        raise ValueError(f"xs and labels must have same length. Got {len(xs)} vs {len(labels)}")
    if len(xs) == 0:
        raise ValueError("xs must be non-empty")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(xs))]
    else:
        colors = list(colors)
        if len(colors) < len(xs):
            raise ValueError(f"colors has {len(colors)} entries but need {len(xs)}")

    for x, label, color in zip(xs, labels, colors, strict=True):
        ax.hist(
            x,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )
        if show_mean_lines:
            m = float(np.mean(x))
            ax.axvline(m, color=color, linestyle="--", linewidth=1.4, alpha=0.95)

    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    # Allow caller to decide whether to draw a per-axis legend. When composing
    # multi-panel figures we may want a single unified legend at the figure level.
    if show_legend:
        ax.legend(loc="best")

    _finalize_figure(fig, save_path=save_path)


def plot_histogram_multi_overlay_ax(
    ax: plt.Axes,
    *,
    xs: list[np.ndarray],
    labels: list[str],
    xlabel: str,
    bins: int = 30,
    colors: Iterable[str | tuple[float, float, float, float]] | None = None,
    show_mean_lines: bool = True,
    show_legend: bool = True,
) -> None:
    """Overlay multiple histograms on an existing axis.

    This is useful when composing multi-panel figures (e.g., a horizontal grid
    containing Fig. 1 + Fig. 2).
    """

    if len(xs) != len(labels):
        raise ValueError(f"xs and labels must have same length. Got {len(xs)} vs {len(labels)}")
    if len(xs) == 0:
        raise ValueError("xs must be non-empty")

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(xs))]
    else:
        colors = list(colors)
        if len(colors) < len(xs):
            raise ValueError(f"colors has {len(colors)} entries but need {len(xs)}")

    for x, label, color in zip(xs, labels, colors, strict=True):
        ax.hist(
            x,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )
        if show_mean_lines:
            m = float(np.mean(x))
            ax.axvline(m, color=color, linestyle="--", linewidth=1.4, alpha=0.95)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    # Allow caller to decide whether to draw a per-axis legend. When composing
    # multi-panel figures we may want a single unified legend at the figure level.
    if show_legend:
        ax.legend(loc="best")


def plot_histogram_overlay(
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    label1: str,
    label2: str,
    title: str,
    xlabel: str,
    bins: int = 30,
    figsize: tuple[int, int] = (10, 6),
    color1: str = COLORS["attack"],
    color2: str = COLORS["counter"],
    show_mean_lines: bool = True,
    save_path: str | None = None,
) -> None:
    """Overlay two histograms to compare distributions."""

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Use density so overlay comparisons don't depend on sample count.
    ax.hist(
        x1,
        bins=bins,
        density=True,
        alpha=0.45,
        color=color1,
        edgecolor="white",
        linewidth=0.7,
        label=label1,
    )
    ax.hist(
        x2,
        bins=bins,
        density=True,
        alpha=0.45,
        color=color2,
        edgecolor="white",
        linewidth=0.7,
        label=label2,
    )

    if show_mean_lines:
        m1 = float(np.mean(x1))
        m2 = float(np.mean(x2))
        ax.axvline(m1, color=color1, linestyle="--", linewidth=1.6, alpha=0.95)
        ax.axvline(m2, color=color2, linestyle="--", linewidth=1.6, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(loc="best")

    _finalize_figure(fig, save_path=save_path)


def plot_histogram(
    *,
    x: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 40,
    figsize: tuple[int, int] = (10, 6),
    color: str = COLORS["neutral"],
    show_mean_line: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot a single histogram."""

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(
        x,
        bins=bins,
        density=True,
        alpha=0.85,
        color=color,
        edgecolor="white",
        linewidth=0.7,
    )

    if show_mean_line:
        m = float(np.mean(x))
        ax.axvline(m, color="#222222", linestyle="--", linewidth=1.6, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")

    _finalize_figure(fig, save_path=save_path)

def main() -> None:
    apply_plot_style()

    plots_dir = _make_plots_dir() if SAVE_PLOTS else None
    if plots_dir is not None:
        print(f"[INFO] Saving plots to: {plots_dir}")

    # ---------------------------
    # LOAD (and align lengths)
    # ---------------------------
    # Each `.npy` is expected to be `[N, D]` (N samples, D feature dims).
    F_clean = load_npy(CLEAN_FOLDER, CLEAN_FILE)
    F_adv = load_npy(ADV_FOLDER, ADV_FILE)
    counter_folders = list(COUNTER_FOLDERS)
    if len(counter_folders) == 0:
        raise ValueError("COUNTER_FOLDERS must contain at least one folder")
    F_ctrs = [load_npy(folder, COUNTER_FILE) for folder in counter_folders]
    y = load_labels(META_PATH)

    # Defensive alignment: some folders may contain slightly different sample counts.
    min_len = min([len(F_clean), len(F_adv), len(y), *[len(F) for F in F_ctrs]])
    F_clean = F_clean[:min_len]
    F_adv = F_adv[:min_len]
    F_ctrs = [F[:min_len] for F in F_ctrs]
    y = y[:min_len]

    # Shared random subsample:
    # - keeps comparisons fair (same indices for clean/adv/counter)
    # - reduces plotting time / visual clutter
    rng = np.random.default_rng(RANDOM_SEED)
    n = min_len if MAX_SAMPLES is None else min(min_len, MAX_SAMPLES)
    idx = np.sort(rng.choice(min_len, size=n, replace=False))

    F_clean = F_clean[idx]
    F_adv = F_adv[idx]
    F_ctrs = [F[idx] for F in F_ctrs]
    y = y[idx]

    # ---------------------------
    # COMPUTE DRIFTS + METRICS
    # ---------------------------

    # Drift vectors / metrics that don't depend on which counter we use.
    # `D_attack`: how much the *attack* moved each sample in feature space.
    # Shape `[N, D]` where each row is a vector pointing from the clean feature to the adversarial feature.
    #   D_attack[i] = F_adv[i] - F_clean[i]
    D_attack = F_adv - F_clean  # adv - clean

    # `cos_clean_adv`: similarity between the clean and adversarial features per sample.
    # Values close to 1.0 mean the attack barely changed the feature direction; smaller values mean larger change.
    cos_clean_adv = cosine_similarity_rows(F_clean, F_adv)

    # `l2_clean_adv`: Euclidean distance between clean and adversarial features per sample.
    # Larger values indicate larger absolute movement (magnitude), regardless of direction.
    l2_clean_adv = l2_distance_rows(F_clean, F_adv)

    # Per-counter drifts/metrics.
    counter_labels = [counter_legend_label(folder) for folder in counter_folders]
    D_recovers: list[np.ndarray] = []
    cos_clean_ctrs: list[np.ndarray] = []
    l2_clean_ctrs: list[np.ndarray] = []
    drift_coss: list[np.ndarray] = []
    alphas: list[np.ndarray] = []

    # Precompute the per-sample squared attack magnitude `||D_attack||^2`.
    # Used to normalize the projection coefficient `alpha` below.
    den = np.sum(D_attack * D_attack, axis=1) + 1e-12
    for F_ctr in F_ctrs:
        # `D_recover`: how much the *countermeasure* moved features, starting from the adversarial point.
        #   D_recover[i] = F_ctr[i] - F_adv[i]
        D_recover = F_ctr - F_adv  # counter - adv

        # `D_net`: total / net movement from clean to countered.
        # This is the overall effect after attack + countermeasure.
        #   D_net[i] = F_ctr[i] - F_clean[i]
        D_net = F_ctr - F_clean  # counter - clean
        D_recovers.append(D_recover)

        cos_clean_ctrs.append(cosine_similarity_rows(F_clean, F_ctr))
        l2_clean_ctrs.append(l2_distance_rows(F_clean, F_ctr))

        # `drift_cos = cos(D_attack, D_recover)` measures alignment between the attack direction
        # and the counter direction (both as vectors in feature space).
        # - near -1: counter tends to reverse the attack direction
        # - near  0: counter moves in an orthogonal direction
        # - near +1: counter moves in the same direction as the attack
        drift_coss.append(cosine_similarity_rows(D_attack, D_recover))

        # `alpha`: scalar projection of the *net* drift onto the attack drift direction.
        # It answers: “after the countermeasure, how much of the remaining change is still along
        # the original attack direction?”
        #   alpha = <D_net, D_attack> / ||D_attack||^2
        # Interpretation (per sample):
        # - alpha ≈ 1  : net movement is similar to the original attack (counter did not undo it)
        # - alpha ≈ 0  : net movement has little component along the attack direction
        # - alpha < 0  : net movement points opposite the attack direction (over-correction)
        alphas.append(np.sum(D_net * D_attack, axis=1) / den)

    print("=== Summary (mean ± std) ===")
    print(f"cos(clean, adv): {cos_clean_adv.mean():.4f} ± {cos_clean_adv.std():.4f}")
    print(f"l2(clean, adv):  {l2_clean_adv.mean():.4f} ± {l2_clean_adv.std():.4f}")
    for lbl, cos_clean_ctr, l2_clean_ctr, drift_cos, alpha in zip(
        counter_labels,
        cos_clean_ctrs,
        l2_clean_ctrs,
        drift_coss,
        alphas,
        strict=True,
    ):
        print(f"--- {lbl} ---")
        print(f"cos(clean, counter): {cos_clean_ctr.mean():.4f} ± {cos_clean_ctr.std():.4f}")
        print(f"l2(clean, counter):  {l2_clean_ctr.mean():.4f} ± {l2_clean_ctr.std():.4f}")
        print(f"drift_cos(att, rec): {drift_cos.mean():.4f} ± {drift_cos.std():.4f}")
        print(f"alpha(net onto att): {alpha.mean():.4f} ± {alpha.std():.4f}")

    def _plot_fig1_drift_pca(ax: plt.Axes, show_legend: bool = True) -> None:
        """Figure 1: Drift space (PCA-2D) on an existing axis."""

        _scatter_drift(
            ax,
            Za[:, 0],
            Za[:, 1],
            label="Adv. Drift",
            color=COLORS["attack"],
            marker="o",
            n=n,
        )

        for i, (Zr, lbl) in enumerate(zip(Zrs, counter_labels, strict=True)):
            src = counter_source_label(counter_folders[i])
            ctr_color = COLOR_CYCLE[(i + 1) % len(COLOR_CYCLE)]
            _scatter_drift(
                ax,
                Zr[:, 0],
                Zr[:, 1],
                label=f"{src} + {lbl}",
                color=ctr_color,
                marker="^",
                n=n,
            )

            # Centroid marker (mean of the 2D-projected drift vectors).
            # This is a visual summary of the “average drift” direction in PCA space; it is not used
            # for computation, but helps interpret whether the counter drifts roughly oppose the attack.
            ax.scatter(
                [float(np.mean(Zr[:, 0]))],
                [float(np.mean(Zr[:, 1]))],
                s=90,
                marker="^",
                color=ctr_color,
                edgecolors="#111111",
                linewidths=0.6,
                alpha=0.95,
                zorder=4,
            )

        # Centroid for the attack drift (same idea as above, but for `D_attack`).
        ax.scatter(
            [float(np.mean(Za[:, 0]))],
            [float(np.mean(Za[:, 1]))],
            s=90,
            marker="o",
            color=COLORS["attack"],
            edgecolors="#111111",
            linewidths=0.6,
            alpha=0.95,
            zorder=4,
        )

        # `explained_variance_ratio_` tells how much of the drift-vector variance is captured by each PC.
        # Higher percentages mean the 2D projection preserves more of the drift distribution structure.
        _ = pca.explained_variance_ratio_

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2", labelpad=2)

        # Tighten limits to reduce empty space while keeping a small padding so points/centroids are not clipped.
        xs = [Za[:, 0], *[Zr[:, 0] for Zr in Zrs]]
        ys = [Za[:, 1], *[Zr[:, 1] for Zr in Zrs]]
        x_min = float(np.min(np.concatenate(xs)))
        x_max = float(np.max(np.concatenate(xs)))
        y_min = float(np.min(np.concatenate(ys)))
        y_max = float(np.max(np.concatenate(ys)))

        x_pad = 0.02 * (x_max - x_min + 1e-12)
        y_pad = 0.02 * (y_max - y_min + 1e-12)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", direction="out", length=5, width=1.0)
        ax.tick_params(axis="both", which="minor", direction="out", length=2.8, width=0.8)
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", alpha=0.12)

        # Per-axis legend may be suppressed when the caller wants a single
        # figure-level legend (e.g., for a 1x2 grid). Draw the axis legend only
        # when requested.
        if show_legend:
            n_entries = 1 + len(counter_labels)
            ncol = 3 if n_entries >= 3 else n_entries
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),
                ncol=ncol,
                borderaxespad=0.0,
                handlelength=1.2,
                markerscale=1.7,
                columnspacing=1.2,
                handletextpad=0.5,
            )

    def _plot_fig2_cosine_hist(ax: plt.Axes, show_legend: bool = True) -> None:
        """Figure 2: Cosine similarity distributions on an existing axis."""

        labels = ["Adversarial"]
        for idx, lbl in enumerate(counter_labels):
            src = counter_source_label(counter_folders[idx])
            labels.append(f"{src} + {lbl}")

        # Draw the histogram overlay. The caller can control whether this axis
        # draws its own legend (useful when composing a multi-panel figure).
        plot_histogram_multi_overlay_ax(
            ax,
            xs=[cos_clean_adv, *cos_clean_ctrs],
            labels=labels,
            xlabel="Cosine similarity",
            bins=30,
            colors=[
                COLORS["attack"],
                *[COLOR_CYCLE[(i + 1) % len(COLOR_CYCLE)] for i in range(len(counter_labels))],
            ],
            show_legend=show_legend,
        )

    # ---------------------------
    # FIGURE 1: Drift space (PCA-2D)
    # ---------------------------
    # What it shows:
    # - Each point is a *drift vector* (not the original feature).
    # - Blue points: `D_attack = adv - clean` (how the attack moved features)
    # - Orange points: `D_recover = counter - adv` (how the countermeasure moved features)
    #
    # Why PCA here:
    # - Drift vectors live in D-dimensional feature space.
    # - PCA projects them to 2D so we can visualize the distribution of directions/magnitudes.
    #
    # Important detail:
    # - We standardize the stacked drift vectors before PCA so that each drift dimension has
    #   comparable scale, making the PCA projection more stable/interpretable.
    #
    # Key terminology for this plot:
    # - “drift vector” = a difference of two feature vectors (e.g., adv-clean), not the feature itself.
    # - PCA axes (PC1/PC2) are directions of maximal variance *within the drift vectors*.
    # - The relative position of clouds indicates how similar the drift distributions are.
    X = np.vstack([D_attack, *D_recovers])
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X)

    # `Z` contains the 2D PCA coordinates for the stacked drifts.
    # First `n` rows correspond to `D_attack`; then each subsequent block of `n` rows corresponds
    # to one `D_recover` in the same order as `COUNTER_FOLDERS`.
    Za = Z[:n]
    Zrs = [Z[n * (i + 1) : n * (i + 2)] for i in range(len(D_recovers))]

    fig, ax = plt.subplots(figsize=(10, 7.4), constrained_layout=True)
    _plot_fig1_drift_pca(ax)

    _finalize_figure(
        fig,
        save_path=None if plots_dir is None else os.path.join(plots_dir, "fig1_drift_space_pca2d.png"),
    )

    # ---------------------------
    # FIGURE 2: Cosine similarity distributions
    # ---------------------------
    # What it shows:
    # - Overlaid histograms of `cos(clean, adv)` and `cos(clean, counter)`.
    #
    # How to read:
    # - If the countermeasure is effective, the `cos(clean, counter)` distribution should
    #   shift right (toward 1.0) relative to `cos(clean, adv)`.
    fig, ax = plt.subplots(figsize=(10, 7.4), constrained_layout=True)
    _plot_fig2_cosine_hist(ax)
    _finalize_figure(
        fig,
        save_path=None if plots_dir is None else os.path.join(plots_dir, "fig2_cosine_similarity_distributions.png"),
    )

    # ---------------------------
    # FIGURE 1+2: Horizontal grid (PCA drift + cosine similarity)
    # ---------------------------
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(20, 7.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )

    # Suppress per-axis legends for the grid. We'll place a single, PCA-only
    # legend at the figure level (user requested only the PCA legend at top).
    _plot_fig1_drift_pca(ax1, show_legend=False)
    _plot_fig2_cosine_hist(ax2, show_legend=False)

    # Remove any remaining axis-level legends on both subplots. This ensures
    # only the PCA legend (placed at the figure level) is visible.
    for a in (ax1, ax2):
        la = a.get_legend()
        if la is not None:
            try:
                la.remove()
            except Exception:
                # Non-fatal: ignore removal failures and continue.
                pass

    # Build a figure-level legend using only the PCA axis handles/labels so
    # the top legend reflects the PCA drift plot entries exclusively.
    handles, labels = ax1.get_legend_handles_labels()
    n_entries = len(labels)
    ncol = 3 if n_entries >= 3 else max(1, n_entries)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        # Nudge the legend upward a bit to make room for a larger font size.
        bbox_to_anchor=(0.5, 1.09),
        ncol=ncol,
        borderaxespad=0.0,
        # Slightly increase handle/marker sizes so the legend is clearer at larger font.
        handlelength=1.6,
        markerscale=3.2,
        columnspacing=1.2,
        handletextpad=0.6,
        fontsize=30,
    )

    # Just in case, remove any axis-level legends that might still exist so
    # the figure-level PCA legend is the only legend shown.
    for a in (ax1, ax2):
        la = a.get_legend()
        if la is not None:
            try:
                la.remove()
            except Exception:
                pass

    _finalize_figure(
        fig,
        save_path=None
        if plots_dir is None
        else os.path.join(plots_dir, "fig1_fig2_horizontal_grid_pca_drift_and_cosine.png"),
    )

    # ---------------------------
    # FIGURE 3: L2 distance distributions
    # ---------------------------
    # What it shows:
    # - Overlaid histograms of `||clean-adv||_2` and `||clean-counter||_2`.
    #
    # How to read:
    # - Smaller `||clean-counter||` than `||clean-adv||` indicates the countermeasure moved
    #   features closer (in Euclidean distance) to their clean baseline.

    plot_histogram_multi_overlay(
        xs=[l2_clean_adv, *l2_clean_ctrs],
        labels=["||clean-adv||", *[f"||clean-{lbl}||" for lbl in counter_labels]],
        title="L2 distance distributions (to clean)",
        xlabel="L2 distance",
        bins=30,
        colors=[COLORS["attack"], *[COLOR_CYCLE[(i + 1) % len(COLOR_CYCLE)] for i in range(len(counter_labels))]],
        save_path=None if plots_dir is None else os.path.join(plots_dir, "fig3_l2_distance_distributions_to_clean.png"),
    )

    # ---------------------------
    # FIGURE 4: Attack vs recovery drift alignment
    # ---------------------------
    # What it shows:
    # - Histogram of `cos(D_attack, D_recover)`.
    #
    # How to read:
    # - Values near `-1` mean the counter drift tends to go *against* the attack drift
    #   (good sign: reversal).
    # - Values near `+1` mean counter drift tends to follow the attack drift (bad sign).
    # - Values near `0` mean counter drift is orthogonal (changes features, but not by undoing
    #   the attack direction).

    plot_histogram_multi_overlay(
        xs=drift_coss,
        labels=[f"cos(D_attack, D_recover) ({lbl})" for lbl in counter_labels],
        title="Alignment between attack and recovery drifts\ncos(D_attack, D_recover)",
        xlabel="cosine",
        bins=40,
        colors=[COLOR_CYCLE[(i + 1) % len(COLOR_CYCLE)] for i in range(len(counter_labels))],
        save_path=None if plots_dir is None else os.path.join(plots_dir, "fig4_attack_recovery_alignment_cosine.png"),
    )

    # ---------------------------
    # FIGURE 5: Alpha projection coefficient
    # ---------------------------
    # What it shows:
    # - Histogram of `alpha`, the projection of the net effect `(counter-clean)` onto the
    #   attack direction `(adv-clean)`.
    #
    # How to read:
    # - `alpha < 0` indicates the countermeasure tends to move opposite to the attack.
    # - `alpha > 0` indicates it tends to move in the same direction as the attack.
    plot_histogram_multi_overlay(
        xs=alphas,
        labels=[f"alpha ({lbl})" for lbl in counter_labels],
        title=(
            "Projection of net drift onto attack drift\n"
            "alpha = <(counter-clean),(adv-clean)> / ||adv-clean||^2"
        ),
        xlabel="alpha",
        bins=40,
        colors=[COLOR_CYCLE[(i + 1) % len(COLOR_CYCLE)] for i in range(len(counter_labels))],
        save_path=None if plots_dir is None else os.path.join(plots_dir, "fig5_alpha_projection_coefficient.png"),
    )

    # ---------------------------
    # OPTIONAL: Per-class improvement boxplot
    # ---------------------------
    # What it shows:
    # - For each class, we compute `delta_cos = cos(clean,counter) - cos(clean,adv)`.
    # - If `delta_cos` is positive, the countermeasure improves similarity to clean.
    # - Boxplots show the per-class distribution of this improvement.
    # For per-class plot, use the first counter variant by default to avoid clutter.

    delta_cos = cos_clean_ctrs[0] - cos_clean_adv

    classes = np.unique(y)
    min_per_class = 10
    keep = np.array([c for c in classes if np.sum(y == c) >= min_per_class])

    if len(keep) > 0:
        data = [delta_cos[y == c] for c in keep]
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        bp = ax.boxplot(
            data,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 1.6},
            whiskerprops={"color": "#444444", "linewidth": 1.2},
            capprops={"color": "#444444", "linewidth": 1.2},
        )
        for box in bp["boxes"]:
            box.set(facecolor=COLORS["counter"], alpha=0.35, edgecolor="#444444", linewidth=1.0)

        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(
            "Per-class improvement: cos(clean,counter) - cos(clean,adv)\n"
            f"(only classes with ≥{min_per_class} samples)"
        )
        # X-axis is the *index* in `keep` (not necessarily the raw class id).
        ax.set_xlabel("class index in keep[]")
        ax.set_ylabel("delta cosine")

        _finalize_figure(
            fig,
            save_path=None
            if plots_dir is None
            else os.path.join(plots_dir, "fig6_per_class_delta_cos_boxplot.png"),
        )
    else:
        print(f"[INFO] No class has ≥{min_per_class} samples in the current subsample.")


if __name__ == "__main__":
    main()