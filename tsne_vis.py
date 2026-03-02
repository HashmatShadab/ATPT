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

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------
# CONFIG
# ---------------------------

# Base directory that contains subfolders like `ADV_Generation_eps_.../image_features/...`.
BASE_DIR = "tsne/vit_l_14_datacomp_1b/Caltech101"

# Feature folders (relative to `BASE_DIR`).
CLEAN_FOLDER = "ADV_Generation_eps_0.0_steps_0"
ADV_FOLDER = "ADV_Generation_eps_4.0_steps_100"
COUNTER_FOLDER = (
    "ADV_Generation_eps_4.0_steps_100_Added_Noise_uniform_Eps_48.0_Tau_Type_normal_num_anchors_1"
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
# PLOTTING STYLE
# ---------------------------

# A small, colorblind-friendly palette (close to Matplotlib "tab" colors, but explicit).
COLORS = {
    "attack": "#1f77b4",  # blue
    "recovery": "#ff7f0e",  # orange
    "counter": "#2ca02c",  # green
    "neutral": "#7f7f7f",  # gray
}


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
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#dddddd",
            "legend.fontsize": 10,
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
    plt.show()


def plot_histogram(
    *,
    x: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 40,
    figsize: tuple[int, int] = (10, 6),
    color: str = COLORS["neutral"],
    show_mean_line: bool = True,
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
    plt.show()

def main() -> None:
    apply_plot_style()

    # ---------------------------
    # LOAD (and align lengths)
    # ---------------------------
    # Each `.npy` is expected to be `[N, D]` (N samples, D feature dims).
    F_clean = load_npy(CLEAN_FOLDER, CLEAN_FILE)
    F_adv = load_npy(ADV_FOLDER, ADV_FILE)
    F_ctr = load_npy(COUNTER_FOLDER, COUNTER_FILE)
    y = load_labels(META_PATH)

    # Defensive alignment: some folders may contain slightly different sample counts.
    min_len = min(len(F_clean), len(F_adv), len(F_ctr), len(y))
    F_clean = F_clean[:min_len]
    F_adv = F_adv[:min_len]
    F_ctr = F_ctr[:min_len]
    y = y[:min_len]

    # Shared random subsample:
    # - keeps comparisons fair (same indices for clean/adv/counter)
    # - reduces plotting time / visual clutter
    rng = np.random.default_rng(RANDOM_SEED)
    n = min_len if MAX_SAMPLES is None else min(min_len, MAX_SAMPLES)
    idx = np.sort(rng.choice(min_len, size=n, replace=False))

    F_clean = F_clean[idx]
    F_adv = F_adv[idx]
    F_ctr = F_ctr[idx]
    y = y[idx]

    # ---------------------------
    # COMPUTE DRIFTS + METRICS
    # ---------------------------

    # Drift vectors (all are `[N, D]`):
    # - Attack drift: how far the adversary moves a feature from its clean baseline.
    # - Recovery drift: how far the countermeasure moves from the attacked feature.
    # - Net drift: where the countermeasure ends up relative to the clean baseline.
    D_attack = F_adv - F_clean  # adv - clean
    D_recover = F_ctr - F_adv  # counter - adv
    D_net = F_ctr - F_clean  # counter - clean

    # Pairwise cosine similarities (per sample):
    # - `cos(clean, adv)`   : similarity after attack (lower often means larger semantic drift)
    # - `cos(clean, counter)`: similarity after recovery (higher means closer to clean)
    cos_clean_adv = cosine_similarity_rows(F_clean, F_adv)
    cos_clean_ctr = cosine_similarity_rows(F_clean, F_ctr)
    cos_adv_ctr = cosine_similarity_rows(F_adv, F_ctr)

    # Pairwise L2 distances (per sample):
    # - These capture magnitude of movement in feature space (scale-dependent).
    l2_clean_adv = l2_distance_rows(F_clean, F_adv)
    l2_clean_ctr = l2_distance_rows(F_clean, F_ctr)
    l2_adv_ctr = l2_distance_rows(F_adv, F_ctr)

    # Drift alignment score:
    # `cos(D_attack, D_recover)` tells whether the countermeasure moves in the same direction
    # as the attack (+1), in the opposite direction (-1), or orthogonally (~0).
    drift_cos = cosine_similarity_rows(D_attack, D_recover)

    # Alpha projection coefficient:
    # Projection of net drift onto the attack direction:
    #   alpha = <(counter-clean), (adv-clean)> / ||adv-clean||^2
    # Interpretation (roughly):
    # - `alpha ≈  1` : counter ends up along the same attack direction (no real undo)
    # - `alpha ≈  0` : counter movement is mostly orthogonal to attack direction
    # - `alpha ≈ -1` : counter tends to reverse the attack direction (strong undo)
    den = np.sum(D_attack * D_attack, axis=1) + 1e-12
    alpha = np.sum(D_net * D_attack, axis=1) / den

    print("=== Summary (mean ± std) ===")
    print(f"cos(clean, adv):     {cos_clean_adv.mean():.4f} ± {cos_clean_adv.std():.4f}")
    print(f"cos(clean, counter): {cos_clean_ctr.mean():.4f} ± {cos_clean_ctr.std():.4f}")
    print(f"cos(adv, counter):   {cos_adv_ctr.mean():.4f} ± {cos_adv_ctr.std():.4f}")
    print(f"l2(clean, adv):      {l2_clean_adv.mean():.4f} ± {l2_clean_adv.std():.4f}")
    print(f"l2(clean, counter):  {l2_clean_ctr.mean():.4f} ± {l2_clean_ctr.std():.4f}")
    print(f"l2(adv, counter):    {l2_adv_ctr.mean():.4f} ± {l2_adv_ctr.std():.4f}")
    print(f"drift_cos(att, rec): {drift_cos.mean():.4f} ± {drift_cos.std():.4f}")
    print(f"alpha(net onto att): {alpha.mean():.4f} ± {alpha.std():.4f}")

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
    X = np.vstack([D_attack, D_recover])
    X = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

    Za = Z[:n]
    Zr = Z[n:]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.scatter(
        Za[:, 0],
        Za[:, 1],
        s=18,
        alpha=0.55,
        color=COLORS["attack"],
        edgecolors="white",
        linewidths=0.25,
        label="attack drift (adv-clean)",
    )
    ax.scatter(
        Zr[:, 0],
        Zr[:, 1],
        s=18,
        alpha=0.55,
        color=COLORS["recovery"],
        edgecolors="white",
        linewidths=0.25,
        label="recovery drift (counter-adv)",
    )
    ax.set_title("Drift space (PCA-2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    plt.show()

    # ---------------------------
    # FIGURE 2: Cosine similarity distributions
    # ---------------------------
    # What it shows:
    # - Overlaid histograms of `cos(clean, adv)` and `cos(clean, counter)`.
    #
    # How to read:
    # - If the countermeasure is effective, the `cos(clean, counter)` distribution should
    #   shift right (toward 1.0) relative to `cos(clean, adv)`.
    plot_histogram_overlay(
        x1=cos_clean_adv,
        x2=cos_clean_ctr,
        label1="cos(clean, adv)",
        label2="cos(clean, counter)",
        title="Cosine similarity distributions (clean vs adv / counter)",
        xlabel="cosine similarity",
        bins=30,
        color1=COLORS["attack"],
        color2=COLORS["counter"],
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
    plot_histogram_overlay(
        x1=l2_clean_adv,
        x2=l2_clean_ctr,
        label1="||clean-adv||",
        label2="||clean-counter||",
        title="L2 distance distributions (to clean)",
        xlabel="L2 distance",
        bins=30,
        color1=COLORS["attack"],
        color2=COLORS["counter"],
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
    plot_histogram(
        x=drift_cos,
        title="Alignment between attack and recovery drifts\ncos(D_attack, D_recover)",
        xlabel="cosine",
        bins=40,
        color=COLORS["neutral"],
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
    plot_histogram(
        x=alpha,
        title=(
            "Projection of net drift onto attack drift\n"
            "alpha = <(counter-clean),(adv-clean)> / ||adv-clean||^2"
        ),
        xlabel="alpha",
        bins=40,
        color=COLORS["neutral"],
    )

    # ---------------------------
    # OPTIONAL: Per-class improvement boxplot
    # ---------------------------
    # What it shows:
    # - For each class, we compute `delta_cos = cos(clean,counter) - cos(clean,adv)`.
    # - If `delta_cos` is positive, the countermeasure improves similarity to clean.
    # - Boxplots show the per-class distribution of this improvement.
    delta_cos = cos_clean_ctr - cos_clean_adv

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
        plt.show()
    else:
        print(f"[INFO] No class has ≥{min_per_class} samples in the current subsample.")


if __name__ == "__main__":
    main()