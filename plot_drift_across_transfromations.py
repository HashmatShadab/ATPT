
# plot_latent_drift_analysis.py
# Comprehensive latent drift visualization and analysis

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except:
    HAS_SEABORN = False

REVERSE_X_TRANSFORMS = {
    "brightness_dark",
    "contrast_low",
    "gamma_bright",
    "hue_negative",
    "saturation_low",
    "sharpness_low",
}

TRANSFORMATION_DISPLAY_NAMES = {
    "brightness_bright": "Increasing Brightness",
    "brightness_dark": "Decreasing Brightness",
    "contrast_high": "Increasing Contrast",
    "contrast_low": "Decreasing Contrast",
    "downsample": "Increasing Downsampling",
    "gamma_bright": "Gamma Brightening",
    "gamma_dark": "Gamma Darkening",
    "gaussian_blur": "Increasing Gaussian Blur",
    "gaussian_noise": "Increasing Gaussian Noise",
    "hue_negative": "Increasing Negative Hue Shift",
    "hue_positive": "Increasing Positive Hue Shift",
    "jpeg": "Increasing JPEG Compression",
    "posterize": "Increasing Posterization",
    "rotation": "Increasing Rotation",
    "saturation_high": "Increasing Saturation",
    "saturation_low": "Decreasing Saturation",
    "sharpness_high": "Increasing Sharpness",
    "sharpness_low": "Decreasing Sharpness",
    "solarize": "Increasing Solarization",
    "translation": "Translation",
    "uniform_noise": "Increasing Uniform Noise",
}

if HAS_SEABORN:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    sns.set_context("paper", font_scale=1.5)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

def load_avg_drift(json_path, transformation_name):
    with open(json_path, "r") as f:
        data = json.load(f)

    drift_dict = data["avg_diff_ratio_after_counter_attack"]

    severity_labels = list(drift_dict.keys())

    # if transformation_name in REVERSE_X_TRANSFORMS:
    #     severity_labels = severity_labels[::-1]

    drift_values = [drift_dict[sev] for sev in severity_labels]

    return severity_labels, drift_values


def get_transformation_name(folder_name):
    prefix = "_Added_Noise_"
    suffix = "_across_severity_levels"

    if prefix in folder_name:
        return folder_name.split(prefix)[1].replace(suffix, "")

    return folder_name


def find_transformation_pairs(root_dir):
    transformations = {}

    for folder in os.listdir(root_dir):
        full_path = os.path.join(root_dir, folder)

        if not os.path.isdir(full_path):
            continue

        transformation = get_transformation_name(folder)
        if transformation == folder:
            continue
        else:
            transformations.setdefault(transformation, {})

        if "ADV_Generation_eps_0.0_steps_0" in folder:
            transformations[transformation]["clean"] = full_path

        elif "ADV_Generation_eps_4.0_steps_100" in folder:
            transformations[transformation]["adv"] = full_path

    return transformations


def normalize_curve(y):
    y = np.array(y)
    if abs(y[0]) < 1e-12:
        return y
    return y / y[0]


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_individual_transformation(
    transformation,
    clean_x,
    clean_y,
    adv_x,
    adv_y,
    save_path,
):
    plt.figure(figsize=(10, 7))

    clean_positions = np.arange(len(clean_y))
    adv_positions = np.arange(len(adv_y))

    # Use distinct darker colors
    clean_color = "#004e82" # Darker Blue
    adv_color = "#9e3d00"   # Darker Red-Orange

    plt.plot(clean_positions, clean_y, marker="o", linewidth=3,
             markersize=8, label="Clean", color=clean_color)

    plt.plot(adv_positions, adv_y, marker="s", linewidth=3,
             markersize=8, label=r"PGD-100 ($\epsilon=4/100$)", color=adv_color)

    if len(clean_y) == len(adv_y):
        plt.fill_between(clean_positions, clean_y, adv_y,
                         color=adv_color, alpha=0.1)

    plt.xticks([])

    plt.xlabel(r"Increasing Severity $\rightarrow$", fontsize=32)
    plt.ylabel(r"Mean Latent Drift $\tau$", fontsize=32)
    
    display_name = TRANSFORMATION_DISPLAY_NAMES.get(transformation, transformation)
    plt.title(display_name, fontsize=32, fontweight="bold", pad=20)
    
    plt.legend(frameon=True, framealpha=0.9, loc="best", fontsize=24)
    # ytick size
    plt.yticks(fontsize=20)

    plt.grid(
        True,
        which='major',
        axis='both',
        linestyle='--',
        linewidth=0.8,
        alpha=0.5
    )

    plt.minorticks_on()

    plt.grid(
        True,
        which='minor',
        axis='both',
        linestyle=':',
        linewidth=0.5,
        alpha=0.3
    )
    # Make axes darker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor("black")
    ax.tick_params(axis='both', colors='black', width=2.0)
    
    sns.despine()
    save_fig(save_path)


def plot_transformation_grid(all_data, transformations_to_plot, save_path):
    """Creates a 3x4 grid of individual transformation plots."""
    fig, axes = plt.subplots(3, 4, figsize=(40, 24))
    axes = axes.flatten()

    # Use darker tones
    clean_color = "#004e82" # Darker Blue
    adv_color = "#9e3d00"   # Darker Red-Orange

    for i, transformation in enumerate(transformations_to_plot):
        if i >= 12:
            break
        
        ax = axes[i]
        data = all_data[transformation]
        
        clean_y = data["clean"]["drift"]
        adv_y = data["adv"]["drift"]
        
        clean_positions = np.arange(len(clean_y))
        adv_positions = np.arange(len(adv_y))

        ax.plot(clean_positions, clean_y, marker="o", linewidth=3,
                 markersize=8, label="Clean", color=clean_color)

        ax.plot(adv_positions, adv_y, marker="s", linewidth=3,
                 markersize=8, label=r"PGD-100 ($\epsilon=4/100$)", color=adv_color)

        if len(clean_y) == len(adv_y):
            ax.fill_between(clean_positions, clean_y, adv_y,
                             color=adv_color, alpha=0.1)

        ax.set_xticks([])
        display_name = TRANSFORMATION_DISPLAY_NAMES.get(transformation, transformation)
        
        if i % 4 == 0:
            ax.set_ylabel(r"Mean Latent Drift $\tau$", fontsize=44)
            ax.tick_params(axis="y", labelsize=24)
        else:
            ax.tick_params(axis="y", labelsize=0) # Hide tick labels for internal plots to save space

        ax.set_xlabel(f"{display_name} $\\rightarrow$", fontsize=40)
        
        ax.legend(loc="lower right", fontsize=34, frameon=True, framealpha=0.9)

        ax.grid(
            True,
            which='major',
            axis='both',
            linestyle='--',
            linewidth=0.8,
            alpha=0.5
        )

        ax.minorticks_on()

        ax.grid(
            True,
            which='minor',
            axis='both',
            linestyle=':',
            linewidth=0.5,
            alpha=0.3
        )
        # Add borders to individual plots and make them darker
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor("black")
        ax.tick_params(axis='both', colors='black', width=2.0)

    # Hide any unused subplots
    for j in range(i + 1, 12):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_all_transformations(transformations_data, save_path, mode="clean",
                             normalize=False):

    plt.figure(figsize=(14, 9))

    if HAS_SEABORN:
        # Use a rich color palette for many lines
        colors = sns.color_palette("husl", len(transformations_data))
    else:
        colors = [None] * len(transformations_data)

    for i, (transformation, data) in enumerate(sorted(transformations_data.items())):

        y = np.array(data[mode]["drift"])

        if normalize:
            y = normalize_curve(y)

        x = np.arange(len(y))

        plt.plot(
            x,
            y,
            linewidth=2.5,
            marker="o",
            markersize=5,
            alpha=0.8,
            color=colors[i],
            label=TRANSFORMATION_DISPLAY_NAMES.get(transformation, transformation),
        )

    plt.xlabel("Severity Index", fontsize=14)
    plt.ylabel("Normalized Drift" if normalize else "Latent Drift", fontsize=14)

    title = f"Transformation Drift: {mode.title()}"
    if normalize:
        title += " (Normalized)"

    plt.title(title, fontsize=20, fontweight="bold", pad=20)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        ncol=1,
        frameon=True,
    )

    plt.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    sns.despine()
    save_fig(save_path)


def plot_metric(transformations_data, key, ylabel, title, save_path):

    plt.figure(figsize=(14, 9))

    if HAS_SEABORN:
        colors = sns.color_palette("husl", len(transformations_data))
    else:
        colors = [None] * len(transformations_data)

    for i, (transformation, data) in enumerate(sorted(transformations_data.items())):

        y = np.array(data[key])
        x = np.arange(len(y))

        plt.plot(
            x,
            y,
            linewidth=2.5,
            marker="o",
            markersize=5,
            alpha=0.8,
            color=colors[i],
            label=TRANSFORMATION_DISPLAY_NAMES.get(transformation, transformation),
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)

    plt.xlabel("Severity Index", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=20, fontweight="bold", pad=20)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        ncol=1,
        frameon=True,
    )

    plt.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    sns.despine()
    save_fig(save_path)


def horizontal_ranking(values, title, xlabel, save_path):

    names = list(values.keys())
    vals = np.array(list(values.values()))

    order = np.argsort(vals)

    names = [names[i] for i in order]
    vals = vals[order]

    display_names = [TRANSFORMATION_DISPLAY_NAMES.get(str(n), n) for n in names]

    height = max(6, len(display_names) * 0.5)
    plt.figure(figsize=(12, height))

    if HAS_SEABORN:
        # Create a color gradient based on values
        norm = plt.Normalize(vals.min(), vals.max())
        colors = plt.cm.viridis(norm(vals))
        bars = plt.barh(display_names, vals, color=colors, edgecolor="black", alpha=0.8)
    else:
        bars = plt.barh(display_names, vals)

    # Add value labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x = width + (max(vals) * 0.01) if width >= 0 else width - (max(vals) * 0.01)
        ha = "left" if width >= 0 else "right"
        plt.text(label_x, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                 va="center", ha=ha, fontsize=11, fontweight="bold")

    plt.xlabel(xlabel, fontsize=14)
    plt.title(title, fontsize=20, fontweight="bold", pad=20)
    
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    sns.despine(left=True, bottom=True)
    
    save_fig(save_path)


def plot_heatmap(matrix, row_labels, title, save_path):

    height = max(8, len(row_labels) * 0.5)
    plt.figure(figsize=(14, height))

    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            cmap="magma",
            yticklabels=row_labels,
            xticklabels=np.arange(matrix.shape[1]),
            annot=True,
            fmt=".2f",
            linewidths=.5,
            cbar_kws={"label": "Value"}
        )
    else:
        plt.imshow(matrix, aspect="auto", cmap="magma")
        plt.yticks(np.arange(len(row_labels)), row_labels)
        plt.colorbar()

    plt.xlabel("Severity Index", fontsize=14)
    plt.ylabel("Transformation", fontsize=14)
    plt.title(title, fontsize=20, fontweight="bold", pad=20)

    save_fig(save_path)


def main(root_dir):

    output_dir = os.path.join(root_dir, "latent_drift_plots")
    os.makedirs(output_dir, exist_ok=True)

    transformations = find_transformation_pairs(root_dir)

    all_data = {}

    final_clean = {}
    final_adv = {}
    final_gap = {}

    auc_gap = {}

    for transformation, paths in sorted(transformations.items()):

        clean_json = os.path.join(
            paths["clean"],
            "diff_ratio_after_counter_attack.json"
        )

        adv_json = os.path.join(
            paths["adv"],
            "diff_ratio_after_counter_attack.json"
        )

        if not os.path.exists(clean_json) or not os.path.exists(adv_json):
            continue

        clean_x, clean_y = load_avg_drift(clean_json, transformation)
        adv_x, adv_y = load_avg_drift(adv_json, transformation)

        diff = []
        rel_diff = []

        for c, a in zip(clean_y, adv_y):

            diff.append(a - c)

            if abs(c) < 1e-12:
                rel_diff.append(np.nan)
            else:
                rel_diff.append(100.0 * (a - c) / c)

        all_data[transformation] = {
            "clean": {
                "severity": clean_x,
                "drift": clean_y,
            },
            "adv": {
                "severity": adv_x,
                "drift": adv_y,
            },
            "diff": diff,
            "rel_diff": rel_diff,
        }

        final_clean[transformation] = clean_y[-1]
        final_adv[transformation] = adv_y[-1]
        final_gap[transformation] = adv_y[-1] - clean_y[-1]

        auc_gap[transformation] = (
            np.trapz(adv_y) - np.trapz(clean_y)
        )

        plot_individual_transformation(
            transformation,
            clean_x,
            clean_y,
            adv_x,
            adv_y,
            os.path.join(output_dir, f"{transformation}.png"),
        )

    # Create 4x4 grid plot of 16 transformations (excluding gaussian_blur)
    grid_transformations = ['brightness_bright', 'brightness_dark', 'contrast_high', 'contrast_low',
                            'sharpness_high', 'sharpness_high', 'saturation_high', 'saturation_low',
                            'gamma_bright', 'gamma_dark',  'hue_negative', 'hue_positive',]
                            # 'posterize', 'rotation',   'downsample', 'translation']

    if grid_transformations:
        plot_transformation_grid(
            all_data,
            grid_transformations,
            os.path.join(output_dir, "mean_latent_drift_across_transformations.png")
        )

    plot_all_transformations(
        all_data,
        os.path.join(output_dir, "all_transformations_clean.png"),
        mode="clean",
    )

    plot_all_transformations(
        all_data,
        os.path.join(output_dir, "all_transformations_adv.png"),
        mode="adv",
    )

    plot_all_transformations(
        all_data,
        os.path.join(output_dir,
                     "all_transformations_clean_normalized.png"),
        mode="clean",
        normalize=True,
    )

    plot_all_transformations(
        all_data,
        os.path.join(output_dir,
                     "all_transformations_adv_normalized.png"),
        mode="adv",
        normalize=True,
    )

    plot_metric(
        all_data,
        "diff",
        "Adv Drift - Clean Drift",
        "Additional Drift Caused by Adversarial Samples",
        os.path.join(output_dir,
                     "all_transformations_adv_minus_clean.png"),
    )

    plot_metric(
        all_data,
        "rel_diff",
        "% Increase Over Clean",
        "Relative Difference (%)",
        os.path.join(output_dir,
                     "all_transformations_relative_difference.png"),
    )

    horizontal_ranking(
        final_clean,
        "Final Severity Ranking (Clean)",
        "Latent Drift",
        os.path.join(output_dir,
                     "final_severity_ranking_clean.png"),
    )

    horizontal_ranking(
        final_adv,
        "Final Severity Ranking (Adversarial)",
        "Latent Drift",
        os.path.join(output_dir,
                     "final_severity_ranking_adv.png"),
    )

    horizontal_ranking(
        final_gap,
        "Adversarial Gap Ranking",
        "Adv - Clean",
        os.path.join(output_dir,
                     "advantage_gap_ranking.png"),
    )

    horizontal_ranking(
        auc_gap,
        "AUC Gap Ranking",
        "AUC(Adv) - AUC(Clean)",
        os.path.join(output_dir,
                     "auc_gap_ranking.png"),
    )

    max_len = max(len(v["diff"]) for v in all_data.values())

    diff_matrix = []
    rel_matrix = []
    names = []

    for name, v in sorted(all_data.items()):

        names.append(name)

        d = np.full(max_len, np.nan)
        r = np.full(max_len, np.nan)

        d[:len(v["diff"])] = v["diff"]
        r[:len(v["rel_diff"])] = v["rel_diff"]

        diff_matrix.append(d)
        rel_matrix.append(r)

    diff_matrix = np.array(diff_matrix)
    rel_matrix = np.array(rel_matrix)

    display_names = [TRANSFORMATION_DISPLAY_NAMES.get(name, name) for name in names]

    plot_heatmap(
        diff_matrix,
        display_names,
        "Adv - Clean Heatmap",
        os.path.join(output_dir,
                     "adv_minus_clean_heatmap.png"),
    )

    plot_heatmap(
        rel_matrix,
        display_names,
        "Relative Difference Heatmap (%)",
        os.path.join(output_dir,
                     "relative_difference_heatmap.png"),
    )

    print(f"Saved all plots to: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="transformation_ablation/vit_l_14_datacomp_1b/DTD",
    )

    args = parser.parse_args()

    main(args.root_dir)
