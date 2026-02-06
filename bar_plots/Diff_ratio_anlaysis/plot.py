#!/usr/bin/env python3
"""
Aggregate diff-ratio + accuracy metrics from experiment folders structured as:

ROOT/
  <DATASET_NAME>/
    ADV_Generation_<ATTACK>_Added_Noise_<NOISE_TYPE>_<NoiseParamName>_<NoiseParamVal>_Tau_Type_<TAU_TYPE>/
      diff_ratio_after_counter_attack.json   (may be missing)

You provide ROOT (e.g., Diffratio_Adv_gen_Results/vit_l_14_datacomp_1b)
We will:
  - iterate datasets (immediate subfolders of ROOT)
  - iterate experiment folders inside each dataset
  - parse folder name to extract: attack, noise_type, noise_param, tau_type
  - read metrics JSON if present; otherwise set metric fields to None
  - save aggregated_results.json under ROOT by default
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np


METRICS_FILENAME = "diff_ratio_after_counter_attack.json"

ATTACK_NAME_MAPPING = {
    "eps_0.0_steps_0": "Clean",
    "eps_1.0_steps_10": "PGD 1/255 (10 steps)",
    "eps_1.0_steps_10_image_only_attack_prm": "PGD 1/255 (10 steps, image only)",
    "eps_1.0_steps_100": "PGD 1/255 (100 steps)",
    "eps_1.0_steps_100_image_only_attack_prm": "PGD 1/255 (100 steps, image only)",
    "eps_4.0_steps_10": "PGD 4/255 (10 steps)",
    "eps_4.0_steps_10_image_only_attack_prm": "PGD 4/255 (10 steps, image only)",
    "eps_4.0_steps_100": "PGD 4/255 (100 steps)",
    "eps_4.0_steps_100_image_only_attack_prm": "PGD 4/255 (100 steps, image only)",
    "eps_8.0_steps_10": "PGD 8/255 (10 steps)",
    "eps_8.0_steps_10_image_only_attack_prm": "PGD 8/255 (10 steps, image only)",
    "eps_8.0_steps_100": "PGD 8/255 (100 steps)",
    "eps_8.0_steps_100_image_only_attack_prm": "PGD 8/255 (100 steps, image only)",
}


def parse_experiment_folder_name(folder_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract parameters from experiment folder name.
    Returns None if it doesn't match the expected ADV_Generation_* pattern.
    """

    # attack: everything between ADV_Generation_ and _Added_Noise_
    m_attack = re.search(r"^ADV_Generation_(.+?)_Added_Noise_", folder_name)
    if not m_attack:
        return None
    attack = m_attack.group(1)

    # noise type: token after Added_Noise_
    m_noise = re.search(r"_Added_Noise_([A-Za-z]+)_", folder_name)
    noise_type = m_noise.group(1) if m_noise else None

    # noise param: Sigma_0.03 or Eps_1.0 (extendable)
    m_param = re.search(r"_(Sigma|Eps)_([\d.]+)(?:_|$)", folder_name)
    noise_param = {"name": None, "value": None}
    if m_param:
        noise_param = {"name": m_param.group(1), "value": float(m_param.group(2))}

    # tau type: everything after Tau_Type_ to end
    m_tau = re.search(r"_Tau_Type_(.+)$", folder_name)
    tau_type = m_tau.group(1) if m_tau else None

    return {
        "attack": attack,
        "noise_type": noise_type,
        "noise_param": noise_param,
        "tau_type": tau_type,
    }


def load_metrics_or_none(exp_folder_path: str) -> Dict[str, Any]:
    """
    Load metrics JSON inside exp folder; if missing, return None-filled fields.
    If JSON exists but is invalid, keep None-filled fields + store error string.
    """
    metrics_path = os.path.join(exp_folder_path, METRICS_FILENAME)

    metrics = {
        "diff_ratio_after_counter_attack": None,
        "avg_diff_ratio_after_counter_attack": None,
        "original_clean_accuracy": None,
        "adversarial_accuracy": None,
        "counter_attack_accuracy": None,
        "metrics_json_present": False,
        "metrics_json_error": None,
    }

    if not os.path.isfile(metrics_path):
        return metrics

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        metrics["metrics_json_error"] = str(e)
        return metrics

    metrics.update({
        "diff_ratio_after_counter_attack": data.get("diff_ratio_after_counter_attack"),
        "avg_diff_ratio_after_counter_attack": data.get("avg_diff_ratio_after_counter_attack"),
        "original_clean_accuracy": data.get("original_clean_accuracy"),
        "adversarial_accuracy": data.get("adversarial_accuracy"),
        "counter_attack_accuracy": data.get("counter_attack_accuracy"),
        "metrics_json_present": True,
        "metrics_json_error": None,
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="../../Diffratio_Adv_gen_Results/delta_clip_l14_224",
        help="Root path containing dataset subfolders (e.g., Diffratio_Adv_gen_Results/vit_l_14_datacomp_1b)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="aggregated_results.json",
        help="Output JSON filename (written under --root unless absolute path)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_path = args.out

    if not os.path.isdir(root):
        raise RuntimeError(f"Root is not a directory: {root}")

    aggregated = {
        "root": root,
        "results": {},  # Hierarchical: model -> dataset -> attack -> noise_type -> noise_param -> tau_type
        "stats": {
            "num_datasets_seen": 0,
            "num_experiment_folders_seen": 0,
            "num_json_present": 0,
            "num_json_missing_or_invalid": 0,
        },
    }

    # Datasets are immediate subfolders under root
    dataset_names = [
        d for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    aggregated["stats"]["num_datasets_seen"] = len(dataset_names)

    model_name = os.path.basename(root)  # e.g., vit_l_14_datacomp_1b

    for dataset in dataset_names:
        dataset_dir = os.path.join(root, dataset)

        # Experiment folders are subfolders under dataset_dir
        for exp_name in sorted(os.listdir(dataset_dir)):
            exp_dir = os.path.join(dataset_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            params = parse_experiment_folder_name(exp_name)
            if params is None:
                continue  # ignore non ADV_Generation_* folders

            metrics = load_metrics_or_none(exp_dir)

            # Extract param components
            attack = params["attack"]
            noise_type = params["noise_type"]
            noise_param_obj = params["noise_param"]  # {"name": "Sigma", "value": 0.03}
            tau_type = params["tau_type"]

            # Filter: remove or don't add values which have Noise uniform and value is 48.0
            if noise_type.lower() == "uniform" and noise_param_obj["value"] == 48.0:
                continue

            # Construct noise_param string key
            if noise_param_obj["name"] and noise_param_obj["value"] is not None:
                noise_param_str = f"{noise_param_obj['name']}_{noise_param_obj['value']}"
            else:
                noise_param_str = "None"

            # Build hierarchy: model -> dataset -> attack -> noise_type -> noise_param -> tau_type
            if model_name not in aggregated["results"]:
                aggregated["results"][model_name] = {}
            if dataset not in aggregated["results"][model_name]:
                aggregated["results"][model_name][dataset] = {}
            if attack not in aggregated["results"][model_name][dataset]:
                aggregated["results"][model_name][dataset][attack] = {}
            if noise_type not in aggregated["results"][model_name][dataset][attack]:
                aggregated["results"][model_name][dataset][attack][noise_type] = {}
            if noise_param_str not in aggregated["results"][model_name][dataset][attack][noise_type]:
                aggregated["results"][model_name][dataset][attack][noise_type][noise_param_str] = {}

            # Save metrics under tau_type
            aggregated["results"][model_name][dataset][attack][noise_type][noise_param_str][tau_type] = {
                "folder": exp_name,
                "path": exp_dir,
                **metrics,
            }

            aggregated["stats"]["num_experiment_folders_seen"] += 1
            if metrics["metrics_json_present"] and metrics["metrics_json_error"] is None:
                aggregated["stats"]["num_json_present"] += 1
            else:
                aggregated["stats"]["num_json_missing_or_invalid"] += 1

    # os.makedirs(out_path, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print(f"[OK] Saved: {out_path}")
    print("[OK] Stats:")
    for k, v in aggregated["stats"].items():
        print(f"  - {k}: {v}")

    # Start plotting
    generate_plots(aggregated, root)


def get_attack_sort_key(atk_name):
    """
    Custom sort key for adversarial attacks.
    Order: attacks containing eps_0.0_steps_0 first, then by eps value, then by steps, then by image_only flag.
    Example: 'eps_1.0_steps_10', 'eps_1.0_steps_100', 'eps_1.0_steps_10_image_only_attack_prm', 'eps_1.0_steps_100_image_only_attack_prm'
    """
    if "eps_0.0_steps_0" in atk_name:
        return (0.0, 0, False)
    
    # eps_1.0_steps_10
    # eps_1.0_steps_10_image_only_attack_prm
    m = re.search(r"eps_([\d.]+)_steps_(\d+)", atk_name)
    if not m:
        return (999.0, 999, False) # Should not happen with current naming
    
    eps = float(m.group(1))
    steps = int(m.group(2))
    is_image_only = "image_only" in atk_name
    
    return (eps, is_image_only, steps)

def get_pretty_attack_name(atk_name):
    if atk_name in ATTACK_NAME_MAPPING:
        return ATTACK_NAME_MAPPING[atk_name]
    
    # Try substring match for Clean
    if "eps_0.0_steps_0" in atk_name:
        return ATTACK_NAME_MAPPING["eps_0.0_steps_0"]
    
    return atk_name

def generate_plots(aggregated: Dict[str, Any], root: str):
    results = aggregated["results"]
    if not results:
        print("No results to plot.")
        return

    # Create output directory for plots
    plots_dir = os.path.join("plots_output", os.path.basename(root))
    os.makedirs(plots_dir, exist_ok=True)
    
    # Subfolder for individual plots
    indiv_plots_dir = os.path.join(plots_dir, "individual_plots")
    os.makedirs(indiv_plots_dir, exist_ok=True)
    
    print(f"Saving plots to: {plots_dir}")
    print(f"Saving individual plots to: {indiv_plots_dir}")

    CLEAN_ATTACK = "eps_0.0_steps_0"

    # results structure: model -> dataset -> attack -> noise_type -> noise_param -> tau_type
    for model_name, datasets in results.items():
        # First, find all unique attacks and noise_types
        all_attacks = set()
        all_noise_types = set()
        all_datasets = sorted(datasets.keys())
        
        # Identify the actual clean attack name (which contains "eps_0.0_steps_0")
        actual_clean_attack = None

        for dataset, attacks in datasets.items():
            for attack, noise_types in attacks.items():
                if "eps_0.0_steps_0" in attack:
                    # In case there are multiple attacks containing "eps_0.0_steps_0", 
                    # we take the first one found or we could handle them differently.
                    # Usually there is only one such attack.
                    actual_clean_attack = attack
                else:
                    all_attacks.add(attack)
                for noise_type in noise_types.keys():
                    all_noise_types.add(noise_type)
        
        if actual_clean_attack is None:
            actual_clean_attack = "eps_0.0_steps_0" # fallback
        
        # for attack in sorted(list(all_attacks)):
        #     for noise_type in sorted(list(all_noise_types)):
        #         print(f"Plotting for Adversarial Attack: {attack}, vs Clean, Noise Type: {noise_type}")
        #
        #         # For each dataset, gather noise_values and corresponding metrics
        #         for metric_name in ["avg_diff_ratio_after_counter_attack", "counter_attack_accuracy"]:
        #             plot_grid(datasets, model_name, attack, actual_clean_attack, noise_type, metric_name, all_datasets, plots_dir, indiv_plots_dir)

        # New: Plot summary grids averaged across datasets for each noise type
        for metric_name in ["avg_diff_ratio_after_counter_attack", "counter_attack_accuracy"]:
            for noise_type in sorted(list(all_noise_types)):
                # plot_noise_summary(datasets, model_name, actual_clean_attack, all_attacks, noise_type, metric_name, plots_dir)
                # Added: Grid of summary plots (average across datasets) for each attack vs clean
                plot_attack_vs_clean_summary_grid(datasets, model_name, actual_clean_attack, all_attacks, noise_type, metric_name, plots_dir, indiv_plots_dir)

def plot_attack_vs_clean_summary_grid(datasets: Dict[str, Any], model_name: str, clean_attack: str, all_attacks: set, noise_type: str, metric_key: str, plots_dir: str, indiv_plots_dir: str):
    """
    Creates a grid of plots, one for each adversarial attack, averaged across all datasets.
    Each subplot compares 'Clean' vs 'one adversarial variant'.
    """
    present_attacks = sorted([atk for atk in all_attacks], key=get_attack_sort_key)
    if not present_attacks:
        return

    # Accumulate average data for each (noise_val, attack_name)
    # (noise_val, attack_name) -> [list of values from different datasets]
    summary_data = {}
    ordered_attacks = [clean_attack] + present_attacks
    
    for dataset, attacks in datasets.items():
        for atk_name in ordered_attacks:
            if atk_name in attacks and noise_type in attacks[atk_name]:
                noise_params = attacks[atk_name][noise_type]
                for noise_param_str, tau_types in noise_params.items():
                    try:
                        noise_val = float(noise_param_str.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    
                    vals = [m.get(metric_key) for m in tau_types.values() if m.get(metric_key) is not None]
                    if vals:
                        avg_val = np.mean(vals)
                        key = (noise_val, atk_name)
                        if key not in summary_data:
                            summary_data[key] = []
                        summary_data[key].append(avg_val)

    if not summary_data:
        return

    # Calculate means across datasets
    attack_means = {} # (noise_val, attack_name) -> mean_val
    for key, vals in summary_data.items():
        attack_means[key] = np.mean(vals)

    unique_noise_vals = sorted(list(set(nv for nv, atk in summary_data.keys())))
    
    num_attacks = len(present_attacks)
    cols = 4
    rows = (num_attacks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
    if metric_key == 'counter_attack_accuracy':
        metric_key = "accuracy_after_noise_addition"
    elif metric_key == 'avg_diff_ratio_after_counter_attack':
        metric_key = "avg_diff_ratio"
    clean_metric_name = metric_key.replace('_', ' ').title()
    fig.suptitle(f"Average Across Datasets. Added Noise: {noise_type}\n Evaluating Metric: {clean_metric_name}", fontsize=20)

    for i, adv_attack in enumerate(present_attacks):
        ax = axes[i // cols, i % cols]
        
        # Data for the individual plot
        plot_data_for_indiv = []
        
        x_indices = np.arange(len(unique_noise_vals))
        width = 0.45
        
        labels = ["Clean", "Adversarial"]
        for j, label in enumerate(labels):
            atk_to_plot = clean_attack if label == "Clean" else adv_attack
            y_vals = []
            idx_list = []
            for idx, nv in enumerate(unique_noise_vals):
                key = (nv, atk_to_plot)
                if key in attack_means:
                    y_vals.append(attack_means[key])
                    idx_list.append(idx)
            
            if y_vals:
                offset = (j - 0.5) * width
                rects = ax.bar(np.array(idx_list) + offset, y_vals, width, label=label,
                               color='skyblue' if label == "Clean" else 'salmon')
                ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=12)
                plot_data_for_indiv.append((idx_list, y_vals, label, offset))

        pretty_adv_name = get_pretty_attack_name(adv_attack)
        ax.set_title(pretty_adv_name, fontsize=16, fontweight='bold')
        ax.set_xlabel(f"{noise_type} Noise", fontsize=14)
        ax.set_ylabel(clean_metric_name)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(v) for v in unique_noise_vals])
        if i == 0:
            ax.legend(fontsize=14, loc='upper left', ncol=2)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max(current_ylim[1] * 1.25, 0.1))

        # --- Create individual figure ---
        metric_indiv_dir = os.path.join(indiv_plots_dir, metric_key)
        os.makedirs(metric_indiv_dir, exist_ok=True)

        fig_indiv, ax_indiv = plt.subplots(figsize=(12, 8))
        for idx_list, y_vals, label, offset in plot_data_for_indiv:
            rects = ax_indiv.bar(np.array(idx_list) + offset, y_vals, width, label=label,
                                 color='skyblue' if label == "Clean" else 'salmon')
            ax_indiv.bar_label(rects, padding=3, fmt='%.2f', fontsize=14)
        
        ax_indiv.set_title(pretty_adv_name, fontsize=22, fontweight='bold', pad=20)
        ax_indiv.set_xlabel(f"{noise_type} Noise", fontsize=20, labelpad=10)
        ax_indiv.set_ylabel(clean_metric_name, fontsize=20, labelpad=10)
        ax_indiv.set_xticks(x_indices)
        ax_indiv.set_xticklabels([str(v) for v in unique_noise_vals], fontsize=16)
        ax_indiv.tick_params(axis='y', labelsize=16)
        ax_indiv.legend(fontsize=18, loc='upper left', ncol=2)
        ax_indiv.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        ax_indiv.set_ylim(ax.get_ylim()) # Match the grid subplot's y-limit
        
        plt.tight_layout()
        indiv_filename = f"{noise_type}_{metric_key}_{adv_attack}_summary.png"
        fig_indiv.savefig(os.path.join(metric_indiv_dir, indiv_filename))
        plt.close(fig_indiv)

    # Hide unused subplots
    for i in range(num_attacks, rows * cols):
        axes[i // cols, i % cols].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    filename = f"{noise_type}_{metric_key}_summary_vs_clean_grid.png"
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  - Saved attack-vs-clean summary grid: {filename}")

def plot_noise_summary(datasets: Dict[str, Any], model_name: str, clean_attack: str, all_attacks: set, noise_type: str, metric_key: str, plots_dir: str):
    """
    Creates a single figure for a specific noise type, averaged across all datasets.
    X-axis: noise parameters.
    Bars: Clean value + all adversarial variants.
    """
    # Accumulate data: (noise_val, attack_name) -> [list of values from different datasets]
    summary_data = {}
    
    # List of all attacks to include: Clean + all others
    ordered_attacks = [clean_attack] + sorted(list(all_attacks), key=get_attack_sort_key)
    
    for dataset, attacks in datasets.items():
        for atk_name in ordered_attacks:
            if atk_name in attacks and noise_type in attacks[atk_name]:
                noise_params = attacks[atk_name][noise_type]
                for noise_param_str, tau_types in noise_params.items():
                    try:
                        noise_val = float(noise_param_str.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    
                    # Average over tau_types
                    vals = [m.get(metric_key) for m in tau_types.values() if m.get(metric_key) is not None]
                    if vals:
                        avg_val = np.mean(vals)
                        key = (noise_val, atk_name)
                        if key not in summary_data:
                            summary_data[key] = []
                        summary_data[key].append(avg_val)

    if not summary_data:
        return

    # Process into means
    unique_noise_vals = sorted(list(set(nv for nv, atk in summary_data.keys())))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    if metric_key == 'counter_attack_accuracy':
        metric_key = "accuracy_after_noise_addition"
    elif metric_key == 'avg_diff_ratio_after_counter_attack':
        metric_key = "avg_diff_ratio"
    clean_metric_name = metric_key.replace('_', ' ').title()
    ax.set_title(f"{model_name} | Noise: {noise_type} | Average Across Datasets\nMetric: {clean_metric_name}", fontsize=16)

    x_indices = np.arange(len(unique_noise_vals))
    num_attacks = len(ordered_attacks)
    total_width = 0.8
    width = total_width / num_attacks
    
    # We might want to filter ordered_attacks to only those that actually have data for this noise_type
    present_attacks = []
    for atk in ordered_attacks:
        if any((nv, atk) in summary_data for nv in unique_noise_vals):
            present_attacks.append(atk)
    
    num_present = len(present_attacks)
    width = total_width / num_present if num_present > 0 else total_width

    for j, atk_name in enumerate(present_attacks):
        y_vals = []
        idx_list = []
        for i, nv in enumerate(unique_noise_vals):
            key = (nv, atk_name)
            if key in summary_data:
                y_vals.append(np.mean(summary_data[key]))
                idx_list.append(i)
        
        if y_vals:
            offset = (j - (num_present - 1) / 2) * width
            label = get_pretty_attack_name(atk_name)
            rects = ax.bar(np.array(idx_list) + offset, y_vals, width, label=label)
            ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=7, rotation=90 if num_present > 3 else 0)

    ax.set_xlabel(f"{noise_type} value")
    ax.set_ylabel(clean_metric_name)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(v) for v in unique_noise_vals])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Increase y-limit
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max(current_ylim[1] * 1.2, 0.1))

    plt.tight_layout()
    
    filename = f"{noise_type}_{metric_key}_summary_all_attacks.png"
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  - Saved summary plot: {filename}")

def plot_grid(datasets: Dict[str, Any], model_name: str, adv_attack: str, clean_attack: str, noise_type: str, metric_key: str, all_datasets: List[str], plots_dir: str, indiv_plots_dir: str):
    num_datasets = len(all_datasets)
    if num_datasets == 0:
        return

    # Add 1 for the "Average" plot
    total_plots = num_datasets + 1
    cols = 3
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    
    # Clean up names for title
    clean_metric_name = metric_key.replace('_', ' ').title()
    pretty_adv_attack = get_pretty_attack_name(adv_attack)
    fig.suptitle(f"{model_name} | Adv: {pretty_adv_attack} vs Clean\nNoise: {noise_type} | Metric: {clean_metric_name}", fontsize=20)

    # Data structure to accumulate values for averaging: 
    # (noise_val, 'clean'|'adv') -> [list of values]
    overall_averages = {}

    for i in range(total_plots):
        ax = axes[i // cols, i % cols]
        
        # List of (noise_value, type('clean'|'adv'), metric_value)
        plot_data = [] 
        
        if i < num_datasets:
            dataset = all_datasets[i]
            title = dataset
            
            # Helper to extract data for a given attack
            def get_attack_data(atk_name, label):
                if dataset in datasets and atk_name in datasets[dataset] and noise_type in datasets[dataset][atk_name]:
                    noise_params = datasets[dataset][atk_name][noise_type]
                    for noise_param_str, tau_types in noise_params.items():
                        try:
                            noise_val = float(noise_param_str.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                        
                        # Average over tau_types since "tau type is not needed"
                        vals = [m.get(metric_key) for m in tau_types.values() if m.get(metric_key) is not None]
                        if vals:
                            avg_val = np.mean(vals)
                            plot_data.append((noise_val, label, avg_val))
                            
                            # Accumulate for overall average
                            key = (noise_val, label)
                            if key not in overall_averages:
                                overall_averages[key] = []
                            overall_averages[key].append(avg_val)

            get_attack_data(clean_attack, "Clean")
            get_attack_data(adv_attack, "Adversarial")
        else:
            # This is the "Average" plot
            title = "AVERAGE ACROSS DATASETS"
            for (noise_val, label), vals in overall_averages.items():
                if vals:
                    plot_data.append((noise_val, label, np.mean(vals)))
        
        if not plot_data:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(title)
            continue
            
        # Group by noise_val and type for bar plotting
        unique_noise_vals = sorted(list(set(nv for nv, label, m in plot_data)))
        labels = ["Clean", "Adversarial"]
        
        # Mapping noise_val to index
        noise_val_to_idx = {val: i for i, val in enumerate(unique_noise_vals)}
        
        x_indices = np.arange(len(unique_noise_vals))
        width = 0.35 # Fixed width for two bars
        
        for j, label in enumerate(labels):
            y_vals = []
            idx_list = []
            for nv in unique_noise_vals:
                match = [m for nv_item, lbl_item, m in plot_data if nv_item == nv and lbl_item == label]
                if match:
                    y_vals.append(match[0])
                    idx_list.append(noise_val_to_idx[nv])
            
            if y_vals:
                offset = (j - 0.5) * width # -0.5*width for Clean, +0.5*width for Adversarial
                rects = ax.bar(np.array(idx_list) + offset, y_vals, width, label=label, 
                               color='skyblue' if label == "Clean" else 'salmon')
                ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=8)
            
        ax.set_title(title, fontweight='bold' if i == num_datasets else 'normal')
        ax.set_xlabel(f"{noise_type} value")
        ax.set_ylabel(metric_key.replace('_', ' ').title())
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(v) for v in unique_noise_vals])
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Increase y-limit to avoid labels hitting the top border
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max(current_ylim[1] * 1.15, 0.1))

        # --- Create individual figure ---
        metric_indiv_dir = os.path.join(indiv_plots_dir, metric_key)
        os.makedirs(metric_indiv_dir, exist_ok=True)

        fig_indiv, ax_indiv = plt.subplots(figsize=(12, 8))
        for j, label in enumerate(labels):
            y_vals = []
            idx_list = []
            for nv in unique_noise_vals:
                match = [m for nv_item, lbl_item, m in plot_data if nv_item == nv and lbl_item == label]
                if match:
                    y_vals.append(match[0])
                    idx_list.append(noise_val_to_idx[nv])
            if y_vals:
                offset = (j - 0.5) * width
                rects = ax_indiv.bar(np.array(idx_list) + offset, y_vals, width, label=label,
                                     color='skyblue' if label == "Clean" else 'salmon')
                ax_indiv.bar_label(rects, padding=3, fmt='%.3f', fontsize=14)
        
        ax_indiv.set_title(title, fontsize=26, fontweight='bold' if i == num_datasets else 'normal', pad=20)
        ax_indiv.set_xlabel(f"{noise_type} value", fontsize=24, labelpad=10)
        ax_indiv.set_ylabel(metric_key.replace('_', ' ').title(), fontsize=24, labelpad=10)
        ax_indiv.set_xticks(x_indices)
        ax_indiv.set_xticklabels([str(v) for v in unique_noise_vals], fontsize=20)
        ax_indiv.tick_params(axis='y', labelsize=20)
        ax_indiv.legend(fontsize=20)
        ax_indiv.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax_indiv.set_ylim(ax.get_ylim())

        plt.tight_layout()
        safe_title = title.replace(" ", "_").lower()
        indiv_filename = f"{adv_attack}_{noise_type}_{metric_key}_{safe_title}.png"
        fig_indiv.savefig(os.path.join(metric_indiv_dir, indiv_filename))
        plt.close(fig_indiv)

    # Hide unused subplots
    for i in range(total_plots, rows * cols):
        axes[i // cols, i % cols].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    filename = f"{adv_attack}_{noise_type}_{metric_key}_vs_clean.png"
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  - Saved plot: {filename}")


if __name__ == "__main__":
    main()
