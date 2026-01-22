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
        default="../../Diffratio_Adv_gen_Results/vit_l_14_datacomp_1b",
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


def generate_plots(aggregated: Dict[str, Any], root: str):
    results = aggregated["results"]
    if not results:
        print("No results to plot.")
        return

    # Create output directory for plots
    plots_dir = os.path.join("plots_output", os.path.basename(root))
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    # results structure: model -> dataset -> attack -> noise_type -> noise_param -> tau_type
    for model_name, datasets in results.items():
        # We want to group by (attack, noise_type)
        # and then by dataset
        
        # First, find all unique attacks and noise_types
        attacks_noise = set()
        all_datasets = sorted(datasets.keys())
        
        for dataset, attacks in datasets.items():
            for attack, noise_types in attacks.items():
                for noise_type in noise_types.keys():
                    attacks_noise.add((attack, noise_type))
        
        for attack, noise_type in sorted(list(attacks_noise)):
            print(f"Plotting for Attack: {attack}, Noise Type: {noise_type}")
            
            # For each dataset, gather noise_values and corresponding metrics
            # metric_types: ('avg_diff_ratio_after_counter_attack', 'counter_attack_accuracy')
            for metric_name in ["avg_diff_ratio_after_counter_attack", "counter_attack_accuracy"]:
                plot_grid(datasets, model_name, attack, noise_type, metric_name, all_datasets, plots_dir)

def plot_grid(datasets: Dict[str, Any], model_name: str, attack: str, noise_type: str, metric_key: str, all_datasets: List[str], plots_dir: str):
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
    fig.suptitle(f"{model_name} evaluated on adversarial attack: {attack}\nNoise added: {noise_type} | Metric: {clean_metric_name}", fontsize=20)

    # Data structure to accumulate values for averaging: (noise_val, tau_type) -> [list of values]
    overall_averages = {}

    for i in range(total_plots):
        ax = axes[i // cols, i % cols]
        
        noise_data = [] # List of (noise_value, tau_type, metric_value)
        
        if i < num_datasets:
            dataset = all_datasets[i]
            title = dataset
            if dataset in datasets and attack in datasets[dataset] and noise_type in datasets[dataset][attack]:
                noise_params = datasets[dataset][attack][noise_type]
                for noise_param_str, tau_types in noise_params.items():
                    # Extract numerical value from Sigma_0.03 or Eps_16.0
                    try:
                        noise_val = float(noise_param_str.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    
                    for tau_type, metrics in tau_types.items():
                        val = metrics.get(metric_key)
                        if val is not None:
                            noise_data.append((noise_val, tau_type, val))
                            # Accumulate for overall average
                            key = (noise_val, tau_type)
                            if key not in overall_averages:
                                overall_averages[key] = []
                            overall_averages[key].append(val)
        else:
            # This is the "Average" plot
            title = "AVERAGE ACROSS DATASETS"
            for (noise_val, tau_type), vals in overall_averages.items():
                if vals:
                    noise_data.append((noise_val, tau_type, np.mean(vals)))
        
        if not noise_data:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(title)
            continue
            
        # Group by noise_val and tau_type for bar plotting
        unique_noise_vals = sorted(list(set(nv for nv, tt, m in noise_data)))
        unique_tau_types = sorted(list(set(tt for nv, tt, m in noise_data)))
        
        # Mapping noise_val to index
        noise_val_to_idx = {val: i for i, val in enumerate(unique_noise_vals)}
        
        # Create a structure for bar positions
        # x_positions: list of indices [0, 1, 2, ...]
        x_indices = np.arange(len(unique_noise_vals))
        width = 0.8 / len(unique_tau_types) if unique_tau_types else 0.8
        
        for j, tt in enumerate(unique_tau_types):
            # Find values for this tau_type at each unique_noise_val
            y_vals = []
            tt_noise_vals = []
            for nv in unique_noise_vals:
                # Find the metric for (nv, tt)
                match = [m for nv_item, tt_item, m in noise_data if nv_item == nv and tt_item == tt]
                if match:
                    y_vals.append(match[0])
                    tt_noise_vals.append(noise_val_to_idx[nv])
            
            if y_vals:
                # Offset for each tau_type
                offset = (j - (len(unique_tau_types) - 1) / 2) * width
                rects = ax.bar(np.array(tt_noise_vals) + offset, y_vals, width, label=tt)
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
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)

    # Hide unused subplots
    for i in range(total_plots, rows * cols):
        axes[i // cols, i % cols].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    filename = f"{model_name}_{attack}_{noise_type}_{metric_key}.png"
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  - Saved plot: {filename}")


if __name__ == "__main__":
    main()
