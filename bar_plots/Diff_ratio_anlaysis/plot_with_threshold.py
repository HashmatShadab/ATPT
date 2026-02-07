# Load zero-shot (ZS) results for a given model.
# These results serve as the baseline (no noise added) for clean and adversarial scenarios.
def get_zs_results(model_name):
    from pathlib import Path
    import json
    import numpy as np

    # MODELS = [
    #     "delta_clip_l14_224",
    #     "fare4",
    #     "ViT-L/14",
    #     "vit_l_14_datacomp_1b",
    # ]
    MODELS = [
             model_name,
    ]

    DATASETS = [
        "DTD",
        "Flower102",
        "Cars",
        "Aircraft",
        "Pets",
        "Caltech101",
        "UCF101",
        "eurosat",
    ]

    PRED_KEYS = ("max_confidence", "prediction", "label")

    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pred_json(obj: dict, *, allow_extra: bool = False) -> dict:
        out = {}
        for k in PRED_KEYS:
            if k not in obj:
                raise KeyError(f"Missing key '{k}' in prediction json.")
            out[k] = obj[k]

        if allow_extra:
            # these only exist for results_original_clean.json (as you described)
            if "correct_clean_indices" in obj:
                out["correct_clean_indices"] = obj["correct_clean_indices"]
            if "incorrect_clean_indices" in obj:
                out["incorrect_clean_indices"] = obj["incorrect_clean_indices"]

        return out

    def diff_ratio(obj: dict) -> dict:
        # note: keeping your spelling "avergae_diff_ratio" as-is
        if "diff_ratio" not in obj or "avergae_diff_ratio" not in obj:
            raise KeyError("Expected keys: 'diff_ratio' and 'avergae_diff_ratio' (note spelling).")
        return {
            "diff_ratio_per_sample": obj["diff_ratio"],
            "diff_ratio_avg": obj["avergae_diff_ratio"],
        }

    def compute_accuracy(preds, labels):
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        return (preds == labels).mean() * 100.0

    """
    ### ZERO SHOT Experiment

    We will load zero shot results for different models and datasets, under clean and adversarial conditions.
    1. We will store true labels for each sample for each dataset.
    2. We will store zero-shot single prediction results for each sample for each dataset under clean and adversarial conditions.
    3. We will store zero-shot max confidence scores for each sample for each dataset under clean and adversarial conditions.
    """

    # Base path templates for each case
    RESULT_PATHS = {
        "zero_shot": {
            "clean": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau/"
                    "{model}/{dataset}/"
                    "Clean/No_Counter_Attack/No_TPT/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            "adversarial_eps4_steps100": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau/"
                    "{model}/{dataset}/"
                    "Adversarial_Eps_4_0_Steps_100/"
                    "No_Counter_Attack/No_TPT/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            "adversarial_eps4_steps100_image_only": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau/"
                    "{model}/{dataset}/"
                    "Adversarial_Eps_4_0_Steps_100_image_only_attack_prm/"
                    "No_Counter_Attack/No_TPT/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
        }
    }

    JSON_KEYMAP = {
        # prediction-style
        "original": "results_original.json",
        "single": "results_single.json",
        "vanilla": "results_vanilla.json",
        "weighted": "results_weighted.json",

        # special prediction json
        "original_clean": "results_original_clean.json",

        # counter-attack json
        "results_counter_attack_diff_ratio": "results_counter_attack_diff_ratio.json",
    }

    PRED_KEYS = ("max_confidence", "prediction", "label")

    def load_one_setting(base_path: str) -> dict:
        """
        base_path: directory for a single (case, model, dataset)
        returns: normalized dict with preds + counter_attack
        """
        base = Path(base_path)

        # --- prediction jsons
        preds = {}
        for key in ["original", "single", "vanilla", "weighted"]:
            fp = base / JSON_KEYMAP[key]
            obj = _load_json(fp)
            preds[key] = _load_pred_json(obj)

        # --- original_clean json (special extras)
        fp = base / JSON_KEYMAP["original_clean"]
        obj = _load_json(fp)
        preds["original_clean"] = _load_pred_json(obj, allow_extra=True)

        # --- counter-attack diff-ratio json
        fp = base / JSON_KEYMAP["results_counter_attack_diff_ratio"]
        obj = _load_json(fp)
        counter_attack = diff_ratio(obj)

        return {"preds": preds, "results_counter_attack_diff_ratio": counter_attack}

    def build_all_data(result_paths: dict, models: list, datasets: list) -> dict:
        DATA = {}

        for case, cfg in result_paths["zero_shot"].items():
            DATA.setdefault(case, {})
            base_template = cfg["base_path"]

            for model in models:
                DATA[case].setdefault(model, {})

                for dataset in datasets:
                    base_path = base_template.format(model=model, dataset=dataset)
                    DATA[case][model][dataset] = load_one_setting(base_path)

        return DATA

    DATA = build_all_data(RESULT_PATHS, MODELS, DATASETS)

    import numpy as np

    case = "clean"
    model = "vit_l_14_datacomp_1b"

    true_labels_data = {}

    zero_shot_clean_preds_data = {}
    zero_shot_clean_max_confidences_data = {}

    zero_shot_clean_preds_vanilla_data = {}
    zero_shot_clean_max_confidences_vanilla_data = {}

    zero_shot_clean_preds_weighted_data = {}
    zero_shot_clean_max_confidences_weighted_data = {}

    for dataset in DATASETS:
        example = DATA[case][model][dataset]
        true_labels_data[dataset] = example["preds"]["original_clean"]["label"]

        zero_shot_clean_preds_data[dataset] = example["preds"]["original_clean"]["prediction"]
        zero_shot_clean_max_confidences_data[dataset] = example["preds"]["original"]["max_confidence"]

        zero_shot_clean_preds_vanilla_data[dataset] = example["preds"]["vanilla"]["prediction"]
        zero_shot_clean_max_confidences_vanilla_data[dataset] = example["preds"]["vanilla"]["max_confidence"]

        zero_shot_clean_preds_weighted_data[dataset] = example["preds"]["weighted"]["prediction"]
        zero_shot_clean_max_confidences_weighted_data[dataset] = example["preds"]["weighted"]["max_confidence"]

        # print accuracy
        print(dataset, compute_accuracy(zero_shot_clean_preds_data[dataset], true_labels_data[dataset]))
        # print mean confidence
        print(dataset, np.mean(zero_shot_clean_max_confidences_data[dataset]))

    case = "adversarial_eps4_steps100"
    model = "vit_l_14_datacomp_1b"

    zero_shot_adv_preds_data = {}
    zero_shot_adv_max_confidences_data = {}

    zero_shot_adv_preds_vanilla_data = {}
    zero_shot_adv_max_confidences_vanilla_data = {}

    zero_shot_adv_preds_weighted_data = {}
    zero_shot_adv_max_confidences_weighted_data = {}

    for dataset in DATASETS:
        example = DATA[case][model][dataset]

        zero_shot_adv_preds_data[dataset] = example["preds"]["original"]["prediction"]
        zero_shot_adv_max_confidences_data[dataset] = example["preds"]["original"]["max_confidence"]

        zero_shot_adv_preds_vanilla_data[dataset] = example["preds"]["vanilla"]["prediction"]
        zero_shot_adv_max_confidences_vanilla_data[dataset] = example["preds"]["vanilla"]["max_confidence"]

        zero_shot_adv_preds_weighted_data[dataset] = example["preds"]["weighted"]["prediction"]
        zero_shot_adv_max_confidences_weighted_data[dataset] = example["preds"]["weighted"]["max_confidence"]

        # print accuracy
        print(dataset, compute_accuracy(zero_shot_adv_preds_data[dataset], true_labels_data[dataset]))
        # print mean confidence
        print(dataset, np.mean(zero_shot_adv_max_confidences_data[dataset]))

    case = "adversarial_eps4_steps100_image_only"
    model = "vit_l_14_datacomp_1b"

    zero_shot_adv_image_only_preds_data = {}
    zero_shot_adv_image_only_max_confidences_data = {}

    zero_shot_adv_image_only_preds_vanilla_data = {}
    zero_shot_adv_image_only_max_confidences_vanilla_data = {}

    zero_shot_adv_image_only_preds_weighted_data = {}
    zero_shot_adv_image_only_max_confidences_weighted_data = {}

    for dataset in DATASETS:
        example = DATA[case][model][dataset]

        zero_shot_adv_image_only_preds_data[dataset] = example["preds"]["original"]["prediction"]
        zero_shot_adv_image_only_max_confidences_data[dataset] = example["preds"]["original"]["max_confidence"]

        zero_shot_adv_image_only_preds_vanilla_data[dataset] = example["preds"]["vanilla"]["prediction"]
        zero_shot_adv_image_only_max_confidences_vanilla_data[dataset] = example["preds"]["vanilla"]["max_confidence"]

        zero_shot_adv_image_only_preds_weighted_data[dataset] = example["preds"]["weighted"]["prediction"]
        zero_shot_adv_image_only_max_confidences_weighted_data[dataset] = example["preds"]["weighted"]["max_confidence"]

        # print accuracy
        print(dataset, compute_accuracy(zero_shot_adv_image_only_preds_data[dataset], true_labels_data[dataset]))
        # print mean confidence
        print(dataset, np.mean(zero_shot_adv_image_only_max_confidences_data[dataset]))

    TRUE_LABELS_DATASET = true_labels_data

    ZS_CLEAN_PREDS_DATASET = zero_shot_clean_preds_data
    ZS_ADV_PREDS_DATASET = zero_shot_adv_preds_data
    ZS_ADV_IMAGE_ONLY_PREDS_DATASET = zero_shot_adv_image_only_preds_data

    # Vanilla
    ZS_CLEAN_PREDS_VANILLA_DATASET = zero_shot_clean_preds_vanilla_data
    ZS_ADV_PREDS_VANILLA_DATASET = zero_shot_adv_preds_vanilla_data
    ZS_ADV_IMAGE_ONLY_PREDS_VANILLA_DATASET = zero_shot_adv_image_only_preds_vanilla_data

    # Weighted
    ZS_CLEAN_PREDS_WEIGHTED_DATASET = zero_shot_clean_preds_weighted_data
    ZS_ADV_PREDS_WEIGHTED_DATASET = zero_shot_adv_preds_weighted_data
    ZS_ADV_IMAGE_ONLY_PREDS_WEIGHTED_DATASET = zero_shot_adv_image_only_preds_weighted_data

    return TRUE_LABELS_DATASET, ZS_CLEAN_PREDS_DATASET, ZS_ADV_PREDS_DATASET, ZS_ADV_IMAGE_ONLY_PREDS_DATASET

import argparse
import json
import os
import re
from typing import Any, Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np


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
        "true_labels": None,
        "original_clean_predictions": None,
        "adversarial_predictions": None,
        "counter_attack_predictions": None,
        "metrics_json_present": False,
        "metrics_json_error": None,
    }

    if not os.path.isfile(metrics_path):
        return metrics

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception as e:
        metrics["metrics_json_error"] = str(e)
        return metrics

    metrics.update({
        "diff_ratio_after_counter_attack": data.get("diff_ratio_after_counter_attack", None),
        "avg_diff_ratio_after_counter_attack": data.get("avg_diff_ratio_after_counter_attack", None),
        "original_clean_accuracy": data.get("original_clean_accuracy", None),
        "adversarial_accuracy": data.get("adversarial_accuracy", None),
        "counter_attack_accuracy": data.get("counter_attack_accuracy", None),
        "true_labels": data.get("true_labels", None),
        "original_clean_predictions": data.get("original_clean_predictions", None),
        "adversarial_predictions": data.get("adversarial_predictions", None),
        "counter_attack_predictions": data.get("counter_attack_predictions", None),
        "metrics_json_present": True,
        "metrics_json_error": None,
    })
    return metrics



METRICS_FILENAME = "diff_ratio_after_counter_attack.json"

# Mapping of dataset folder names to more descriptive labels for plots (if needed).
ATTACK_NAME_MAPPING = {
    "eps_0.0_steps_0": "Clean",
    "eps_4.0_steps_100": "PGD 4/255 (100 steps)",
}




# Traverse the root directory to aggregate experiment results.
# Experiments are expected to be organized as: root/dataset/experiment_folder/metrics.json
def get_aggregated_results(root: str, selected_attacks: Optional[List[str]] = None) -> dict:
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
            
            # Mapping: 'eps_0.0_steps_0_image_only_attack_prm' -> 'eps_0.0_steps_0'
            if attack == "eps_0.0_steps_0_image_only_attack_prm":
                attack = "eps_0.0_steps_0"

            # Filtering: if selected_attacks is provided, only include those attacks
            if selected_attacks is not None and attack not in selected_attacks:
                continue

            noise_type = params["noise_type"]
            noise_param_obj = params["noise_param"]  # {"name": "Sigma", "value": 0.03}
            tau_type = params["tau_type"]

            # # Filter: remove or don't add values which have Noise uniform and value is 48.0
            # if noise_type.lower() == "uniform" and noise_param_obj["value"] == 48.0:
            #     continue

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

    return aggregated

def format_tau_label_from_dr(dr_label: str):
    """
    Convert DR_* label into a clean tau-based mathematical label.

    Examples:
    DR_uniform_Eps_1.0_normal_anchors -> τ(𝒰, ε = 1/255)
    DR_gaussian_Sigma_0.06_noisy      -> τ(𝒩, σ = 0.06)
    """

    parts = dr_label.split("_")

    if len(parts) < 4 or parts[0] != "DR":
        raise ValueError(f"Invalid DR label format: {dr_label}")

    noise_type = parts[1].lower()
    param_name = parts[2]
    param_value = parts[3]

    # Noise distribution
    if noise_type == "gaussian":
        noise_symbol = r"\mathcal{N}"
    elif noise_type == "uniform":
        noise_symbol = r"\mathcal{U}"
    else:
        noise_symbol = noise_type.capitalize()

    # Noise parameter
    if param_name == "Sigma":
        # keep sigma as-is
        param_str = rf"\sigma = {param_value}"
    elif param_name == "Eps":
        # convert to k/255
        k = int(float(param_value))
        param_str = rf"\epsilon = {k}/255"
    else:
        param_str = f"{param_name} = {param_value}"

    return rf"Threshold $\tau({noise_symbol},\ {param_str})$"

def format_tau_label_from_ca(dr_label: str):
    """
    Convert DR_* label into a clean tau-based mathematical label.

    Examples:
    DR_uniform_Eps_1.0_normal_anchors -> τ(𝒰, ε = 1/255)
    DR_gaussian_Sigma_0.06_noisy      -> τ(𝒩, σ = 0.06)
    """

    parts = dr_label.split("_")

    if len(parts) < 4 or parts[0] != "CA":
        raise ValueError(f"Invalid CA label format: {dr_label}")

    noise_type = parts[1].lower()
    param_name = parts[2]
    param_value = parts[3]

    # Noise distribution
    if noise_type == "gaussian":
        noise_symbol = r"\mathcal{N}"
    elif noise_type == "uniform":
        noise_symbol = r"\mathcal{U}"
    else:
        noise_symbol = noise_type.capitalize()

    # Noise parameter
    if param_name == "Sigma":
        # keep sigma as-is
        param_str = rf"\sigma = {param_value}"
    elif param_name == "Eps":
        # convert to k/255
        k = int(float(param_value))
        param_str = rf"\epsilon = {k}/255"
    else:
        param_str = f"{param_name} = {param_value}"

    return rf"Additive Noise $({noise_symbol},\ {param_str})$"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_diff_ratio",
        type=str,
        default="../../Diffratio_Adv_gen_Results/vit_l_14_datacomp_1b",
        help="Root path containing dataset subfolders (e.g., Diffratio_Adv_gen_Results/vit_l_14_datacomp_1b)",
    )
    parser.add_argument(
        "--out_diff_ratio",
        type=str,
        default="aggregated_results_with_diff_ratio.json",
        help="Output JSON filename (written under --root unless absolute path)",
    )
    parser.add_argument(
        "--root_results",
        type=str,
        default="../../Diffratio_Adv_gen_Results_v2/vit_l_14_datacomp_1b",
        help="Root path containing dataset subfolders (e.g., Diffratio_Adv_gen_Results/vit_l_14_datacomp_1b)",
    )

    parser.add_argument(
        "--out_results",
        type=str,
        default="aggregated_results_v2.json",
        help="Output JSON filename (written under --root unless absolute path)",
    )
    args = parser.parse_args()

    root_diff_ratio = os.path.abspath(args.root_diff_ratio)
    root_results = os.path.abspath(args.root_results)
    selected_attacks = ['eps_0.0_steps_0', 'eps_4.0_steps_100']
    # 1. Load data from two different result directories (e.g., different noise configurations).
    print(f"Loading diff ratio results from: {root_diff_ratio}")
    diff_ratio_dic = get_aggregated_results(root_diff_ratio, selected_attacks=selected_attacks)
    print(f"Loading counter-attack results from: {root_results}")
    results_dic = get_aggregated_results(root_results, selected_attacks=selected_attacks)

    # 2. Load Zero-Shot baseline results for comparison.
    print("Loading Zero-Shot results...")
    TRUE_LABELS_DATASET, ZS_CLEAN_PREDS_DATASET, ZS_ADV_PREDS_DATASET, ZS_ADV_IMAGE_ONLY_PREDS_DATASET = get_zs_results("vit_l_14_datacomp_1b")
    print("Zero-Shot results loaded.")

    # Helper to compute accuracy percentage from predictions and labels.
    def compute_accuracy(preds, labels):
        if preds is None or labels is None:
            return 0.0
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        if len(preds) == 0:
            return 0.0
        return (preds == labels).mean() * 100.0

    # Range of thresholds for the Diff Ratio.
    # For each sample: if sample_diff_ratio < threshold, use ZS prediction; otherwise, use Counter-Attack prediction.
    thresholds = [0.0, 0.1,  0.2, 0.3,  0.4,  0.5, 0.6, 0.7,  0.8, 0.85, 0.9,  1.0]
    
    model_name = "vit_l_14_datacomp_1b"
    datasets = list(results_dic['results'][model_name].keys())
    
    # Structure to hold results: dataset -> attack -> dr_label -> ca_label -> threshold -> list of accuracies
    final_results = {}

    # Iterate through each dataset to process samples.
    print(f"\nProcessing {len(datasets)} datasets...")
    for dataset in datasets:
        print(f"  Dataset: {dataset}")
        true_labels = TRUE_LABELS_DATASET.get(dataset)
        if true_labels is None:
            print(f"    Warning: No true labels found for {dataset}. Skipping.")
            continue
            
        final_results.setdefault(dataset, {})
            
        for attack in selected_attacks:
            if attack not in results_dic['results'][model_name][dataset]:
                continue
            
            print(f"    Attack: {attack}")
            
            # Select the appropriate Zero-Shot baseline based on the attack type.
            if attack == 'eps_0.0_steps_0':
                zs_preds = ZS_CLEAN_PREDS_DATASET.get(dataset)
            else:
                zs_preds = ZS_ADV_PREDS_DATASET.get(dataset)
                
            if zs_preds is None:
                continue
                
            final_results[dataset].setdefault(attack, {})
            
            # Process each counter-attack configuration in results_dic.
            res_dataset_attack = results_dic['results'].get(model_name, {}).get(dataset, {}).get(attack, {})
            
            for res_noise_type in res_dataset_attack:
                res_noise_params = res_dataset_attack[res_noise_type]
                for res_noise_param in res_noise_params:
                    res_tau_types = res_noise_params[res_noise_param]
                    for res_tau_type in res_tau_types:
                        res_entry = res_tau_types[res_tau_type]
                        ca_preds = res_entry.get('counter_attack_predictions')
                        
                        ca_label = f"CA_{res_noise_type}_{res_noise_param}_{res_tau_type}"
                        print(f"      Counter-Attack: {ca_label}")
                        
                        if ca_preds is None:
                            print(f"        Warning: No counter-attack predictions found for {ca_label}. Skipping.")
                            continue
                            
                        # Now iterate over each noise configuration in diff_ratio_dic for diff ratios.
                        dr_noise_types = diff_ratio_dic['results'].get(model_name, {}).get(dataset, {}).get(attack, {})
                        
                        for noise_type in dr_noise_types:
                            dr_noise_params = dr_noise_types[noise_type]
                            for noise_param in dr_noise_params:
                                dr_tau_types = dr_noise_params[noise_param]
                                for tau_type in dr_tau_types:
                                    dr_label = f"DR_{noise_type}_{noise_param}_{tau_type}"
                                    print(f"        Diff-Ratio: {dr_label}")
                                    
                                    dr_entry = dr_tau_types[tau_type]
                                    diff_ratios = dr_entry.get('diff_ratio_after_counter_attack')
                                    
                                    if diff_ratios is None:
                                        print(f"          Warning: No diff ratios found for {dr_label}. Skipping.")
                                        continue

                                    # Verify data consistency
                                    if len(ca_preds) != len(diff_ratios) or len(ca_preds) != len(zs_preds):
                                        print(f"          Warning: Length mismatch. ca_preds: {len(ca_preds)}, diff_ratios: {len(diff_ratios)}, zs_preds: {len(zs_preds)}. Skipping.")
                                        continue

                                    # To avoid huge nested dicts, let's name the keys appropriately.
                                    # dr_label = f"DR_{noise_type}_{noise_param}_{tau_type}"
                                    # ca_label = f"CA_{res_noise_type}_{res_noise_param}_{res_tau_type}"
                                    
                                    final_results[dataset][attack].setdefault(dr_label, {})
                                    final_results[dataset][attack][dr_label].setdefault(ca_label, {})

                                    # Apply thresholding logic per sample.
                                    for threshold in thresholds:
                                        combined_preds = [
                                            zs_preds[i] if diff_ratios[i] < threshold else ca_preds[i]
                                            for i in range(len(zs_preds))
                                        ]
                                        acc = compute_accuracy(combined_preds, true_labels)
                                        final_results[dataset][attack][dr_label][ca_label].setdefault(threshold, [])
                                        final_results[dataset][attack][dr_label][ca_label][threshold].append(acc)
    
    # 3. Create final_results_average_dic by averaging across datasets.
    print("\nAveraging results across datasets...")
    final_results_average_dic = {}
    
    for dataset in final_results:
        for attack in final_results[dataset]:
            final_results_average_dic.setdefault(attack, {})
            for dr_label in final_results[dataset][attack]:
                final_results_average_dic[attack].setdefault(dr_label, {})
                for ca_label in final_results[dataset][attack][dr_label]:
                    final_results_average_dic[attack][dr_label].setdefault(ca_label, {})
                    for threshold in final_results[dataset][attack][dr_label][ca_label]:
                        final_results_average_dic[attack][dr_label][ca_label].setdefault(threshold, [])
                        # Extend the list of accuracies for this config across all datasets
                        final_results_average_dic[attack][dr_label][ca_label][threshold].extend(
                            final_results[dataset][attack][dr_label][ca_label][threshold]
                        )

    # Compute the mean for each configuration
    for attack in final_results_average_dic:
        for dr_label in final_results_average_dic[attack]:
            for ca_label in final_results_average_dic[attack][dr_label]:
                for threshold in final_results_average_dic[attack][dr_label][ca_label]:
                    acc_list = final_results_average_dic[attack][dr_label][ca_label][threshold]
                    if acc_list:
                        final_results_average_dic[attack][dr_label][ca_label][threshold] = np.mean(acc_list)
                    else:
                        final_results_average_dic[attack][dr_label][ca_label][threshold] = 0.0

    print("Averaging complete.")
    
    # --- PLOTTING LOGIC ---
    print("\nGenerating bar plots...")
    
    # Create base output directory
    output_base_dir = "plots_threshold_bar"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # We want to plot thresholds on x-axis, average accuracy on y-axis.
    # For each (dr_label, ca_label) pair, we create one plot.
    # Each plot will show 'Clean' and 'PGD 4/255 (100 steps)' accuracies.
    
    # Identify all (dr_label, ca_label) combinations across all attacks
    combinations = set()
    ca_to_drs = {} # Map ca_label to list of dr_labels for grid plotting
    
    # Tracking best configurations per CA label
    # ca_label -> {'net': (score, dr_label), 'clean': (score, dr_label), 'pgd': (score, dr_label)}
    best_configs = {}

    for attack in final_results_average_dic:
        for dr_label in final_results_average_dic[attack]:
            for ca_label in final_results_average_dic[attack][dr_label]:
                combinations.add((dr_label, ca_label))
                if ca_label not in ca_to_drs:
                    ca_to_drs[ca_label] = set()
                ca_to_drs[ca_label].add(dr_label)
                
                # Initialize best tracking for this ca_label
                if ca_label not in best_configs:
                    best_configs[ca_label] = {
                        'net': (-1.0, None),
                        'clean': (-1.0, None),
                        'pgd': (-1.0, None)
                    }
                
                # Calculate max scores for this (dr_label, ca_label)
                # Note: We need both attacks to compute Net Avg properly
                attack_keys = ["eps_0.0_steps_0", "eps_4.0_steps_100"]
                if all(ak in final_results_average_dic and dr_label in final_results_average_dic[ak] and ca_label in final_results_average_dic[ak][dr_label] for ak in attack_keys):
                    clean_accs = [final_results_average_dic[attack_keys[0]][dr_label][ca_label][t] for t in thresholds]
                    pgd_accs = [final_results_average_dic[attack_keys[1]][dr_label][ca_label][t] for t in thresholds]
                    net_accs = [(c + p) / 2.0 for c, p in zip(clean_accs, pgd_accs)]
                    
                    max_net = max(net_accs)
                    max_clean = max(clean_accs)
                    max_pgd = max(pgd_accs)
                    
                    if max_net > best_configs[ca_label]['net'][0]:
                        best_configs[ca_label]['net'] = (max_net, dr_label)
                    if max_clean > best_configs[ca_label]['clean'][0]:
                        best_configs[ca_label]['clean'] = (max_clean, dr_label)
                    if max_pgd > best_configs[ca_label]['pgd'][0]:
                        best_configs[ca_label]['pgd'] = (max_pgd, dr_label)
    
    # Sort for consistency
    for ca in ca_to_drs:
        ca_to_drs[ca] = sorted(list(ca_to_drs[ca]))
                
    for dr_label, ca_label in combinations:
        plt.figure(figsize=(20, 10))
        
        # Prepare data for this combination
        # x_values: thresholds
        # y_values: accuracies for each attack
        
        # Determine available thresholds (should be same for all)
        sorted_thresholds = sorted(thresholds)
        x = np.arange(len(sorted_thresholds))
        width = 0.4  # width of the bars
        
        found_data = False
        
        # Store accuracies for net average calculation
        # attack_key -> list of accuracies across thresholds
        attack_accs = {}
        
        # Plot bars for each attack
        # We use ATTACK_NAME_MAPPING to identify 'Clean' and 'PGD'
        # eps_0.0_steps_0 -> Clean
        # eps_4.0_steps_100 -> PGD 4/255 (100 steps)
        
        attack_keys = ["eps_0.0_steps_0", "eps_4.0_steps_100"]
        colors = ['skyblue', 'salmon']
        
        for i, attack_key in enumerate(attack_keys):
            if attack_key in final_results_average_dic and \
               dr_label in final_results_average_dic[attack_key] and \
               ca_label in final_results_average_dic[attack_key][dr_label]:
                
                accs = [final_results_average_dic[attack_key][dr_label][ca_label][t] for t in sorted_thresholds]
                attack_accs[attack_key] = accs
                label = ATTACK_NAME_MAPPING.get(attack_key, attack_key)
                
                # Plot Bars
                offset = (i - 0.5) * width
                bars = plt.bar(x + offset, accs, width, label=f"{label}", color=colors[i], alpha=0.7)
                
                # Add text value on the bar
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                             f'{height:.1f}', ha='center', va='bottom', fontsize=16)
                
                found_data = True

        if not found_data:
            plt.close()
            continue

        # Plot Net Accuracy Line (average of Clean and PGD)
        if len(attack_accs) == len(attack_keys):
            # Calculate net average accuracy across thresholds
            net_accs = [
                (attack_accs[attack_keys[0]][j] + attack_accs[attack_keys[1]][j]) / 2.0
                for j in range(len(sorted_thresholds))
            ]
            
            plt.plot(x, net_accs, marker='o', color='purple', label="Net Average", linewidth=2)
            
            # Add line plots for Clean and Adversarial alone
            if attack_keys[0] in attack_accs:
                offset_clean = (0 - 0.5) * width
                plt.plot(x + offset_clean, attack_accs[attack_keys[0]], marker='s', color='blue', label="Clean", linewidth=1.5, linestyle='--')
            if attack_keys[1] in attack_accs:
                offset_pgd = (1 - 0.5) * width
                plt.plot(x + offset_pgd, attack_accs[attack_keys[1]], marker='^', color='red', label="Adversarial", linewidth=1.5, linestyle='--')
            
            # Highlight the dot with the highest score and mention the threshold
            max_net_acc = max(net_accs)
            max_indices = [idx for idx, val in enumerate(net_accs) if val == max_net_acc]
            for max_idx in max_indices:
                plt.plot(x[max_idx], net_accs[max_idx], marker='*', color='gold', markersize=15, markeredgecolor='black', zorder=5)
                # Annotate with threshold value
                # plt.annotate(f"T: {sorted_thresholds[max_idx]}",
                #              xy=(x[max_idx], net_accs[max_idx]),
                #              xytext=(0, 10),
                #              textcoords='offset points',
                #              ha='center',
                #              fontweight='bold',
                #              color='purple',
                #              fontsize=14)

            # Highlight Clean line max
            if attack_keys[0] in attack_accs:
                clean_accs = attack_accs[attack_keys[0]]
                max_clean_acc = max(clean_accs)
                max_clean_indices = [idx for idx, val in enumerate(clean_accs) if val == max_clean_acc]
                offset_clean = (0 - 0.5) * width
                for max_idx in max_clean_indices:
                    plt.plot(x[max_idx] + offset_clean, clean_accs[max_idx], marker='*', color='gold', markersize=15, markeredgecolor='black', zorder=5)
                    # plt.annotate(f"T: {sorted_thresholds[max_idx]}",
                    #              xy=(x[max_idx] + offset_clean, clean_accs[max_idx]),
                    #              xytext=(0, -15),
                    #              textcoords='offset points',
                    #              ha='center',
                    #              fontweight='bold',
                    #              color='blue',
                    #              fontsize=9)

            # Highlight PGD line max
            if attack_keys[1] in attack_accs:
                pgd_accs = attack_accs[attack_keys[1]]
                max_pgd_acc = max(pgd_accs)
                max_pgd_indices = [idx for idx, val in enumerate(pgd_accs) if val == max_pgd_acc]
                offset_pgd = (1 - 0.5) * width
                for max_idx in max_pgd_indices:
                    plt.plot(x[max_idx] + offset_pgd, pgd_accs[max_idx], marker='*', color='gold', markersize=15, markeredgecolor='black', zorder=5)
                    # plt.annotate(f"T: {sorted_thresholds[max_idx]}",
                    #              xy=(x[max_idx] + offset_pgd, pgd_accs[max_idx]),
                    #              xytext=(0, 10),
                    #              textcoords='offset points',
                    #              ha='center',
                    #              fontweight='bold',
                    #              color='red',
                    #              fontsize=9)
                             
        elif len(attack_accs) > 0:
            # If only one attack is present, just plot its line (fallback)
            for attack_key, accs in attack_accs.items():
                label = ATTACK_NAME_MAPPING.get(attack_key, attack_key)
                plt.plot(x, accs, marker='o', label=f"{label} (Line)", linewidth=2)
                
                max_acc = max(accs)
                max_indices = [idx for idx, val in enumerate(accs) if val == max_acc]
                for max_idx in max_indices:
                    plt.plot(x[max_idx], accs[max_idx], marker='*', color='gold', markersize=15, markeredgecolor='black', zorder=5)
                    # plt.annotate(f"Threshold: {sorted_thresholds[max_idx]}",
                    #              xy=(x[max_idx], accs[max_idx]),
                    #              xytext=(0, 10),
                    #              textcoords='offset points',
                    #              ha='center',
                    #              fontweight='bold')
            
        plt.xlabel(rf"Threshold $\tau$", fontsize=24)
        plt.ylabel('Average Accuracy (%)', fontsize=24)
        plt.title(f'{format_tau_label_from_ca(ca_label)}\n{format_tau_label_from_dr(dr_label)}', fontsize=20)
        plt.xticks(x, [str(t) for t in sorted_thresholds], fontsize=20)
        # Calculate dynamic y-limits
        all_values = []
        for acc_list in attack_accs.values():
            all_values.extend(acc_list)
        if 'net_accs' in locals() or 'net_accs' in globals():
            all_values.extend(net_accs)
        
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            plt.ylim(max(min_val - 10, 0), max_val + 10)
        else:
            plt.ylim(40, 80)
        
        plt.legend(fontsize=18)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot in structured directory
        plot_dir = os.path.join(output_base_dir, ca_label, dr_label)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "accuracy_vs_threshold.png")
        
        plt.savefig(plot_path)
        
        # Check if this is a "best" plot for the current ca_label and save with specific name
        if ca_label in best_configs:
            if dr_label == best_configs[ca_label]['net'][1]:
                best_net_path = os.path.join(output_base_dir, ca_label, "best_net_avg.png")
                plt.savefig(best_net_path)
                print(f"  Saved best net plot: {best_net_path}")
            if dr_label == best_configs[ca_label]['clean'][1]:
                best_clean_path = os.path.join(output_base_dir, ca_label, "best_clean.png")
                plt.savefig(best_clean_path)
                print(f"  Saved best clean plot: {best_clean_path}")
            if dr_label == best_configs[ca_label]['pgd'][1]:
                best_pgd_path = os.path.join(output_base_dir, ca_label, "best_adversarial.png")
                plt.savefig(best_pgd_path)
                print(f"  Saved best adversarial plot: {best_pgd_path}")

        plt.close()
        print(f"  Saved plot: {plot_path}")
        
    print("\nAll individual plots generated.")
    
    # --- GRID PLOTTING LOGIC ---
    print("\nGenerating grid plots per CA label...")
    for ca_label in sorted(ca_to_drs.keys()):
        dr_labels = ca_to_drs[ca_label]
        num_plots = len(dr_labels)
        if num_plots == 0:
            continue
            
        # Determine grid size
        cols = 4
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 15, rows * 12), squeeze=False)
        fig.suptitle(f'Grid Plot: Accuracy vs Threshold | CA: {ca_label}', fontsize=32, fontweight='bold')
        
        for idx, dr_label in enumerate(dr_labels):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            
            sorted_thresholds = sorted(thresholds)
            x = np.arange(len(sorted_thresholds))
            width = 0.4
            
            attack_accs = {}
            attack_keys = ["eps_0.0_steps_0", "eps_4.0_steps_100"]
            colors = ['skyblue', 'salmon']
            
            for i, attack_key in enumerate(attack_keys):
                if attack_key in final_results_average_dic and \
                   dr_label in final_results_average_dic[attack_key] and \
                   ca_label in final_results_average_dic[attack_key][dr_label]:
                    
                    accs = [final_results_average_dic[attack_key][dr_label][ca_label][t] for t in sorted_thresholds]
                    attack_accs[attack_key] = accs
                    label = ATTACK_NAME_MAPPING.get(attack_key, attack_key)
                    
                    offset = (i - 0.5) * width
                    bars = ax.bar(x + offset, accs, width, label=f"{label}", color=colors[i], alpha=0.7)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            if len(attack_accs) == len(attack_keys):
                net_accs = [(attack_accs[attack_keys[0]][j] + attack_accs[attack_keys[1]][j]) / 2.0 for j in range(len(sorted_thresholds))]
                ax.plot(x, net_accs, marker='o', color='purple', label="Net Average", linewidth=2)
                
                offset_clean = (0 - 0.5) * width
                ax.plot(x + offset_clean, attack_accs[attack_keys[0]], marker='s', color='blue', label="Clean", linewidth=1.5, linestyle='--')
                
                offset_pgd = (1 - 0.5) * width
                ax.plot(x + offset_pgd, attack_accs[attack_keys[1]], marker='^', color='red', label="PGD", linewidth=1.5, linestyle='--')
                
                # Highlight Net Max
                max_net_acc = max(net_accs)
                for max_idx in [i for i, v in enumerate(net_accs) if v == max_net_acc]:
                    ax.plot(x[max_idx], net_accs[max_idx], marker='*', color='gold', markersize=15, markeredgecolor='black', zorder=5)
                    ax.annotate(f"T: {sorted_thresholds[max_idx]}", xy=(x[max_idx], net_accs[max_idx]), xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold', color='purple', fontsize=10)
                
                # Highlight Clean Max
                clean_accs = attack_accs[attack_keys[0]]
                max_clean_acc = max(clean_accs)
                for max_idx in [i for i, v in enumerate(clean_accs) if v == max_clean_acc]:
                    ax.plot(x[max_idx] + offset_clean, clean_accs[max_idx], marker='*', color='gold', markersize=12, markeredgecolor='black', zorder=5)
                
                # Highlight PGD Max
                pgd_accs = attack_accs[attack_keys[1]]
                max_pgd_acc = max(pgd_accs)
                for max_idx in [i for i, v in enumerate(pgd_accs) if v == max_pgd_acc]:
                    ax.plot(x[max_idx] + offset_pgd, pgd_accs[max_idx], marker='*', color='gold', markersize=12, markeredgecolor='black', zorder=5)

            # Calculate dynamic y-limits for subplot
            all_values_subplot = []
            for acc_list in attack_accs.values():
                all_values_subplot.extend(acc_list)
            if 'net_accs' in locals():
                all_values_subplot.extend(net_accs)
            
            if all_values_subplot:
                min_val = min(all_values_subplot)
                max_val = max(all_values_subplot)
                ax.set_ylim(min_val - 10, max_val + 10)
            else:
                ax.set_ylim(0, 100)

            ax.set_xlabel('Threshold')
            ax.set_ylabel('Average Accuracy (%)')
            ax.set_title(f'{dr_label}', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels([str(t) for t in sorted_thresholds])
            ax.legend(fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        # Hide empty subplots
        for i in range(idx + 1, rows * cols):
            axes.flatten()[i].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        grid_plot_path = os.path.join(output_base_dir, ca_label, "grid_accuracy_vs_threshold.png")
        plt.savefig(grid_plot_path)
        plt.close()
        print(f"  Saved grid plot: {grid_plot_path}")

    print("\nAll grid plots generated.")
    
    # Optional: Print some results to verify
    for attack in final_results_average_dic:
        print(f"\nAttack: {attack}")
        for dr_label in list(final_results_average_dic[attack].keys())[:1]: # Print first DR label only
            for ca_label in list(final_results_average_dic[attack][dr_label].keys())[:1]: # Print first CA label only
                print(f"  Configuration: {dr_label} | {ca_label}")
                for threshold in sorted(final_results_average_dic[attack][dr_label][ca_label].keys()):
                    avg_acc = final_results_average_dic[attack][dr_label][ca_label][threshold]
                    print(f"    Threshold {threshold}: {avg_acc:.2f}%")
    
    



if __name__ == "__main__":
    main()
