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

    return aggregated


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
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    model_name = "vit_l_14_datacomp_1b"
    datasets = list(results_dic['results'][model_name].keys())
    
    # Structure to hold results: attack -> noise_type -> noise_param -> threshold -> list of accuracies (one per dataset)
    final_results = {}

    # Iterate through each dataset to process samples.
    print(f"\nProcessing {len(datasets)} datasets...")
    for dataset in datasets:
        print(f"  Dataset: {dataset}")
        true_labels = TRUE_LABELS_DATASET.get(dataset)
        if true_labels is None:
            print(f"    Warning: No true labels found for {dataset}. Skipping.")
            continue
            
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
                
            final_results.setdefault(attack, {})
            
            # Process each noise configuration in diff_ratio_dic.
            # We iterate over diff_ratio_dic to use all available diff ratios for thresholding.
            dr_noise_types = diff_ratio_dic['results'].get(model_name, {}).get(dataset, {}).get(attack, {})
            
            for noise_type in dr_noise_types:
                final_results[attack].setdefault(noise_type, {})
                
                dr_noise_params = dr_noise_types[noise_type]
                for noise_param in dr_noise_params:
                    dr_tau_types = dr_noise_params[noise_param]
                    for tau_type in dr_tau_types:
                        dr_entry = dr_tau_types[tau_type]
                        diff_ratios = dr_entry.get('diff_ratio_after_counter_attack')
                        
                        if diff_ratios is None:
                            continue
                            
                        # Now find the corresponding ca_preds in results_dic.
                        # We try to find a match for the same noise_type and tau_type.
                        # If the exact noise_param isn't found, we pick the first available noise_param in results_dic for that noise_type.
                        res_dataset_attack = results_dic['results'].get(model_name, {}).get(dataset, {}).get(attack, {})
                        res_noise_type_dict = res_dataset_attack.get(noise_type, {})
                        
                        ca_preds = None
                        matched_param = None
                        # Try exact match for noise_param first
                        if noise_param in res_noise_type_dict:
                            res_entry = res_noise_type_dict[noise_param].get(tau_type)
                            if res_entry:
                                ca_preds = res_entry.get('counter_attack_predictions')
                                matched_param = noise_param
                        
                        # Fallback: pick the first available noise_param for the same noise_type and tau_type
                        if ca_preds is None:
                            for any_param in res_noise_type_dict:
                                res_entry = res_noise_type_dict[any_param].get(tau_type)
                                if res_entry:
                                    ca_preds = res_entry.get('counter_attack_predictions')
                                    if ca_preds is not None:
                                        matched_param = any_param
                                        break
                                        
                        if ca_preds is None:
                            print(f"      No counter-attack predictions found for {noise_type} {tau_type}. Skipping.")
                            continue

                        print(f"      Matched: DR({noise_param}) -> CA({matched_param}) [{noise_type}, {tau_type}]")

                        # Verify data consistency across baseline, counter-attack, and diff ratios.
                        if len(ca_preds) != len(diff_ratios) or len(ca_preds) != len(zs_preds):
                            print(f"Warning: length mismatch for {dataset} {attack} {noise_type} {noise_param}")
                            continue
                            
                        final_results[attack][noise_type].setdefault(noise_param, {})
                        
                        # Apply thresholding logic per sample.
                        for threshold in thresholds:
                            combined_preds = [
                                zs_preds[i] if diff_ratios[i] < threshold else ca_preds[i]
                                for i in range(len(zs_preds))
                            ]
                            acc = compute_accuracy(combined_preds, true_labels)
                            # Store accuracy to later average across all datasets.
                            final_results[attack][noise_type][noise_param].setdefault(threshold, [])
                            final_results[attack][noise_type][noise_param][threshold].append(acc)

    # 3. Aggregate results across datasets and generate plots for each attack type.
    for attack in final_results:
        plt.figure(figsize=(12, 8))
        attack_label = ATTACK_NAME_MAPPING.get(attack, attack)
        
        for noise_type in final_results[attack]:
            for noise_param in final_results[attack][noise_type]:
                avg_accs = []
                valid_thresholds = []
                for threshold in thresholds:
                    acc_list = final_results[attack][noise_type][noise_param].get(threshold, [])
                    if acc_list:
                        avg_accs.append(np.mean(acc_list))
                        valid_thresholds.append(threshold)
                
                if avg_accs:
                    plt.plot(valid_thresholds, avg_accs, marker='o', label=f"{noise_type} {noise_param}")
        
        plt.title(f"Accuracy vs Diff Ratio Threshold ({attack_label})")
        plt.xlabel("Threshold (if diff_ratio < threshold use ZS, else CA)")
        plt.ylabel("Average Accuracy (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        safe_attack_name = attack.replace('.', '_')
        plot_path = f"accuracy_vs_threshold_{safe_attack_name}.png"
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        plt.close()

    # Save aggregated results to JSON
    output_data = {}
    for attack, nt_dict in final_results.items():
        output_data[attack] = {}
        for nt, np_dict in nt_dict.items():
            output_data[attack][nt] = {}
            for np_val, t_dict in np_dict.items():
                output_data[attack][nt][np_val] = {
                    str(t): np.mean(accs) for t, accs in t_dict.items() if accs
                }
    
    with open("threshold_analysis_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    print("Saved aggregated results to threshold_analysis_results.json")

if __name__ == "__main__":
    main()
