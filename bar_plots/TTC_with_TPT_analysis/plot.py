"""TPT analysis: aggregation + plotting.

High-level flow
----------------
1) Load zero-shot predictions for each dataset (single / vanilla / weighted).
2) Load TPT predictions for each dataset (e.g. `tpt`, `rtpt`).
3) Produce plots:
   - zero-shot avg-across-datasets
   - TPT avg-across-datasets
   - TPT avg-across-datasets includes both `tpt` and `rtpt` in the same plot

Folder layout reference (examples)
---------------------------------
- `Final_Results_corrected_ca_tau_Counter_Attack/vit_l_14_datacomp_1b/<Dataset Name>/...`
  contains json files like:
  - `results_original_clean.json`
  - `results_single.json`
  - `results_original.json`
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

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


def get_zs_results(model_name: str) -> Dict[str, Any]:
    """Load zero-shot predictions + labels for all datasets.

    This function reads the json files produced by your zero-shot runs and returns
    a single dictionary that is later used by the aggregation / plotting helpers.
    """

    # MODELS = [
    #     "delta_clip_l14_224",
    #     "fare4",
    #     "ViT-L/14",
    #     "vit_l_14_datacomp_1b",
    # ]
    # Keep the structure consistent with earlier scripts that supported multiple models.
    MODELS = [model_name]

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
            out[k] = obj[k]

        return out

    def diff_ratio(obj: dict) -> dict:
        # note: keeping your spelling "avergae_diff_ratio" as-is
        if "diff_ratio" not in obj or "avergae_diff_ratio" not in obj:
            raise KeyError("Expected keys: 'diff_ratio' and 'avergae_diff_ratio' (note spelling).")
        return {
            "diff_ratio_per_sample": obj["diff_ratio"],
            "diff_ratio_avg": obj["avergae_diff_ratio"],
        }

    # NOTE: Prefer the global `compute_accuracy` / `accuracy_percent` helpers below.

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
        preds["original_clean"] = _load_pred_json(obj)

        # NEW: correctness masks based on original_clean
        # --------------------------------------------------
        oc_preds = preds["original_clean"]["prediction"]
        oc_labels = preds["original_clean"]["label"]

        assert len(oc_preds) == len(oc_labels), \
            "Prediction and label lengths do not match in original_clean"

        original_clean_correct = [
            (p == y) for p, y in zip(oc_preds, oc_labels)
        ]


        preds["original_clean_correct"] = original_clean_correct


        return {"preds": preds}

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

    ZS_DATA = build_all_data(RESULT_PATHS, MODELS, DATASETS)

    import numpy as np

    case = "clean"
    model = model_name

    true_labels_data = {}

    zero_shot_clean_preds_data = {}
    zero_shot_clean_correct_preds = {}
    zero_shot_clean_max_confidences_data = {}

    zero_shot_clean_preds_vanilla_data = {}
    zero_shot_clean_max_confidences_vanilla_data = {}

    zero_shot_clean_preds_weighted_data = {}
    zero_shot_clean_max_confidences_weighted_data = {}

    for dataset in DATASETS:
        example = ZS_DATA[case][model][dataset]
        true_labels_data[dataset] = example["preds"]["original_clean"]["label"]

        zero_shot_clean_preds_data[dataset] = example["preds"]["original_clean"]["prediction"]
        zero_shot_clean_max_confidences_data[dataset] = example["preds"]["original"]["max_confidence"]
        zero_shot_clean_correct_preds[dataset] = example["preds"]["original_clean_correct"]

        zero_shot_clean_preds_vanilla_data[dataset] = example["preds"]["vanilla"]["prediction"]
        zero_shot_clean_max_confidences_vanilla_data[dataset] = example["preds"]["vanilla"]["max_confidence"]

        zero_shot_clean_preds_weighted_data[dataset] = example["preds"]["weighted"]["prediction"]
        zero_shot_clean_max_confidences_weighted_data[dataset] = example["preds"]["weighted"]["max_confidence"]

        # print accuracy
        print(dataset, compute_accuracy(zero_shot_clean_preds_data[dataset], true_labels_data[dataset]))
        # print mean confidence
        print(dataset, np.mean(zero_shot_clean_max_confidences_data[dataset]))

    case = "adversarial_eps4_steps100"
    model = model_name

    zero_shot_adv_preds_data = {}
    zero_shot_adv_max_confidences_data = {}

    zero_shot_adv_preds_vanilla_data = {}
    zero_shot_adv_max_confidences_vanilla_data = {}

    zero_shot_adv_preds_weighted_data = {}
    zero_shot_adv_max_confidences_weighted_data = {}

    for dataset in DATASETS:
        example = ZS_DATA[case][model][dataset]

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
    model = model_name

    zero_shot_adv_image_only_preds_data = {}
    zero_shot_adv_image_only_max_confidences_data = {}

    zero_shot_adv_image_only_preds_vanilla_data = {}
    zero_shot_adv_image_only_max_confidences_vanilla_data = {}

    zero_shot_adv_image_only_preds_weighted_data = {}
    zero_shot_adv_image_only_max_confidences_weighted_data = {}

    for dataset in DATASETS:
        example = ZS_DATA[case][model][dataset]

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

    ZS_ADV_PREDS_DATASET = zero_shot_adv_preds_data
    ZS_ADV_IMAGE_ONLY_PREDS_DATASET = zero_shot_adv_image_only_preds_data
    ZS_CLEAN_CORRECT_PREDS = zero_shot_clean_correct_preds

    ZS_CLEAN_PREDS_DATASET = zero_shot_clean_preds_data
    # Vanilla
    ZS_CLEAN_PREDS_VANILLA_DATASET = zero_shot_clean_preds_vanilla_data
    ZS_ADV_PREDS_VANILLA_DATASET = zero_shot_adv_preds_vanilla_data
    ZS_ADV_IMAGE_ONLY_PREDS_VANILLA_DATASET = zero_shot_adv_image_only_preds_vanilla_data

    # Weighted
    ZS_CLEAN_PREDS_WEIGHTED_DATASET = zero_shot_clean_preds_weighted_data
    ZS_ADV_PREDS_WEIGHTED_DATASET = zero_shot_adv_preds_weighted_data
    ZS_ADV_IMAGE_ONLY_PREDS_WEIGHTED_DATASET = zero_shot_adv_image_only_preds_weighted_data

    return_dic = {
        "zero_shot_clean": ZS_CLEAN_PREDS_DATASET,
        "zero_shot_adv": ZS_ADV_PREDS_DATASET,
        "zero_shot_adv_image_only": ZS_ADV_IMAGE_ONLY_PREDS_DATASET,
        "true_labels": TRUE_LABELS_DATASET,
        "zero_shot_clean_correct_preds": ZS_CLEAN_CORRECT_PREDS,
        "zero_shot_clean_vanilla": ZS_CLEAN_PREDS_VANILLA_DATASET,
        "zero_shot_adv_vanilla": ZS_ADV_PREDS_VANILLA_DATASET,
        "zero_shot_adv_image_only_vanilla": ZS_ADV_IMAGE_ONLY_PREDS_VANILLA_DATASET,
        "zero_shot_clean_weighted": ZS_CLEAN_PREDS_WEIGHTED_DATASET,
        "zero_shot_adv_weighted": ZS_ADV_PREDS_WEIGHTED_DATASET,
        "zero_shot_adv_image_only_weighted": ZS_ADV_IMAGE_ONLY_PREDS_WEIGHTED_DATASET,
    }

    return return_dic

def compute_accuracy(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return (preds == labels).mean() * 100.0


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

METRICS_FILENAME = "diff_ratio_after_counter_attack.json"

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

def get_tpt_results(model_name: str) -> Dict[str, Any]:
    """Load TPT predictions for all datasets.

    Returns a nested dictionary keyed as:
      tpt_dic[attack][model][dataset][tpt_type]["preds"][variant] -> prediction payload
    """

    # MODELS = [
    #     "delta_clip_l14_224",
    #     "fare4",
    #     "ViT-L/14",
    #     "vit_l_14_datacomp_1b",
    # ]
    # Keep the list wrapper for backward compatibility with earlier multi-model scripts.
    MODELS = [model_name]

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

    tpt_types = ["tpt", "rtpt"]

    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pred_json(obj: dict, *, allow_extra: bool = False) -> dict:
        # TPT prediction jsons can contain multiple keys; we keep them all.
        # (The downstream plotting code expects at least `prediction` and `label`.)
        return dict(obj)
    # NOTE: Prefer the global `compute_accuracy` / `accuracy_percent` helpers below.

    """
    ### ZERO SHOT Experiment

    We will load zero shot results for different models and datasets, under clean and adversarial conditions.
    1. We will store true labels for each sample for each dataset.
    2. We will store zero-shot single prediction results for each sample for each dataset under clean and adversarial conditions.
    3. We will store zero-shot max confidence scores for each sample for each dataset under clean and adversarial conditions.
    """

    # Base path templates for each case
    RESULT_PATHS = {
        "TPT_TYPE": {
            "clean": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau_Counter_Attack_then_TPT/"
                    "{model}/{dataset}/"
                    "Clean/Counter_Attack/Eps_4_0_Steps_5_Alpha_1_0/tau_100_0_beta_2_0_weighted_pertrubation_True/TPT/Optimization_Loss_{tpt_type}_LR_0_005_Optimization_Steps_1_View_Selection_Fraction_0_1/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            "adversarial_eps4_steps100": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau_Counter_Attack_then_TPT/"
                    "{model}/{dataset}/"
                    "Adversarial_Eps_4_0_Steps_100/"
                    "Counter_Attack/Eps_4_0_Steps_5_Alpha_1_0/tau_100_0_beta_2_0_weighted_pertrubation_True/TPT/Optimization_Loss_{tpt_type}_LR_0_005_Optimization_Steps_1_View_Selection_Fraction_0_1/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            # "adversarial_eps4_steps100_image_only": {
            #     "base_path": (
            #         "../../Final_Results_corrected_ca_tau_Counter_Attack_then_TPT/"
            #         "{model}/{dataset}/"
            #         "Adversarial_Eps_4_0_Steps_100_image_only_attack_prm/"
            #         "Counter_Attack/Eps_4_0_Steps_5_Alpha_1_0/tau_100_beta_2_0_weighted_pertrubation_True/No_TPT/"
            #         "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
            #     ),
            # },
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
        # "results_counter_attack_diff_ratio": "results_counter_attack_diff_ratio.json",
    }


    def load_one_setting(base_path: str) -> dict:
        """
        base_path: directory for a single (case, model, dataset)
        returns: normalized dict with preds + counter_attack
        """
        base = Path(base_path)

        # --- prediction jsons
        preds = {}
        for key in ["original"]:
            fp = base / JSON_KEYMAP[key]
            obj = _load_json(fp)
            preds[key] = _load_pred_json(obj)

        for key in ["single", "vanilla", "weighted"]:
            fp = base / JSON_KEYMAP[key]
            obj = _load_json(fp)
            preds[key] = _load_pred_json(obj)

        # --- original_clean json (special extras)
        fp = base / JSON_KEYMAP["original_clean"]
        obj = _load_json(fp)
        preds["original_clean"] = _load_pred_json(obj)

        # --- counter-attack diff-ratio json
        # fp = base / JSON_KEYMAP["results_counter_attack_diff_ratio"]
        # obj = _load_json(fp)
        # counter_attack = diff_ratio(obj)

        return {"preds": preds}

    def build_all_data(result_paths: dict, models: list, datasets: list, tpt_types: list) -> dict:
        DATA = {}

        for case, cfg in result_paths["TPT_TYPE"].items():
            DATA.setdefault(case, {})
            base_template = cfg["base_path"]

            for model in models:
                DATA[case].setdefault(model, {})

                for dataset in datasets:
                    DATA[case][model].setdefault(dataset, {})

                    for tpt_type in tpt_types:
                        base_path = base_template.format(
                            model=model,
                            dataset=dataset,
                            tpt_type=tpt_type,

                        )
                        DATA[case][model][dataset][tpt_type] = load_one_setting(base_path)

        return DATA

    TPT_DATA = build_all_data(RESULT_PATHS, MODELS, DATASETS, tpt_types)

    return TPT_DATA


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AOM analysis plotting / aggregation")
    parser.add_argument("--model-name", type=str, default="vit_l_14_datacomp_1b")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of dataset names to plot (e.g. 'DTD,Cars'). "
            "If omitted, all available datasets are used."
        ),
    )
    parser.add_argument(
        "--diff-ratio-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for diff-ratio based routing. If diff_ratio > threshold, use TPT prediction; "
            "otherwise use zero-shot prediction."
        ),
    )
    args = parser.parse_args()

    model_name = args.model_name
    diff_ratio_threshold = float(args.diff_ratio_threshold)

    zero_shot_dic = get_zs_results(model_name)
    # True labels for all datasets
    TRUE_LABELS_DATASET = zero_shot_dic["true_labels"]
    # Zero-Shot clean predictions for all datasets
    ZS_CLEAN_PREDS_DATASET = zero_shot_dic["zero_shot_clean"]
    # Zero-Shot adversarial predictions for all datasets
    ZS_ADV_PREDS_DATASET = zero_shot_dic["zero_shot_adv"]
    # Zero-Shot clean correct predictions for all datasets
    ZS_CLEAN_CORRECT_DATASET = zero_shot_dic["zero_shot_clean_correct_preds"]

    # Zero-shot clean vanilla predictions for all datasets
    ZS_CLEAN_PREDS_VANILLA_DATASET = zero_shot_dic["zero_shot_clean_vanilla"]
    # Zero-shot adversarial vanilla predictions for all datasets
    ZS_ADV_PREDS_VANILLA_DATASET = zero_shot_dic["zero_shot_adv_vanilla"]
    # Zero-shot clean weighted predictions for all datasets
    ZS_CLEAN_PREDS_WEIGHTED_DATASET = zero_shot_dic["zero_shot_clean_weighted"]
    # Zero-shot adversarial weighted predictions for all datasets
    ZS_ADV_PREDS_WEIGHTED_DATASET = zero_shot_dic["zero_shot_adv_weighted"]

    root_diff_ratio = "../../Diffratio_Adv_gen_Results/vit_l_14_datacomp_1b"
    selected_attacks = ['eps_0.0_steps_0', 'eps_4.0_steps_100']
    print(f"Loading diff ratio results from: {root_diff_ratio}")
    diff_ratio_dic = get_aggregated_results(root_diff_ratio, selected_attacks=selected_attacks)

    diff_ratio_dic = diff_ratio_dic["results"]["vit_l_14_datacomp_1b"]

    final_diff_ratio_dic = {}

    for dataset_key, dataset_value in diff_ratio_dic.items():
        final_diff_ratio_dic[dataset_key] = {}
        for attack_key, attack_value in dataset_value.items():
            if attack_key == 'eps_0.0_steps_0':
                attack_name = "Clean"
            else:
                attack_name = "Adversarial"
            final_diff_ratio_dic[dataset_key][attack_name] = {}
            for noise_type_key, noise_type_value in attack_value.items():
                if noise_type_key == 'gaussian':
                    noise_type_name = "Gaussian"
                else:
                    noise_type_name = "Uniform"
                final_diff_ratio_dic[dataset_key][attack_name][noise_type_name] = {}
                for noise_param_key, noise_param_value in noise_type_value.items():
                    for tau_type_key, tau_type_value in noise_param_value.items():
                        diff_ratio = tau_type_value.get("diff_ratio_after_counter_attack", None)
                        final_diff_ratio_dic[dataset_key][attack_name][noise_type_name][noise_param_key] = diff_ratio

    # ------------------------------------------------------------
    # TPT results
    # ------------------------------------------------------------
    tpt_dic = get_tpt_results(model_name)

    # ============================================================
    # Helpers + plotting utilities
    # ============================================================
    # Everything below is pure-Python manipulation / plotting.
    # The helpers are intentionally kept small and explicit so it is easy
    # to sanity-check aggregation behavior.

    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


    def compute_accuracy(preds, labels):
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        return (preds == labels).mean() * 100.0


    # ============================================================
    # Assumptions (already available in your runtime as you said)
    # ------------------------------------------------------------
    # TRUE_LABELS_DATASET: dict[dataset] -> list[int] ground-truth labels
    # ZS_CLEAN_PREDS_DATASET: dict[dataset] -> list[int] zero-shot clean preds
    # ZS_ADV_PREDS_DATASET: dict[dataset] -> list[int] zero-shot adv preds
    # tpt_dic: dict[attack][model][dataset][tpt_type]["preds"][pred_variant] -> dict with keys incl: "prediction", "label"
    # ============================================================

    # ----------------------------
    # Small utilities
    # ----------------------------
    def _as_np(x):
        return np.asarray(x)


    def _ensure_same_len(*arrays, name="arrays"):
        lens = [len(a) for a in arrays]
        if len(set(lens)) != 1:
            raise ValueError(f"Length mismatch in {name}: {lens}")


    def _attack_to_diff_ratio_attack_name(attack: str) -> str:
        if attack == "clean":
            return "Clean"
        if attack == "adversarial_eps4_steps100":
            return "Adversarial"
        raise KeyError(f"Unsupported attack='{attack}' for diff-ratio mapping")


    def _get_diff_ratio_per_sample(
            *,
            dataset: str,
            attack: str,
            noise_type: str = "Uniform",
            noise_value: str = "Eps_24.0",
    ):
        """Fetch per-sample diff-ratio array from `final_diff_ratio_dic`.

        `final_diff_ratio_dic` is built earlier as:
          final_diff_ratio_dic[dataset][{"Clean"|"Adversarial"}][{"Gaussian"|"Uniform"}][noise_value] -> list[float]
        """
        attack_name = _attack_to_diff_ratio_attack_name(attack)
        try:
            dr = final_diff_ratio_dic[dataset][attack_name][noise_type][noise_value]
        except KeyError as e:
            raise KeyError(
                f"Missing diff-ratio for dataset='{dataset}', attack='{attack_name}', noise_type='{noise_type}', noise_value='{noise_value}'. "
                f"Available keys (dataset-level)={list(final_diff_ratio_dic.get(dataset, {}).keys())}"
            ) from e
        return dr


    def _mix_preds_by_diff_ratio(
            *,
            diff_ratio_per_sample,
            threshold: float,
            zs_preds,
            tpt_preds,
    ):
        """Per-sample router: if diff_ratio > threshold => TPT else zero-shot."""
        _ensure_same_len(diff_ratio_per_sample, zs_preds, tpt_preds, name="diff_ratio vs preds")
        out = []
        for r, zs_p, tpt_p in zip(diff_ratio_per_sample, zs_preds, tpt_preds):
            # If diff-ratio is missing/None, default to zero-shot (conservative).
            if r is None:
                out.append(zs_p)
                continue
            try:
                rv = float(r)
            except Exception:
                out.append(zs_p)
                continue
            # out.append(tpt_p if (rv > threshold) else zs_p)
            out.append(tpt_p if (rv > threshold) else zs_p)

        return out


    def accuracy_percent(preds, labels):
        preds = _as_np(preds)
        labels = _as_np(labels)
        _ensure_same_len(preds, labels, name="preds vs labels")
        return float((preds == labels).mean() * 100.0)


    def avg_across_datasets(metric_by_dataset, datasets):
        """
        metric_by_dataset: dict[dataset] -> float
        Returns mean over datasets ignoring NaNs.
        """
        vals = [metric_by_dataset[d] for d in datasets]
        vals = np.asarray(vals, dtype=float)
        return float(np.nanmean(vals))


    def _get_zero_shot_preds_for_variant(*, attack: str, dataset: str, zs_variant: str):
        """Return zero-shot predictions for a given variant.

        zs_variant:
          - 'single'   -> ZS_*_PREDS_DATASET
          - 'vanilla'  -> ZS_*_PREDS_VANILLA_DATASET
          - 'weighted' -> ZS_*_PREDS_WEIGHTED_DATASET
        """
        if zs_variant not in {"single", "vanilla", "weighted"}:
            raise KeyError(f"Unsupported zs_variant='{zs_variant}'")

        if attack == "clean":
            if zs_variant == "single":
                return ZS_CLEAN_PREDS_DATASET[dataset]
            if zs_variant == "vanilla":
                return ZS_CLEAN_PREDS_VANILLA_DATASET[dataset]
            return ZS_CLEAN_PREDS_WEIGHTED_DATASET[dataset]

        if attack == "adversarial_eps4_steps100":
            if zs_variant == "single":
                return ZS_ADV_PREDS_DATASET[dataset]
            if zs_variant == "vanilla":
                return ZS_ADV_PREDS_VANILLA_DATASET[dataset]
            return ZS_ADV_PREDS_WEIGHTED_DATASET[dataset]

        raise KeyError(f"Unsupported attack='{attack}' for zero-shot baseline mapping")


    def save_bar_plot(
            path,
            title,
            labels,
            values,
            ylabel="Accuracy (%)",
            ylim=(0, 120),
            value_fmt="{:.2f}",
            colors=None,
            hatches=None,
            bar_edgecolor="#1a1a1a",
            bar_linewidth=0.7,
            grid_axis="y",
            grid_alpha=0.25,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # --- style (local, self-contained)
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.size": 11,
            "hatch.linewidth": 1.0,
        })

        # Default palette (colorblind-friendly-ish)
        # - prefer explicit mapping by common labels
        if colors is None:
            label_to_color = {
                "Clean": "#0072B2",        # Okabe-Ito blue
                "Adversarial": "#D55E00",  # Okabe-Ito vermillion
                "Zero-shot": "#4D4D4D",    # neutral gray
                "Single": "#0072B2",
                "Vanilla": "#009E73",     # green
                "Weighted": "#E69F00",    # orange

                # Expanded labels used by TPT plots
                "Zero-shot (Single)": "#0072B2",
                "Zero-shot (Vanilla)": "#009E73",
                "Zero-shot (Weighted)": "#E69F00",
                "TPT Single": "#0072B2",
                "TPT Vanilla": "#009E73",
                "TPT Weighted": "#E69F00",
            }
            colors = [label_to_color.get(str(l), "#56B4E9") for l in labels]  # fallback: sky blue

        if hatches is None:
            label_to_hatch = {
                "Clean": "///",
                "Adversarial": "\\\\",
                "Zero-shot": "",
                "Single": "///",
                "Vanilla": "xx",
                "Weighted": "..",
                "Zero-shot (Single)": "///",
                "Zero-shot (Vanilla)": "xx",
                "Zero-shot (Weighted)": "..",
                "TPT Single": "///",
                "TPT Vanilla": "xx",
                "TPT Weighted": "..",
            }
            hatches = [label_to_hatch.get(str(l), "") for l in labels]
        if len(hatches) != len(labels):
            raise ValueError("hatches must have same length as labels")

        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor=bar_edgecolor,
            linewidth=bar_linewidth,
            width=0.62,
            alpha=0.95,
        )

        for rect, h in zip(bars, hatches):
            if h:
                rect.set_hatch(h)

        # Grid (y only by default)
        ax.grid(True, axis=grid_axis, alpha=grid_alpha, linestyle="-")

        # Add value labels on top of bars
        y_min, y_max = ylim
        y_range = float(y_max - y_min) if (y_max is not None and y_min is not None) else None
        pad = 0.012 * y_range if y_range else 0.0
        for rect, v in zip(bars, values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + pad,
                value_fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=10,
                color="#222222",
                fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
                clip_on=True,
            )

        ax.set_xticks(x)
        # Rotate only if labels are long
        max_label_len = max((len(str(l)) for l in labels), default=0)
        if max_label_len >= 10:
            ax.set_xticklabels(labels, rotation=20, ha="right")
        else:
            ax.set_xticklabels(labels)

        ax.set_ylabel(ylabel, fontweight="semibold")
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="semibold", pad=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"[Saved] {path}")


    def save_grouped_bar_plot(
            path,
            title,
            x_labels,
            series,
            ylabel="Accuracy (%)",
            ylim=(0, 100),
            value_fmt="{:.2f}",
            colors=None,
            hatches=None,
            bar_edgecolor="#1a1a1a",
            bar_linewidth=0.7,
            grid_axis="y",
            grid_alpha=0.25,
            legend_loc="upper center",
    ):
        """
        Save grouped bar plot.

        x_labels: list[str] categories on x-axis.
        series: list of dicts, each:
          {"name": str, "values": list[float] (same length as x_labels)}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not series:
            raise ValueError("series must be non-empty")
        n = len(x_labels)
        for s in series:
            if len(s["values"]) != n:
                raise ValueError(f"Length mismatch in series '{s.get('name', '?')}'. Expected {n} values.")

        # --- style (local, self-contained)
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "font.size": 14,
            "hatch.linewidth": 1.0,
        })

        if colors is None:
            # Consistent with save_bar_plot; fixed mapping for common names
            name_to_color = {
                "tpt": "#0072B2",   # blue
                "rtpt": "#D55E00",  # vermillion
                "TPT": "#0072B2",
                "rTPT": "#D55E00",

                # requested unified baseline series
                "Zero-shot": "#4D4D4D",  # neutral gray
                "zero-shot": "#4D4D4D",

                # Zero-shot series names (avg plot)
                "Original": "#4D4D4D",  # neutral gray
                "Single": "#0072B2",    # blue
                "Vanilla": "#009E73",   # green
                "Weighted": "#E69F00",  # orange
            }
            colors = [name_to_color.get(str(s["name"]), "#56B4E9") for s in series]
        if len(colors) != len(series):
            raise ValueError("colors must have same length as series")

        if hatches is None:
            # Keep series visually separable in grayscale/print.
            base = ["///", "\\\\", "xx", "..", "++", "oo"]
            hatches = [base[i % len(base)] for i in range(len(series))]
        if len(hatches) != len(series):
            raise ValueError("hatches must have same length as series")

        fig, ax = plt.subplots(figsize=(24, 6))
        x = np.arange(n)

        # bar geometry
        total_width = 0.72
        bar_width = total_width / max(len(series), 1)
        offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * bar_width

        bars_by_series = []
        for i, s in enumerate(series):
            vals = np.asarray(s["values"], dtype=float)
            bars = ax.bar(
                x + offsets[i],
                vals,
                width=bar_width * 0.95,
                label=str(s["name"]),
                color=colors[i],
                edgecolor=bar_edgecolor,
                linewidth=bar_linewidth,
                hatch=hatches[i],
                alpha=0.95,
            )
            bars_by_series.append((bars, vals))

        ax.grid(True, axis=grid_axis, alpha=grid_alpha, linestyle="-")

        # annotate
        y_min, y_max = ylim
        y_range = float(y_max - y_min) if (y_max is not None and y_min is not None) else None
        pad = 0.012 * y_range if y_range else 0.0
        for (bars, vals) in bars_by_series:
            for rect, v in zip(bars, vals):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    rect.get_height() + pad,
                    value_fmt.format(float(v)),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="#222222",
                    clip_on=True,
                )

        ax.set_xticks(x)
        max_label_len = max((len(str(l)) for l in x_labels), default=0)
        if max_label_len >= 10:
            ax.set_xticklabels(x_labels,  ha="center")
        else:
            ax.set_xticklabels(x_labels)

        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.legend(loc=legend_loc, ncol=3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"[Saved] {path}")


    def save_line_plot(
            path,
            title,
            x,
            series,
            xlabel="tau threshold",
            ylabel="Average Accuracy (%)",
            ylim=(0, 100),
            xlim=(0.0, 1.0),
            grid_alpha=0.25,
            legend_loc="best",
    ):
        """Save a line plot.

        series: list of dicts: {"name": str, "y": list[float]}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not series:
            raise ValueError("series must be non-empty")

        x = np.asarray(x, dtype=float)
        for s in series:
            y = np.asarray(s["y"], dtype=float)
            if len(y) != len(x):
                raise ValueError(
                    f"Length mismatch in series '{s.get('name', '?')}'. Expected {len(x)} points, got {len(y)}."
                )

        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 12,
        })

        name_to_color = {
            "Zero-shot": "#4D4D4D",
            "TPT": "#0072B2",
            "R-TPT": "#D55E00",
        }

        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for s in series:
            name = str(s["name"])
            y = np.asarray(s["y"], dtype=float)
            ax.plot(
                x,
                y,
                label=name,
                linewidth=2.0,
                color=name_to_color.get(name, None),
            )

        # Ensure ticks correspond exactly to the provided threshold grid.
        # Without this, Matplotlib may auto-format labels (e.g., rounding 0.85 to 0.8).
        ax.set_xticks(x)
        xticklabels = []
        for xv in x:
            # Show 1 decimal when it is exact (e.g., 0.8), otherwise show 2 decimals (e.g., 0.85)
            if np.isclose(xv, round(float(xv), 1)):
                xticklabels.append(f"{float(xv):.1f}")
            else:
                xticklabels.append(f"{float(xv):.2f}")
        ax.set_xticklabels(xticklabels)

        ax.grid(True, alpha=grid_alpha, linestyle="-")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc=legend_loc)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"[Saved] {path}")


    def save_threshold_sweep_bar_plot(
            path,
            title,
            thresholds,
            series,
            xlabel="tau threshold",
            ylabel="Average Accuracy (%) across datasets",
            ylim=(0, 100),
            legend_loc="best",
            grid_alpha=0.25,
    ):
        """Save a grouped bar plot for a threshold sweep.

        thresholds: list[float]
        series: list of dicts: {"name": str, "y": list[float]}

        Note: Intended for *coarse* threshold grids (e.g., 0.0..1.0 with step 0.1).
        """
        thresholds = [float(x) for x in thresholds]
        # Use an adaptive formatter so values like 0.85 are not shown as 0.8.
        x_labels = [f"{x:.1f}" if np.isclose(x, round(x, 1)) else f"{x:.2f}" for x in thresholds]
        series_bar = [{"name": s["name"], "values": [float(y) for y in s["y"]]} for s in series]

        save_grouped_bar_plot(
            path,
            title,
            x_labels=x_labels,
            series=series_bar,
            ylabel=ylabel,
            ylim=ylim,
            legend_loc=legend_loc,
        )




    # ============================================================
    # 1) Zero-shot results (avg across datasets) [NO conservative]
    # ============================================================
    def compute_zero_shot_avg_results(datasets):
        per_ds = {}
        for ds in datasets:
            labels = TRUE_LABELS_DATASET[ds]

            # Original
            clean_preds = ZS_CLEAN_PREDS_DATASET[ds]
            adv_preds = ZS_ADV_PREDS_DATASET[ds]

            # Vanilla
            clean_preds_vanilla = ZS_CLEAN_PREDS_VANILLA_DATASET[ds]
            adv_preds_vanilla = ZS_ADV_PREDS_VANILLA_DATASET[ds]

            # Weighted
            clean_preds_weighted = ZS_CLEAN_PREDS_WEIGHTED_DATASET[ds]
            adv_preds_weighted = ZS_ADV_PREDS_WEIGHTED_DATASET[ds]

            _ensure_same_len(labels, clean_preds, adv_preds, name=f"{ds} zs original preds")
            _ensure_same_len(labels, clean_preds_vanilla, adv_preds_vanilla, name=f"{ds} zs vanilla preds")
            _ensure_same_len(labels, clean_preds_weighted, adv_preds_weighted, name=f"{ds} zs weighted preds")

            per_ds[ds] = {
                "num_samples": len(labels),
                "single": {
                    "zs_clean_acc": accuracy_percent(clean_preds, labels),
                    "zs_adv_acc": accuracy_percent(adv_preds, labels),
                },
                "vanilla": {
                    "zs_clean_acc": accuracy_percent(clean_preds_vanilla, labels),
                    "zs_adv_acc": accuracy_percent(adv_preds_vanilla, labels),
                },
                "weighted": {
                    "zs_clean_acc": accuracy_percent(clean_preds_weighted, labels),
                    "zs_adv_acc": accuracy_percent(adv_preds_weighted, labels),
                },
            }

        avg = {
            "single": {
                "avg_zs_clean_acc": avg_across_datasets({d: per_ds[d]["single"]["zs_clean_acc"] for d in datasets}, datasets),
                "avg_zs_adv_acc": avg_across_datasets({d: per_ds[d]["single"]["zs_adv_acc"] for d in datasets}, datasets),
            },
            "vanilla": {
                "avg_zs_clean_acc": avg_across_datasets({d: per_ds[d]["vanilla"]["zs_clean_acc"] for d in datasets}, datasets),
                "avg_zs_adv_acc": avg_across_datasets({d: per_ds[d]["vanilla"]["zs_adv_acc"] for d in datasets}, datasets),
            },
            "weighted": {
                "avg_zs_clean_acc": avg_across_datasets({d: per_ds[d]["weighted"]["zs_clean_acc"] for d in datasets}, datasets),
                "avg_zs_adv_acc": avg_across_datasets({d: per_ds[d]["weighted"]["zs_adv_acc"] for d in datasets}, datasets),
            },
        }

        print("[Zero-shot] Avg across datasets:")
        for variant in ["single", "vanilla", "weighted"]:
            print(f"  - {variant}:")
            print(f"      clean: {avg[variant]['avg_zs_clean_acc']:.2f}")
            print(f"      adv  : {avg[variant]['avg_zs_adv_acc']:.2f}")

        return {"per_dataset": per_ds, "avg": avg}


    def plot_zero_shot_avg_results(
            *,
            out_root="Results",
            datasets=None,
    ):
        """Compute + save the zero-shot avg-across-datasets plot."""
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        if datasets is None:
            datasets = list(TRUE_LABELS_DATASET.keys())

        zs = compute_zero_shot_avg_results(datasets)

        # Plot grouped bars: x = attack condition, series = {original, vanilla, weighted}
        x_labels = ["Clean", "Adversarial"]
        series = [
            {
                "name": "Single",
                "values": [
                    zs["avg"]["single"]["avg_zs_clean_acc"],
                    zs["avg"]["single"]["avg_zs_adv_acc"],
                ],
            },
            {
                "name": "Vanilla",
                "values": [
                    zs["avg"]["vanilla"]["avg_zs_clean_acc"],
                    zs["avg"]["vanilla"]["avg_zs_adv_acc"],
                ],
            },
            {
                "name": "Weighted",
                "values": [
                    zs["avg"]["weighted"]["avg_zs_clean_acc"],
                    zs["avg"]["weighted"]["avg_zs_adv_acc"],
                ],
            },
        ]
        save_grouped_bar_plot(
            out_root / "zero_shot_avg_across_datasets.png",
            "Zero-shot (Average across Datasets)",
            x_labels=x_labels,
            series=series,
            ylabel="Average Accuracy (%)",
            ylim=(0, 100),
        )

        # Per-dataset plots
        per_ds_root = out_root / "per_dataset"
        per_ds_root.mkdir(parents=True, exist_ok=True)
        for ds in datasets:
            series_ds = [
                {
                    "name": "Single",
                    "values": [
                        zs["per_dataset"][ds]["single"]["zs_clean_acc"],
                        zs["per_dataset"][ds]["single"]["zs_adv_acc"],
                    ],
                },
                {
                    "name": "Vanilla",
                    "values": [
                        zs["per_dataset"][ds]["vanilla"]["zs_clean_acc"],
                        zs["per_dataset"][ds]["vanilla"]["zs_adv_acc"],
                    ],
                },
                {
                    "name": "Weighted",
                    "values": [
                        zs["per_dataset"][ds]["weighted"]["zs_clean_acc"],
                        zs["per_dataset"][ds]["weighted"]["zs_adv_acc"],
                    ],
                },
            ]

            save_grouped_bar_plot(
                per_ds_root / f"zero_shot_{ds}.png",
                f"Zero-shot ({ds})",
                x_labels=x_labels,
                series=series_ds,
                ylabel="Accuracy (%)",
                ylim=(0, 100),
            )

        return zs


    # ============================================================
    # 2) TPT results (avg across datasets) with folder structure
    #    Model/
    #       attack_name/
    #           plots + json
    #
    # NOTE: NO conservative metrics are computed/saved/plotted.
    # ============================================================
    def plot_tpt_avg_results_and_plots(
            *,
            out_root="Results",
            only_models=None,  # e.g. ["vit_l_14_datacomp_1b"]
            datasets=None,
    ):
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        if datasets is None:
            datasets = list(TRUE_LABELS_DATASET.keys())

        # ----- TPT summary -----
        attacks = list(tpt_dic.keys())  # e.g. ["clean", "adversarial_eps4_steps100"]
        for attack in attacks:
            print(f"\n[TPT] Processing attack='{attack}'")
            models = list(tpt_dic[attack].keys())

            for model in models:
                if only_models is not None and model not in only_models:
                    continue

                print(f"  [Model] {model}")
                model_root = out_root / model

                # sanity: all datasets exist
                for ds in datasets:
                    if ds not in tpt_dic[attack][model]:
                        raise KeyError(f"Missing dataset '{ds}' under tpt_dic['{attack}']['{model}'].")

                # get tpt_types from first dataset
                tpt_types = list(tpt_dic[attack][model][datasets[0]].keys())  # e.g. ["tpt", "rtpt"]

                # We want a single plot per (model, attack) where the legend includes BOTH TPT types.
                # Folder layout:
                #   Results/{model}/{attack}/avg_acc_across_datasets.png
                tpt_root = model_root / attack
                tpt_root.mkdir(parents=True, exist_ok=True)

                # Requested layout:
                #   - legend = {Zero-shot, TPT, rTPT}
                #   - x-axis  = {Single, Vanilla, Weighted}
                variants = ["single", "vanilla", "weighted"]

                tpt_type_to_name = {
                    "tpt": "TPT",
                    "rtpt": "R-TPT",
                }

                # pred variants available (use first dataset as reference)
                ref_ds = datasets[0]
                pred_variants_by_type = {
                    t: set(tpt_dic[attack][model][ref_ds][t]["preds"].keys())
                    for t in tpt_types
                }

                print(f"    [TPT Types] available = {tpt_types}")
                for t, vs in pred_variants_by_type.items():
                    print(f"      - {t}: pred_variants={sorted(vs)}")
                print(f"      x-axis variants (target) = {variants}")

                summary = {
                    "attack": attack,
                    "model": model,
                    "tpt_types": tpt_types,
                    "datasets": datasets,
                    "x_variants": variants,
                    "per_pred_variant": {},
                }

                # Compute avg accuracy across datasets for each variant, for:
                #   - zero-shot baseline
                #   - each available tpt_type (e.g. tpt, rtpt)
                for v in variants:
                    # ----- zero-shot baseline
                    zs_per_ds = {}
                    for ds in datasets:
                        labels = TRUE_LABELS_DATASET[ds]
                        preds = _get_zero_shot_preds_for_variant(
                            attack=attack,
                            dataset=ds,
                            zs_variant=v,
                        )
                        _ensure_same_len(preds, labels, name=f"{attack}/{model}/{ds}/zero-shot/{v}")
                        zs_per_ds[ds] = accuracy_percent(preds, labels)
                    zs_avg = avg_across_datasets(zs_per_ds, datasets)

                    per_variant = {
                        "zero_shot": {
                            "avg_acc_across_datasets": zs_avg,
                            "per_dataset": zs_per_ds,
                        }
                    }

                    # ----- TPT types
                    for t in tpt_types:
                        t_per_ds = {}
                        if v in pred_variants_by_type.get(t, set()):
                            for ds in datasets:
                                labels = TRUE_LABELS_DATASET[ds]
                                obj = tpt_dic[attack][model][ds][t]["preds"][v]
                                preds = obj["prediction"]
                                _ensure_same_len(preds, labels, name=f"{attack}/{model}/{ds}/{t}/{v}")
                                t_per_ds[ds] = accuracy_percent(preds, labels)
                            t_avg = avg_across_datasets(t_per_ds, datasets)
                        else:
                            t_avg = float("nan")

                        per_variant[t] = {
                            "avg_acc_across_datasets": t_avg,
                            "per_dataset": t_per_ds,
                        }

                    summary["per_pred_variant"][v] = per_variant

                    tpt_print = " | ".join(
                        [
                            f"{t} avg_acc={per_variant[t]['avg_acc_across_datasets']:.2f}"
                            for t in tpt_types
                        ]
                    )
                    print(f"      [{v}] zero-shot avg_acc={zs_avg:.2f} | {tpt_print}")

                # Plot grouped bars: x = variant, series = {Zero-shot, TPT, rTPT}
                x_labels = ["Single", "Vanilla", "Weighted"]
                series = [
                    {
                        "name": "Zero-shot",
                        "values": [summary["per_pred_variant"][v]["zero_shot"]["avg_acc_across_datasets"] for v in variants],
                    },
                ]
                for t in tpt_types:
                    series.append(
                        {
                            "name": tpt_type_to_name.get(t, str(t)),
                            "values": [summary["per_pred_variant"][v][t]["avg_acc_across_datasets"] for v in variants],
                        }
                    )

                save_grouped_bar_plot(
                    tpt_root / "avg_acc_across_datasets.png",
                    f"Average accuracy across Datasets",
                    x_labels=x_labels,
                    series=series,
                    ylabel="Average Accuracy (%)",
                    ylim=(0, 100),
                    legend_loc="upper center",
                )

                # Per-dataset plots
                per_ds_root = tpt_root / "per_dataset"
                per_ds_root.mkdir(parents=True, exist_ok=True)
                for ds in datasets:
                    series_ds = [
                        {
                            "name": "Zero-shot",
                            "values": [
                                summary["per_pred_variant"][v]["zero_shot"]["per_dataset"][ds]
                                for v in variants
                            ],
                        },
                    ]
                    for t in tpt_types:
                        if ds not in summary["per_pred_variant"][variants[0]].get(t, {}).get("per_dataset", {}):
                            # If a whole TPT type is missing for this dataset/variant, keep behavior consistent with avg plots.
                            values = [float("nan") for _ in variants]
                        else:
                            values = [
                                summary["per_pred_variant"][v][t]["per_dataset"].get(ds, float("nan"))
                                for v in variants
                            ]
                        series_ds.append(
                            {
                                "name": tpt_type_to_name.get(t, str(t)),
                                "values": values,
                            }
                        )

                    save_grouped_bar_plot(
                        per_ds_root / f"avg_acc_{ds}.png",
                        f"Accuracy ({attack}, {model}, {ds})",
                        x_labels=x_labels,
                        series=series_ds,
                        ylabel="Accuracy (%)",
                        ylim=(0, 100),
                        legend_loc="upper center",
                    )

        print(f"\n[Done] Outputs written to: {out_root.resolve()}")


    def plot_tpt_avg_results_and_plots_threshold(
            *,
            out_root="Results",
            only_models=None,  # e.g. ["vit_l_14_datacomp_1b"]
            datasets=None,
    ):
        """Thresholded TPT plots.

        This function is intentionally separate from `plot_tpt_avg_results_and_plots()` so
        the default avg plots stay clean and threshold logic is isolated.
        """
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        if datasets is None:
            datasets = list(TRUE_LABELS_DATASET.keys())

        attacks = list(tpt_dic.keys())
        for attack in attacks:
            print(f"\n[TPT-THRESH] Processing attack='{attack}'")
            models = list(tpt_dic[attack].keys())

            for model in models:
                if only_models is not None and model not in only_models:
                    continue

                print(f"  [Model] {model}")
                model_root = out_root / model

                # sanity: all datasets exist
                for ds in datasets:
                    if ds not in tpt_dic[attack][model]:
                        raise KeyError(f"Missing dataset '{ds}' under tpt_dic['{attack}']['{model}'].")

                tpt_types = list(tpt_dic[attack][model][datasets[0]].keys())
                # Put *all* thresholding-related artifacts under a dedicated subfolder.
                # This keeps the main Results/<model>/<attack>/ directory clean.
                tpt_root = model_root / attack
                thresholding_root = tpt_root / "thresholding"
                thresholding_root.mkdir(parents=True, exist_ok=True)

                variants = ["single", "vanilla", "weighted"]

                # ------------------------------------------------------------
                # Diff-ratio configurations
                # ------------------------------------------------------------
                # Automatically evaluate all available diff-ratio configs.
                # We take an intersection across datasets to avoid missing-key failures.
                dr_attack_name = _attack_to_diff_ratio_attack_name(attack)
                configs_by_ds = []
                for ds in datasets:
                    ds_attack_dic = final_diff_ratio_dic.get(ds, {}).get(dr_attack_name, {})
                    ds_configs = set()
                    for noise_type_name, noise_vals in ds_attack_dic.items():
                        for noise_value_name in noise_vals.keys():
                            ds_configs.add((str(noise_type_name), str(noise_value_name)))
                    configs_by_ds.append(ds_configs)

                common_configs = set.intersection(*configs_by_ds) if configs_by_ds else set()
                if not common_configs:
                    raise KeyError(
                        f"No common diff-ratio configs found for attack='{attack}' (diff-ratio attack='{dr_attack_name}') "
                        f"across datasets={datasets}."
                    )

                # Stable order for output folders.
                diff_ratio_configs = sorted(common_configs, key=lambda x: (x[0], x[1]))
                tpt_type_to_name = {
                    "tpt": "TPT",
                    "rtpt": "R-TPT",
                }

                ref_ds = datasets[0]
                pred_variants_by_type = {
                    t: set(tpt_dic[attack][model][ref_ds][t]["preds"].keys())
                    for t in tpt_types
                }

                print(f"    [TPT Types] available = {tpt_types}")
                for t, vs in pred_variants_by_type.items():
                    print(f"      - {t}: pred_variants={sorted(vs)}")
                print(f"      x-axis variants (target) = {variants}")

                def plot_thresholded_bar_results(*, out_dir: Path, summary_obj: dict):
                    """Single-threshold grouped-bar plot (backward compatible output name)."""
                    x_labels_local = ["Single", "Vanilla", "Weighted"]

                    thr_series_local = []
                    for t in tpt_types:
                        thr_series_local.append(
                            {
                                "name": f"Thresholded ({tpt_type_to_name.get(t, str(t))})",
                                "values": [
                                    summary_obj["per_pred_variant"][v]["thresholded"][t]["avg_acc_across_datasets"]
                                    for v in variants
                                ],
                            }
                        )

                    save_grouped_bar_plot(
                        out_dir / f"avg_acc_across_datasets_thresholded_thr_{diff_ratio_threshold}.png",
                        f"Thresholded avg acc across Datasets (thr={diff_ratio_threshold}, {diff_ratio_noise_type}/{diff_ratio_noise_value})",
                        x_labels=x_labels_local,
                        series=thr_series_local,
                        ylabel="Average Accuracy (%)",
                        ylim=(0, 100),
                        legend_loc="upper center",
                    )

                    # Per-dataset thresholded bars
                    per_ds_root = out_dir / "per_dataset"
                    per_ds_root.mkdir(parents=True, exist_ok=True)
                    for ds in datasets:
                        thr_series_ds = []
                        for t in tpt_types:
                            thr_series_ds.append(
                                {
                                    "name": f"Thresholded ({tpt_type_to_name.get(t, str(t))})",
                                    "values": [
                                        summary_obj["per_pred_variant"][v]["thresholded"][t]["per_dataset"].get(ds, float("nan"))
                                        for v in variants
                                    ],
                                }
                            )

                        save_grouped_bar_plot(
                            per_ds_root / f"thresholded_{ds}_thr_{diff_ratio_threshold}.png",
                            f"Thresholded acc ({attack}, {model}, {ds}) (thr={diff_ratio_threshold}, {diff_ratio_noise_type}/{diff_ratio_noise_value})",
                            x_labels=x_labels_local,
                            series=thr_series_ds,
                            ylabel="Accuracy (%)",
                            ylim=(0, 100),
                            legend_loc="upper center",
                        )

                def plot_threshold_sweep_curves(*, out_dir: Path):
                    """For each variant, plot accuracy vs tau-threshold + bar version."""
                    # Use only the requested threshold grid (no dense sweep).
                    thresholds = np.asarray([0.0, 0.1, 0.2, 0.25, 0.30, 0.4, 0.6, 0.8, 0.85, 0.9, 1.0], dtype=float)
                    thresholds_bar = thresholds

                    for v in variants:
                        # Zero-shot is independent of threshold
                        zs_per_ds = {}
                        for ds in datasets:
                            labels = TRUE_LABELS_DATASET[ds]
                            zs_preds = _get_zero_shot_preds_for_variant(
                                attack=attack,
                                dataset=ds,
                                zs_variant=v,
                            )
                            _ensure_same_len(zs_preds, labels, name=f"{attack}/{model}/{ds}/zero-shot/{v}")
                            zs_per_ds[ds] = accuracy_percent(zs_preds, labels)
                        zs_avg = avg_across_datasets(zs_per_ds, datasets)
                        zs_curve = [zs_avg for _ in thresholds]

                        series_curves = [
                            {"name": "Zero-shot", "y": zs_curve},
                        ]

                        for t in ["tpt", "rtpt"]:
                            if t not in tpt_types:
                                continue
                            if v not in pred_variants_by_type.get(t, set()):
                                continue

                            y_vals = []
                            for thr in thresholds:
                                thr_per_ds = {}
                                for ds in datasets:
                                    labels = TRUE_LABELS_DATASET[ds]
                                    zs_preds = _get_zero_shot_preds_for_variant(
                                        attack=attack,
                                        dataset=ds,
                                        zs_variant=v,
                                    )
                                    tpt_preds = tpt_dic[attack][model][ds][t]["preds"][v]["prediction"]
                                    diff_ratio_per_sample = _get_diff_ratio_per_sample(
                                        dataset=ds,
                                        attack=attack,
                                        noise_type=diff_ratio_noise_type,
                                        noise_value=diff_ratio_noise_value,
                                    )
                                    mixed = _mix_preds_by_diff_ratio(
                                        diff_ratio_per_sample=diff_ratio_per_sample,
                                        threshold=float(thr),
                                        zs_preds=zs_preds,
                                        tpt_preds=tpt_preds,
                                    )
                                    _ensure_same_len(mixed, labels, name=f"{attack}/{model}/{ds}/threshold_sweep/{t}/{v}")
                                    thr_per_ds[ds] = accuracy_percent(mixed, labels)
                                y_vals.append(avg_across_datasets(thr_per_ds, datasets))

                            series_curves.append(
                                {
                                    "name": tpt_type_to_name.get(t, str(t)),
                                    "y": y_vals,
                                }
                            )

                        save_line_plot(
                            out_dir / f"threshold_sweep_{v}.png",
                            f"Threshold sweep ({attack}, {v}, {diff_ratio_noise_type}_{diff_ratio_noise_value})",
                            x=thresholds,
                            series=series_curves,
                            xlabel="tau threshold",
                            ylabel="Average Accuracy (%) across datasets",
                            ylim=(0, 100),
                            xlim=(0.0, 1.0),
                            legend_loc="best",
                        )

                        series_bars = [
                            {"name": s["name"], "y": [float(y) for y in s["y"]]}
                            for s in series_curves
                        ]
                        save_threshold_sweep_bar_plot(
                            out_dir / f"threshold_sweep_bar_{v}.png",
                            f"Threshold sweep (bars) ({attack}, {v}, {diff_ratio_noise_type}_{diff_ratio_noise_value})",
                            thresholds=thresholds_bar,
                            series=series_bars,
                            xlabel="tau threshold",
                            ylabel="Average Accuracy (%) across datasets",
                            ylim=(0, 100),
                            legend_loc="upper center",
                            grid_alpha=0.25,
                        )

                        # Per-dataset threshold sweep: BAR plots (requested)
                        per_ds_root = out_dir / "per_dataset" / v
                        per_ds_root.mkdir(parents=True, exist_ok=True)
                        for ds in datasets:
                            zs_val = zs_per_ds[ds]
                            series_ds = [
                                {"name": "Zero-shot", "y": [zs_val for _ in thresholds]},
                            ]

                            for t in ["tpt", "rtpt"]:
                                if t not in tpt_types:
                                    continue
                                if v not in pred_variants_by_type.get(t, set()):
                                    continue
                                y_vals_ds = []
                                for thr in thresholds:
                                    labels = TRUE_LABELS_DATASET[ds]
                                    zs_preds = _get_zero_shot_preds_for_variant(
                                        attack=attack,
                                        dataset=ds,
                                        zs_variant=v,
                                    )
                                    tpt_preds = tpt_dic[attack][model][ds][t]["preds"][v]["prediction"]
                                    diff_ratio_per_sample = _get_diff_ratio_per_sample(
                                        dataset=ds,
                                        attack=attack,
                                        noise_type=diff_ratio_noise_type,
                                        noise_value=diff_ratio_noise_value,
                                    )
                                    mixed = _mix_preds_by_diff_ratio(
                                        diff_ratio_per_sample=diff_ratio_per_sample,
                                        threshold=float(thr),
                                        zs_preds=zs_preds,
                                        tpt_preds=tpt_preds,
                                    )
                                    _ensure_same_len(mixed, labels, name=f"{attack}/{model}/{ds}/threshold_sweep_per_dataset/{t}/{v}")
                                    y_vals_ds.append(accuracy_percent(mixed, labels))

                                series_ds.append(
                                    {
                                        "name": tpt_type_to_name.get(t, str(t)),
                                        "y": y_vals_ds,
                                    }
                                )

                            save_threshold_sweep_bar_plot(
                                per_ds_root / f"threshold_sweep_{ds}.png",
                                f"Threshold sweep ({attack}, {model}, {ds}, {v}, {diff_ratio_noise_type}_{diff_ratio_noise_value})",
                                thresholds=thresholds,
                                series=series_ds,
                                ylabel="Accuracy (%)",
                                ylim=(0, 100),
                                legend_loc="best",
                            )

                def _compute_threshold_sweep_series(*, v: str, noise_type: str, noise_value: str):
                    """Compute sweep series for a given prediction variant + diff-ratio config.

                    Returns:
                      (thresholds: np.ndarray, series_curves: list[dict], best_text: str)
                    """
                    thresholds = np.asarray([0.0, 0.1, 0.4, 0.6, 0.8, 0.85, 0.9, 1.0], dtype=float)

                    # Zero-shot is independent of threshold
                    zs_per_ds = {}
                    for ds in datasets:
                        labels = TRUE_LABELS_DATASET[ds]
                        zs_preds = _get_zero_shot_preds_for_variant(
                            attack=attack,
                            dataset=ds,
                            zs_variant=v,
                        )
                        _ensure_same_len(zs_preds, labels, name=f"{attack}/{model}/{ds}/zero-shot/{v}")
                        zs_per_ds[ds] = accuracy_percent(zs_preds, labels)
                    zs_avg = avg_across_datasets(zs_per_ds, datasets)
                    zs_curve = [zs_avg for _ in thresholds]

                    series_curves = [
                        {"name": "Zero-shot", "y": zs_curve},
                    ]

                    best_acc = -float("inf")
                    best_thr = None
                    best_name = None

                    for t in ["tpt", "rtpt"]:
                        if t not in tpt_types:
                            continue
                        if v not in pred_variants_by_type.get(t, set()):
                            continue

                        y_vals = []
                        for thr in thresholds:
                            thr_per_ds = {}
                            for ds in datasets:
                                labels = TRUE_LABELS_DATASET[ds]
                                zs_preds = _get_zero_shot_preds_for_variant(
                                    attack=attack,
                                    dataset=ds,
                                    zs_variant=v,
                                )
                                tpt_preds = tpt_dic[attack][model][ds][t]["preds"][v]["prediction"]
                                diff_ratio_per_sample = _get_diff_ratio_per_sample(
                                    dataset=ds,
                                    attack=attack,
                                    noise_type=noise_type,
                                    noise_value=noise_value,
                                )
                                mixed = _mix_preds_by_diff_ratio(
                                    diff_ratio_per_sample=diff_ratio_per_sample,
                                    threshold=float(thr),
                                    zs_preds=zs_preds,
                                    tpt_preds=tpt_preds,
                                )
                                _ensure_same_len(mixed, labels, name=f"{attack}/{model}/{ds}/threshold_sweep/{t}/{v}")
                                thr_per_ds[ds] = accuracy_percent(mixed, labels)
                            y_vals.append(avg_across_datasets(thr_per_ds, datasets))

                        display_name = tpt_type_to_name.get(t, str(t))
                        series_curves.append(
                            {
                                "name": display_name,
                                "y": y_vals,
                            }
                        )

                        # best over thresholds for this method
                        y_arr = np.asarray(y_vals, dtype=float)
                        if np.all(np.isnan(y_arr)):
                            continue
                        idx = int(np.nanargmax(y_arr))
                        if float(y_arr[idx]) > best_acc:
                            best_acc = float(y_arr[idx])
                            best_thr = float(thresholds[idx])
                            best_name = display_name

                    if best_name is None:
                        best_text = "best: n/a"
                    else:
                        best_text = f"best: {best_name} {best_acc:.2f}% @ thr={best_thr:g}"

                    return thresholds, series_curves, best_text

                def plot_threshold_sweep_grid(*, out_dir: Path):
                    """Grid plot: one figure per variant, each subplot = one noise config.

                    NOTE: Subplots are BAR plots (grouped bars over the threshold grid).
                    """
                    n = len(diff_ratio_configs)
                    if n <= 0:
                        return

                    ncols = 3
                    nrows = int(np.ceil(n / float(ncols)))

                    for v in variants:
                        fig, axes = plt.subplots(
                            nrows=nrows,
                            ncols=ncols,
                            figsize=(5.0 * ncols, 3.6 * nrows),
                            sharex=True,
                            sharey=True,
                        )
                        axes = np.asarray(axes).reshape(-1)

                        legend_handles = None
                        legend_labels = None

                        def _plot_threshold_sweep_bars_on_ax(*, ax, thresholds, series_curves):
                            # Convert to grouped-bar inputs.
                            thresholds = [float(x) for x in thresholds]
                            # Adaptive tick labels so values like 0.85 don't show as 0.8.
                            x_labels = [f"{x:.1f}" if np.isclose(x, round(x, 1)) else f"{x:.2f}" for x in thresholds]

                            # We plot bars on integer x positions.
                            x = np.arange(len(thresholds), dtype=float)
                            n_series = max(1, len(series_curves))
                            group_width = 0.82
                            bar_w = group_width / float(n_series)
                            offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_w

                            name_to_color = {
                                "Zero-shot": "#4D4D4D",
                                "TPT": "#0072B2",
                                "R-TPT": "#D55E00",
                            }

                            handles = []
                            labels = []
                            for si, s in enumerate(series_curves):
                                name = str(s["name"])
                                y = np.asarray(s["y"], dtype=float)
                                h = ax.bar(
                                    x + offsets[si],
                                    y,
                                    width=bar_w,
                                    label=name,
                                    color=name_to_color.get(name, None),
                                )
                                handles.append(h)
                                labels.append(name)

                            ax.set_xticks(x)
                            ax.set_xticklabels(x_labels)
                            ax.grid(True, axis="y", alpha=0.25, linestyle="-")
                            return handles, labels

                        for i, (nt, nv) in enumerate(diff_ratio_configs):
                            ax = axes[i]
                            thresholds, series_curves, best_text = _compute_threshold_sweep_series(
                                v=v,
                                noise_type=nt,
                                noise_value=nv,
                            )

                            _plot_threshold_sweep_bars_on_ax(
                                ax=ax,
                                thresholds=thresholds,
                                series_curves=series_curves,
                            )

                            ax.set_title(f"{nt}_{nv} | {best_text}")
                            ax.set_ylim(0, 100)

                            if legend_handles is None:
                                legend_handles, legend_labels = ax.get_legend_handles_labels()

                        # Hide unused axes
                        for j in range(n, len(axes)):
                            axes[j].axis("off")

                        fig.suptitle(f"Threshold sweep grid (bars) ({attack}, {model}, {v})", y=0.995)
                        fig.text(0.5, 0.04, "tau threshold", ha="center")
                        fig.text(0.04, 0.5, "Average Accuracy (%) across datasets", va="center", rotation="vertical")

                        if legend_handles is not None:
                            fig.legend(
                                legend_handles,
                                legend_labels,
                                loc="upper center",
                                ncol=min(len(legend_labels), 4),
                                bbox_to_anchor=(0.5, 0.98),
                            )

                        fig.tight_layout(rect=[0.05, 0.06, 0.995, 0.94])
                        out_path = out_dir / f"threshold_sweep_grid_{v}.png"
                        fig.savefig(out_path, dpi=180)
                        plt.close(fig)

                for diff_ratio_noise_type, diff_ratio_noise_value in diff_ratio_configs:
                    # Single folder: <noise_type>_<noise_value>
                    diff_ratio_out_dir = thresholding_root / f"{diff_ratio_noise_type}_{diff_ratio_noise_value}"
                    diff_ratio_out_dir.mkdir(parents=True, exist_ok=True)

                    summary = {
                        "attack": attack,
                        "model": model,
                        "tpt_types": tpt_types,
                        "datasets": datasets,
                        "x_variants": variants,
                        "thresholding": {
                            "enabled": True,
                            "diff_ratio_threshold": diff_ratio_threshold,
                            "diff_ratio_source": {
                                "noise_type": diff_ratio_noise_type,
                                "noise_value": diff_ratio_noise_value,
                            },
                        },
                        "per_pred_variant": {},
                    }

                    for v in variants:
                        zs_per_ds = {}
                        for ds in datasets:
                            labels = TRUE_LABELS_DATASET[ds]
                            zs_preds = _get_zero_shot_preds_for_variant(
                                attack=attack,
                                dataset=ds,
                                zs_variant=v,
                            )
                            _ensure_same_len(zs_preds, labels, name=f"{attack}/{model}/{ds}/zero-shot/{v}")
                            zs_per_ds[ds] = accuracy_percent(zs_preds, labels)
                        zs_avg = avg_across_datasets(zs_per_ds, datasets)

                        per_variant = {
                            "zero_shot": {
                                "avg_acc_across_datasets": zs_avg,
                                "per_dataset": zs_per_ds,
                            }
                        }

                        for t in tpt_types:
                            thr_per_ds = {}
                            if v in pred_variants_by_type.get(t, set()):
                                for ds in datasets:
                                    labels = TRUE_LABELS_DATASET[ds]
                                    zs_preds = _get_zero_shot_preds_for_variant(
                                        attack=attack,
                                        dataset=ds,
                                        zs_variant=v,
                                    )
                                    tpt_preds = tpt_dic[attack][model][ds][t]["preds"][v]["prediction"]
                                    diff_ratio_per_sample = _get_diff_ratio_per_sample(
                                        dataset=ds,
                                        attack=attack,
                                        noise_type=diff_ratio_noise_type,
                                        noise_value=diff_ratio_noise_value,
                                    )
                                    mixed = _mix_preds_by_diff_ratio(
                                        diff_ratio_per_sample=diff_ratio_per_sample,
                                        threshold=diff_ratio_threshold,
                                        zs_preds=zs_preds,
                                        tpt_preds=tpt_preds,
                                    )
                                    _ensure_same_len(mixed, labels, name=f"{attack}/{model}/{ds}/thresholded/{t}/{v}")
                                    thr_per_ds[ds] = accuracy_percent(mixed, labels)
                                thr_avg = avg_across_datasets(thr_per_ds, datasets)
                            else:
                                thr_avg = float("nan")

                            per_variant.setdefault("thresholded", {})[t] = {
                                "avg_acc_across_datasets": thr_avg,
                                "per_dataset": thr_per_ds,
                            }

                        summary["per_pred_variant"][v] = per_variant

                    # plot_thresholded_bar_results(out_dir=diff_ratio_out_dir, summary_obj=summary)
                    plot_threshold_sweep_curves(out_dir=diff_ratio_out_dir)

                # One grid figure per variant with subplots for all noise configs
                plot_threshold_sweep_grid(out_dir=thresholding_root)

        print(f"\n[Done] Threshold outputs written to: {out_root.resolve()}")

    # ============================================================
    # RUN (restrict to vit_l_14_datacomp_1b as per your earlier example)
    # ============================================================
    # ============================================================
    # RUN (sequence: zero-shot first, then TPT)
    # ============================================================
    out_root = "Results"
    if args.datasets is None:
        datasets = list(TRUE_LABELS_DATASET.keys())
    else:
        datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
        missing = [d for d in datasets if d not in TRUE_LABELS_DATASET]
        if missing:
            raise KeyError(f"Unknown dataset(s) in --datasets: {missing}. Available: {sorted(TRUE_LABELS_DATASET.keys())}")

    plot_zero_shot_avg_results(
        out_root=out_root,
        datasets=datasets,
    )

    plot_tpt_avg_results_and_plots(
        out_root=out_root,
        only_models=["vit_l_14_datacomp_1b"],
        datasets=datasets,
    )

    plot_tpt_avg_results_and_plots_threshold(
        out_root=out_root,
        only_models=["vit_l_14_datacomp_1b"],
        datasets=datasets,
    )




