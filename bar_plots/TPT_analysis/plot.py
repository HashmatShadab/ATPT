"""TPT analysis: aggregation + plotting.

High-level flow
----------------
1) Load zero-shot predictions for each dataset (single / vanilla / weighted).
2) Load TPT predictions for each dataset (e.g. `tpt`, `rtpt`).
3) Produce plots:
   - zero-shot avg-across-datasets
   - TPT avg-across-datasets
   - TPT-type comparison (`tpt` vs `rtpt`)

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
        # "UCF101",
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
        # "UCF101",
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
                    "../../Final_Results_corrected_ca_tau/"
                    "{model}/{dataset}/"
                    "Clean/No_Counter_Attack/TPT/Optimization_Loss_{tpt_type}_LR_0_005_Optimization_Steps_1_View_Selection_Fraction_0_1/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            "adversarial_eps4_steps100": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau/"
                    "{model}/{dataset}/"
                    "Adversarial_Eps_4_0_Steps_100/"
                    "No_Counter_Attack/TPT/Optimization_Loss_{tpt_type}_LR_0_005_Optimization_Steps_1_View_Selection_Fraction_0_1/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            # "adversarial_eps4_steps100_image_only": {
            #     "base_path": (
            #         "../../Final_Results_corrected_ca_tau_AOM/"
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
    args = parser.parse_args()

    model_name = args.model_name

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


    def save_json(path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"[Saved] {path}")


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
            ylim=(0, 100),
            value_fmt="{:.2f}",
            colors=None,
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

        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor=bar_edgecolor,
            linewidth=bar_linewidth,
            width=0.62,
        )

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
                clip_on=True,
            )

        ax.set_xticks(x)
        # Rotate only if labels are long
        max_label_len = max((len(str(l)) for l in labels), default=0)
        if max_label_len >= 10:
            ax.set_xticklabels(labels, rotation=20, ha="right")
        else:
            ax.set_xticklabels(labels)

        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(title)
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

        fig, ax = plt.subplots(figsize=(10, 4.9))
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
        """Compute + save the zero-shot avg-across-datasets JSON + plot."""
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        if datasets is None:
            datasets = list(TRUE_LABELS_DATASET.keys())

        zs = compute_zero_shot_avg_results(datasets)
        save_json(out_root / "zero_shot_avg_across_datasets.json", zs)

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

        return zs


    # ============================================================
    # 2) TPT results (avg across datasets) with folder structure
    #    Model/
    #       TPT Type
    #           attack_name/
    #               plots + json
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

                for tpt_type in tpt_types:
                    print(f"    [TPT Type] {tpt_type}")
                    tpt_root = model_root / tpt_type / attack
                    tpt_root.mkdir(parents=True, exist_ok=True)

                    # pred variants available (use first dataset as reference)
                    ref_ds = datasets[0]
                    pred_variants_all = list(tpt_dic[attack][model][ref_ds][tpt_type]["preds"].keys())

                    # Requested layout:
                    #   - legend = {Zero-shot, tpt_type}
                    #   - x-axis = {Single, Vanilla, Weighted}
                    variants = ["single", "vanilla", "weighted"]

                    print(f"      pred_variants (available in TPT) = {pred_variants_all}")
                    print(f"      x-axis variants (target)         = {variants}")

                    summary = {
                        "attack": attack,
                        "model": model,
                        "tpt_type": tpt_type,
                        "datasets": datasets,
                        "x_variants": variants,
                        "per_pred_variant": {},
                    }

                    # Compute avg accuracy across datasets for each variant, for:
                    #   - zero-shot baseline
                    #   - the selected TPT type (tpt/rtpt)
                    for v in variants:
                        # zero-shot
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

                        # tpt_type (may be missing for some variants in some runs)
                        tpt_per_ds = {}
                        if v in pred_variants_all:
                            for ds in datasets:
                                labels = TRUE_LABELS_DATASET[ds]
                                obj = tpt_dic[attack][model][ds][tpt_type]["preds"][v]
                                preds = obj["prediction"]
                                _ensure_same_len(preds, labels, name=f"{attack}/{model}/{ds}/{tpt_type}/{v}")
                                tpt_per_ds[ds] = accuracy_percent(preds, labels)
                            tpt_avg = avg_across_datasets(tpt_per_ds, datasets)
                        else:
                            tpt_avg = float("nan")

                        summary["per_pred_variant"][v] = {
                            "zero_shot": {
                                "avg_acc_across_datasets": zs_avg,
                                "per_dataset": zs_per_ds,
                            },
                            "tpt": {
                                "avg_acc_across_datasets": tpt_avg,
                                "per_dataset": tpt_per_ds,
                            },
                        }

                        print(f"      [{v}] zero-shot avg_acc={zs_avg:.2f} | {tpt_type} avg_acc={tpt_avg:.2f}")

                    # Save json
                    save_json(tpt_root / "tpt_avg_across_datasets.json", summary)

                    # Plot grouped bars: x = variant, series = {Zero-shot, tpt_type}
                    x_labels = ["Single", "Vanilla", "Weighted"]
                    series = [
                        {
                            "name": "Zero-shot",
                            "values": [summary["per_pred_variant"][v]["zero_shot"]["avg_acc_across_datasets"] for v in variants],
                        },
                        {
                            "name": tpt_type.upper(),
                            "values": [summary["per_pred_variant"][v]["tpt"]["avg_acc_across_datasets"] for v in variants],
                        },
                    ]

                    save_grouped_bar_plot(
                        tpt_root / "avg_acc_across_datasets.png",
                        f"Average accuracy across Datasets",
                        x_labels=x_labels,
                        series=series,
                        ylabel="Average Accuracy (%)",
                        ylim=(0, 100),
                        legend_loc="upper center",
                    )

        print(f"\n[Done] Outputs written to: {out_root.resolve()}")


    # ============================================================
    # 3) TPT type comparison (tpt vs rtpt), avg across datasets
    #    Output folder (in addition to existing structure):
    #       out_root/
    #         TPT_Type_Comparison/
    #           Model/
    #             attack/
    #               tpt_type_comparison_avg_across_datasets.json
    #               tpt_type_comparison_avg_across_datasets.png
    #
    # Each bar plot contains averages across datasets for both TPT types.
    # ============================================================
    def plot_tpt_type_comparison_avg_across_datasets(
            *,
            out_root="Results",
            only_models=None,
            datasets=None,
            tpt_types_order=None,
    ):
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        if datasets is None:
            datasets = list(TRUE_LABELS_DATASET.keys())
        if tpt_types_order is None:
            tpt_types_order = ["tpt", "rtpt"]

        comparison_root = out_root / "TPT_vs_RTPT_Comparison"
        comparison_root.mkdir(parents=True, exist_ok=True)

        attacks = list(tpt_dic.keys())
        for attack in attacks:
            print(f"\n[TPT Type Comparison] Processing attack='{attack}'")
            models = list(tpt_dic[attack].keys())

            for model in models:
                if only_models is not None and model not in only_models:
                    continue

                # sanity: datasets exist
                for ds in datasets:
                    if ds not in tpt_dic[attack][model]:
                        raise KeyError(f"Missing dataset '{ds}' under tpt_dic['{attack}']['{model}'].")

                # discover available tpt types
                available_tpt_types = list(tpt_dic[attack][model][datasets[0]].keys())
                tpt_types = [t for t in tpt_types_order if t in available_tpt_types]
                if len(tpt_types) < 2:
                    print(f"  [Skip] model={model}: need both tpt types for comparison, found={available_tpt_types}")
                    continue

                # Requested layout:
                #   - legend = {Zero-shot, tpt, rtpt}
                #   - x-axis = {Single, Vanilla, Weighted}
                variants = ["single", "vanilla", "weighted"]

                # discover which variants exist for both tpt types (to avoid KeyErrors)
                ref_ds = datasets[0]
                pred_variants_sets = []
                for t in tpt_types:
                    pred_variants_sets.append(set(tpt_dic[attack][model][ref_ds][t]["preds"].keys()))
                pred_variants_intersection = set.intersection(*pred_variants_sets)
                available_variants = [v for v in variants if v in pred_variants_intersection]
                if not available_variants:
                    print(f"  [Skip] model={model}: none of {variants} available for BOTH tpt types; intersection={sorted(pred_variants_intersection)}")
                    continue

                model_attack_root = comparison_root / model / attack
                model_attack_root.mkdir(parents=True, exist_ok=True)

                summary = {
                    "attack": attack,
                    "model": model,
                    "datasets": datasets,
                    "tpt_types": tpt_types,
                    "x_variants": available_variants,
                    "avg_acc_across_datasets": {},
                }

                # Zero-shot avg per variant (shared between tpt/rtpt)
                summary["avg_acc_across_datasets"]["zero_shot"] = {}
                for v in available_variants:
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
                    summary["avg_acc_across_datasets"]["zero_shot"][v] = {
                        "avg_acc_across_datasets": avg_across_datasets(zs_per_ds, datasets),
                        "per_dataset": zs_per_ds,
                    }

                # Compute avg acc for each (tpt_type, variant)
                for t in tpt_types:
                    summary["avg_acc_across_datasets"][t] = {}
                    for v in available_variants:
                        per_ds_metrics = {}
                        for ds in datasets:
                            labels = TRUE_LABELS_DATASET[ds]
                            obj = tpt_dic[attack][model][ds][t]["preds"][v]
                            preds = obj["prediction"]
                            _ensure_same_len(preds, labels, name=f"{attack}/{model}/{ds}/{t}/{v}")
                            per_ds_metrics[ds] = accuracy_percent(preds, labels)

                        summary["avg_acc_across_datasets"][t][v] = {
                            "avg_acc_across_datasets": avg_across_datasets(per_ds_metrics, datasets),
                            "per_dataset": per_ds_metrics,
                        }

                save_json(model_attack_root / "tpt_type_comparison_avg_across_datasets.json", summary)

                # Plot grouped bars: x = variants, series = {Zero-shot, tpt, rtpt}
                x_labels = [v.capitalize() for v in available_variants]
                series = [
                    {
                        "name": "Zero-shot",
                        "values": [summary["avg_acc_across_datasets"]["zero_shot"][v]["avg_acc_across_datasets"] for v in available_variants],
                    },
                ]
                for t in tpt_types:
                    series.append({
                        "name": t,
                        "values": [summary["avg_acc_across_datasets"][t][v]["avg_acc_across_datasets"] for v in available_variants],
                    })

                save_grouped_bar_plot(
                    model_attack_root / "tpt_type_comparison_avg_across_datasets.png",
                    f"{model} | {attack} | Avg accuracy across datasets: TPT type comparison",
                    x_labels=x_labels,
                    series=series,
                    ylabel="Accuracy (%)",
                    ylim=(0, 100),
                    legend_loc="best",
                )


    # ============================================================
    # RUN (restrict to vit_l_14_datacomp_1b as per your earlier example)
    # ============================================================
    # ============================================================
    # RUN (sequence: zero-shot first, then TPT)
    # ============================================================
    out_root = "Results"
    datasets = list(TRUE_LABELS_DATASET.keys())

    plot_zero_shot_avg_results(
        out_root=out_root,
        datasets=datasets,
    )

    plot_tpt_avg_results_and_plots(
        out_root=out_root,
        only_models=["vit_l_14_datacomp_1b"],
        datasets=datasets,
    )

    plot_tpt_type_comparison_avg_across_datasets(
        out_root=out_root,
        only_models=["vit_l_14_datacomp_1b"],
        datasets=datasets,
    )




