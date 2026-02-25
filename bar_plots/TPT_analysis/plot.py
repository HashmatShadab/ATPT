"""
Default TTC

Final_Results_corrected_ca_tau_Counter_Attack/vit_l_14_datacomp_1b/<Dataset Name>/Adversarial_Eps_4_0_Steps_100/Counter_Attack/Eps_4_0_Steps_5_Alpha_1_0/tau_0_2_beta_2_0_weighted_pertrubation_True/No_TPT/Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01
- results_original_clean.json
- results_single.json
- results_original.json
- results_counter_attack_diff_ratio.json

Final_Results_corrected_ca_tau_Counter_Attack/vit_l_14_datacomp_1b/<Dataset Name>/Clean/Counter_Attack/Eps_4_0_Steps_5_Alpha_1_0/tau_0_2_beta_2_0_weighted_pertrubation_True/No_TPT/Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01
- results_original_clean.json
- results_single.json
- results_original.json
- results_counter_attack_diff_ratio.json
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

    ZS_CLEAN_PREDS_DATASET = zero_shot_clean_preds_data
    ZS_ADV_PREDS_DATASET = zero_shot_adv_preds_data
    ZS_ADV_IMAGE_ONLY_PREDS_DATASET = zero_shot_adv_image_only_preds_data
    ZS_CLEAN_CORRECT_PREDS = zero_shot_clean_correct_preds

    # Vanilla
    ZS_CLEAN_PREDS_VANILLA_DATASET = zero_shot_clean_preds_vanilla_data
    ZS_ADV_PREDS_VANILLA_DATASET = zero_shot_adv_preds_vanilla_data
    ZS_ADV_IMAGE_ONLY_PREDS_VANILLA_DATASET = zero_shot_adv_image_only_preds_vanilla_data

    # Weighted
    ZS_CLEAN_PREDS_WEIGHTED_DATASET = zero_shot_clean_preds_weighted_data
    ZS_ADV_PREDS_WEIGHTED_DATASET = zero_shot_adv_preds_weighted_data
    ZS_ADV_IMAGE_ONLY_PREDS_WEIGHTED_DATASET = zero_shot_adv_image_only_preds_weighted_data

    return_dic = {"zero_shot_clean": ZS_CLEAN_PREDS_DATASET, "zero_shot_adv": ZS_ADV_PREDS_DATASET,
                  "zero_shot_adv_image_only": ZS_ADV_IMAGE_ONLY_PREDS_DATASET,
                  "true_labels": TRUE_LABELS_DATASET,
                  "zero_shot_clean_correct_preds": ZS_CLEAN_CORRECT_PREDS,}

    return return_dic



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

def get_tpt_results(model_name):
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
        # "UCF101",
        "eurosat",
    ]

    tpt_types = ["tpt", "rtpt"]

    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pred_json(obj: dict, *, allow_extra: bool = False) -> dict:
        out = {}
        for k in obj.keys():
            out[k] = obj[k]


        return out



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





    tpt_dic = get_tpt_results(
        model_name,

    )

    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


    def compute_accuracy(preds, labels):
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        return (preds == labels).mean() * 100.0


    import os
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt


    # ============================================================
    # Assumptions (already available in your runtime as you said)
    # ------------------------------------------------------------
    # TRUE_LABELS_DATASET: dict[dataset] -> list[int] ground-truth labels
    # ZS_CLEAN_PREDS_DATASET: dict[dataset] -> list[int] zero-shot clean preds
    # ZS_ADV_PREDS_DATASET: dict[dataset] -> list[int] zero-shot adv preds
    # final_diff_ratio_dic: dict[dataset][Clean/Adversarial][NoiseType][NoiseValue] -> list[float] per-sample diff ratio
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


    # ============================================================
    # 1) Zero-shot results (avg across datasets) [NO conservative]
    # ============================================================
    def compute_zero_shot_avg_results(datasets):
        per_ds = {}
        for ds in datasets:
            labels = TRUE_LABELS_DATASET[ds]
            clean_preds = ZS_CLEAN_PREDS_DATASET[ds]
            adv_preds = ZS_ADV_PREDS_DATASET[ds]
            _ensure_same_len(labels, clean_preds, adv_preds, name=f"{ds} zs preds")

            clean_acc = accuracy_percent(clean_preds, labels)
            adv_acc = accuracy_percent(adv_preds, labels)

            per_ds[ds] = {
                "zs_clean_acc": clean_acc,
                "zs_adv_acc": adv_acc,
                "num_samples": len(labels),
            }

        avg = {
            "avg_zs_clean_acc": avg_across_datasets({d: per_ds[d]["zs_clean_acc"] for d in datasets}, datasets),
            "avg_zs_adv_acc": avg_across_datasets({d: per_ds[d]["zs_adv_acc"] for d in datasets}, datasets),
        }

        print("[Zero-shot] Avg across datasets:")
        for k, v in avg.items():
            print(f"  - {k}: {v:.2f}")

        return {"per_dataset": per_ds, "avg": avg}


    def plot_zero_shot_avg_results(
            *,
            out_root="Results_AvgAcrossDatasets",
            datasets=None,
    ):
        """Compute + save the zero-shot avg-across-datasets JSON + plot."""
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        if datasets is None:
            datasets = list(TRUE_LABELS_DATASET.keys())

        zs = compute_zero_shot_avg_results(datasets)
        save_json(out_root / "zero_shot_avg_across_datasets.json", zs)

        save_bar_plot(
            out_root / "zero_shot_avg_across_datasets.png",
            "Zero-shot (avg across datasets)",
            labels=["Clean", "Adversarial"],
            values=[
                zs["avg"]["avg_zs_clean_acc"],
                zs["avg"]["avg_zs_adv_acc"],
            ],
            ylabel="Accuracy (%)",
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
            out_root="Results_AvgAcrossDatasets",
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

                    # For TPT bar plots: only plot single, vanilla, weighted, and original.
                    # The "original" baseline should be displayed as "Zero-shot".
                    allowed_pred_variants = ["original", "single", "vanilla", "weighted"]
                    pred_variants = [pv for pv in allowed_pred_variants if pv in pred_variants_all]
                    print(f"      pred_variants (available) = {pred_variants_all}")
                    print(f"      pred_variants (plotted)   = {pred_variants}")

                    pred_variant_display = {
                        "original": "Zero-shot",
                        "single": "Single",
                        "vanilla": "Vanilla",
                        "weighted": "Weighted",
                    }

                    summary = {
                        "attack": attack,
                        "model": model,
                        "tpt_type": tpt_type,
                        "datasets": datasets,
                        "pred_variants": pred_variants,
                        "pred_variant_display": {pv: pred_variant_display.get(pv, pv) for pv in pred_variants},
                        "per_pred_variant": {},
                    }

                    for pv in pred_variants:
                        per_ds_metrics = {}
                        for ds in datasets:
                            obj = tpt_dic[attack][model][ds][tpt_type]["preds"][pv]

                            # Source-of-truth labels (avoid any mismatch/leakage)
                            labels = TRUE_LABELS_DATASET[ds]
                            preds = obj["prediction"]
                            _ensure_same_len(preds, labels, name=f"{attack}/{model}/{ds}/{tpt_type}/{pv}")

                            acc = accuracy_percent(preds, labels)
                            per_ds_metrics[ds] = {
                                "acc": acc,
                                "num_samples": len(labels),
                            }

                        avg_acc = avg_across_datasets({d: per_ds_metrics[d]["acc"] for d in datasets}, datasets)

                        summary["per_pred_variant"][pv] = {
                            "per_dataset": per_ds_metrics,
                            "avg_acc_across_datasets": avg_acc,
                        }

                        print(f"      [{pv}] avg_acc={avg_acc:.2f}")

                    # Save json
                    save_json(tpt_root / "tpt_avg_across_datasets.json", summary)

                    # Plot avg accuracy across datasets per pred_variant
                    labels_pv = [pred_variant_display.get(pv, pv) for pv in pred_variants]
                    values_avg = [summary["per_pred_variant"][pv]["avg_acc_across_datasets"] for pv in pred_variants]

                    save_bar_plot(
                        tpt_root / "avg_acc_across_datasets.png",
                        f"{model} | {tpt_type} | {attack} | Avg accuracy across datasets",
                        labels=labels_pv,
                        values=values_avg,
                        ylabel="Accuracy (%)",
                        ylim=(0, 100),
                    )

        print(f"\n[Done] Outputs written to: {out_root.resolve()}")


    # ============================================================
    # RUN (restrict to vit_l_14_datacomp_1b as per your earlier example)
    # ============================================================
    # ============================================================
    # RUN (sequence: zero-shot first, then TPT)
    # ============================================================
    out_root = "Results_AvgAcrossDatasets"
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




