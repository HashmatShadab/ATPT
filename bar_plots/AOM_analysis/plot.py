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

def create_image_grid(image_paths, output_path, cols=3):
    """
    Creates a grid of images.
    """
    if not image_paths:
        return

    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)

    rows = (len(images) + cols - 1) // cols
    grid_width = cols * max_width
    grid_height = rows * max_height

    new_im = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    for i, im in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(im, (col * max_width, row * max_height))

    new_im.save(output_path)
    print(f"Saved grid to: {output_path}")

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


def get_aom_results(model_name):
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

    anchor_noises = ["Sigma_0_06", "Sigma_0_12", "Sigma_0_18", "Sigma_0_24", "Eps_16_0", "Eps_32_0", "Eps_40_0", "Eps_48_0"]
    normalize_embeddings = ["False", "True"]

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
        "AOM": {
            "clean": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau_AOM/"
                    "{model}/{dataset}/"
                    "Clean/No_Counter_Attack/No_TPT/Image_Feature_Purify/Type_{noisy}_anchor_Anchors_10_Alpha_ablation_{noise_anchor}_threshold_0_0_normalize_embeddings_{normalize_embeddings}/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            "adversarial_eps4_steps100": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau_AOM/"
                    "{model}/{dataset}/"
                    "Adversarial_Eps_4_0_Steps_100/"
                    "No_Counter_Attack/No_TPT/Image_Feature_Purify/Type_{noisy}_anchor_Anchors_10_Alpha_ablation_{noise_anchor}_threshold_0_0_normalize_embeddings_{normalize_embeddings}/"
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
            preds[key] = {}
            for alpha, value in obj.items():
                preds[key][alpha] = _load_pred_json(value)

        # --- original_clean json (special extras)
        fp = base / JSON_KEYMAP["original_clean"]
        obj = _load_json(fp)
        preds["original_clean"] = _load_pred_json(obj)

        # --- counter-attack diff-ratio json
        # fp = base / JSON_KEYMAP["results_counter_attack_diff_ratio"]
        # obj = _load_json(fp)
        # counter_attack = diff_ratio(obj)

        return {"preds": preds}

    def build_all_data(result_paths: dict, models: list, datasets: list, anchor_noises: list, normalize_embeddings: list) -> dict:
        DATA = {}

        for case, cfg in result_paths["AOM"].items():
            DATA.setdefault(case, {})
            base_template = cfg["base_path"]

            for model in models:
                DATA[case].setdefault(model, {})

                for dataset in datasets:
                    DATA[case][model].setdefault(dataset, {})
                    for noise_anchor in anchor_noises:
                        if noise_anchor.startswith("Sigma"):
                            noise_anchor_name = "noisy"
                        else:
                            noise_anchor_name = "uniform"
                        DATA[case][model][dataset].setdefault(f"{noise_anchor_name}_{noise_anchor}", {})
                        for normalize_embedding in normalize_embeddings:
                            base_path = base_template.format(
                                model=model,
                                dataset=dataset,
                                noisy=noise_anchor_name,
                                noise_anchor=noise_anchor,
                                normalize_embeddings=normalize_embedding,
                            )
                            DATA[case][model][dataset][f"{noise_anchor_name}_{noise_anchor}"][normalize_embedding] = load_one_setting(base_path)

        return DATA

    AOM_DATA = build_all_data(RESULT_PATHS, MODELS, DATASETS, anchor_noises, normalize_embeddings)

    return AOM_DATA

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


    aom_dic = get_aom_results(
        model_name,

    )

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Dict, List


    # -------------------------
    # Plot styling
    # -------------------------
    def apply_plot_style():
        """Apply a consistent, publication-friendly Matplotlib style."""
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "font.size": 11,
        })


    # -------------------------
    # Utils
    # -------------------------
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


    def compute_accuracy(preds, labels):
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        return (preds == labels).mean() * 100.0


    def conservative_accuracy_from_mask(preds, labels, clean_correct_mask):
        """
        Conservative accuracy:
        - Evaluate correctness only on indices where clean ZS was correct
        - Normalize by TOTAL number of samples
        """
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        mask = np.asarray(clean_correct_mask, dtype=bool)

        assert len(preds) == len(labels) == len(mask)

        total = len(preds)
        if total == 0:
            return np.nan

        # Correct predictions ONLY where clean was correct
        num_correct = (preds[mask] == labels[mask]).sum()

        return float(num_correct / total * 100.0)


    def alpha_sort_key(a: str):
        try:
            return float(a)
        except Exception:
            return float("inf")


    # -------------------------
    # Zero-shot aggregation
    # -------------------------
    def aggregate_zero_shot(TRUE_LABELS_DATASET,
                            ZS_CLEAN_PREDS_DATASET,
                            ZS_ADV_PREDS_DATASET):

        datasets = list(TRUE_LABELS_DATASET.keys())

        clean_accs, adv_accs = [], []

        for d in datasets:
            labels = TRUE_LABELS_DATASET[d]
            clean_accs.append(
                compute_accuracy(ZS_CLEAN_PREDS_DATASET[d], labels)
            )
            adv_accs.append(
                compute_accuracy(ZS_ADV_PREDS_DATASET[d], labels)
            )

        return {
            "clean_avg": float(np.mean(clean_accs)),
            "adv_avg": float(np.mean(adv_accs)),
            "clean_std": float(np.std(clean_accs)),
            "adv_std": float(np.std(adv_accs)),
        }


    # -------------------------
    # AOM aggregation
    # -------------------------
    def aggregate_aom_conservative_accuracy(aom_dic,
                                            model_name,
                                            datasets,
                                            ZS_CLEAN_CORRECT_DATASET):
        """
        Returns:
        AOM_AVG[attack][noise_anchor][normalize] = {
            "alphas": [...],
            "avg": [...]
        }
        """

        AOM_AVG = {}

        for attack in aom_dic.keys():
            if model_name not in aom_dic[attack]:
                continue

            AOM_AVG.setdefault(attack, {})

            # discover anchors
            for d in datasets:
                anchors = list(aom_dic[attack][model_name][d].keys())
                break

            for noise_anchor in anchors:
                AOM_AVG[attack].setdefault(noise_anchor, {})

                normalize_keys = aom_dic[attack][model_name][d][noise_anchor].keys()

                for normalize in normalize_keys:
                    # discover alpha keys
                    for d in datasets:
                        try:
                            alpha_keys = list(
                                aom_dic[attack][model_name][d]
                                [noise_anchor][normalize]
                                ["preds"]["single"].keys()
                            )
                            break
                        except Exception:
                            continue

                    alpha_keys = sorted(alpha_keys, key=alpha_sort_key)
                    alphas = [float(a) for a in alpha_keys]

                    avg_vals = []

                    for a in alpha_keys:
                        per_dataset_acc = []

                        for d in datasets:
                            entry = (
                                aom_dic[attack][model_name][d]
                                [noise_anchor][normalize]
                                ["preds"]["single"][a]
                            )

                            preds = entry["prediction"]
                            labels = entry["label"]
                            clean_mask = ZS_CLEAN_CORRECT_DATASET[d]

                            acc = conservative_accuracy_from_mask(
                                preds, labels, clean_mask
                            )
                            per_dataset_acc.append(acc)

                        avg_vals.append(float(np.nanmean(per_dataset_acc)))

                    AOM_AVG[attack][noise_anchor][normalize] = {
                        "alphas": alphas,
                        "avg": avg_vals,
                    }

        return AOM_AVG


    # -------------------------
    # Plotting
    # -------------------------
    def plot_aom_bars(
            alphas,
            clean_avg,
            adv_avg,
            title,
            outpath,
            *,
            note_alpha0_is_zs: bool = True,
    ):
        """Grouped bar plot: for each alpha show Clean vs Adversarial avg accuracy.

        Important: `alpha == 0.0` corresponds to zero-shot results (not AOM), and should
        be labeled accordingly.
        """

        ensure_dir(os.path.dirname(outpath))

        alphas = [float(a) for a in alphas]
        clean_avg = np.asarray(clean_avg, dtype=float)
        adv_avg = np.asarray(adv_avg, dtype=float)

        apply_plot_style()

        fig, ax = plt.subplots(figsize=(10.2, 4.9))

        x = np.arange(len(alphas))
        width = 0.38

        # Colorblind-friendly palette (Okabe-Ito)
        clean_color = "#0072B2"  # blue
        adv_color = "#D55E00"    # vermillion

        clean_bars = ax.bar(
            x - width / 2,
            clean_avg,
            width=width,
            label="Clean",
            color=clean_color,
            edgecolor="black",
            linewidth=0.6,
        )
        adv_bars = ax.bar(
            x + width / 2,
            adv_avg,
            width=width,
            label="Adversarial",
            color=adv_color,
            edgecolor="black",
            linewidth=0.6,
        )

        # tick labels
        tick_labels = []
        for a in alphas:
            if note_alpha0_is_zs and abs(a - 0.0) < 1e-12:
                tick_labels.append("0.0\n(ZS)")
            else:
                tick_labels.append(f"{a:g}")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)

        max_val = float(np.nanmax([np.nanmax(clean_avg), np.nanmax(adv_avg)]))
        ax.set_ylim(0, max(100.0, max_val + 4.0))
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"Average accuracy across datasets (\%)")
        ax.set_title(title)

        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=2,
            handlelength=1.4,
            columnspacing=1.2,
        )

        # value labels above bars
        def _add_bar_value_labels(bar_container):
            for rect in bar_container:
                h = rect.get_height()
                if h is None or not np.isfinite(h):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + 0.25,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#222222",
                    clip_on=True,
                )

        _add_bar_value_labels(clean_bars)
        _add_bar_value_labels(adv_bars)

        if note_alpha0_is_zs:
            fig.text(
                0.995,
                0.01,
                "ZS = Zero-shot baseline (alpha = 0.0)",
                ha="right",
                va="bottom",
                fontsize=10,
                color="#444444",
            )

        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        fig.savefig(outpath, dpi=250)
        plt.close(fig)


    def _parse_noise_param_key(k: str) -> Optional[Dict[str, float]]:
        """Parse keys like 'Sigma_0.06' or 'Eps_32.0' into {'name': 'Sigma', 'value': 0.06}."""
        if not isinstance(k, str) or "_" not in k:
            return None
        name, rest = k.split("_", 1)
        try:
            val = float(rest)
        except Exception:
            return None
        return {"name": name, "value": val}


    def _find_noise_param_key(keys, *, name: str, value: float, tol: float = 1e-9) -> Optional[str]:
        """Find a matching noise-param key in `final_diff_ratio_dic` robustly (float formatting differs)."""
        best = None
        best_dist = float("inf")
        for k in keys:
            parsed = _parse_noise_param_key(k)
            if not parsed:
                continue
            if parsed["name"] != name:
                continue
            dist = abs(parsed["value"] - float(value))
            if dist < best_dist:
                best = k
                best_dist = dist
        if best is None:
            return None
        if best_dist <= tol:
            return best
        # still allow a close match (e.g., 32.0 vs 32)
        if best_dist <= 1e-3:
            return best
        return None


    def compute_threshold_mixed_accuracy(
            *,
            aom_preds: List[int],
            zs_preds: List[int],
            labels: List[int],
            diff_ratio: List[float],
            threshold: float,
    ) -> float:
        """Accuracy when selecting between AOM and ZS prediction per-sample by diff-ratio threshold."""
        aom_preds = np.asarray(aom_preds)
        zs_preds = np.asarray(zs_preds)
        labels = np.asarray(labels)
        diff_ratio = np.asarray(diff_ratio, dtype=float)

        assert len(aom_preds) == len(zs_preds) == len(labels) == len(diff_ratio), (
            f"Length mismatch: aom={len(aom_preds)} zs={len(zs_preds)} labels={len(labels)} diff={len(diff_ratio)}"
        )

        use_aom = diff_ratio > float(threshold)
        mixed = np.where(use_aom, aom_preds, zs_preds)
        return float((mixed == labels).mean() * 100.0)


    def compute_threshold_mixed_conservative_accuracy(
            *,
            aom_preds: List[int],
            zs_preds: List[int],
            labels: List[int],
            diff_ratio: List[float],
            threshold: float,
            clean_correct_mask: List[bool],
    ) -> float:
        """Conservative accuracy for threshold mixing.

        Conservative accuracy definition used elsewhere in this script:
        - Evaluate correctness only on indices where clean ZS was correct
        - Normalize by TOTAL number of samples
        """
        aom_preds = np.asarray(aom_preds)
        zs_preds = np.asarray(zs_preds)
        labels = np.asarray(labels)
        diff_ratio = np.asarray(diff_ratio, dtype=float)

        assert len(aom_preds) == len(zs_preds) == len(labels) == len(diff_ratio), (
            f"Length mismatch: aom={len(aom_preds)} zs={len(zs_preds)} labels={len(labels)} diff={len(diff_ratio)}"
        )

        use_aom = diff_ratio > float(threshold)
        mixed = np.where(use_aom, aom_preds, zs_preds)

        return conservative_accuracy_from_mask(mixed, labels, clean_correct_mask)


    def plot_threshold_mixing_bars(
            *,
            thresholds: List[float],
            alpha_to_acc: Dict[float, List[float]],
            title: str,
            outpath: str,
            ylabel: str = "Average accuracy across datasets (%)",
    ):
        """Grouped bar plot: x-axis = $\tau$ thresholds, bars = different $\alpha$ values."""
        ensure_dir(os.path.dirname(outpath))

        apply_plot_style()

        fig, ax = plt.subplots(figsize=(13, 6))

        thresholds = [float(t) for t in thresholds]
        x = np.arange(len(thresholds))

        alphas_sorted = sorted(alpha_to_acc.keys())
        n_series = max(1, len(alphas_sorted))

        group_width = 0.82
        bar_width = group_width / n_series

        # Colorblind-friendly cycle for multiple series
        series_colors = [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#009E73",  # bluish green
            "#CC79A7",  # reddish purple
            "#F0E442",  # yellow
            "#56B4E9",  # sky blue
        ]

        def _add_bar_value_labels(bar_container):
            for rect in bar_container:
                h = rect.get_height()
                if h is None or not np.isfinite(h):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + 0.20,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#222222",
                    clip_on=True,
                )

        for i, alpha in enumerate(alphas_sorted):
            y = np.asarray(alpha_to_acc[alpha], dtype=float)
            if y.shape[0] != len(thresholds):
                raise ValueError(
                    f"alpha_to_acc length mismatch for alpha={alpha}: "
                    f"got {y.shape[0]} values but {len(thresholds)} thresholds"
                )
            offset = -group_width / 2 + (i + 0.5) * bar_width
            color = series_colors[i % len(series_colors)]
            bars = ax.bar(
                x + offset,
                y,
                width=bar_width,
                label=rf"$\alpha={float(alpha):g}$",
                color=color,
                edgecolor="black",
                linewidth=0.55,
            )
            _add_bar_value_labels(bars)

        ax.set_xticks(x)
        # Mark \tau-threshold=0.0 explicitly as the original AOM setting (no thresholding).
        tick_labels = []
        for t in thresholds:
            if abs(float(t) - 0.0) < 1e-12:
                tick_labels.append("0.0\n(AOM)")
            else:
                tick_labels.append(f"{t:.2f}".rstrip("0").rstrip("."))
        ax.set_xticklabels(tick_labels, fontsize=16)
        ax.set_xlabel(r"$\tau_{\mathrm{threshold}}$", fontsize=20)
        ax.set_ylabel(ylabel.replace("(%)", r"(%)"), fontsize=18)
        ax.set_title(title, pad=30)

        # Corner note clarifying that \tau-threshold=0.0 corresponds to original AOM (no thresholding).
        note_text = r"$\tau_{\mathrm{threshold}}=0.0$ corresponds to original AOM (no thresholding)"
        fig.text(
            0.995,
            0.01,
            note_text,
            ha="right",
            va="bottom",
            fontsize=16,
            color="#000000",
            alpha=0.35,
        )
        ax.legend(
            frameon=True,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=min(4, n_series),
            columnspacing=1.0,
            handlelength=1.3,
            fontsize=10,
        )

        # Extra bottom margin for the explanatory note.
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(outpath, dpi=250)
        plt.close(fig)


    # -------------------------
    # Main execution
    # -------------------------
    def run_all_plots(model_name, out_root="plots"):
        datasets = list(TRUE_LABELS_DATASET.keys())

        # Zero-shot baselines
        zs = aggregate_zero_shot(
            TRUE_LABELS_DATASET,
            ZS_CLEAN_PREDS_DATASET,
            ZS_ADV_PREDS_DATASET
        )

        # AOM conservative aggregation
        aom_avg = aggregate_aom_conservative_accuracy(
            aom_dic,
            model_name,
            datasets,
            ZS_CLEAN_CORRECT_DATASET
        )

        # We want a single plot per (noise_anchor, normalize) showing both conditions.
        clean_key = "clean"
        adv_key = "adversarial_eps4_steps100"

        if clean_key not in aom_avg or adv_key not in aom_avg:
            raise KeyError(
                f"Expected AOM results for both '{clean_key}' and '{adv_key}'. "
                f"Found keys: {sorted(aom_avg.keys())}"
            )

        for noise_anchor, anchor_obj in aom_avg[clean_key].items():
            for normalize, clean_series in anchor_obj.items():
                if noise_anchor not in aom_avg[adv_key] or normalize not in aom_avg[adv_key][noise_anchor]:
                    continue

                adv_series = aom_avg[adv_key][noise_anchor][normalize]

                # Build alpha -> avg maps
                clean_map = {float(a): float(v) for a, v in zip(clean_series["alphas"], clean_series["avg"])}
                adv_map = {float(a): float(v) for a, v in zip(adv_series["alphas"], adv_series["avg"])}

                all_alphas = sorted(set(clean_map.keys()) | set(adv_map.keys()))
                if 0.0 not in all_alphas:
                    all_alphas = [0.0] + all_alphas

                # alpha=0.0 is explicitly zero-shot baseline
                clean_vals = []
                adv_vals = []
                for a in all_alphas:
                    if abs(a - 0.0) < 1e-12:
                        clean_vals.append(zs["clean_avg"])
                        adv_vals.append(zs["adv_avg"])
                    else:
                        clean_vals.append(clean_map.get(a, np.nan))
                        adv_vals.append(adv_map.get(a, np.nan))

                outdir = os.path.join(
                    out_root,
                    model_name,
                    "AOM",
                    noise_anchor,
                    f"normalize_{normalize}"
                )
                outpath = os.path.join(outdir, "clean_vs_adversarial_bars.png")

                title = (
                    "Average accuracy across Datasets\n"
                    f"Anchor={noise_anchor}, Normalize={normalize}"
                )

                plot_aom_bars(
                    all_alphas,
                    clean_vals,
                    adv_vals,
                    title,
                    outpath,
                    note_alpha0_is_zs=True,
                )

                print(f"[Saved] {outpath}")

        # ------------------------------------------------------------------
        # NEW: Thresholded mixing analysis (AOM vs ZS) using diff-ratio per sample
        # Settings requested: normalize=True and anchors {Sigma_0.06, uniform eps32}
        # ------------------------------------------------------------------
        target_normalize = "True"
        target_anchors = ["noisy_Sigma_0_06", "noisy_Sigma_0_12", "noisy_Sigma_0_18", "uniform_Eps_32_0", "uniform_Eps_40_0", "uniform_Eps_48_0"]

        # Collect per-anchor threshold plots so we can generate grids at the end.
        # Structure: plots_for_grid[diff_attack]["standard"|"conservative"] -> [image paths]
        plots_for_grid = {
            "Clean": {"standard": [], "conservative": []},
            "Adversarial": {"standard": [], "conservative": []},
            "Average": {"standard": [], "conservative": []},
        }

        # Attack mapping between AOM keys and diff-ratio keys / ZS predictions
        attack_specs = [
            {
                "aom_attack": "clean",
                "diff_attack": "Clean",
                "zs_preds": ZS_CLEAN_PREDS_DATASET,
            },
            {
                "aom_attack": "adversarial_eps4_steps100",
                "diff_attack": "Adversarial",
                "zs_preds": ZS_ADV_PREDS_DATASET,
            },
        ]

        # Thresholds requested (fixed)
        thresholds = [0.0, 0.2, 0.40, 0.60, 0.80, 0.85, 0.90, 0.95,  1.0]

        # Only evaluate these alpha values for threshold plots (requested)
        target_threshold_alphas = [1.0, 1.2, 1.4]

        # Helper: get diff-ratio vector for a dataset + anchor selection
        def _get_diff_ratio_vector(dataset: str, *, diff_attack: str, anchor_key: str) -> Optional[List[float]]:
            if dataset not in final_diff_ratio_dic:
                return None
            if diff_attack not in final_diff_ratio_dic[dataset]:
                return None

            if anchor_key == "noisy_Sigma_0_06":
                noise_type = "Gaussian"
                name, value = "Sigma", 0.06
            elif anchor_key == "noisy_Sigma_0_12":
                noise_type = "Gaussian"
                name, value = "Sigma", 0.12
            elif anchor_key == "noisy_Sigma_0_18":
                noise_type = "Gaussian"
                name, value = "Sigma", 0.18
            elif anchor_key == "uniform_Eps_32_0":
                noise_type = "Uniform"
                name, value = "Eps", 32.0
            elif anchor_key == "uniform_Eps_40_0":
                noise_type = "Uniform"
                name, value = "Eps", 40.0
            elif anchor_key == "uniform_Eps_48_0":
                noise_type = "Uniform"
                name, value = "Eps", 48.0
            else:
                return None

            noise_obj = final_diff_ratio_dic[dataset][diff_attack].get(noise_type, {})
            key = _find_noise_param_key(noise_obj.keys(), name=name, value=value)
            if key is None:
                return None
            return noise_obj.get(key, None)

        for anchor_key in target_anchors:
            # Short, presentation-friendly anchor description (used in titles).
            if anchor_key == "noisy_Sigma_0_06":
                anchor_desc = r"Gaussian ($\sigma=0.06$)"
            elif anchor_key == "noisy_Sigma_0_12":
                anchor_desc = r"Gaussian ($\sigma=0.12$)"
            elif anchor_key == "noisy_Sigma_0_18":
                anchor_desc = r"Gaussian ($\sigma=0.18$)"
            elif anchor_key == "uniform_Eps_32_0":
                anchor_desc = r"Uniform ($\epsilon=32/255$)"
            elif anchor_key == "uniform_Eps_40_0":
                anchor_desc = r"Uniform ($\epsilon=40/255$)"
            elif anchor_key == "uniform_Eps_48_0":
                anchor_desc = r"Uniform ($\epsilon=48/255$)"
            else:
                anchor_desc = str(anchor_key)

            # Keep computed series so we can optionally plot the Clean/Adversarial average.
            computed_by_attack = {}

            for spec in attack_specs:
                aom_attack = spec["aom_attack"]
                diff_attack = spec["diff_attack"]
                zs_preds_by_dataset = spec["zs_preds"]

                # Build alpha list from AOM predictions for this anchor/normalize
                alpha_keys = None
                for d in datasets:
                    try:
                        alpha_keys = list(
                            aom_dic[aom_attack][model_name][d][anchor_key][target_normalize]
                            ["preds"]["single"].keys()
                        )
                        break
                    except Exception:
                        continue
                if not alpha_keys:
                    continue

                # Restrict to requested alphas only
                alpha_keys = [a for a in alpha_keys if float(a) in set(target_threshold_alphas)]
                if not alpha_keys:
                    continue
                alpha_keys = sorted(alpha_keys, key=alpha_sort_key)
                alphas = [float(a) for a in alpha_keys]

                alpha_to_acc = {float(a): [] for a in alphas}
                alpha_to_cons_acc = {float(a): [] for a in alphas}

                for t in thresholds:
                    for a in alpha_keys:
                        per_dataset_acc = []
                        per_dataset_cons_acc = []
                        for d in datasets:
                            try:
                                entry = (
                                    aom_dic[aom_attack][model_name][d][anchor_key][target_normalize]
                                    ["preds"]["single"][a]
                                )
                            except Exception:
                                continue

                            aom_preds = entry["prediction"]
                            labels = TRUE_LABELS_DATASET[d]
                            zs_preds = zs_preds_by_dataset[d]

                            clean_mask = ZS_CLEAN_CORRECT_DATASET[d]

                            dr = _get_diff_ratio_vector(d, diff_attack=diff_attack, anchor_key=anchor_key)
                            if dr is None:
                                continue

                            acc = compute_threshold_mixed_accuracy(
                                aom_preds=aom_preds,
                                zs_preds=zs_preds,
                                labels=labels,
                                diff_ratio=dr,
                                threshold=float(t),
                            )
                            per_dataset_acc.append(acc)

                            cons_acc = compute_threshold_mixed_conservative_accuracy(
                                aom_preds=aom_preds,
                                zs_preds=zs_preds,
                                labels=labels,
                                diff_ratio=dr,
                                threshold=float(t),
                                clean_correct_mask=clean_mask,
                            )
                            per_dataset_cons_acc.append(cons_acc)

                        alpha_to_acc[float(a)].append(float(np.nanmean(per_dataset_acc)) if per_dataset_acc else np.nan)
                        alpha_to_cons_acc[float(a)].append(
                            float(np.nanmean(per_dataset_cons_acc)) if per_dataset_cons_acc else np.nan
                        )

                computed_by_attack[diff_attack] = {
                    "alpha_to_acc": alpha_to_acc,
                    "alpha_to_cons_acc": alpha_to_cons_acc,
                }

                outdir = os.path.join(
                    out_root,
                    model_name,
                    "AOM",
                    "threshold_mixing",
                    anchor_key,
                    f"normalize_{target_normalize}",
                    diff_attack,
                )
                outpath = os.path.join(outdir, "accuracy_vs_threshold.png")
                outpath_cons = os.path.join(outdir, "conservative_accuracy_vs_threshold.png")

                # NOTE: Avoid raw strings for the newline; in a raw string, "\n" is literal.
                title = (
                    r"$\tau$-thresholded AOM (Label Leakage): use AOM if $\tau > \tau_{\mathrm{threshold}}$ else Zero-shot"
                    "\n"
                    rf"Anchor: {anchor_desc}"
                )

                title_cons = (
                    r"$\tau$-thresholded AOM: use AOM if $\tau > \tau_{\mathrm{threshold}}$ else Zero-shot"
                    "\n"
                    rf"Anchor: {anchor_desc}"
                )

                plot_threshold_mixing_bars(
                    thresholds=thresholds,
                    alpha_to_acc=alpha_to_acc,
                    title=title,
                    outpath=outpath,
                )

                plot_threshold_mixing_bars(
                    thresholds=thresholds,
                    alpha_to_acc=alpha_to_cons_acc,
                    title=title_cons,
                    outpath=outpath_cons,
                    ylabel="Average accuracy across datasets (%)",
                )

                # Add to grid lists (keep anchor order as in `target_anchors`).
                if diff_attack in plots_for_grid:
                    if os.path.exists(outpath):
                        plots_for_grid[diff_attack]["standard"].append(outpath)
                    if os.path.exists(outpath_cons):
                        plots_for_grid[diff_attack]["conservative"].append(outpath_cons)

                print(f"[Saved] {outpath}")
                print(f"[Saved] {outpath_cons}")

            # NEW: Clean/Adversarial average plots (single plot rather than separate).
            if "Clean" in computed_by_attack and "Adversarial" in computed_by_attack:
                clean_std = computed_by_attack["Clean"]["alpha_to_acc"]
                adv_std = computed_by_attack["Adversarial"]["alpha_to_acc"]
                clean_cons = computed_by_attack["Clean"]["alpha_to_cons_acc"]
                adv_cons = computed_by_attack["Adversarial"]["alpha_to_cons_acc"]

                common_alphas = sorted(set(clean_std.keys()) & set(adv_std.keys()))
                if common_alphas:
                    alpha_to_acc_avg = {}
                    alpha_to_cons_acc_avg = {}

                    for a in common_alphas:
                        c = np.asarray(clean_std[a], dtype=float)
                        v = np.asarray(adv_std[a], dtype=float)
                        alpha_to_acc_avg[float(a)] = list(((c + v) / 2.0).astype(float))

                        c2 = np.asarray(clean_cons[a], dtype=float)
                        v2 = np.asarray(adv_cons[a], dtype=float)
                        alpha_to_cons_acc_avg[float(a)] = list(((c2 + v2) / 2.0).astype(float))

                    avg_attack = "Average"
                    outdir_avg = os.path.join(
                        out_root,
                        model_name,
                        "AOM",
                        "threshold_mixing",
                        anchor_key,
                        f"normalize_{target_normalize}",
                        avg_attack,
                    )
                    outpath_avg = os.path.join(outdir_avg, "accuracy_vs_threshold.png")
                    outpath_avg_cons = os.path.join(outdir_avg, "conservative_accuracy_vs_threshold.png")

                    title_avg = (
                        r"$\tau$-thresholded AOM (Avg. Clean/Adv.): use AOM if $\tau > \tau_{\mathrm{threshold}}$ else Zero-shot"
                        "\n"
                        rf"Anchor: {anchor_desc}"
                    )
                    title_avg_cons = (
                        r"$\tau$-thresholded AOM (Conservative, Avg. Clean/Adv.): use AOM if $\tau > \tau_{\mathrm{threshold}}$ else Zero-shot"
                        "\n"
                        rf"Anchor: {anchor_desc}"
                    )

                    plot_threshold_mixing_bars(
                        thresholds=thresholds,
                        alpha_to_acc=alpha_to_acc_avg,
                        title=title_avg,
                        outpath=outpath_avg,
                    )
                    plot_threshold_mixing_bars(
                        thresholds=thresholds,
                        alpha_to_acc=alpha_to_cons_acc_avg,
                        title=title_avg_cons,
                        outpath=outpath_avg_cons,
                        ylabel="Average accuracy across datasets (%)",
                    )

                    if os.path.exists(outpath_avg):
                        plots_for_grid[avg_attack]["standard"].append(outpath_avg)
                    if os.path.exists(outpath_avg_cons):
                        plots_for_grid[avg_attack]["conservative"].append(outpath_avg_cons)

                    print(f"[Saved] {outpath_avg}")
                    print(f"[Saved] {outpath_avg_cons}")

        # -------------------------
        # NEW: Create grids for threshold-mixing plots
        # -------------------------
        for diff_attack in ["Clean", "Adversarial", "Average"]:
            std_paths = plots_for_grid.get(diff_attack, {}).get("standard", [])
            cons_paths = plots_for_grid.get(diff_attack, {}).get("conservative", [])

            if not std_paths and not cons_paths:
                continue

            grid_dir = os.path.join(
                out_root,
                model_name,
                "AOM",
                "threshold_mixing",
                "grids",
                f"normalize_{target_normalize}",
                diff_attack,
            )
            ensure_dir(grid_dir)

            # Use 3 columns by default (works well for 6 anchors).
            grid_cols = 3

            if std_paths:
                grid_out = os.path.join(grid_dir, "accuracy_vs_threshold_grid.png")
                create_image_grid(std_paths, grid_out, cols=grid_cols)

            if cons_paths:
                grid_out_cons = os.path.join(grid_dir, "conservative_accuracy_vs_threshold_grid.png")
                create_image_grid(cons_paths, grid_out_cons, cols=grid_cols)


    run_all_plots(model_name)









