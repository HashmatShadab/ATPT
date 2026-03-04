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
import math
from typing import Any, Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Map adversarial `attack_key` identifiers to human-readable legend labels.
# (Used for plots where we compare Clean vs a single adversarial setting.)
ATTACK_KEY_LEGEND_MAPPING = {
    "eps_4.0_steps_100": "PGD-100 (ε=4/255)",
    "eps_8.0_steps_100": "PGD-100 (ε=8/255)",
}


# ATTACK_NAME_MAPPING = {
#     "eps_0.0_steps_0": "Clean",
#
#     # Epsilon 1/255
#     "eps_1.0_steps_10": "PGD-10 (ε=1/255)",
#     "eps_1.0_steps_10_image_only_attack_prm": "PGD-10 (ε=1/255, Img)",
#     "eps_1.0_steps_100": "PGD-100 (ε=1/255)",
#     "eps_1.0_steps_100_image_only_attack_prm": "PGD-100 (ε=1/255, Img)",
#
#     # Epsilon 4/255
#     "eps_4.0_steps_10": "PGD-10 (ε=4/255)",
#     "eps_4.0_steps_10_image_only_attack_prm": "PGD-10 (ε=4/255, Img)",
#     "eps_4.0_steps_100": "PGD-100 (ε=4/255)",
#     "eps_4.0_steps_100_image_only_attack_prm": "PGD-100 (ε=4/255, Img)",
#
#     # Epsilon 8/255
#     "eps_8.0_steps_10": "PGD-10 (ε=8/255)",
#     "eps_8.0_steps_10_image_only_attack_prm": "PGD-10 (ε=8/255, Img)",
#     "eps_8.0_steps_100": "PGD-100 (ε=8/255)",
#     "eps_8.0_steps_100_image_only_attack_prm": "PGD-100 (ε=8/255, Img)",
# }

def sanitize_for_path(name: str) -> str:
    """Make a string safe to use as a Windows folder/file name component."""
    # Windows invalid chars: < > : " / \ | ? *  plus control chars
    safe = re.sub(r"[<>:\"/\\|?*]", "_", str(name))
    safe = re.sub(r"\s+", "_", safe).strip("._ ")
    return safe or "unnamed"

def create_image_grid(image_paths, output_path, cols=4, scale: float = 1.0):
    """Creates a grid of images.

    Args:
        image_paths: List of image file paths.
        output_path: Output path for the grid image.
        cols: Number of columns in the grid.
        scale: Optional upscaling factor applied to each tile to improve readability.
    """
    if not image_paths:
        return

    images = [Image.open(x) for x in image_paths]
    if scale and scale != 1.0:
        scaled = []
        for im in images:
            w, h = im.size
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            scaled.append(im.resize(new_size, resample=Image.Resampling.LANCZOS))
        images = scaled
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


            if attack == "eps_0.0_steps_100":
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

######################################################################################
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

def get_tau(dataset, noise_type, noise_value, dic):
    tau_clean = np.array(
        dic[dataset]["Clean"][noise_type][noise_value]["diff_ratio"]
    )
    tau_adv = np.array(
        dic[dataset]["Adversarial"][noise_type][noise_value]["diff_ratio"]
    )
    return tau_clean, tau_adv

"""
Analysis A.1 — Mean τ and τ-gap vs noise
"""
def compute_mean_tau_and_gap(dataset, noise_type, dic):
    """
    Compute the mean latent drift (τ) for clean and adversarial samples, and
    their separation (gap), as a function of noise magnitude.

    This analysis quantifies the *average* behavior of CLIP representations
    under increasing random noise. For each noise value, it computes:
        (i) the mean τ for clean inputs,
        (ii) the mean τ for adversarial inputs, and
        (iii) the gap Δτ = mean(τ_adv) − mean(τ_clean).

    The goal is to identify noise regimes where adversarial samples exhibit
    significantly higher latent instability than clean samples. A small or
    negative gap indicates a "false stability" regime where noise fails to
    expose adversarial vulnerability, while a large positive gap indicates
    a regime where adversarial representations destabilize much more rapidly.

    This analysis corresponds to the main trend plots shown in the paper and
    motivates the existence of a critical noise threshold beyond which τ
    becomes informative.

    Args:
        dataset (str): Dataset name (e.g., 'DTD', 'Cars', 'Flower102').
        noise_type (str): Noise distribution type ('Uniform' or 'Gaussian').

    Returns:
        dict: Mapping from noise value (e.g., 'Eps_24.0') to a dictionary with:
            - mean_clean (float): Mean τ for clean samples.
            - mean_adv (float): Mean τ for adversarial samples.
            - gap (float): mean_adv − mean_clean.
            - std_clean (float): Standard deviation of clean τ.
            - std_adv (float): Standard deviation of adversarial τ.
    """

    results = {}
    noise_values = dic[dataset]["Clean"][noise_type].keys()

    for nv in noise_values:
        tau_c, tau_a = get_tau(dataset, noise_type, nv, dic)
        print(noise_type, nv)
        results[nv] = {
            "mean_clean": tau_c.mean(),
            "mean_adv": tau_a.mean(),
            "gap": tau_a.mean() - tau_c.mean(),
            "std_clean": tau_c.std(),
            "std_adv": tau_a.std(),
        }
    return results

"""
Analysis A.2 — Distribution-level separation (stats test)
"""

def tau_distribution_stats(dataset, noise_type, noise_value):
    """
    Compare the full τ distributions of clean and adversarial samples at a
    fixed noise magnitude using non-parametric statistics.

    Rather than relying on averages, this analysis examines whether the τ
    values of adversarial samples are systematically larger than those of
    clean samples across the dataset. It reports robust statistics such as
    medians and interquartile ranges (IQR), and performs a Mann–Whitney U
    test to assess whether the two distributions differ significantly.

    This analysis is critical for ruling out the possibility that observed
    mean differences are driven by a small number of outliers. Strong and
    consistent distributional separation supports the claim that τ captures
    genuine representation instability at the per-sample level.

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type ('Uniform' or 'Gaussian').
        noise_value (str): Noise magnitude identifier (e.g., 'Eps_4.0').

    Returns:
        dict: Dictionary containing:
            - median_clean (float): Median τ for clean samples.
            - median_adv (float): Median τ for adversarial samples.
            - iqr_clean (float): Interquartile range of clean τ.
            - iqr_adv (float): Interquartile range of adversarial τ.
            - p_value (float): Mann–Whitney U test p-value comparing distributions.
    """

    tau_c, tau_a = get_tau(dataset, noise_type, noise_value)

    stat, pval = mannwhitneyu(tau_c, tau_a, alternative="two-sided")

    return {
        "median_clean": np.median(tau_c),
        "median_adv": np.median(tau_a),
        "iqr_clean": np.percentile(tau_c, 75) - np.percentile(tau_c, 25),
        "iqr_adv": np.percentile(tau_a, 75) - np.percentile(tau_a, 25),
        "p_value": pval,
    }

"""
Analysis A.3 — ROC-AUC(τ) vs noise 
"""
def compute_auc_tau(dataset, noise_type):
    """
    Measure how well τ separates clean and adversarial samples using ROC–AUC
    as a function of noise magnitude.

    For each noise value, τ is treated as a scalar score, and the task is to
    discriminate between clean (negative class) and adversarial (positive
    class) samples. The ROC–AUC quantifies the probability that a randomly
    chosen adversarial sample has a higher τ than a randomly chosen clean
    sample.

    An AUC near 0.5 indicates that τ provides no discriminative signal
    (false-stability regime), while an AUC approaching 1.0 indicates strong
    and reliable separation. Plotting AUC versus noise magnitude reveals
    the noise threshold at which τ becomes an effective indicator of
    adversarial instability.

    This analysis provides the most principled justification for selecting
    high probing noise when computing τ.

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type ('Uniform' or 'Gaussian').

    Returns:
        dict: Mapping from noise value to ROC–AUC score.
    """
    aucs = {}
    noise_values = final_diff_ratio_dic[dataset]["Clean"][noise_type].keys()

    for nv in noise_values:
        tau_c, tau_a = get_tau(dataset, noise_type, nv)

        scores = np.concatenate([tau_c, tau_a])
        labels = np.concatenate([
            np.zeros(len(tau_c)),
            np.ones(len(tau_a))
        ])

        aucs[nv] = roc_auc_score(labels, scores)

    return aucs

"""
Analysis A.4 — Paired τ-gap and fraction(τ_adv > τ_clean)
"""
def paired_tau_gap(dataset, noise_type):
    """
    Perform a paired per-sample comparison of τ between clean and adversarial
    inputs originating from the same image.

    For each sample i, this analysis computes:
        g_i = τ_adv_i − τ_clean_i

    It then summarizes how often and by how much adversarial samples exhibit
    larger latent drift than their clean counterparts. Because the comparison
    is paired, this analysis is more sensitive and robust than pooled
    distribution comparisons.

    This analysis directly answers the question:
        "For how many images does the adversarial representation become more
         unstable than the clean one under noise?"

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type.

    Returns:
        dict: Mapping from noise value to:
            - median_gap (float): Median of τ_adv − τ_clean.
            - mean_gap (float): Mean of τ_adv − τ_clean.
            - fraction_adv_greater (float): Fraction of samples where τ_adv > τ_clean.
    """
    gaps = {}
    noise_values = final_diff_ratio_dic[dataset]["Clean"][noise_type].keys()

    for nv in noise_values:
        tau_c, tau_a = get_tau(dataset, noise_type, nv)
        gap = tau_a - tau_c

        gaps[nv] = {
            "median_gap": np.median(gap),
            "mean_gap": gap.mean(),
            "fraction_adv_greater": np.mean(gap > 0),
        }

    return gaps

"""
Analysis A.5 — τ-threshold gating statistics
"""
def tau_threshold_stats(dataset, noise_type, noise_value, quantile=0.95):
    """
    Evaluate τ-based thresholding as a discriminator between clean and
    adversarial samples at a fixed noise magnitude.

    A threshold τ* is defined as a high quantile (e.g., 95th percentile) of
    the clean τ distribution. Samples with τ > τ* are flagged as unstable.

    This analysis computes:
        - False Positive Rate (FPR): fraction of clean samples exceeding τ*.
        - True Positive Rate (TPR): fraction of adversarial samples exceeding τ*.

    By fixing the acceptable clean false-positive rate, this analysis shows
    how effectively adversarial samples can be detected at different noise
    levels and directly informs the design of τ-gated test-time defenses.

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type.
        noise_value (str): Noise magnitude identifier.
        quantile (float): Quantile of clean τ used to define τ*.

    Returns:
        dict: Dictionary with:
            - tau_threshold (float): Selected τ* value.
            - FPR_clean (float): False positive rate on clean samples.
            - TPR_adv (float): True positive rate on adversarial samples.
    """
    tau_c, tau_a = get_tau(dataset, noise_type, noise_value)

    tau_star = np.quantile(tau_c, quantile)

    FPR = np.mean(tau_c > tau_star)
    TPR = np.mean(tau_a > tau_star)

    return {
        "tau_threshold": tau_star,
        "FPR_clean": FPR,
        "TPR_adv": TPR,
    }


"""
Analysis B.1 — Accuracy recovery vs noise (dataset-level)
"""
def accuracy_recovery(dataset, noise_type):
    """
    Measure dataset-level accuracy recovery of adversarial samples as a
    function of noise magnitude.

    This analysis compares the classification accuracy of adversarial inputs
    before and after noise addition. It quantifies whether noise levels that
    induce high τ (latent instability) also lead to improved prediction
    accuracy, thereby linking representation-level behavior to task-level
    robustness.

    This serves as a sanity check that the identified noise regimes are not
    only analytically meaningful but also practically beneficial.

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type.

    Returns:
        dict: Mapping from noise value to:
            - acc_before (float): Adversarial accuracy before noise.
            - acc_after (float): Adversarial accuracy after noise.
            - delta_acc (float): Accuracy improvement due to noise.
    """
    results = {}
    noise_values = final_diff_ratio_dic[dataset]["Adversarial"][noise_type].keys()

    for nv in noise_values:
        info = final_diff_ratio_dic[dataset]["Adversarial"][noise_type][nv]

        acc_before = info["accuracy_before_noise_addition"]
        acc_after = info["accuracy_after_noise_addition"]

        results[nv] = {
            "acc_before": acc_before,
            "acc_after": acc_after,
            "delta_acc": acc_after - acc_before,
        }

    return results

"""
Analysis B.2 — Per-sample recovery vs τ (key new analysis)
"""
def tau_vs_recovery(dataset, noise_type, noise_value):
    """
    Analyze the relationship between τ and per-sample adversarial recovery
    after noise addition.

    Adversarial samples are split into two groups:
        - recovered: prediction after noise matches the ground truth,
        - not recovered: prediction remains incorrect.

    The τ distributions of these two groups are compared to assess whether
    recovered samples tend to exhibit higher latent instability under noise.

    Importantly, this analysis does not claim that τ deterministically predicts
    recovery; rather, it demonstrates a statistical association between large
    τ values and successful robustness restoration.

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type.
        noise_value (str): Noise magnitude identifier.

    Returns:
        dict: Dictionary with:
            - median_tau_recovered (float): Median τ for recovered samples.
            - median_tau_not_recovered (float): Median τ for non-recovered samples.
            - fraction_recovered (float): Fraction of adversarial samples recovered.
    """
    tau_adv = np.array(
        final_diff_ratio_dic[dataset]["Adversarial"][noise_type][noise_value]["diff_ratio"]
    )
    preds_after = np.array(
        final_diff_ratio_dic[dataset]["Adversarial"][noise_type][noise_value]["predictions_after_noise_addition"]
    )

    gt = np.array(TRUE_LABELS_DATASET[dataset])
    recovered = preds_after == gt

    return {
        "median_tau_recovered": np.median(tau_adv[recovered]),
        "median_tau_not_recovered": np.median(tau_adv[~recovered]),
        "fraction_recovered": np.mean(recovered),
    }

"""
Analysis B.3 — Recovery probability as function of τ (binned)
"""
def recovery_probability_vs_tau(dataset, noise_type, noise_value, n_bins=5):
    """
    Estimate the probability of adversarial recovery as a function of τ by
    binning samples according to their latent drift.

    Adversarial samples are partitioned into bins based on τ quantiles, and
    the fraction of recovered samples is computed within each bin. This
    analysis reveals whether higher τ values are associated with a greater
    likelihood of correct prediction after noise.

    The resulting monotonic trend (if present) provides further evidence that
    τ captures meaningful representation instability relevant to robustness,
    while avoiding any claim of hard thresholds or deterministic behavior.

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type.
        noise_value (str): Noise magnitude identifier.
        n_bins (int): Number of τ bins.

    Returns:
        list: Recovery probabilities for each τ bin (ordered from low to high τ).
    """
    tau_adv = np.array(
        final_diff_ratio_dic[dataset]["Adversarial"][noise_type][noise_value]["diff_ratio"]
    )
    preds_after = np.array(
        final_diff_ratio_dic[dataset]["Adversarial"][noise_type][noise_value]["predictions_after_noise_addition"]
    )
    gt = np.array(TRUE_LABELS_DATASET[dataset])
    recovered = preds_after == gt

    bins = np.quantile(tau_adv, np.linspace(0, 1, n_bins + 1))
    probs = []

    for i in range(n_bins):
        mask = (tau_adv >= bins[i]) & (tau_adv < bins[i + 1])
        probs.append(np.mean(recovered[mask]) if mask.any() else np.nan)

    return probs

"""
Analysis C.1 — Explicit τ-gated defense simulation
"""
def tau_gated_accuracy(dataset, noise_type, noise_value, quantile=0.95):
    """
    Simulate a τ-gated test-time defense and evaluate its impact on clean and
    adversarial accuracy.

    A τ threshold τ* is computed from the clean τ distribution. For adversarial
    samples:
        - if τ ≤ τ*, the original adversarial prediction is retained;
        - if τ > τ*, the prediction after noise addition is used.

    This analysis demonstrates how τ-based gating preserves clean accuracy
    while selectively applying noise where it is most beneficial for
    adversarial robustness. It directly mirrors the decision logic used in
    the proposed defense.

    Args:
        dataset (str): Dataset name.
        noise_type (str): Noise distribution type.
        noise_value (str): Noise magnitude identifier.
        quantile (float): Quantile of clean τ used to define τ*.

    Returns:
        dict: Dictionary with:
            - clean_acc (float): Clean accuracy (zero-shot).
            - adv_acc_before (float): Adversarial accuracy before noise.
            - adv_acc_after_gated (float): Adversarial accuracy with τ-gated noise.
            - adv_acc_after_always_noise (float): Adversarial accuracy if noise is
              applied to all samples unconditionally.
    """
    tau_c = np.array(
        final_diff_ratio_dic[dataset]["Clean"][noise_type][noise_value]["diff_ratio"]
    )
    tau_a = np.array(
        final_diff_ratio_dic[dataset]["Adversarial"][noise_type][noise_value]["diff_ratio"]
    )

    tau_star = np.quantile(tau_c, quantile)

    clean_preds = np.array(ZS_CLEAN_PREDS_DATASET[dataset])
    adv_preds_before = np.array(ZS_ADV_PREDS_DATASET[dataset])
    adv_preds_after = np.array(
        final_diff_ratio_dic[dataset]["Adversarial"][noise_type][noise_value]["predictions_after_noise_addition"]
    )
    gt = np.array(TRUE_LABELS_DATASET[dataset])

    # Gated predictions
    gated_preds = adv_preds_before.copy()
    gated_preds[tau_a > tau_star] = adv_preds_after[tau_a > tau_star]

    return {
        "clean_acc": np.mean(clean_preds == gt),
        "adv_acc_before": np.mean(adv_preds_before == gt),
        "adv_acc_after_gated": np.mean(gated_preds == gt),
        "adv_acc_after_always_noise": np.mean(adv_preds_after == gt),
    }

def exclude_noise_values(final_diff_ratio_dic, noise_type, exclude_list):
    """
    Remove selected noise values from final_diff_ratio_dic while keeping
    all other available noise magnitudes intact.

    Args:
        final_diff_ratio_dic (dict): Original diff-ratio dictionary.
        noise_type (str): 'Uniform' or 'Gaussian'.
        exclude_list (list[str]): Noise keys to remove (e.g., ['Eps_48.0']).

    Returns:
        dict: Filtered dictionary with the same structure.
    """
    filtered = {}

    for dataset in final_diff_ratio_dic:
        filtered[dataset] = {"Clean": {}, "Adversarial": {}}

        for attack in ["Clean", "Adversarial"]:
            filtered[dataset][attack][noise_type] = {}

            for nv, content in final_diff_ratio_dic[dataset][attack][noise_type].items():
                if nv not in exclude_list:
                    filtered[dataset][attack][noise_type][nv] = content

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AOM analysis plotting / aggregation")
    parser.add_argument("--model-name", type=str, default="vit_l_14_datacomp_1b")


    args = parser.parse_args()

    model_name = args.model_name

    # Zero-shot results are required for some of the later analyses in this file.
    # For A.1 (mean τ / τ-gap curves) we can run purely from the diff-ratio dumps.
    try:
        zero_shot_dic = get_zs_results(model_name)
        # True labels for all datasets
        TRUE_LABELS_DATASET = zero_shot_dic["true_labels"]
        # Zero-Shot clean predictions for all datasets
        ZS_CLEAN_PREDS_DATASET = zero_shot_dic["zero_shot_clean"]
        # Zero-Shot adversarial predictions for all datasets
        ZS_ADV_PREDS_DATASET = zero_shot_dic["zero_shot_adv"]
        # Zero-Shot clean correct predictions for all datasets
        ZS_CLEAN_CORRECT_DATASET = zero_shot_dic["zero_shot_clean_correct_preds"]
    except FileNotFoundError as e:
        print(f"[WARN] Zero-shot result file not found; continuing without ZS data. ({e})")
        zero_shot_dic = None


    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_diff_ratio = os.path.abspath(os.path.join(script_dir, "..", "..", "Diffratio_v3", model_name))
    selected_attacks = ['eps_0.0_steps_0', 'eps_4.0_steps_100']

    # The analysis code below compares exactly two conditions:
    #   - Clean (eps_0.0_steps_0)
    #   - A single adversarial setting (one of the eps_*_steps_* keys)
    # We keep internal dictionary keys stable as {"Clean", "Adversarial"} so that
    # downstream analysis functions work, but we also preserve the selected
    # adversarial `attack_key` for clarity in saved figure filenames.
    clean_attack_key = 'eps_0.0_steps_0'
    adv_attack_keys = [k for k in selected_attacks if k != clean_attack_key]
    if len(adv_attack_keys) != 1:
        raise ValueError(
            f"Expected exactly one adversarial attack_key in selected_attacks (besides '{clean_attack_key}'), "
            f"but got: {adv_attack_keys}"
        )
    selected_adv_attack_key = adv_attack_keys[0]
    print(f"Loading diff ratio results from: {root_diff_ratio}")
    diff_ratio_dic = get_aggregated_results(root_diff_ratio, selected_attacks=selected_attacks)

    diff_ratio_dic = diff_ratio_dic["results"][model_name]

    final_diff_ratio_dic = {}

    for dataset_key, dataset_value in diff_ratio_dic.items():
        final_diff_ratio_dic[dataset_key] = {}
        for attack_key, attack_value in dataset_value.items():
            if attack_key == clean_attack_key:
                attack_name = "Clean"
            else:
                # Treat the (single) selected adversarial setting as "Adversarial"
                # so that the analysis utilities (which assume a clean-vs-adv split)
                # remain consistent.
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
                        acc_after_noise_addition = tau_type_value.get("counter_attack_accuracy", None)
                        acc_before_noise_addition = tau_type_value.get("adversarial_accuracy", None)
                        predictions_after_noise_addition = tau_type_value.get("counter_attack_predictions", None)
                        predictions_before_noise_addition = tau_type_value.get("adversarial_predictions", None)
                        final_diff_ratio_dic[dataset_key][attack_name][noise_type_name][noise_param_key] = {"diff_ratio": diff_ratio, "accuracy_after_noise_addition": acc_after_noise_addition,
                                                                                                            "accuracy_before_noise_addition": acc_before_noise_addition,
                                                                                                            "predictions_after_noise_addition": predictions_after_noise_addition,
                                                                                                            "predictions_before_noise_addition": predictions_before_noise_addition}

    # UNIFORM_NOISE_EXCLUDE = [
    #     "Eps_4.0",
    #     "Eps_48.0",
    # ]

    UNIFORM_NOISE_EXCLUDE = []
    GAUSSIAN_NOISE_EXCLUDE = []

    if UNIFORM_NOISE_EXCLUDE:
        final_diff_ratio_dic = exclude_noise_values(
            final_diff_ratio_dic,
            noise_type="Uniform",
            exclude_list=UNIFORM_NOISE_EXCLUDE
        )

    if GAUSSIAN_NOISE_EXCLUDE:
        final_diff_ratio_dic = exclude_noise_values(
            final_diff_ratio_dic,
            noise_type="Gaussian",
            exclude_list=GAUSSIAN_NOISE_EXCLUDE
        )


    # A.1

    def compute_mean_tau_all_datasets(noise_type, dic):
        all_results = {}

        for D in dic.keys():
            all_results[D] = compute_mean_tau_and_gap(D, noise_type, dic)

        return all_results


    def aggregate_across_datasets(all_results):
        noise_values = list(next(iter(all_results.values())).keys())

        aggregated = {}

        for nv in noise_values:
            clean_means = []
            adv_means = []
            gaps = []

            for D in all_results:
                stats = all_results[D][nv]
                clean_means.append(stats["mean_clean"])
                adv_means.append(stats["mean_adv"])
                gaps.append(stats["gap"])

            aggregated[nv] = {
                "mean_clean": np.mean(clean_means),
                "mean_adv": np.mean(adv_means),
                "mean_gap": np.mean(gaps),
                "std_gap": np.std(gaps),
            }

        return aggregated


    def eps_from_key(k):
        """Parse noise magnitude strings like 'Eps_4.0' or 'Sigma_0.06' into floats."""
        if k.startswith("Eps_"):
            return float(k.replace("Eps_", ""))
        if k.startswith("Sigma_"):
            return float(k.replace("Sigma_", ""))
        m = re.search(r"([\d.]+)", k)
        if not m:
            raise ValueError(f"Could not parse noise magnitude from key: {k}")
        return float(m.group(1))


    def compute_curve_std(all_results, noise_type):
        std_clean = {}
        std_adv = {}

        for nv in all_results[next(iter(all_results))]:
            clean_vals = []
            adv_vals = []

            for D in all_results:
                clean_vals.append(all_results[D][nv]["mean_clean"])
                adv_vals.append(all_results[D][nv]["mean_adv"])

            std_clean[nv] = np.std(clean_vals)
            std_adv[nv] = np.std(adv_vals)

        return std_clean, std_adv


    def run_a1_and_save_plots(*, noise_type: str, dic: Dict[str, Any], model_name: str, analysis_name: str = "A1", attack_key: str = ""):
        # Guard: skip noise types not present
        sample_dataset = next(iter(dic.keys()))
        if noise_type not in dic[sample_dataset]["Clean"]:
            print(f"[A1] Noise type '{noise_type}' not found in results. Skipping.")
            return

        out_dir = os.path.join("plots_output", analysis_name, model_name, noise_type)
        os.makedirs(out_dir, exist_ok=True)

        all_results = compute_mean_tau_all_datasets(noise_type, dic)
        avg_results = aggregate_across_datasets(all_results)
        sorted_keys = sorted(avg_results.keys(), key=eps_from_key)

        eps = [eps_from_key(k) for k in sorted_keys]

        mean_clean = [avg_results[k]["mean_clean"] for k in sorted_keys]
        mean_adv = [avg_results[k]["mean_adv"] for k in sorted_keys]

        std_clean_dict, std_adv_dict = compute_curve_std(all_results, noise_type)
        std_clean = [std_clean_dict[k] for k in sorted_keys]
        std_adv = [std_adv_dict[k] for k in sorted_keys]

        # Save aggregated stats
        stats_out = {
            "analysis": analysis_name,
            "model": model_name,
            "noise_type": noise_type,
            "attack_key": attack_key,
            "noise_keys_sorted": sorted_keys,
            "eps": eps,
            "avg_results": avg_results,
            "per_dataset_results": all_results,
        }
        stats_name = f"a1_stats_{attack_key}.json" if attack_key else "a1_stats.json"
        stats_path = os.path.join(out_dir, stats_name)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_out, f, indent=2)
        print(f"[A1] Saved stats: {stats_path}")

        def _set_reasonable_ylim(ax, *, y_values, include_zero: bool, symmetric_about_zero: bool = False, min_span: float = 0.1):
            """Make y-axis limits stable/legible (avoid overly tight autoscale)."""
            y = np.asarray(y_values, dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                return

            y_min = float(np.min(y))
            y_max = float(np.max(y))

            if include_zero:
                y_min = min(y_min, 0.0)
                y_max = max(y_max, 0.0)

            span = y_max - y_min
            if span < min_span:
                center = (y_max + y_min) / 2.0
                half = min_span / 2.0
                y_min = center - half
                y_max = center + half
                span = y_max - y_min

            pad = 0.08 * span
            y_min -= pad
            y_max += pad

            if symmetric_about_zero:
                max_abs = max(abs(y_min), abs(y_max), min_span / 2.0)
                y_min = -max_abs
                y_max = max_abs

            ax.set_ylim(y_min, y_max)

        TICK_LABEL_FONTSIZE = 18

        def _set_tick_label_fontsize(ax, fontsize: int = TICK_LABEL_FONTSIZE):
            ax.tick_params(axis="both", which="major", labelsize=fontsize)

        # Plot: mean τ curves
        fig = plt.figure(figsize=(7, 5))
        plt.plot(
            eps, mean_clean,
            marker='o',
            linewidth=2.5,
            label='Clean'
        )
        plt.fill_between(
            eps,
            np.array(mean_clean) - np.array(std_clean),
            np.array(mean_clean) + np.array(std_clean),
            alpha=0.25
        )

        # Use a human-readable legend label for the selected adversarial setting.
        # Fall back to the raw key if we don't have an explicit mapping.
        if attack_key:
            adv_label = ATTACK_KEY_LEGEND_MAPPING.get(attack_key, f"Adversarial ({attack_key})")
        else:
            adv_label = "Adversarial"
        plt.plot(
            eps, mean_adv,
            marker='s',
            linewidth=2.5,
            label=adv_label,
        )
        plt.fill_between(
            eps,
            np.array(mean_adv) - np.array(std_adv),
            np.array(mean_adv) + np.array(std_adv),
            alpha=0.25
        )

        if noise_type == "Gaussian":
            plt.xlabel("Noise Strength σ", fontsize=20)
        else:
            plt.xlabel("Noise Strength ε (/255)", fontsize=20)
        plt.ylabel("Mean Latent Drift τ", fontsize=20)
        plt.title(f"A1: Average Noise–τ Response Across Datasets ({noise_type})", fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.5)

        # Keep y-axis range from becoming overly tight (more consistent across runs)
        ax = plt.gca()
        _set_tick_label_fontsize(ax)
        y_upper_clean = np.array(mean_clean) + np.array(std_clean)
        y_upper_adv = np.array(mean_adv) + np.array(std_adv)
        y_lower_clean = np.array(mean_clean) - np.array(std_clean)
        y_lower_adv = np.array(mean_adv) - np.array(std_adv)
        _set_reasonable_ylim(
            ax,
            y_values=np.concatenate([y_upper_clean, y_upper_adv, y_lower_clean, y_lower_adv]),
            include_zero=True,
            symmetric_about_zero=False,
            min_span=0.1,
        )
        plt.tight_layout()

        tau_curve_name = f"a1_mean_tau_curve_{attack_key}.png" if attack_key else "a1_mean_tau_curve.png"
        tau_curve_path = os.path.join(out_dir, tau_curve_name)
        fig.savefig(tau_curve_path, dpi=200)
        plt.close(fig)
        print(f"[A1] Saved plot: {tau_curve_path}")

        # Plot: mean τ gap
        mean_gap = [avg_results[k]["mean_gap"] for k in sorted_keys]
        fig = plt.figure(figsize=(7, 6))
        plt.plot(eps, mean_gap, marker='d', linewidth=2.5)
        plt.axhline(0, linestyle='--', linewidth=1)

        if noise_type == "Gaussian":
            plt.xlabel("Noise Strength σ", fontsize=20)
        else:
            plt.xlabel("Noise Strength ε (/255)", fontsize=20)
        plt.ylabel("Mean Latent Drift τ", fontsize=20)
        plt.title(f"A1: Adversarial–Clean τ Gap (Averaged Across Datasets) ({noise_type})")
        plt.grid(True, linestyle="--", alpha=0.5)

        # Keep y-axis range stable and always centered around 0 for gap plots
        ax = plt.gca()
        _set_tick_label_fontsize(ax)
        _set_reasonable_ylim(
            ax,
            y_values=np.asarray(mean_gap, dtype=float),
            include_zero=True,
            symmetric_about_zero=True,
            min_span=0.1,
        )
        plt.tight_layout()

        gap_curve_name = f"a1_tau_gap_curve_{attack_key}.png" if attack_key else "a1_tau_gap_curve.png"
        gap_curve_path = os.path.join(out_dir, gap_curve_name)
        fig.savefig(gap_curve_path, dpi=200)
        plt.close(fig)
        print(f"[A1] Saved plot: {gap_curve_path}")

        # ------------------------------------------------------------------
        # Per-dataset plots + grids
        # ------------------------------------------------------------------
        per_dataset_root = os.path.join(out_dir, "per_dataset")
        os.makedirs(per_dataset_root, exist_ok=True)

        per_dataset_tau_paths: List[str] = []
        per_dataset_gap_paths: List[str] = []

        GRID_COLS = 4
        GRID_SCALE = 1.5

        # Preserve deterministic ordering in grids
        for dataset_name in sorted(all_results.keys()):
            dataset_stats = all_results[dataset_name]
            dataset_sorted_keys = sorted(dataset_stats.keys(), key=eps_from_key)
            dataset_eps = [eps_from_key(k) for k in dataset_sorted_keys]

            ds_mean_clean = [dataset_stats[k]["mean_clean"] for k in dataset_sorted_keys]
            ds_mean_adv = [dataset_stats[k]["mean_adv"] for k in dataset_sorted_keys]
            ds_std_clean = [dataset_stats[k]["std_clean"] for k in dataset_sorted_keys]
            ds_std_adv = [dataset_stats[k]["std_adv"] for k in dataset_sorted_keys]
            ds_gap = [dataset_stats[k]["gap"] for k in dataset_sorted_keys]

            dataset_dir = os.path.join(per_dataset_root, sanitize_for_path(dataset_name))
            os.makedirs(dataset_dir, exist_ok=True)

            # Per-dataset: mean τ curves
            fig = plt.figure(figsize=(7, 6))
            plt.plot(
                dataset_eps,
                ds_mean_clean,
                marker='o',
                linewidth=2.5,
                label='Clean',
            )
            plt.fill_between(
                dataset_eps,
                np.array(ds_mean_clean) - np.array(ds_std_clean),
                np.array(ds_mean_clean) + np.array(ds_std_clean),
                alpha=0.25,
            )

            if attack_key:
                adv_label = ATTACK_KEY_LEGEND_MAPPING.get(attack_key, f"Adversarial ({attack_key})")
            else:
                adv_label = "Adversarial"
            plt.plot(
                dataset_eps,
                ds_mean_adv,
                marker='s',
                linewidth=2.5,
                label=adv_label,
            )
            plt.fill_between(
                dataset_eps,
                np.array(ds_mean_adv) - np.array(ds_std_adv),
                np.array(ds_mean_adv) + np.array(ds_std_adv),
                alpha=0.25,
            )

            if noise_type == "Gaussian":
                plt.xlabel("Noise Strength σ", fontsize=28)
            else:
                plt.xlabel("Noise Strength ε (/255)", fontsize=28)
            plt.ylabel("Mean Latent Drift τ", fontsize=28)
            if dataset_name == "eurosat":
                dataset_name = "EuroSAT"
            plt.title(f"{dataset_name}", fontsize=32, fontweight="bold")
            plt.legend(fontsize=20)
            plt.grid(True, linestyle="--", alpha=0.5)

            ax = plt.gca()
            _set_tick_label_fontsize(ax)
            y_upper_clean = np.array(ds_mean_clean) + np.array(ds_std_clean)
            y_upper_adv = np.array(ds_mean_adv) + np.array(ds_std_adv)
            y_lower_clean = np.array(ds_mean_clean) - np.array(ds_std_clean)
            y_lower_adv = np.array(ds_mean_adv) - np.array(ds_std_adv)
            _set_reasonable_ylim(
                ax,
                y_values=np.concatenate([y_upper_clean, y_upper_adv, y_lower_clean, y_lower_adv]),
                include_zero=True,
                symmetric_about_zero=False,
                min_span=0.1,
            )
            plt.tight_layout()

            ds_tau_name = f"a1_mean_tau_curve_{sanitize_for_path(dataset_name)}_{attack_key}.png" if attack_key else f"a1_mean_tau_curve_{sanitize_for_path(dataset_name)}.png"
            ds_tau_path = os.path.join(dataset_dir, ds_tau_name)
            fig.savefig(ds_tau_path, dpi=200)
            plt.close(fig)
            per_dataset_tau_paths.append(ds_tau_path)

            # Per-dataset: τ gap
            fig = plt.figure(figsize=(7, 4))
            plt.plot(dataset_eps, ds_gap, marker='d', linewidth=2.5)
            plt.axhline(0, linestyle='--', linewidth=1)

            if noise_type == "Gaussian":
                plt.xlabel("Noise Strength σ", fontsize=20)
            else:
                plt.xlabel("Noise Strength ε (/255)", fontsize=20)
            plt.ylabel("Mean Latent Drift τ", fontsize=20)
            plt.title(f"{dataset_name}", fontsize=28)
            plt.grid(True, linestyle="--", alpha=0.5)

            ax = plt.gca()
            _set_tick_label_fontsize(ax)
            _set_reasonable_ylim(
                ax,
                y_values=np.asarray(ds_gap, dtype=float),
                include_zero=True,
                symmetric_about_zero=True,
                min_span=0.1,
            )
            plt.tight_layout()

            ds_gap_name = f"a1_tau_gap_curve_{sanitize_for_path(dataset_name)}_{attack_key}.png" if attack_key else f"a1_tau_gap_curve_{sanitize_for_path(dataset_name)}.png"
            ds_gap_path = os.path.join(dataset_dir, ds_gap_name)
            fig.savefig(ds_gap_path, dpi=200)
            plt.close(fig)
            per_dataset_gap_paths.append(ds_gap_path)

        # Grid images across datasets (one for τ curve, one for τ gap)
        if per_dataset_tau_paths:
            tau_grid_name = f"a1_mean_tau_curve_grid_{attack_key}.png" if attack_key else "a1_mean_tau_curve_grid.png"
            tau_grid_path = os.path.join(out_dir, tau_grid_name)
            create_image_grid(per_dataset_tau_paths, tau_grid_path, cols=GRID_COLS, scale=GRID_SCALE)
            print(f"[A1] Saved per-dataset grid: {tau_grid_path}")

        if per_dataset_gap_paths:
            gap_grid_name = f"a1_tau_gap_curve_grid_{attack_key}.png" if attack_key else "a1_tau_gap_curve_grid.png"
            gap_grid_path = os.path.join(out_dir, gap_grid_name)
            create_image_grid(per_dataset_gap_paths, gap_grid_path, cols=GRID_COLS, scale=GRID_SCALE)
            print(f"[A1] Saved per-dataset grid: {gap_grid_path}")

    # Run A.1
    for noise_type in ["Uniform", "Gaussian"]:
        run_a1_and_save_plots(
            noise_type=noise_type,
            dic=final_diff_ratio_dic,
            model_name=model_name,
            analysis_name="A1",
            attack_key=selected_adv_attack_key,
        )

    # ------------------------------------------------------------------
    # A.2: Distribution plots for different noise types + strengths
    # ------------------------------------------------------------------
    def run_a2_and_save_plots(
        *,
        noise_type: str,
        dic: Dict[str, Any],
        model_name: str,
        analysis_name: str = "A2",
        attack_key: str = "",
        bins: int = 70,
    ):
        # Guard: skip noise types not present
        sample_dataset = next(iter(dic.keys()))
        if noise_type not in dic[sample_dataset]["Clean"]:
            print(f"[A2] Noise type '{noise_type}' not found in results. Skipping.")
            return

        out_dir = os.path.join("plots_output", analysis_name, model_name, noise_type)
        os.makedirs(out_dir, exist_ok=True)

        # Human-readable label for the selected adversarial setting
        if attack_key:
            adv_label = ATTACK_KEY_LEGEND_MAPPING.get(attack_key, f"Adversarial ({attack_key})")
        else:
            adv_label = "Adversarial"

        def _collect_tau_across_datasets(final_diff_ratio_dic, noise_value: str, attack: str) -> np.ndarray:
            taus: List[float] = []
            for dset in final_diff_ratio_dic:
                taus.extend(final_diff_ratio_dic[dset][attack][noise_type][noise_value]["diff_ratio"])
            return np.asarray(taus, dtype=float)

        def _compute_delta_tau(final_diff_ratio_dic, noise_value: str) -> np.ndarray:
            delta_tau_all: List[float] = []
            for dset in final_diff_ratio_dic:
                tau_clean = np.asarray(final_diff_ratio_dic[dset]["Clean"][noise_type][noise_value]["diff_ratio"], dtype=float)
                tau_adv = np.asarray(final_diff_ratio_dic[dset]["Adversarial"][noise_type][noise_value]["diff_ratio"], dtype=float)
                # Safety: skip if shapes mismatch (should not happen, but protects downstream plotting)
                if tau_clean.shape != tau_adv.shape:
                    continue
                delta_tau_all.extend((tau_adv - tau_clean).tolist())
            return np.asarray(delta_tau_all, dtype=float)

        def _summarize(arr: np.ndarray) -> Dict[str, Any]:
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {"count": 0}
            q = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95]).tolist()
            return {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "quantiles": {"q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4]},
            }

        noise_values = sorted(dic[sample_dataset]["Clean"][noise_type].keys(), key=eps_from_key)
        if not noise_values:
            print(f"[A2] No noise magnitudes found for noise_type='{noise_type}'.")
            return

        overlay_paths: List[str] = []
        delta_paths: List[str] = []

        a2_stats: Dict[str, Any] = {
            "analysis": analysis_name,
            "model": model_name,
            "noise_type": noise_type,
            "attack_key": attack_key,
            "bins": bins,
            "noise_values_sorted": noise_values,
            "per_noise_value": {},
        }

        for noise_value in noise_values:
            # Collect taus
            tau_clean = _collect_tau_across_datasets(dic, noise_value, "Clean")
            tau_adv = _collect_tau_across_datasets(dic, noise_value, "Adversarial")
            delta_tau = _compute_delta_tau(dic, noise_value)

            # Choose a shared binning range for overlay histograms
            combined = np.concatenate([tau_clean[np.isfinite(tau_clean)], tau_adv[np.isfinite(tau_adv)]])
            if combined.size == 0:
                print(f"[A2] Empty τ arrays for {noise_type} {noise_value}; skipping plots.")
                continue
            x_min = float(np.min(combined))
            x_max = float(np.max(combined))
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                # Fallback to matplotlib auto-binning
                hist_range = None
            else:
                # Pad slightly so bars don't clip at extremes
                span = x_max - x_min
                hist_range = (x_min - 0.02 * span, x_max + 0.02 * span)

            # X-axis label for noise strength
            if noise_type == "Gaussian":
                strength_label = f"σ = {noise_value.replace('Sigma_', '')}"
            else:
                strength_label = f"ε = {noise_value.replace('Eps_', '')}/255"

            # (i) τ Distribution Overlay
            fig = plt.figure(figsize=(6.5, 4.5))
            plt.hist(tau_clean, bins=bins, range=hist_range, alpha=0.60, density=True, label="Clean")
            plt.hist(tau_adv, bins=bins, range=hist_range, alpha=0.60, density=True, label=adv_label)
            plt.xlabel("Latent Drift τ")
            plt.ylabel("Density")
            plt.title(f"A2: τ Distribution Overlay ({noise_type}, {strength_label})")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            overlay_name = (
                f"a2_tau_distribution_overlay_{noise_value}_{attack_key}.png"
                if attack_key else f"a2_tau_distribution_overlay_{noise_value}.png"
            )
            overlay_path = os.path.join(out_dir, overlay_name)
            fig.savefig(overlay_path, dpi=200)
            plt.close(fig)
            overlay_paths.append(overlay_path)

            # (ii) Sample-wise Δτ Distribution
            fig = plt.figure(figsize=(6.5, 4.5))
            plt.hist(delta_tau, bins=bins, alpha=0.80)
            plt.axvline(0, color="black", linestyle="--", linewidth=1)
            plt.xlabel("Δτ = τ_adv − τ_clean")
            plt.ylabel("Count")
            plt.title(f"A2: Sample-wise τ Separation ({noise_type}, {strength_label})")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            delta_name = (
                f"a2_delta_tau_distribution_{noise_value}_{attack_key}.png"
                if attack_key else f"a2_delta_tau_distribution_{noise_value}.png"
            )
            delta_path = os.path.join(out_dir, delta_name)
            fig.savefig(delta_path, dpi=200)
            plt.close(fig)
            delta_paths.append(delta_path)

            a2_stats["per_noise_value"][noise_value] = {
                "strength": eps_from_key(noise_value),
                "tau_clean": _summarize(tau_clean),
                "tau_adv": _summarize(tau_adv),
                "delta_tau": _summarize(delta_tau),
                "overlay_plot": os.path.basename(overlay_path),
                "delta_plot": os.path.basename(delta_path),
            }

        # Save aggregated A2 stats
        stats_name = f"a2_stats_{attack_key}.json" if attack_key else "a2_stats.json"
        stats_path = os.path.join(out_dir, stats_name)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(a2_stats, f, indent=2)
        print(f"[A2] Saved stats: {stats_path}")

        # Create grids across noise magnitudes
        GRID_COLS = 5
        GRID_SCALE = 1.5

        if overlay_paths:
            grid_name = f"a2_tau_distribution_overlay_grid_{attack_key}.png" if attack_key else "a2_tau_distribution_overlay_grid.png"
            grid_path = os.path.join(out_dir, grid_name)
            create_image_grid(overlay_paths, grid_path, cols=GRID_COLS, scale=GRID_SCALE)
            print(f"[A2] Saved grid: {grid_path}")

        if delta_paths:
            grid_name = f"a2_delta_tau_distribution_grid_{attack_key}.png" if attack_key else "a2_delta_tau_distribution_grid.png"
            grid_path = os.path.join(out_dir, grid_name)
            create_image_grid(delta_paths, grid_path, cols=GRID_COLS, scale=GRID_SCALE)
            print(f"[A2] Saved grid: {grid_path}")

    #Run A.2
    for noise_type in ["Uniform", "Gaussian"]:
        run_a2_and_save_plots(
            noise_type=noise_type,
            dic=final_diff_ratio_dic,
            model_name=model_name,
            analysis_name="A2",
            attack_key=selected_adv_attack_key,
            bins=70,
        )

