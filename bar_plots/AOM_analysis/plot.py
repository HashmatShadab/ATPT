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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    images = []
    try:
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

        new_im = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        for i, im in enumerate(images):
            row = i // cols
            col = i % cols
            new_im.paste(im, (col * max_width, row * max_height))

        new_im.save(output_path)
        print(f"Saved grid to: {output_path}")
    finally:
        for im in images:
            try:
                im.close()
            except Exception:
                pass

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
    parser.add_argument(
        "--grid-normalize",
        type=str,
        default="true",
        choices=["both", "true", "false"],
        help="Which normalize setting(s) to include in the final grid image.",
    )


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

        # --- style
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "font.size": 11,
        })

        fig, ax = plt.subplots(figsize=(9.5, 4.8))

        x = np.arange(len(alphas))
        width = 0.38

        # Colors chosen to be print-friendly and distinct
        clean_color = "#4C72B0"
        adv_color = "#DD8452"

        clean_bars = ax.bar(x - width / 2, clean_avg, width=width, label="Clean", color=clean_color)
        adv_bars = ax.bar(x + width / 2, adv_avg, width=width, label="Adversarial", color=adv_color)

        # Average accuracy line (centered between the two bars)
        avg_acc = 0.5 * (clean_avg + adv_avg)
        ax.plot(
            x,
            avg_acc,
            color="#333333",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="Average (Clean+Adv)/2",
            zorder=3,
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

        max_val = float(np.nanmax([np.nanmax(clean_avg), np.nanmax(adv_avg), np.nanmax(avg_acc)]))
        ax.set_ylim(0, max(100.0, max_val + 4.0))
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Average accuracy across datasets (%)")

        # Best average accuracy + alpha (finite values only)
        best_suffix = ""
        finite_mask = np.isfinite(avg_acc)
        if np.any(finite_mask):
            best_idx = int(np.nanargmax(np.where(finite_mask, avg_acc, -np.inf)))
            best_alpha = float(alphas[best_idx])
            best_val = float(avg_acc[best_idx])
            best_suffix = f"\nBest average = {best_val:.1f}% @ alpha = {best_alpha:g}"

        ax.set_title(f"{title}{best_suffix}")
        ax.legend(
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
        )

        # value labels above bars
        def _add_bar_value_labels(bar_container):
            for rect in bar_container:
                h = rect.get_height()
                if h is None or not np.isfinite(h):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h,
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

        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(outpath, dpi=250)
        plt.close(fig)


    # -------------------------
    # Main execution
    # -------------------------
    def _format_anchor_label(noise_anchor: str) -> str:
        # noise_anchor examples: "noisy_Sigma_0_06", "uniform_Eps_16_0"
        dist = None
        rest = noise_anchor
        if "_" in noise_anchor:
            prefix, maybe_rest = noise_anchor.split("_", 1)
            if prefix in {"noisy", "uniform"}:
                dist = prefix
                rest = maybe_rest

        dist_label = {"noisy": "Gaussian", "uniform": "Uniform"}.get(dist, "")

        if rest.startswith("Sigma_"):
            val = rest[len("Sigma_"):].replace("_", ".")
            core = rf"$\sigma={val}$"
        elif rest.startswith("Eps_"):
            val = rest[len("Eps_"):].replace("_", ".")
            core = rf"$\epsilon={val}$"
        else:
            core = rest

        if dist_label:
            return f"{dist_label} ({core})"
        return str(core)


    def _format_normalize_label(normalize: str) -> str:
        if str(normalize).lower() == "true":
            return r"$z/\|z\|_2$"
        return "None"





    def run_all_plots(model_name, out_root="plots", *, grid_normalize: str = "both"):
        datasets = list(TRUE_LABELS_DATASET.keys())

        grid_normalize = str(grid_normalize).lower().strip()
        if grid_normalize not in {"both", "true", "false"}:
            raise ValueError("grid_normalize must be one of: 'both', 'true', 'false'")

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

        saved_paths_by_anchor = {}

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
                    "Average accuracy across datasets\n"
                    rf"Anchor: {_format_anchor_label(noise_anchor)}   |   Normalize: {_format_normalize_label(normalize)}"
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

                saved_paths_by_anchor.setdefault(noise_anchor, {})[str(normalize)] = outpath

        # --- Create a single grid:
        # - If grid_normalize == 'both': 4 columns (two anchors per row)
        #   [A False, A True, B False, B True]
        # - If grid_normalize == 'false' or 'true': 2 columns (two anchors per row)
        #   [A False, B False]  or  [A True, B True]
        anchors_sorted = sorted(saved_paths_by_anchor.keys())
        grid_paths = []
        for i in range(0, len(anchors_sorted), 2):
            row_anchors = anchors_sorted[i:i + 2]
            for a in row_anchors:
                per_norm = saved_paths_by_anchor.get(a, {})
                if grid_normalize in {"both", "false"}:
                    p_false = per_norm.get("False")
                    if p_false is not None:
                        grid_paths.append(p_false)
                if grid_normalize in {"both", "true"}:
                    p_true = per_norm.get("True")
                    if p_true is not None:
                        grid_paths.append(p_true)

        if grid_paths:
            cols = 4 if grid_normalize == "both" else 2
            if grid_normalize == "both":
                grid_name = "grid_all_noise_anchors_normalize.png"
            else:
                grid_name = f"grid_all_noise_anchors_normalize_{grid_normalize}.png"
            grid_out = os.path.join(out_root, model_name, "AOM", grid_name)
            create_image_grid(grid_paths, grid_out, cols=cols, scale=1.0)


    run_all_plots(model_name, grid_normalize=args.grid_normalize)









