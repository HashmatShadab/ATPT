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
    model = model_name

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
    model = model_name

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
    model = model_name

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

    return_dic = {"zero_shot_clean": ZS_CLEAN_PREDS_DATASET, "zero_shot_adv": ZS_ADV_PREDS_DATASET,
                  "zero_shot_adv_image_only": ZS_ADV_IMAGE_ONLY_PREDS_DATASET,
                  "true_labels": TRUE_LABELS_DATASET}

    return return_dic


def get_ttc_results(model_name):
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
        "ttc": {
            "clean": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau_Counter_Attack/"
                    "{model}/{dataset}/"
                    "Clean/Counter_Attack/Eps_4_0_Steps_5_Alpha_1_0/tau_100_0_beta_2_0_weighted_pertrubation_True/No_TPT/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            "adversarial_eps4_steps100": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau_Counter_Attack/"
                    "{model}/{dataset}/"
                    "Adversarial_Eps_4_0_Steps_100/"
                    "Counter_Attack/Eps_4_0_Steps_5_Alpha_1_0/tau_100_0_beta_2_0_weighted_pertrubation_True/No_TPT/"
                    "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
                ),
            },
            # "adversarial_eps4_steps100_image_only": {
            #     "base_path": (
            #         "../../Final_Results_corrected_ca_tau_Counter_Attack/"
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

        for case, cfg in result_paths["ttc"].items():
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
    model = model_name

    true_labels_data = {}

    # Clean zero shot
    zero_shot_clean_preds_data = {}
    zero_shot_clean_max_confidences_data = {}


    # Clean single TTC
    ttc_clean_preds_single_data = {}
    ttc_clean_max_confidences_single_data = {}

    # Clean Vanilla TTC
    ttc_clean_preds_vanilla_data = {}
    ttc_clean_max_confidences_vanilla_data = {}

    # Clean Weighted TTC
    ttc_clean_preds_weighted_data = {}
    ttc_clean_max_confidences_weighted_data = {}

    for dataset in DATASETS:
        example = DATA[case][model][dataset]
        true_labels_data[dataset] = example["preds"]["original_clean"]["label"]

        zero_shot_clean_preds_data[dataset] = example["preds"]["original_clean"]["prediction"]
        zero_shot_clean_max_confidences_data[dataset] = example["preds"]["original_clean"]["max_confidence"]


        ttc_clean_preds_single_data[dataset] = example["preds"]["single"]["prediction"]
        ttc_clean_max_confidences_single_data[dataset] = example["preds"]["single"]["max_confidence"]

        ttc_clean_preds_vanilla_data[dataset] = example["preds"]["vanilla"]["prediction"]
        ttc_clean_max_confidences_vanilla_data[dataset] = example["preds"]["vanilla"]["max_confidence"]

        ttc_clean_preds_weighted_data[dataset] = example["preds"]["weighted"]["prediction"]
        ttc_clean_max_confidences_weighted_data[dataset] = example["preds"]["weighted"]["max_confidence"]

        # # print accuracy
        # print(dataset, compute_accuracy(zero_shot_clean_preds_data[dataset], true_labels_data[dataset]))
        # print(dataset, compute_accuracy(ttc_clean_preds_single_data[dataset], true_labels_data[dataset]))
        # print(dataset, compute_accuracy(ttc_clean_preds_vanilla_data[dataset], true_labels_data[dataset]))
        # print(dataset, compute_accuracy(ttc_clean_preds_weighted_data[dataset], true_labels_data[dataset]))
        # # print mean confidence
        # print(dataset, np.mean(zero_shot_clean_max_confidences_data[dataset]))
        # print(dataset, np.mean(ttc_clean_max_confidences_single_data[dataset]))
        # print(dataset, np.mean(ttc_clean_max_confidences_vanilla_data[dataset]))
        # print(dataset, np.mean(ttc_clean_max_confidences_weighted_data[dataset]))


    case = "adversarial_eps4_steps100"
    model = model_name

    # Adversarial zero shot
    zero_shot_adv_preds_data = {}
    zero_shot_adv_max_confidences_data = {}

    # Adversarial single TTC
    ttc_adv_preds_single_data = {}
    ttc_adv_max_confidences_single_data = {}

    # Adversarial Vanilla TTC
    ttc_adv_preds_vanilla_data = {}
    ttc_adv_max_confidences_vanilla_data = {}

    # Adversarial Weighted TTC
    ttc_adv_preds_weighted_data = {}
    ttc_adv_max_confidences_weighted_data = {}


    for dataset in DATASETS:
        example = DATA[case][model][dataset]

        zero_shot_adv_preds_data[dataset] = example["preds"]["original"]["prediction"]
        zero_shot_adv_max_confidences_data[dataset] = example["preds"]["original"]["max_confidence"]

        ttc_adv_preds_single_data[dataset] = example["preds"]["single"]["prediction"]
        ttc_adv_max_confidences_single_data[dataset] = example["preds"]["single"]["max_confidence"]

        ttc_adv_preds_vanilla_data[dataset] = example["preds"]["vanilla"]["prediction"]
        ttc_adv_max_confidences_vanilla_data[dataset] = example["preds"]["vanilla"]["max_confidence"]

        ttc_adv_preds_weighted_data[dataset] = example["preds"]["weighted"]["prediction"]
        ttc_adv_max_confidences_weighted_data[dataset] = example["preds"]["weighted"]["max_confidence"]

        # print accuracy
        print(dataset, compute_accuracy(zero_shot_adv_preds_data[dataset], true_labels_data[dataset]))
        # print(dataset, compute_accuracy(ttc_adv_preds_single_data[dataset], true_labels_data[dataset]))
        # print(dataset, compute_accuracy(ttc_adv_preds_vanilla_data[dataset], true_labels_data[dataset]))
        # print(dataset, compute_accuracy(ttc_adv_preds_weighted_data[dataset], true_labels_data[dataset]))
        # # print mean confidence
        # print(dataset, np.mean(zero_shot_adv_max_confidences_data[dataset]))
        # print(dataset, np.mean(ttc_adv_max_confidences_single_data[dataset]))
        # print(dataset, np.mean(ttc_adv_max_confidences_vanilla_data[dataset]))
        # print(dataset, np.mean(ttc_adv_max_confidences_weighted_data[dataset]))

    case = "adversarial_eps4_steps100_image_only"
    model = model_name

    # # Adversarial image only zero shot
    # zero_shot_adv_image_only_preds_data = {}
    # zero_shot_adv_image_only_max_confidences_data = {}
    #
    # # Adversarial image only single TTC
    # ttc_adv_image_only_preds_single_data = {}
    # ttc_adv_image_only_max_confidences_single_data = {}
    #
    # # Adversarial image only Vanilla TTC
    # ttc_adv_image_only_preds_vanilla_data = {}
    # ttc_adv_image_only_max_confidences_vanilla_data = {}
    #
    # # Adversarial image only Weighted TTC
    # ttc_adv_image_only_preds_weighted_data = {}
    # ttc_adv_image_only_max_confidences_weighted_data = {}
    #
    #
    # for dataset in DATASETS:
    #     example = DATA[case][model][dataset]
    #
    #     zero_shot_adv_image_only_preds_data[dataset] = example["preds"]["original"]["prediction"]
    #     zero_shot_adv_image_only_max_confidences_data[dataset] = example["preds"]["original"]["max_confidence"]
    #
    #     ttc_adv_image_only_preds_single_data[dataset] = example["preds"]["single"]["prediction"]
    #     ttc_adv_image_only_max_confidences_single_data[dataset] = example["preds"]["single"]["max_confidence"]
    #
    #     ttc_adv_image_only_preds_vanilla_data[dataset] = example["preds"]["vanilla"]["prediction"]
    #     ttc_adv_image_only_max_confidences_vanilla_data[dataset] = example["preds"]["vanilla"]["max_confidence"]
    #
    #     ttc_adv_image_only_preds_weighted_data[dataset] = example["preds"]["weighted"]["prediction"]
    #     ttc_adv_image_only_max_confidences_weighted_data[dataset] = example["preds"]["weighted"]["max_confidence"]
    #
    #     # print accuracy
    #     print(dataset, compute_accuracy(zero_shot_adv_image_only_preds_data[dataset], true_labels_data[dataset]))
    #     print(dataset, compute_accuracy(ttc_adv_image_only_preds_single_data[dataset], true_labels_data[dataset]))
    #     print(dataset, compute_accuracy(ttc_adv_image_only_preds_vanilla_data[dataset], true_labels_data[dataset]))
    #     print(dataset, compute_accuracy(ttc_adv_image_only_preds_weighted_data[dataset], true_labels_data[dataset]))
    #     # print mean confidence
    #     print(dataset, np.mean(zero_shot_adv_image_only_max_confidences_data[dataset]))
    #     print(dataset, np.mean(ttc_adv_image_only_max_confidences_single_data[dataset]))
    #     print(dataset, np.mean(ttc_adv_image_only_max_confidences_vanilla_data[dataset]))
    #     print(dataset, np.mean(ttc_adv_image_only_max_confidences_weighted_data[dataset]))

    TRUE_LABELS_DATASET = true_labels_data

    # Zero Shot
    ZS_CLEAN_PREDS_DATASET = zero_shot_clean_preds_data
    ZS_ADV_PREDS_DATASET = zero_shot_adv_preds_data
    # ZS_ADV_IMAGE_ONLY_PREDS_DATASET = zero_shot_adv_image_only_preds_data

    # Single
    TTC_CLEAN_PREDS_SINGLE_DATASET = ttc_clean_preds_single_data
    TTC_ADV_PREDS_SINGLE_DATASET = ttc_adv_preds_single_data
    # TTC_ADV_IMAGE_ONLY_PREDS_SINGLE_DATASET = ttc_adv_image_only_preds_single_data

    # Vanilla
    TTC_CLEAN_PREDS_VANILLA_DATASET = ttc_clean_preds_vanilla_data
    TTC_ADV_PREDS_VANILLA_DATASET = ttc_adv_preds_vanilla_data
    # TTC_ADV_IMAGE_ONLY_PREDS_VANILLA_DATASET = ttc_adv_image_only_preds_vanilla_data

    # Weighted
    TTC_CLEAN_PREDS_WEIGHTED_DATASET = ttc_clean_preds_weighted_data
    TTC_ADV_PREDS_WEIGHTED_DATASET = ttc_adv_preds_weighted_data
    # TTC_ADV_IMAGE_ONLY_PREDS_WEIGHTED_DATASET = ttc_adv_image_only_preds_weighted_data

    # create a dcitionary to return this in structured format, return only single setting in ttc and zs

    return_dic = {"zero_shot_clean": ZS_CLEAN_PREDS_DATASET, "zero_shot_adv": ZS_ADV_PREDS_DATASET,
                 "ttc_single_clean": TTC_CLEAN_PREDS_SINGLE_DATASET, "ttc_single_adv": TTC_ADV_PREDS_SINGLE_DATASET,
                 "true_labels": TRUE_LABELS_DATASET}
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
    model_name = "vit_l_14_datacomp_1b"
    zero_shot_dic = get_zs_results(model_name)
    ttc_dic = get_ttc_results(model_name)

    # True labels for all datasets
    TRUE_LABELS_DATASET = zero_shot_dic["true_labels"]
    # Zero-Shot clean predictions for all datasets
    ZS_CLEAN_PREDS_DATASET = zero_shot_dic["zero_shot_clean"]
    # Zero-Shot adversarial predictions for all datasets
    ZS_ADV_PREDS_DATASET = zero_shot_dic["zero_shot_adv"]

    # TTC clean predictions for all datasets
    TTC_CLEAN_PREDS_SINGLE_DATASET = ttc_dic["ttc_single_clean"]
    # TTC adversarial predictions for all datasets
    TTC_ADV_PREDS_SINGLE_DATASET = ttc_dic["ttc_single_adv"]


    # compute zero shot accuracy
    def compute_accuracy(preds, labels):
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        return (preds == labels).mean() * 100.0

    import numpy as np

    datasets = list(TRUE_LABELS_DATASET.keys())


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



    # Calculate Average Accuracies
    def get_avg_acc(preds_dict, labels_dict):
        accs = []
        for dataset in datasets:
            acc = compute_accuracy(preds_dict[dataset], labels_dict[dataset])
            accs.append(acc)
        return np.mean(accs)

    # Zero-Shot Average Accuracies
    zs_avg_clean = get_avg_acc(ZS_CLEAN_PREDS_DATASET, TRUE_LABELS_DATASET)
    zs_avg_adv = get_avg_acc(ZS_ADV_PREDS_DATASET, TRUE_LABELS_DATASET)

    # TTC Average Accuracies
    ttc_avg_clean = get_avg_acc(TTC_CLEAN_PREDS_SINGLE_DATASET, TRUE_LABELS_DATASET)
    ttc_avg_adv = get_avg_acc(TTC_ADV_PREDS_SINGLE_DATASET, TRUE_LABELS_DATASET)

    print(f"Zero-Shot Clean: {zs_avg_clean:.2f}, Adv: {zs_avg_adv:.2f}")
    print(f"TTC Clean: {ttc_avg_clean:.2f}, Adv: {ttc_avg_adv:.2f}")

    output_dir = "ttc_withtout_threshold"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Zero-Shot Average Accuracy
    plt.figure(figsize=(6, 6))
    labels = ['Clean', 'Adversarial']
    values = [zs_avg_clean, zs_avg_adv]
    # Professional colors
    bars = plt.bar(labels, values, color=['#4682B4', '#FF7F50'], edgecolor='black', linewidth=0.5)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.title('Average Zero-Shot Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 85)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'average_zs_accuracy.png'), dpi=300)
    plt.close()

    # Plot 2: TTC Average Accuracy
    plt.figure(figsize=(6, 6))
    labels = ['Clean', 'Adversarial']
    values = [ttc_avg_clean, ttc_avg_adv]
    # Professional colors
    bars = plt.bar(labels, values, color=['#4682B4', '#FF7F50'], edgecolor='black', linewidth=0.5)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.title('Average TTC Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 85)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'average_ttc_accuracy.png'), dpi=300)
    plt.close()

    print(f"Plots saved in {output_dir}")

    diff_ratio_thresholds = [
        0.0, 0.1, 0.2, 0.25, 0.3, 0.4,
        0.5, 0.6,  0.7,
     0.8, 0.85, 0.9, 1.0
    ]


    def format_noise_math(noise_type: str, noise_param: str) -> str:
        """
        Convert (noise_type, noise_param) into a LaTeX-style mathematical string
        using standard noise distribution symbols.

        Examples:
        - ("Gaussian", "Sigma_0.12") -> r"$\mathcal{N}(0, \sigma^2),\ \sigma=0.12$"
        - ("Uniform", "Eps_48.0")   -> r"$\mathcal{U}(-\epsilon, \epsilon),\ \epsilon=48/255$"
        """

        if noise_param is None:
            return noise_type

        try:
            name, value = noise_param.split("_")
            value = float(value)
        except Exception:
            return noise_type

        # Gaussian noise
        if noise_type.lower() == "gaussian" and name.lower() == "sigma":
            return rf"Threshold $\tau(\mathcal{{N}}, \sigma = {value})$"

        # Uniform noise
        if noise_type.lower() == "uniform" and name.lower() == "eps":
            if value > 1:
                return rf"Threshold $\tau(\mathcal{{U}}, \epsilon = {int(value)}/255)$"
            else:
                return rf"Threshold $\tau(\mathcal{{U}},\ \epsilon = {value})$"

        return noise_type


    def gated_accuracy(
            diff_ratios,
            zs_preds,
            ttc_preds,
            labels,
            threshold,
            logic="greater"
    ):
        """
        diff_ratios: list[float]
        zs_preds: list[int]
        ttc_preds: list[int]
        labels: list[int]
        logic: "greater" or "less_equal"
        """
        correct = 0
        total = len(labels)

        for i in range(total):
            if logic == "greater":
                if diff_ratios[i] > threshold:
                    pred = ttc_preds[i]
                else:
                    pred = zs_preds[i]
            else: # logic == "less_equal"
                if diff_ratios[i] <= threshold:
                    pred = ttc_preds[i]
                else:
                    pred = zs_preds[i]

            if pred == labels[i]:
                correct += 1

        return 100.0 * correct / total


    # --- Original Gating Logic (TTC if > threshold) ---
    output_root = os.path.join("ttc_with_our_threshold", model_name)
    os.makedirs(output_root, exist_ok=True)

    # --- Opposite Gating Logic (TTC if <= threshold) ---
    output_root_default = os.path.join("ttc_with_reported_threshold", model_name)
    os.makedirs(output_root_default, exist_ok=True)

    for noise_type in ["Gaussian", "Uniform"]:
        for noise_param in next(
                iter(final_diff_ratio_dic.values())
        )["Clean"].get(noise_type, {}).keys():

            for current_output_root, current_logic in [
                (output_root, "greater"),
                (output_root_default, "less_equal")
            ]:
                clean_accs = []
                adv_accs = []

                for tau in diff_ratio_thresholds:
                    clean_dataset_accs = []
                    adv_dataset_accs = []

                    for dataset in datasets:
                        # --- Clean ---
                        diff_clean = final_diff_ratio_dic[dataset]["Clean"][noise_type][noise_param]
                        acc_clean = gated_accuracy(
                            diff_clean,
                            ZS_CLEAN_PREDS_DATASET[dataset],
                            TTC_CLEAN_PREDS_SINGLE_DATASET[dataset],
                            TRUE_LABELS_DATASET[dataset],
                            tau,
                            logic=current_logic
                        )
                        clean_dataset_accs.append(acc_clean)

                        # --- Adversarial ---
                        diff_adv = final_diff_ratio_dic[dataset]["Adversarial"][noise_type][noise_param]
                        acc_adv = gated_accuracy(
                            diff_adv,
                            ZS_ADV_PREDS_DATASET[dataset],
                            TTC_ADV_PREDS_SINGLE_DATASET[dataset],
                            TRUE_LABELS_DATASET[dataset],
                            tau,
                            logic=current_logic
                        )
                        adv_dataset_accs.append(acc_adv)

                    clean_accs.append(np.mean(clean_dataset_accs))
                    adv_accs.append(np.mean(adv_dataset_accs))

                clean_accs = np.array(clean_accs)
                adv_accs = np.array(adv_accs)
                net_accs = (clean_accs + adv_accs) / 2

                # Identify max values and corresponding thresholds
                max_clean_idx = np.argmax(clean_accs)
                max_adv_idx = np.argmax(adv_accs)
                max_net_idx = np.argmax(net_accs)

                max_clean_val = clean_accs[max_clean_idx]
                max_clean_thr = diff_ratio_thresholds[max_clean_idx]

                max_adv_val = adv_accs[max_adv_idx]
                max_adv_thr = diff_ratio_thresholds[max_adv_idx]

                max_net_val = net_accs[max_net_idx]
                max_net_thr = diff_ratio_thresholds[max_net_idx]

                # ---- Plot ----
                save_dir = os.path.join(current_output_root, noise_type, noise_param)
                os.makedirs(save_dir, exist_ok=True)

                x = np.arange(len(diff_ratio_thresholds))
                width = 0.35

                plt.figure(figsize=(12, 8))
                # Professional colors: Steel Blue and Coral
                bars1 = plt.bar(x - width / 2, clean_accs, width, label="Clean", color='#4682B4', edgecolor='black', linewidth=0.5, alpha=0.7)
                bars2 = plt.bar(x + width / 2, adv_accs, width, label="Adversarial", color='#FF7F50', edgecolor='black', linewidth=0.5, alpha=0.7)

                plt.xticks(x, diff_ratio_thresholds, fontsize=18)
                plt.xlabel(r"$\tau_{\text{threshold}}$", fontsize=24)
                plt.ylabel("Average Accuracy (%)", fontsize=24)

                if current_logic == "greater":
                    logic_str = r"{Apply Counter Attack if $\tau > \tau_{threshold}$; otherwise Zero-Shot (Ours)}"
                else:
                    logic_str = r"{Apply Counter Attack if $\tau \leq \tau_{threshold}$; otherwise Zero-Shot (Baseline)}"

                # First two lines (normal)
                plt.title(
                    f"Test Time Counter Attack {logic_str}\n"
                    f"{format_noise_math(noise_type, noise_param)}",
                    fontsize=12
                )

                # Third line (colored separately)
                plt.text(
                    0.5, .92,
                    rf"Best Average Performance: {max_net_val:.1f}% at $\tau_{{\text{{threshold}}}} = {max_net_thr}$",
                    color="blue",
                    fontsize=12,
                    ha="center",
                    va="bottom",
                    transform=plt.gca().transAxes
                )

                plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))
                plt.ylim(0, 85) # Increased to give space for text
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                def add_labels(bars):
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                                 f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

                add_labels(bars1)
                add_labels(bars2)

                plt.tight_layout()

                plt.savefig(os.path.join(save_dir, "accuracy_vs_threshold.png"), dpi=300)
                # plt.savefig(os.path.join(save_dir, "accuracy_vs_threshold.pdf"))
                plt.close()

                print(f"Saved plots to: {save_dir}")

    # --- Create Grids ---
    print("\nGenerating grids...")
    for root_dir in ["ttc_with_our_threshold", "ttc_with_reported_threshold"]:
        for noise_type in ["Gaussian", "Uniform"]:
            plots_to_grid = []
            
            # Walk through the model directory in the root
            model_path = os.path.join(root_dir, model_name, noise_type)
            if not os.path.exists(model_path):
                continue
                
            for noise_param in sorted(os.listdir(model_path)):
                param_path = os.path.join(model_path, noise_param)
                if not os.path.isdir(param_path):
                    continue
                
                img_path = os.path.join(param_path, "accuracy_vs_threshold.png")
                if os.path.exists(img_path):
                    plots_to_grid.append(img_path)
            
            if plots_to_grid:
                grid_name = f"{noise_type.lower()}_plots_grid.png"
                grid_output = os.path.join(root_dir, model_name, grid_name)
                create_image_grid(plots_to_grid, grid_output, cols=2)






