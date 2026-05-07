from pathlib import Path
import json
import numpy as np

from collections import defaultdict
import numpy as np


MODELS = [
    "delta_clip_l14_224",
    "fare4",
    "ViT-L/14",
    "vit_l_14_datacomp_1b",
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


def get_zs_results():
    from pathlib import Path
    import json
    import numpy as np

    MODELS = [
        "delta_clip_l14_224",
        "fare4",
        "ViT-L/14",
        "vit_l_14_datacomp_1b",
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
            "adversarial_eps8_steps100": {
                "base_path": (
                    "../../Final_Results_corrected_ca_tau/"
                    "{model}/{dataset}/"
                    "Adversarial_Eps_8_0_Steps_100/"
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

    case = "adversarial_eps8_steps100"
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



def load_one_setting(base_path: str, *, load_counter_attack: bool = True) -> dict:
    """
    base_path: directory for a single (method, case, model, dataset, [eps/sigma])
    returns: normalized dict with preds (+ optional counter_attack)
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

    out = {"preds": preds}

    # --- counter-attack diff-ratio json (only in counter-attack runs)
    if load_counter_attack:
        fp = base / JSON_KEYMAP["results_counter_attack_diff_ratio"]
        if fp.exists():
            obj = _load_json(fp)
            out["results_counter_attack_diff_ratio"] = diff_ratio(obj)
        else:
            out["results_counter_attack_diff_ratio"] = None

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


# Base path templates for each case
RESULT_PATHS = {
    # -------------------------
    # Zero-shot (no counter-attack)
    # -------------------------
    # "zero_shot": {
    #     "clean": {
    #         "base_path": (
    #             "../../Final_Results_corrected_ca_tau/"
    #             "{model}/{dataset}/"
    #             "Clean/No_Counter_Attack/No_TPT/"
    #             "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
    #         ),
    #     },
    #     "adversarial_eps4_steps100": {
    #         "base_path": (
    #             "../../Final_Results_corrected_ca_tau/"
    #             "{model}/{dataset}/"
    #             "Adversarial_Eps_4_0_Steps_100/"
    #             "No_Counter_Attack/No_TPT/"
    #             "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
    #         ),
    #     },
    #     "adversarial_eps4_steps100_image_only": {
    #         "base_path": (
    #             "../../Final_Results_corrected_ca_tau/"
    #             "{model}/{dataset}/"
    #             "Adversarial_Eps_4_0_Steps_100_image_only_attack_prm/"
    #             "No_Counter_Attack/No_TPT/"
    #             "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
    #         ),
    #     },
    # },

    # -------------------------
    # Zero-shot + Uniform Noise (Single noisy image for diff ratio)
    # EPS: 4/8/12/16  -> use {eps} in the path
    # -------------------------
    # "zero_shot_uniform_single": {
    #     "clean": {
    #         "base_path": (
    #             "../../Final_Results_corrected_ca_tau/"
    #             "{model}/{dataset}/"
    #             "Clean/Counter_Attack/"
    #             "Eps_{eps}_0_Steps_0_Alpha_1_0/"
    #             "tau_0_2_beta_2_0_weighted_pertrubation_True/"
    #             "No_TPT/"
    #             "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
    #         ),
    #     },
    #     "adversarial_eps4_steps100": {
    #         "base_path": (
    #             "../../Final_Results_corrected_ca_tau/"
    #             "{model}/{dataset}/"
    #             "Adversarial_Eps_4_0_Steps_100/"
    #             "Counter_Attack/"
    #             "Eps_{eps}_0_Steps_0_Alpha_1_0/"
    #             "tau_0_2_beta_2_0_weighted_pertrubation_True/"
    #             "No_TPT/"
    #             "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
    #         ),
    #     },
    #     "adversarial_eps4_steps100_image_only": {
    #         "base_path": (
    #             "../../Final_Results_corrected_ca_tau/"
    #             "{model}/{dataset}/"
    #             "Adversarial_Eps_4_0_Steps_100_image_only_attack_prm/"
    #             "Counter_Attack/"
    #             "Eps_{eps}_0_Steps_0_Alpha_1_0/"
    #             "tau_0_2_beta_2_0_weighted_pertrubation_True/"
    #             "No_TPT/"
    #             "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
    #         ),
    #     },
    # },

    # -------------------------
    # Zero-shot + Uniform Noise (Normal anchors for diff ratio)
    # EPS: 4/8/12/16
    # -------------------------
    "zero_shot_uniform_anchors": {
        "clean": {
            "base_path": (
                "../../Final_Results_corrected_ca_tau/"
                "{model}/{dataset}/"
                "Clean/Counter_Attack/"
                "Eps_{eps}_0_Steps_0_Alpha_1_0/"
                "Tau_normal_anchors_num_anchors_10_tauthresh_0_2_beta_2_0_weighted_pertrubation_True/"
                "No_TPT/"
                "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
            ),
        },
        "adversarial_eps8_steps100": {
            "base_path": (
                "../../Final_Results_corrected_ca_tau/"
                "{model}/{dataset}/"
                "Adversarial_Eps_8_0_Steps_100/"
                "Counter_Attack/"
                "Eps_{eps}_0_Steps_0_Alpha_1_0/"
                "Tau_normal_anchors_num_anchors_10_tauthresh_0_2_beta_2_0_weighted_pertrubation_True/"
                "No_TPT/"
                "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
            ),
        },
        "adversarial_eps4_steps100_image_only": {
            "base_path": (
                "../../Final_Results_corrected_ca_tau/"
                "{model}/{dataset}/"
                "Adversarial_Eps_4_0_Steps_100_image_only_attack_prm/"
                "Counter_Attack/"
                "Eps_{eps}_0_Steps_0_Alpha_1_0/"
                "Tau_normal_anchors_num_anchors_10_tauthresh_0_2_beta_2_0_weighted_pertrubation_True/"
                "No_TPT/"
                "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
            ),
        },
    },

    # -------------------------
    # Zero-shot + Gaussian Noise (anchors for diff ratio)
    # SIGMA placeholder -> use {sigma} in the path
    # -------------------------
    "zero_shot_gaussian_anchors": {
        "clean": {
            "base_path": (
                "../../Final_Results_corrected_ca_tau/"
                "{model}/{dataset}/"
                "Clean/Counter_Attack/"
                "Init_Sigma_0_{sigma}_Eps_4_0_Steps_0_Alpha_1_0/"
                "Tau_noisy_num_anchors_10_tauthresh_0_2_beta_2_0_weighted_pertrubation_True/"
                "No_TPT/"
                "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
            ),
        },
        "adversarial_eps8_steps100": {
            "base_path": (
                "../../Final_Results_corrected_ca_tau/"
                "{model}/{dataset}/"
                "Adversarial_Eps_8_0_Steps_100/"
                "Counter_Attack/"
                "Init_Sigma_0_{sigma}_Eps_4_0_Steps_0_Alpha_1_0/"
                "Tau_noisy_num_anchors_10_tauthresh_0_2_beta_2_0_weighted_pertrubation_True/"
                "No_TPT/"
                "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
            ),
        },
        "adversarial_eps4_steps100_image_only": {
            "base_path": (
                "../../Final_Results_corrected_ca_tau/"
                "{model}/{dataset}/"
                "Adversarial_Eps_4_0_Steps_100_image_only_attack_prm/"
                "Counter_Attack/"
                "Init_Sigma_0_{sigma}_Eps_4_0_Steps_0_Alpha_1_0/"
                "Tau_noisy_num_anchors_10_tauthresh_0_2_beta_2_0_weighted_pertrubation_True/"
                "No_TPT/"
                "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01"
            ),
        },
    },
}


# %%
# Map semantic keys -> filenames
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



def build_all_data(
    result_paths: dict,
    models: list,
    datasets: list,
    *,
    uniform_eps_list=('4', '8', '12', '16', '24', '32'),
    gaussian_sigma_list=('03', '06', '12', '18'),
) -> dict:
    """
    Builds:
      DATA[method][case][model][dataset] = setting (for methods without eps/sigma)
      DATA[method][case][model][dataset][eps] = setting (uniform methods)
      DATA[method][case][model][dataset][sigma] = setting (gaussian methods)
    """
    DATA = {}

    for method, method_cfg in result_paths.items():
        DATA.setdefault(method, {})

        for case, cfg in method_cfg.items():
            DATA[method].setdefault(case, {})
            base_template = cfg["base_path"]

            # Decide what parameter this method needs (if any)
            needs_eps = "{eps}" in base_template
            needs_sigma = "{sigma}" in base_template

            # Counter-attack is present in these new methods (paths include Counter_Attack)
            # But we’ll still guard by checking file existence.
            load_ca = "Counter_Attack" in base_template

            for model in models:
                DATA[method][case].setdefault(model, {})

                for dataset in datasets:
                    DATA[method][case][model].setdefault(dataset, {})

                    if needs_eps:
                        # Uniform eps sweep
                        for eps in uniform_eps_list:
                            base_path = base_template.format(model=model, dataset=dataset, eps=eps)
                            DATA[method][case][model][dataset][eps] = load_one_setting(
                                base_path,
                                load_counter_attack=load_ca
                            )

                    elif needs_sigma:
                        # Gaussian sigma sweep
                        for sigma in gaussian_sigma_list:
                            base_path = base_template.format(model=model, dataset=dataset, sigma=sigma)
                            DATA[method][case][model][dataset][sigma] = load_one_setting(
                                base_path,
                                load_counter_attack=load_ca
                            )

                    else:
                        # No sweep param
                        base_path = base_template.format(model=model, dataset=dataset)
                        DATA[method][case][model][dataset] = load_one_setting(
                            base_path,
                            load_counter_attack=load_ca
                        )

    return DATA

MODELS = [
    "vit_l_14_datacomp_1b",
]

DATA_2 = build_all_data(RESULT_PATHS, MODELS, DATASETS)


TRUE_LABELS_DATASET, ZS_CLEAN_PREDS_DATASET, ZS_ADV_PREDS_DATASET, ZS_ADV_IMAGE_ONLY_PREDS_DATASET = get_zs_results()

def _print_setting_accuracy(setting, prefix):
    if "preds" not in setting:
        return

    accuracies = {}
    for pred_type, pred_data in setting["preds"].items():
        if pred_type != "single":
            continue
        if "prediction" in pred_data and "label" in pred_data:
            acc = compute_accuracy(pred_data["prediction"], pred_data["label"])
            # acc_check = compute_accuracy(pred_data["prediction"], true_labels)
            accuracies[pred_type] = f"{acc:.2f}"

    if accuracies:
        print(f"{prefix} -> Accuracies: {accuracies}")

def accuracy_with_diff_ratio(setting, prefix, diff_ratios, true_labels, zs_predictions, thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    if "preds" not in setting:
        return

    accuracies = {}
    for pred_type, pred_data in setting["preds"].items():
        if pred_type != "single":
            continue
        if "prediction" in pred_data and "label" in pred_data:
            acc = compute_accuracy(pred_data["prediction"],true_labels)
            # acc_check = compute_accuracy(pred_data["prediction"], true_labels)
            accuracies[pred_type] = f"{acc:.2f}"

            for threshold in thresholds:
                diff_threshold_preds = [
                zs_predictions[i] if diff_ratios[i] < threshold else pred_data["prediction"][i]
                for i in range(len(pred_data["prediction"]))]

                acc_diff_ratio = compute_accuracy(diff_threshold_preds, true_labels)
                accuracies[f"{pred_type}_diff_ratio_{threshold:.2f}"] = f"{acc_diff_ratio:.2f}"
            avg_diff_ratio = np.mean(diff_ratios).item()
            accuracies[f"diff_ratio_avg"] = avg_diff_ratio

    if accuracies:
        print(f"{prefix}")
        for key, value in accuracies.items():
            print(f"{key}: {value}")
    return accuracies


def print_accuracies(data):
    """
    Iterates through the DATA structure and stores accuracy results.
    DATA structure:
      [method][case][model][dataset][param]
    """
    for method, cases in data.items():
        print(f"\nMethod: {method}")

        for case, models in cases.items():
            print(f"  Case: {case}")

            if case == "clean":
                ZS_PREDICTIONS = ZS_CLEAN_PREDS_DATASET
            elif case == "adversarial_eps4_steps100":
                ZS_PREDICTIONS = ZS_ADV_PREDS_DATASET
            elif case == "adversarial_eps4_steps100_image_only":
                ZS_PREDICTIONS = ZS_ADV_IMAGE_ONLY_PREDS_DATASET
            else:
                continue

            for model, datasets in models.items():
                print(f"    Model: {model}")

                for dataset, content in datasets.items():
                    if "preds" in content:
                        # direct setting (no eps/sigma)
                        acc_dict = accuracy_with_diff_ratio(
                            content,
                            f"{method}|{case}|{model}|{dataset}",
                            diff_ratios=None,
                            true_labels=TRUE_LABELS_DATASET[dataset],
                            zs_predictions=ZS_PREDICTIONS[dataset],
                        )
                        ACC_RESULTS[method][case][model][dataset]["NA"]["NA"] = acc_dict
                        continue

                    # ---- nested eps/sigma case ----
                    print(f"      Dataset: {dataset}")

                    diff_ratio_dict = {}
                    diff_ratio_avg_dict = {}

                    # collect diff ratios for all params
                    for param, setting in content.items():
                        dr = setting["results_counter_attack_diff_ratio"]["diff_ratio_per_sample"]
                        diff_ratio_dict[param] = dr
                        diff_ratio_avg_dict[param] = float(np.mean(dr))

                    # add zero baseline
                    ref_param = next(iter(diff_ratio_dict))
                    diff_ratio_dict["0"] = [0.0] * len(diff_ratio_dict[ref_param])
                    diff_ratio_avg_dict["0"] = 0.0

                    # evaluate: predictions from `param`, diff-ratio from `diff_ratio_key`
                    for param, setting in content.items():
                        for diff_ratio_key, diff_ratios in diff_ratio_dict.items():
                            acc_dict = accuracy_with_diff_ratio(
                                setting,
                                f"{method}|{case}|{model}|{dataset}|param={param}|diff={diff_ratio_key}",
                                diff_ratios,
                                TRUE_LABELS_DATASET[dataset],
                                ZS_PREDICTIONS[dataset],
                            )

                            ACC_RESULTS[method][case][model][dataset][param][diff_ratio_key] = acc_dict


from collections import defaultdict

ACC_RESULTS = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict)
            )
        )
    )
)

print_accuracies(DATA_2)

import json
import numpy as np
from collections import defaultdict

def _to_jsonable(x):
    """Convert numpy/scalars/etc to JSON-serializable python types."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def recursive_to_dict(obj):
    """Recursively convert defaultdicts to dicts and numpy scalars to python scalars."""
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {str(k): recursive_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_dict(v) for v in obj]
    return _to_jsonable(obj)

# ---- save ----
ACC_RESULTS_PATH = "acc_results.json"
acc_results_dict = recursive_to_dict(ACC_RESULTS)

with open(ACC_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(acc_results_dict, f, indent=2)

print("Saved:", ACC_RESULTS_PATH)

import json

ACC_RESULTS_PATH = "acc_results.json"

with open(ACC_RESULTS_PATH, "r", encoding="utf-8") as f:
    ACC_RESULTS_LOADED = json.load(f)

print("Loaded keys:", list(ACC_RESULTS_LOADED.keys())[:5])
