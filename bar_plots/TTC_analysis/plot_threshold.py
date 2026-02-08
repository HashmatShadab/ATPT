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

        # print accuracy
        print(dataset, compute_accuracy(zero_shot_clean_preds_data[dataset], true_labels_data[dataset]))
        print(dataset, compute_accuracy(ttc_clean_preds_single_data[dataset], true_labels_data[dataset]))
        print(dataset, compute_accuracy(ttc_clean_preds_vanilla_data[dataset], true_labels_data[dataset]))
        print(dataset, compute_accuracy(ttc_clean_preds_weighted_data[dataset], true_labels_data[dataset]))
        # print mean confidence
        print(dataset, np.mean(zero_shot_clean_max_confidences_data[dataset]))
        print(dataset, np.mean(ttc_clean_max_confidences_single_data[dataset]))
        print(dataset, np.mean(ttc_clean_max_confidences_vanilla_data[dataset]))
        print(dataset, np.mean(ttc_clean_max_confidences_weighted_data[dataset]))


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
        print(dataset, compute_accuracy(ttc_adv_preds_single_data[dataset], true_labels_data[dataset]))
        print(dataset, compute_accuracy(ttc_adv_preds_vanilla_data[dataset], true_labels_data[dataset]))
        print(dataset, compute_accuracy(ttc_adv_preds_weighted_data[dataset], true_labels_data[dataset]))
        # print mean confidence
        print(dataset, np.mean(zero_shot_adv_max_confidences_data[dataset]))
        print(dataset, np.mean(ttc_adv_max_confidences_single_data[dataset]))
        print(dataset, np.mean(ttc_adv_max_confidences_vanilla_data[dataset]))
        print(dataset, np.mean(ttc_adv_max_confidences_weighted_data[dataset]))

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

    # Zero-Shot clean predictions for all datasets from TTC dict
    ZS_TTC_CLEAN_PREDS_DATASET = ttc_dic["zero_shot_clean"]
    # Zero-Shot adversarial predictions for all datasets from TTC dict
    ZS_TTC_ADV_PREDS_DATASET = ttc_dic["zero_shot_adv"]

    # compute zero shot accuracy





