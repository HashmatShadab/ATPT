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


# %% [markdown]
# ## Inspect one experiment setting + compute accuracies

# %%
import numpy as np

# %%
model = "vit_l_14_datacomp_1b"
case = "clean"


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




# ## Zero-Shot Experiment Plots (Evaluation on Single image (No Ensembling of Augmnetations))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Data Preparation ---
datasets = DATASETS
# Single: No Augmentation Ensembling
clean_accs_single = [compute_accuracy(ZS_CLEAN_PREDS_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_accs_single = [compute_accuracy(ZS_ADV_PREDS_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_img_accs_single = [compute_accuracy(ZS_ADV_IMAGE_ONLY_PREDS_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]

# Vanilla
clean_accs_vanilla = [compute_accuracy(ZS_CLEAN_PREDS_VANILLA_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_accs_vanilla = [compute_accuracy(ZS_ADV_PREDS_VANILLA_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_img_accs_vanilla = [compute_accuracy(ZS_ADV_IMAGE_ONLY_PREDS_VANILLA_DATASET[d], TRUE_LABELS_DATASET[d]) for d in
                        datasets]

# Weighted
clean_accs_weighted = [compute_accuracy(ZS_CLEAN_PREDS_WEIGHTED_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_accs_weighted = [compute_accuracy(ZS_ADV_PREDS_WEIGHTED_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_img_accs_weighted = [compute_accuracy(ZS_ADV_IMAGE_ONLY_PREDS_WEIGHTED_DATASET[d], TRUE_LABELS_DATASET[d]) for d in
                         datasets]


# --- 2. Plotting ---

def plot_ensembling_comparison(data_groups, title, ylabel="Accuracy (%)", ylim_top=100.0, filename=None):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(datasets))
    width = 0.25

    comp_labels = ['Single', 'Vanilla', 'Weighted']
    comp_colors = ["#4A90E2", "#50E3C2", "#F5A623"]  # Distinct palette for ensembling
    comp_patterns = ["", "//", ".."]

    # Create the bars
    rects1 = ax.bar(x - width, data_groups[0], width, label=comp_labels[0],
                    color=comp_colors[0], hatch=comp_patterns[0], edgecolor='white', linewidth=1)
    rects2 = ax.bar(x, data_groups[1], width, label=comp_labels[1],
                    color=comp_colors[1], hatch=comp_patterns[1], edgecolor='white', linewidth=1)
    rects3 = ax.bar(x + width, data_groups[2], width, label=comp_labels[2],
                    color=comp_colors[2], hatch=comp_patterns[2], edgecolor='white', linewidth=1)

    # Styling
    ax.set_title(title, fontsize=18, pad=25, color='#222222')
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha='center', fontsize=11)
    ax.set_ylim(0, ylim_top)

    # Add light horizontal grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
    ax.set_axisbelow(True)
    sns.despine()

    # Numeric Labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#444444')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#DDDDDD', fontsize=10)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


# --- 3. Execute Comparison Plots ---

# Clean Setting
plot_ensembling_comparison([clean_accs_single, clean_accs_vanilla, clean_accs_weighted],
                           f"Ensembling Comparison: Clean Setting ({model})", filename=f"zs_aug_ensemble_{model}_clean_comparison.png")

# Adversarial (Image-Text) Setting
plot_ensembling_comparison([adv_accs_single, adv_accs_vanilla, adv_accs_weighted],
                           f"Ensembling Comparison: Adversarial (Image-Text) ({model})", filename=f"zs_aug_ensemble_{model}_adv_img_text_comparison.png")

# Adversarial (Image-Only) Setting
plot_ensembling_comparison([adv_img_accs_single, adv_img_accs_vanilla, adv_img_accs_weighted],
                           f"Ensembling Comparison: Adversarial (Image-Only) ({model})", filename=f"zs_aug_ensemble_{model}_adv_img_only_comparison.png")

# --- 4. Global Average Summary ---

plt.figure(figsize=(10, 6))
conditions = ['Clean', 'Adv (Img-Txt)', 'Adv (Img)']
comp_methods = ['Single', 'Vanilla', 'Weighted']
comp_colors = ["#4A90E2", "#50E3C2", "#F5A623"]

avg_data = {
    'Single': [np.mean(clean_accs_single), np.mean(adv_accs_single), np.mean(adv_img_accs_single)],
    'Vanilla': [np.mean(clean_accs_vanilla), np.mean(adv_accs_vanilla), np.mean(adv_img_accs_vanilla)],
    'Weighted': [np.mean(clean_accs_weighted), np.mean(adv_accs_weighted), np.mean(adv_img_accs_weighted)]
}

x = np.arange(len(conditions))
width = 0.25

for i, method in enumerate(comp_methods):
    plt.bar(x + (i - 1) * width, avg_data[method], width, label=method,
            color=comp_colors[i], edgecolor='white', linewidth=1.5)

plt.title(f"Overall Ensembling Performance Summary ({model})", fontsize=18, pad=20)
plt.ylabel("Global Mean Accuracy (%)", fontsize=12)
plt.xticks(x, conditions)
plt.ylim(0, 100)
plt.legend()

for i in range(len(conditions)):
    for j, method in enumerate(comp_methods):
        val = avg_data[method][i]
        plt.text(i + (j - 1) * width, val + 1, f"{val:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

sns.despine()
plt.tight_layout()
plt.savefig(f"zs_aug_ensemble_{model}_overall_performance_summary.png", bbox_inches='tight')
plt.show()
