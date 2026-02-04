from pathlib import Path
import json
import numpy as np
import argparse

MODELS = [
    "delta_clip_l14_224",
    "fare4",
    "ViT-L/14",
    "vit_l_14_datacomp_1b",
]

MODEL_NAME_MAP = {
    "delta_clip_l14_224": "Δ-CLIP (DataComp-1B)",
    "fare4": "FARE-4",
    "ViT-L/14": "CLIP",
    "vit_l_14_datacomp_1b": "CLIP (DataComp-1B)"
}


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

# --- Argparse ---
parser = argparse.ArgumentParser(description="Zero-shot single plots")
parser.add_argument("--model", type=str, default="ViT-L/14", help="Model name")
parser.add_argument("--legend_y", type=float, default=1.1, help="Vertical position of the legend")
parser.add_argument("--summary_xtick_fontsize", type=int, default=16, help="Font size for x-axis ticks in summary plots")
args = parser.parse_args()

model = args.model
safe_model_name = model.replace("/", "-")

DATA = build_all_data(RESULT_PATHS, MODELS, DATASETS)


# %% [markdown]
# ## Inspect one experiment setting + compute accuracies

# %%
import numpy as np

# %%
case = "clean"
# model = "vit_l_14_datacomp_1b" # Removed hardcoded model


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
# model = "vit_l_14_datacomp_1b" # Removed hardcoded model

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
# model = "vit_l_14_datacomp_1b" # Removed hardcoded model

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

# --- 1. Style & Aesthetic Configuration ---
sns.set_theme(style="white") # Clean background
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

# Custom Palette: Deep Blue, Soft Coral, Sage Green
colors = ["#2A5A8A", "#D9534F", "#5CB85C"]
patterns = ["", "//", "oo"] # Clean, Diagonal Stripes, Small Circles
labels = ['Clean', 'Adv (Image-Text)', 'Adv (Image)']
datasets = list(DATASETS)

# --- 2. Data Preparation ---
clean_accs = [compute_accuracy(ZS_CLEAN_PREDS_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_accs = [compute_accuracy(ZS_ADV_PREDS_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]
adv_img_accs = [compute_accuracy(ZS_ADV_IMAGE_ONLY_PREDS_DATASET[d], TRUE_LABELS_DATASET[d]) for d in datasets]

clean_confs = [np.mean(zero_shot_clean_max_confidences_data[d]) for d in datasets]
adv_confs = [np.mean(zero_shot_adv_max_confidences_data[d]) for d in datasets]
adv_img_confs = [np.mean(zero_shot_adv_image_only_max_confidences_data[d]) for d in datasets]

def plot_styled_bars(data_groups, title, ylabel, ylim_top, filename=None, legend_y=1.1):
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    x = np.arange(len(datasets))
    width = 0.30

    # Create the bars with patterns
    rects1 = ax.bar(x - width, data_groups[0], width, label=labels[0],
                    color=colors[0], hatch=patterns[0], edgecolor='white', linewidth=1)
    rects2 = ax.bar(x, data_groups[1], width, label=labels[1],
                    color=colors[1], hatch=patterns[1], edgecolor='white', linewidth=1)
    rects3 = ax.bar(x + width, data_groups[2], width, label=labels[2],
                    color=colors[2], hatch=patterns[2], edgecolor='white', linewidth=1)

    # Styling
    ax.set_title(title, fontsize=18, pad=35, color='#222222')
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha='center', fontsize=16)
    ax.set_ylim(0, ylim_top)

    # Add light horizontal grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
    ax.set_axisbelow(True) # Ensure grid is behind bars

    # Remove top/right spines
    sns.despine()

    # Numeric Labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12,  color='#444444')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y),
              frameon=True, facecolor='white', edgecolor='#DDDDDD', fontsize=16, ncol=3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

# --- 3. Execute Plots ---
# model = "vit_l_14_datacomp_1b" # Removed hardcoded model
display_name = MODEL_NAME_MAP.get(model, model)

# Accuracy Plot
plot_styled_bars([clean_accs, adv_accs, adv_img_accs],
                 f"{display_name}: Accuracy Under Clean/Adversarial Setting ",
                 "Zero-Shot Accuracy (%)", 110.0, filename=f"zs_single_{safe_model_name}_accuracy_plot.png",
                 legend_y=args.legend_y)

# Confidence Plot
plot_styled_bars([clean_confs, adv_confs, adv_img_confs],
                 f"{display_name}: Prediction Confidence",
                 "Mean Max Confidence", 1.15, filename=f"zs_single_{safe_model_name}_confidence_plot.png",
                 legend_y=args.legend_y)

# --- 4. Global Average Plot (Final Summary) ---
plt.figure(figsize=(10, 6))
avg_values = [np.mean(clean_accs), np.mean(adv_accs), np.mean(adv_img_accs)]
avg_labels = ['Clean', 'Adv (Image-Txt)', 'Adv (Image)']

# Barplot with patterns for the summary
for i in range(len(avg_values)):
    plt.bar(avg_labels[i], avg_values[i], color=colors[i],
            hatch=patterns[i], edgecolor='white', width=0.6, linewidth=1.5)

plt.title(f"{display_name}: Average Accuracy across Datasets", fontsize=18, pad=20)
plt.ylabel("Average Zero-Shot Accuracy (%)", fontsize=16)
plt.ylim(0, max(avg_values) * 1.05)

for i, v in enumerate(avg_values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom',  fontsize=16)

plt.xticks(fontsize=args.summary_xtick_fontsize)
sns.despine()
plt.tight_layout()
plt.savefig(f"zs_single_{safe_model_name}_overall_performance_summary.png", bbox_inches='tight')
plt.show()

# --- 5. Global Average Confidence Plot (Final Summary) ---
plt.figure(figsize=(10, 6))
avg_conf_values = [np.mean(clean_confs), np.mean(adv_confs), np.mean(adv_img_confs)]
avg_conf_labels =  ['Clean', 'Adv (Image-Txt)', 'Adv (Image)']

# Barplot with patterns for the summary
for i in range(len(avg_conf_values)):
    plt.bar(avg_conf_labels[i], avg_conf_values[i], color=colors[i],
            hatch=patterns[i], edgecolor='white', width=0.6, linewidth=1.5)

plt.title(f"{display_name}: Average Confidence across Datasets", fontsize=18, pad=20)
plt.ylabel("Average Confidence", fontsize=16)
plt.ylim(0, 1.05)

for i, v in enumerate(avg_conf_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom',  fontsize=16)

plt.xticks(fontsize=args.summary_xtick_fontsize)
sns.despine()
plt.tight_layout()
plt.savefig(f"zs_single_{safe_model_name}_overall_confidence_summary.png", bbox_inches='tight')
plt.show()