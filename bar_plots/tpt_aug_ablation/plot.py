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
import os
import re
from typing import Any, Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Map adversarial `attack_key` identifiers to human-readable legend labels.
# (Used for plots where we compare Clean vs a single adversarial setting.)
ATTACK_KEY_LEGEND_MAPPING = {
    "eps_4.0_steps_100": "PGD 4/255 (100 steps)",
    "eps_8.0_steps_100": "PGD 8/255 (100 steps)",
}


# Human-friendly names for plotting only (keys are on-disk augmentation pool identifiers).
AUGMENTATION_PLOT_NAMES = {
    "all": "Combined",
    "photometric_tpt": "Photometric",
    "geometric_tpt": "Geometric",
}


def augmentation_plot_name(aug: str) -> str:
    return AUGMENTATION_PLOT_NAMES.get(aug, aug)


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


def get_zs_aug_results(model_name):
    from pathlib import Path
    import json

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

    # Keep this list minimal for the diff-ratio grid/plots (requested).
    # NOTE: these are the on-disk augmentation pool identifiers.
    AUGMENTATIONS = ["all", "photometric_tpt", "geometric_tpt"]

    # Optional on-disk cache: loading all jsons for all augmentations is slow.
    # Cache is stored next to this plotting script.
    cache_path = Path(__file__).with_name(
        f"zs_aug_results_cache_{sanitize_for_path(model_name)}.json"
    )

    def _try_load_cache() -> Optional[dict]:
        try:
            if cache_path.exists():
                with cache_path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                # very lightweight schema check
                if isinstance(obj, dict) and obj.get("model_name") == model_name and "data" in obj:
                    return obj["data"]
        except Exception as e:
            print(f"[WARN] Failed to load ZS aug cache at {cache_path}: {e}")
        return None

    def _try_save_cache(data: dict) -> None:
        try:
            payload = {
                "model_name": model_name,
                "augmentations": AUGMENTATIONS,
                "datasets": DATASETS,
                "data": data,
            }
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"[INFO] Saved ZS aug cache to: {cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to save ZS aug cache at {cache_path}: {e}")

    # Try cache first (this call can be very slow otherwise).
    cached = _try_load_cache()
    if cached is not None:
        # Backward/forward safety: if an older cache exists with different augmentation set,
        # ignore it and rebuild.
        # (This script's correctness depends on matching the current AUGMENTATIONS list.)
        try:
            if set(cached.get("zero_shot_aug_clean", {}).keys()) == set(AUGMENTATIONS):
                return cached
        except Exception:
            pass
        print(
            f"[INFO] Ignoring cached ZS aug data due to AUGMENTATIONS mismatch. Rebuilding: {AUGMENTATIONS}"
        )

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
    # IMPORTANT: build paths relative to this file to avoid dependence on CWD.
    results_root = Path(__file__).resolve().parents[2] / "Final_Results_corrected_ca_tau_aug_pool_ablation"

    RESULT_PATHS = {
        "zero_shot": {
            "clean": {
                "base_path": (
                    str(
                        results_root
                        / "{model}"
                        / "{dataset}"
                        / "Clean"
                        / "No_Counter_Attack"
                        / "No_TPT"
                        / "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01_augmentation_pool_{augment}"
                    )
                ),
            },
            "adversarial_eps4_steps100": {
                "base_path": (
                    str(
                        results_root
                        / "{model}"
                        / "{dataset}"
                        / "Adversarial_Eps_4_0_Steps_100"
                        / "No_Counter_Attack"
                        / "No_TPT"
                        / "Inference_Ensemble_all_weighted_rtpt_topk_20_softmaxtemp_0_01_augmentation_pool_{augment}"
                    )
                ),
            },
            # "adversarial_eps4_steps100_image_only": {
            #     "base_path": (
            #         "../../Final_Results_corrected_ca_tau/"
            #         "{model}/{dataset}/"
            #         "Adversarial_Eps_4_0_Steps_100_image_only_attack_prm/"
            #         "No_Counter_Attack/No_TPT/"
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
        "results_counter_attack_diff_ratio": "results_images_diff_ratio_tpt_noisy_anchors.json",

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
            if not fp.exists() and key == "original":
                # Some clean runs only store `results_original_clean.json`.
                # Fall back to that file if `results_original.json` is missing.
                fp_alt = base / JSON_KEYMAP["original_clean"]
                if fp_alt.exists():
                    fp = fp_alt
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

        # aug diff ratio
        fp = base / JSON_KEYMAP["results_counter_attack_diff_ratio"]
        obj = _load_json(fp)
        diff_ratio_per_sample = diff_ratio(obj)["diff_ratio_per_sample"]


        return {"preds": preds, "diff_ratio_per_sample": diff_ratio_per_sample}

    def build_all_data(result_paths: dict, models: list, datasets: list) -> dict:
        DATA = {}

        for case, cfg in result_paths["zero_shot"].items():
            DATA.setdefault(case, {})
            base_template = cfg["base_path"]

            for model in models:
                DATA[case].setdefault(model, {})

                for dataset in datasets:
                    DATA[case][model].setdefault(dataset, {})
                    for augment in AUGMENTATIONS:
                        base_path = base_template.format(model=model, dataset=dataset, augment=augment)
                        DATA[case][model][dataset][augment] = load_one_setting(base_path)

        return DATA

    ZS_DATA = build_all_data(RESULT_PATHS, MODELS, DATASETS)

    import numpy as np



    model = model_name

    # We return dictionaries keyed by augmentation and then dataset.
    # Example: out["zero_shot_aug_clean"][augment][dataset] -> list[int]
    true_labels_data: Dict[str, list] = {}

    out: Dict[str, Any] = {
        "true_labels": true_labels_data,
        "zero_shot_aug_clean": {a: {} for a in AUGMENTATIONS},
        "zero_shot_aug_adv": {a: {} for a in AUGMENTATIONS},
        "zero_shot_aug_clean_correct_preds": {a: {} for a in AUGMENTATIONS},
        "zero_shot_aug_vanilla_clean": {a: {} for a in AUGMENTATIONS},
        "zero_shot_aug_vanilla_adv": {a: {} for a in AUGMENTATIONS},
        "zero_shot_aug_weighted_clean": {a: {} for a in AUGMENTATIONS},
        "zero_shot_aug_weighted_adv": {a: {} for a in AUGMENTATIONS},
        "diff_ratio_per_sample_clean": {a: {} for a in AUGMENTATIONS},
        "diff_ratio_per_sample_adv": {a: {} for a in AUGMENTATIONS},
    }

    # Labels are identical across augmentations; take from the first augmentation available.
    first_aug = AUGMENTATIONS[0]
    for dataset in DATASETS:
        example = ZS_DATA["clean"][model][dataset][first_aug]
        true_labels_data[dataset] = example["preds"]["original_clean"]["label"]

    # Fill outputs per augmentation
    for augment in AUGMENTATIONS:
        # --- clean
        for dataset in DATASETS:
            example = ZS_DATA["clean"][model][dataset][augment]
            out["zero_shot_aug_clean"][augment][dataset] = example["preds"]["original_clean"]["prediction"]
            out["zero_shot_aug_clean_correct_preds"][augment][dataset] = example["preds"]["original_clean_correct"]
            out["zero_shot_aug_vanilla_clean"][augment][dataset] = example["preds"]["vanilla"]["prediction"]
            out["zero_shot_aug_weighted_clean"][augment][dataset] = example["preds"]["weighted"]["prediction"]
            out["diff_ratio_per_sample_clean"][augment][dataset] = example.get("diff_ratio_per_sample")

        # --- adversarial
        for dataset in DATASETS:
            example = ZS_DATA["adversarial_eps4_steps100"][model][dataset][augment]
            out["zero_shot_aug_adv"][augment][dataset] = example["preds"]["original"]["prediction"]
            out["zero_shot_aug_vanilla_adv"][augment][dataset] = example["preds"]["vanilla"]["prediction"]
            out["zero_shot_aug_weighted_adv"][augment][dataset] = example["preds"]["weighted"]["prediction"]
            out["diff_ratio_per_sample_adv"][augment][dataset] = example.get("diff_ratio_per_sample")

        # Optional per-augmentation summary prints (kept minimal)
        try:
            d0 = DATASETS[0]
            acc = compute_accuracy(out["zero_shot_aug_clean"][augment][d0], true_labels_data[d0])
            print(f"[ZS_AUG] augment={augment} {d0} clean acc={acc:.2f}")
        except Exception:
            pass

    _try_save_cache(out)
    return out


def compute_accuracy(preds, labels) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.shape[0] != labels.shape[0]:
        raise ValueError(f"Pred/label length mismatch: {preds.shape[0]} vs {labels.shape[0]}")
    return float((preds == labels).mean() * 100.0)

def aggregate_avg_accuracy_across_datasets(zs_aug_dic):
    """
    Returns:
        avg_acc[augment][setting] = average accuracy over datasets
        setting ∈ {
            clean_original, clean_vanilla, clean_weighted,
            adv_original, adv_vanilla, adv_weighted
        }
    """
    datasets = list(zs_aug_dic["true_labels"].keys())
    augmentations = list(zs_aug_dic["zero_shot_aug_clean"].keys())

    avg_acc = {}

    for aug in augmentations:
        avg_acc[aug] = {
            "clean_original": [],
            "clean_vanilla": [],
            "clean_weighted": [],
            "adv_original": [],
            "adv_vanilla": [],
            "adv_weighted": [],
        }

        for dset in datasets:
            labels = zs_aug_dic["true_labels"][dset]

            # ---- CLEAN ----
            avg_acc[aug]["clean_original"].append(
                compute_accuracy(
                    zs_aug_dic["zero_shot_aug_clean"][aug][dset], labels
                )
            )
            avg_acc[aug]["clean_vanilla"].append(
                compute_accuracy(
                    zs_aug_dic["zero_shot_aug_vanilla_clean"][aug][dset], labels
                )
            )
            avg_acc[aug]["clean_weighted"].append(
                compute_accuracy(
                    zs_aug_dic["zero_shot_aug_weighted_clean"][aug][dset], labels
                )
            )

            # ---- ADVERSARIAL ----
            avg_acc[aug]["adv_original"].append(
                compute_accuracy(
                    zs_aug_dic["zero_shot_aug_adv"][aug][dset], labels
                )
            )
            avg_acc[aug]["adv_vanilla"].append(
                compute_accuracy(
                    zs_aug_dic["zero_shot_aug_vanilla_adv"][aug][dset], labels
                )
            )
            avg_acc[aug]["adv_weighted"].append(
                compute_accuracy(
                    zs_aug_dic["zero_shot_aug_weighted_adv"][aug][dset], labels
                )
            )

        # Convert to mean
        for k in avg_acc[aug]:
            avg_acc[aug][k] = np.mean(avg_acc[aug][k])

    return avg_acc


def aggregate_accuracy_by_dataset_and_augmentation(zs_aug_dic: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute accuracy per dataset (no cross-dataset averaging).

    Returns:
        acc[dataset][augment][setting] = accuracy (%)

        setting ∈ {
            clean_original, clean_vanilla, clean_weighted,
            adv_original, adv_vanilla, adv_weighted
        }
    """

    datasets = list(zs_aug_dic["true_labels"].keys())
    augmentations = list(zs_aug_dic["zero_shot_aug_clean"].keys())

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dset in datasets:
        labels = zs_aug_dic["true_labels"][dset]
        out[dset] = {}
        for aug in augmentations:
            out[dset][aug] = {
                "clean_original": compute_accuracy(zs_aug_dic["zero_shot_aug_clean"][aug][dset], labels),
                "clean_vanilla": compute_accuracy(zs_aug_dic["zero_shot_aug_vanilla_clean"][aug][dset], labels),
                "clean_weighted": compute_accuracy(zs_aug_dic["zero_shot_aug_weighted_clean"][aug][dset], labels),
                "adv_original": compute_accuracy(zs_aug_dic["zero_shot_aug_adv"][aug][dset], labels),
                "adv_vanilla": compute_accuracy(zs_aug_dic["zero_shot_aug_vanilla_adv"][aug][dset], labels),
                "adv_weighted": compute_accuracy(zs_aug_dic["zero_shot_aug_weighted_adv"][aug][dset], labels),
            }

    return out


def aggregate_avg_diff_ratio_across_datasets(zs_aug_dic):
    """Aggregate average diff-ratio across datasets for each augmentation.

    We compute, for each augmentation and dataset, the mean over samples of
    `diff_ratio_per_sample_{clean,adv}` (when available), then average those
    dataset-level means across datasets.

    Returns:
        avg_diff[augment][setting] = average diff ratio over datasets
        setting ∈ {clean, adv}
    """

    datasets = list(zs_aug_dic["true_labels"].keys())
    augmentations = list(zs_aug_dic["zero_shot_aug_clean"].keys())

    def _safe_mean(x) -> float:
        if x is None:
            return float("nan")
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.nanmean(arr))

    avg_diff = {}

    for aug in augmentations:
        clean_vals = []
        adv_vals = []

        for dset in datasets:
            clean_list = zs_aug_dic.get("diff_ratio_per_sample_clean", {}).get(aug, {}).get(dset)
            adv_list = zs_aug_dic.get("diff_ratio_per_sample_adv", {}).get(aug, {}).get(dset)

            c = _safe_mean(clean_list)
            a = _safe_mean(adv_list)
            if np.isfinite(c):
                clean_vals.append(c)
            if np.isfinite(a):
                adv_vals.append(a)

        avg_diff[aug] = {
            "clean": float(np.mean(clean_vals)) if clean_vals else float("nan"),
            "adv": float(np.mean(adv_vals)) if adv_vals else float("nan"),
        }

    return avg_diff


def aggregate_diff_ratio_by_dataset_and_augmentation(
    zs_aug_dic: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean diff-ratio per dataset and augmentation (no cross-dataset averaging).

    Returns:
        diff[dataset][augment][setting] = mean diff ratio

        setting ∈ {clean, adv}

    Notes:
        - If a diff-ratio list is missing/empty for a dataset+augmentation, the value is NaN.
    """

    datasets = list(zs_aug_dic.get("true_labels", {}).keys())
    augmentations = list(zs_aug_dic.get("zero_shot_aug_clean", {}).keys())

    def _safe_mean(x) -> float:
        if x is None:
            return float("nan")
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.nanmean(arr))

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dset in datasets:
        out[dset] = {}
        for aug in augmentations:
            clean_list = zs_aug_dic.get("diff_ratio_per_sample_clean", {}).get(aug, {}).get(dset)
            adv_list = zs_aug_dic.get("diff_ratio_per_sample_adv", {}).get(aug, {}).get(dset)
            out[dset][aug] = {
                "clean": _safe_mean(clean_list),
                "adv": _safe_mean(adv_list),
            }

    return out


def _combine_predictions_by_diff_ratio_threshold(
    original_preds: List[int],
    modified_preds: List[int],
    diff_ratios: List[float],
    threshold: float,
) -> List[int]:
    """Per-sample routing based on diff ratio.

    For each sample i:
      - if diff_ratios[i] < threshold: use original_preds[i]
      - else: use modified_preds[i]

    This matches the requested behavior for vanilla/weighted predictions.
    """
    if len(original_preds) != len(modified_preds) or len(original_preds) != len(diff_ratios):
        raise ValueError(
            "Length mismatch for threshold routing: "
            f"original={len(original_preds)} modified={len(modified_preds)} diff_ratios={len(diff_ratios)}"
        )

    # Ensure we can compare diff ratios even if they were loaded as strings.
    dr = np.asarray(diff_ratios, dtype=float)
    out = []
    for i in range(len(original_preds)):
        out.append(original_preds[i] if dr[i] < threshold else modified_preds[i])
    return out


def aggregate_thresholded_accuracy_across_datasets(
    zs_aug_dic: Dict[str, Any],
    thresholds: List[float],
) -> Dict[str, Dict[str, Dict[Any, float]]]:
    """Compute average accuracy across datasets under per-sample diff-ratio threshold routing.

    Returns:
        avg_acc_thr[augment][setting][threshold] = average accuracy over datasets

        setting ∈ {
            clean_original,
            adv_original,
            clean_vanilla_thr,
            clean_weighted_thr,
            adv_vanilla_thr,
            adv_weighted_thr,
        }
    """
    datasets = list(zs_aug_dic["true_labels"].keys())
    augmentations = list(zs_aug_dic["zero_shot_aug_clean"].keys())

    out: Dict[str, Dict[str, Dict[Any, float]]] = {}

    for aug in augmentations:
        out[aug] = {
            "clean_original": {},
            "adv_original": {},
            "clean_vanilla_thr": {},
            "clean_weighted_thr": {},
            "adv_vanilla_thr": {},
            "adv_weighted_thr": {},
        }

        # Pre-compute per-dataset baselines for stability.
        clean_original_accs = []
        adv_original_accs = []
        for dset in datasets:
            labels = zs_aug_dic["true_labels"][dset]
            clean_orig = zs_aug_dic["zero_shot_aug_clean"][aug][dset]
            adv_orig = zs_aug_dic["zero_shot_aug_adv"][aug][dset]
            clean_original_accs.append(compute_accuracy(clean_orig, labels))
            adv_original_accs.append(compute_accuracy(adv_orig, labels))

        out[aug]["clean_original"]["_mean"] = float(np.mean(clean_original_accs))
        out[aug]["adv_original"]["_mean"] = float(np.mean(adv_original_accs))

        # Thresholded routing for vanilla/weighted.
        for thr in thresholds:
            clean_vanilla_accs = []
            clean_weighted_accs = []
            adv_vanilla_accs = []
            adv_weighted_accs = []

            for dset in datasets:
                labels = zs_aug_dic["true_labels"][dset]

                # ---- CLEAN ----
                clean_orig = zs_aug_dic["zero_shot_aug_clean"][aug][dset]
                clean_vanilla = zs_aug_dic["zero_shot_aug_vanilla_clean"][aug][dset]
                clean_weighted = zs_aug_dic["zero_shot_aug_weighted_clean"][aug][dset]
                dr_clean = zs_aug_dic.get("diff_ratio_per_sample_clean", {}).get(aug, {}).get(dset)
                if dr_clean is None:
                    raise KeyError(f"Missing diff ratios for clean: aug={aug} dataset={dset}")

                clean_vanilla_routed = _combine_predictions_by_diff_ratio_threshold(
                    clean_orig, clean_vanilla, dr_clean, thr
                )
                clean_weighted_routed = _combine_predictions_by_diff_ratio_threshold(
                    clean_orig, clean_weighted, dr_clean, thr
                )

                clean_vanilla_accs.append(compute_accuracy(clean_vanilla_routed, labels))
                clean_weighted_accs.append(compute_accuracy(clean_weighted_routed, labels))

                # ---- ADVERSARIAL ----
                adv_orig = zs_aug_dic["zero_shot_aug_adv"][aug][dset]
                adv_vanilla = zs_aug_dic["zero_shot_aug_vanilla_adv"][aug][dset]
                adv_weighted = zs_aug_dic["zero_shot_aug_weighted_adv"][aug][dset]
                dr_adv = zs_aug_dic.get("diff_ratio_per_sample_adv", {}).get(aug, {}).get(dset)
                if dr_adv is None:
                    raise KeyError(f"Missing diff ratios for adv: aug={aug} dataset={dset}")

                adv_vanilla_routed = _combine_predictions_by_diff_ratio_threshold(
                    adv_orig, adv_vanilla, dr_adv, thr
                )
                adv_weighted_routed = _combine_predictions_by_diff_ratio_threshold(
                    adv_orig, adv_weighted, dr_adv, thr
                )

                adv_vanilla_accs.append(compute_accuracy(adv_vanilla_routed, labels))
                adv_weighted_accs.append(compute_accuracy(adv_weighted_routed, labels))

            out[aug]["clean_vanilla_thr"][thr] = float(np.mean(clean_vanilla_accs))
            out[aug]["clean_weighted_thr"][thr] = float(np.mean(clean_weighted_accs))
            out[aug]["adv_vanilla_thr"][thr] = float(np.mean(adv_vanilla_accs))
            out[aug]["adv_weighted_thr"][thr] = float(np.mean(adv_weighted_accs))

    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPT augmentaation analysis plotting / aggregation")
    parser.add_argument("--model-name", type=str, default="vit_l_14_datacomp_1b")
    parser.add_argument("--out-dir", type=str, default=os.path.join("plots_output", "tpt_aug_ablation"))
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--diff-ratio-thresholds",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,1.0",
        help=(
            "Comma-separated diff-ratio thresholds. For each sample: if diff_ratio < threshold, "
            "use original prediction; else use vanilla/weighted prediction."
        ),
    )

    parser.add_argument(
        "--dataset-grid-cols",
        type=int,
        default=4,
        help="Number of columns for the per-dataset plot grid image.",
    )
    parser.add_argument(
        "--dataset-grid-scale",
        type=float,
        default=1.0,
        help="Optional scaling factor applied to each tile when building the per-dataset grid image.",
    )


    args = parser.parse_args()

    model_name = args.model_name


    # Zero-shot single, vanilla, weighted results with different type of augmentations used create multiple views of the original clean/adversarial image
    zero_shot_aug_dic = get_zs_aug_results(model_name)


    def plot_avg_accuracy_by_augmentation(avg_acc, out_dir, model_name):
        os.makedirs(out_dir, exist_ok=True)

        # Slightly nicer default look without adding new dependencies.
        plt.style.use("seaborn-v0_8-whitegrid")

        augmentations = list(avg_acc.keys())
        x = np.arange(len(augmentations))
        width = 0.12

        fig, ax = plt.subplots(figsize=(14, 6.5))

        settings = [
            # Muted palette + consistent semantics (Clean=blue, Adv=orange/red)
            ("clean_original", "Clean – Original", "#4C72B0"),
            ("clean_vanilla", "Clean – Vanilla", "#9AB6DF"),
            ("clean_weighted", "Clean – Weighted", "#2F5597"),
            ("adv_original", "Adv – Original", "#DD8452"),
            ("adv_vanilla", "Adv – Vanilla", "#F0B38C"),
            ("adv_weighted", "Adv – Weighted", "#C44E52"),
        ]

        # Keep y-limits stable and leave headroom for value labels.
        all_vals = [avg_acc[aug][k] for aug in augmentations for k, _, _ in settings]
        ymin = max(0.0, float(np.floor(min(all_vals) / 5.0) * 5.0))
        ymax = float(np.ceil(max(all_vals) / 5.0) * 5.0) + 3.0
        ax.set_ylim(ymin, ymax)

        def _add_value_labels(bars, *, fmt: str = "{:.1f}", fontsize: int = 8):
            """Add numbers on top of bars (matplotlib-version-safe)."""
            for b in bars:
                h = b.get_height()
                if not np.isfinite(h):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.2,
                    fmt.format(h),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    color="#222222",
                    rotation=0,
                    clip_on=False,
                )

        for i, (key, label, color) in enumerate(settings):
            values = [avg_acc[aug][key] for aug in augmentations]
            bars = ax.bar(
                x + (i - 2.5) * width,
                values,
                width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
            _add_value_labels(bars)

        ax.set_xticks(x)
        ax.set_xticklabels(augmentations, rotation=15, ha="right")
        ax.set_ylabel("Average Accuracy (%)", labelpad=8, fontsize=12)
        ax.set_xlabel("Augmentation Type", fontsize=12)
        # ax.set_title(
        #     f"Average Clean and Adversarial Accuracy Across Datasets\nModel: {model_name}",
        #     fontsize=14,
        #     pad=18,
        # )

        ax.tick_params(axis="both", which="major", labelsize=10)

        ax.legend(
            ncol=6,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            fontsize=10,
            columnspacing=1.2,
            handlelength=1.4,
            borderpad=0.6,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", visible=False)

        # Improve readability a bit
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"avg_accuracy_by_augmentation_{model_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
        plt.close()


    def plot_accuracy_by_augmentation_per_dataset(
        acc_by_dataset: Dict[str, Dict[str, Dict[str, float]]],
        out_dir: str,
        model_name: str,
        *,
        grid_cols: int = 4,
        grid_scale: float = 1.0,
    ) -> None:
        """Create one accuracy plot per dataset + a grid image that contains all dataset plots."""

        os.makedirs(out_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")

        # Ensure stable ordering (match the order used in the loading section when possible).
        dataset_names = list(acc_by_dataset.keys())
        if not dataset_names:
            return

        # Augmentations from first dataset.
        augmentations = list(acc_by_dataset[dataset_names[0]].keys())
        x = np.arange(len(augmentations))
        width = 0.12

        settings = [
            ("clean_original", "Clean – Original", "#4C72B0"),
            ("clean_vanilla", "Clean – Vanilla", "#9AB6DF"),
            ("clean_weighted", "Clean – Weighted", "#2F5597"),
            ("adv_original", "Adv – Original", "#DD8452"),
            ("adv_vanilla", "Adv – Vanilla", "#F0B38C"),
            ("adv_weighted", "Adv – Weighted", "#C44E52"),
        ]

        def _add_value_labels(ax, bars, *, fmt: str = "{:.1f}", fontsize: int = 8):
            for b in bars:
                h = b.get_height()
                if not np.isfinite(h):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.2,
                    fmt.format(h),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    color="#222222",
                    rotation=0,
                    clip_on=False,
                )

        saved_paths: List[str] = []

        for dset in dataset_names:
            # Compute y-limits per dataset with a little headroom.
            all_vals = [
                acc_by_dataset[dset][aug][k]
                for aug in augmentations
                for k, _, _ in settings
                if k in acc_by_dataset[dset][aug]
            ]
            ymin = max(0.0, float(np.floor(min(all_vals) / 5.0) * 5.0)) if all_vals else 0.0
            ymax = (float(np.ceil(max(all_vals) / 5.0) * 5.0) + 3.0) if all_vals else 100.0

            fig, ax = plt.subplots(figsize=(14, 6.5))
            ax.set_ylim(ymin, ymax)

            for i, (key, label, color) in enumerate(settings):
                values = [float(acc_by_dataset[dset][aug][key]) for aug in augmentations]
                bars = ax.bar(
                    x + (i - 2.5) * width,
                    values,
                    width,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                )
                _add_value_labels(ax, bars)

            ax.set_xticks(x)
            ax.set_xticklabels(augmentations, rotation=15, ha="right")
            ax.set_ylabel("Accuracy (%)", labelpad=8, fontsize=12)
            ax.set_xlabel("Augmentation Type", fontsize=12)
            ax.set_title(f"{dset} | Model: {model_name}", fontsize=13, pad=12)

            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.legend(
                ncol=6,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                frameon=True,
                fancybox=True,
                framealpha=0.95,
                fontsize=10,
                columnspacing=1.2,
                handlelength=1.4,
                borderpad=0.6,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.grid(axis="x", visible=False)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            plt.tight_layout()
            out_path = os.path.join(
                out_dir,
                f"accuracy_by_augmentation_{sanitize_for_path(dset)}_{model_name}.png",
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {out_path}")
            saved_paths.append(out_path)
            plt.close(fig)

        # Grid over datasets
        grid_out = os.path.join(out_dir, f"accuracy_by_augmentation_all_datasets_grid_{model_name}.png")
        create_image_grid(saved_paths, grid_out, cols=max(1, int(grid_cols)), scale=float(grid_scale))


    def plot_avg_diff_ratio_by_augmentation(avg_diff, out_dir, model_name):
        os.makedirs(out_dir, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")

        augmentations = list(avg_diff.keys())
        x = np.arange(len(augmentations))
        width = 0.36

        fig, ax = plt.subplots(figsize=(12.5, 5.8))

        settings = [
            ("clean", "Clean", "#4C72B0"),
            ("adv", "Adversarial", "#DD8452"),
        ]

        all_vals = [avg_diff[aug][k] for aug in augmentations for k, _, _ in settings]
        finite = [v for v in all_vals if np.isfinite(v)]
        if finite:
            ymin = max(0.0, float(np.floor(min(finite) * 20.0) / 20.0))
            ymax = float(np.ceil(max(finite) * 20.0) / 20.0) + 0.05
            ax.set_ylim(ymin, min(1.0, ymax) if ymax <= 1.0 else ymax)

        def _add_value_labels(bars, *, fmt: str = "{:.3f}", fontsize: int = 9):
            for b in bars:
                h = b.get_height()
                if not np.isfinite(h):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.01,
                    fmt.format(h),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    color="#222222",
                    rotation=0,
                    clip_on=False,
                )

        for i, (key, label, color) in enumerate(settings):
            values = [avg_diff[aug][key] for aug in augmentations]
            bars = ax.bar(
                x + (i - 0.5) * width,
                values,
                width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
            _add_value_labels(bars)

        ax.set_xticks(x)
        ax.set_xticklabels([augmentation_plot_name(a) for a in augmentations], rotation=0, ha="center")
        ax.set_ylabel("Average Diff Ratio", labelpad=10, fontsize=14)
        ax.set_xlabel("Augmentation Type", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)

        ax.legend(
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            fontsize=12,
            columnspacing=1.4,
            handlelength=1.6,
            borderpad=0.6,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", visible=False)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"avg_diff_ratio_by_augmentation_{model_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
        plt.close()


    def plot_diff_ratio_by_augmentation_per_dataset(
        diff_by_dataset: Dict[str, Dict[str, Dict[str, float]]],
        out_dir: str,
        model_name: str,
        *,
        grid_cols: int = 4,
        grid_scale: float = 1.0,
    ) -> None:
        """Create one diff-ratio plot per dataset + a grid image that contains all dataset plots."""

        os.makedirs(out_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")

        # Font sizes tuned for readability once stitched into a multi-panel grid.
        label_fontsize = 36
        tick_fontsize = 28
        legend_fontsize = 28
        title_fontsize = 42
        value_label_fontsize = 24

        dataset_names = list(diff_by_dataset.keys())
        if not dataset_names:
            return

        augmentations = list(diff_by_dataset[dataset_names[0]].keys())
        x = np.arange(len(augmentations))
        width = 0.36

        settings = [
            ("clean", "Clean", "#4C72B0"),
            ("adv", "Adversarial", "#DD8452"),
        ]

        def _add_value_labels(ax, bars, *, fmt: str = "{:.3f}", fontsize: int = value_label_fontsize):
            for b in bars:
                h = b.get_height()
                if not np.isfinite(h):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.01,
                    fmt.format(h),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    color="#222222",
                    rotation=0,
                    clip_on=False,
                )

        saved_paths: List[str] = []

        for dset in dataset_names:
            all_vals = [
                float(diff_by_dataset[dset][aug][k])
                for aug in augmentations
                for k, _, _ in settings
                if k in diff_by_dataset[dset][aug]
            ]
            finite = [v for v in all_vals if np.isfinite(v)]

            fig, ax = plt.subplots(figsize=(14, 8))
            # if finite:
            #     ymin = max(0.0, float(np.floor(min(finite) * 20.0) / 20.0))
            #     ymax = float(np.ceil(max(finite) * 20.0) / 20.0) + 0.05
            #     ax.set_ylim(ymin, min(1.0, ymax) if ymax <= 1.0 else ymax)

            for i, (key, label, color) in enumerate(settings):
                values = [float(diff_by_dataset[dset][aug][key]) for aug in augmentations]
                bars = ax.bar(
                    x + (i - 0.5) * width,
                    values,
                    width,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                )
                _add_value_labels(ax, bars)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [augmentation_plot_name(a) for a in augmentations],
                rotation=0,
                ha="center",
                fontsize=tick_fontsize,
            )
            ax.set_ylabel("Mean Latent Drift", labelpad=10, fontsize=label_fontsize)
            ax.set_xlabel("Stochastic Transformation", fontsize=label_fontsize)
            ax.set_title(f"{dset}", fontsize=title_fontsize, pad=22)

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
            ax.legend(
                ncol=2,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.05),
                frameon=True,
                fancybox=True,
                framealpha=0.95,
                fontsize=legend_fontsize,
                columnspacing=1.4,
                handlelength=1.6,
                borderpad=0.6,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.grid(axis="x", visible=False)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            # plt.tight_layout()
            out_path = os.path.join(
                out_dir,
                f"diff_ratio_by_augmentation_{sanitize_for_path(dset)}_{model_name}.png",
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {out_path}")
            saved_paths.append(out_path)
            plt.close(fig)

        grid_out = os.path.join(out_dir, f"diff_ratio_by_augmentation_all_datasets_grid_{model_name}.png")
        create_image_grid(saved_paths, grid_out, cols=max(1, int(grid_cols)), scale=float(grid_scale))

    avg_acc = aggregate_avg_accuracy_across_datasets(zero_shot_aug_dic)
    avg_diff = aggregate_avg_diff_ratio_across_datasets(zero_shot_aug_dic)

    plot_avg_accuracy_by_augmentation(
        avg_acc,
        out_dir=args.out_dir,
        model_name=model_name,
    )

    # --- NEW: per-dataset plots + grid ---
    acc_by_dataset = aggregate_accuracy_by_dataset_and_augmentation(zero_shot_aug_dic)
    plot_accuracy_by_augmentation_per_dataset(
        acc_by_dataset,
        out_dir=args.out_dir,
        model_name=model_name,
        grid_cols=args.dataset_grid_cols,
        grid_scale=args.dataset_grid_scale,
    )

    plot_avg_diff_ratio_by_augmentation(
        avg_diff,
        out_dir=args.out_dir,
        model_name=model_name,
    )

    # --- NEW: per-dataset diff-ratio plots + grid ---
    diff_by_dataset = aggregate_diff_ratio_by_dataset_and_augmentation(zero_shot_aug_dic)
    plot_diff_ratio_by_augmentation_per_dataset(
        diff_by_dataset,
        out_dir=args.out_dir,
        model_name=model_name,
        grid_cols=args.dataset_grid_cols,
        grid_scale=args.dataset_grid_scale,
    )

    def plot_accuracy_vs_diff_ratio_threshold(avg_acc_thr, thresholds, out_dir, model_name):
        """Bar plots: accuracy under per-sample diff-ratio routing across thresholds.

        NOTE on which diff ratio is used:
          - For CLEAN plots we use `diff_ratio_per_sample_clean[aug][dataset]`
          - For ADV plots we use `diff_ratio_per_sample_adv[aug][dataset]`

        These per-sample diff ratios are loaded from `results_images_diff_ratio_tpt_noisy_anchors.json`
        under each augmentation pool folder.
        """
        os.makedirs(out_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")

        augmentations = list(avg_acc_thr.keys())
        thr_vals = [float(t) for t in thresholds]
        x = np.arange(len(thr_vals))

        colors = {
            "clean_vanilla_thr": "#4C72B0",
            "clean_weighted_thr": "#2F5597",
            "adv_vanilla_thr": "#DD8452",
            "adv_weighted_thr": "#C44E52",
        }

        settings = [
            ("clean_vanilla_thr", "Clean – Vanilla (routed)"),
            ("clean_weighted_thr", "Clean – Weighted (routed)"),
            ("adv_vanilla_thr", "Adv – Vanilla (routed)"),
            ("adv_weighted_thr", "Adv – Weighted (routed)"),
        ]

        def _add_value_labels(ax, bars, *, fmt: str = "{:.1f}", fontsize: int = 7):
            for b in bars:
                h = b.get_height()
                if not np.isfinite(h):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.15,
                    fmt.format(h),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    color="#222222",
                    rotation=90,
                    clip_on=False,
                )

        for aug in augmentations:
            # Wider figure since threshold list can be long.
            fig, ax = plt.subplots(figsize=(max(12.0, 0.65 * len(thr_vals)), 6.4))

            width = 0.18

            # Baselines (horizontal lines)
            clean_orig = float(avg_acc_thr[aug]["clean_original"]["_mean"])
            adv_orig = float(avg_acc_thr[aug]["adv_original"]["_mean"])
            ax.axhline(
                clean_orig,
                color="#9AB6DF",
                linestyle="--",
                linewidth=1.2,
                label="Clean – Original (baseline)",
                zorder=1,
            )
            ax.axhline(
                adv_orig,
                color="#F0B38C",
                linestyle="--",
                linewidth=1.2,
                label="Adv – Original (baseline)",
                zorder=1,
            )

            for i, (key, label) in enumerate(settings):
                y = [float(avg_acc_thr[aug][key][t]) for t in thr_vals]
                bars = ax.bar(
                    x + (i - 1.5) * width,
                    y,
                    width=width,
                    color=colors[key],
                    edgecolor="white",
                    linewidth=0.6,
                    label=label,
                    zorder=2,
                )
                # Labeling every bar can be too dense with many thresholds.
                if len(thr_vals) <= 12:
                    _add_value_labels(ax, bars)

            ax.set_xlabel(
                "Diff-ratio threshold (per-sample routing)",
                fontsize=12,
            )
            ax.set_ylabel("Average accuracy across datasets (%)", fontsize=12)
            ax.set_title(
                "Threshold routing using per-sample diff ratios\n"
                f"Augmentation pool: {aug} | Model: {model_name}\n"
                "Diff-ratio source: results_images_diff_ratio_tpt_noisy_anchors.json",
                fontsize=12,
                pad=14,
            )

            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:g}" for t in thr_vals], rotation=0)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.grid(axis="x", visible=False)
            ax.legend(ncol=2, fontsize=9, frameon=True)
            ax.margins(x=0.01)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            plt.tight_layout()
            out_path = os.path.join(
                out_dir,
                f"accuracy_vs_diff_ratio_threshold_bar_{sanitize_for_path(aug)}_{model_name}.png",
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {out_path}")
            if args.show:
                plt.show()
            plt.close(fig)

    # --- NEW: threshold routing plots for vanilla/weighted using per-sample diff ratios ---
    thresholds = [float(x.strip()) for x in args.diff_ratio_thresholds.split(",") if x.strip()]
    avg_acc_thr = aggregate_thresholded_accuracy_across_datasets(zero_shot_aug_dic, thresholds)
    plot_accuracy_vs_diff_ratio_threshold(avg_acc_thr, thresholds, out_dir=args.out_dir, model_name=model_name)
