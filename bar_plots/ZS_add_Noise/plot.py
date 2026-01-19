import json
import os

ACC_RESULTS_PATH = "acc_results.json"
OUTPUT_DIR = "plots_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ACC_RESULTS_PATH, "r", encoding="utf-8") as f:
    ACC_RESULTS_LOADED = json.load(f)

print("Loaded keys:", list(ACC_RESULTS_LOADED.keys())[:5])

def parse_threshold_key(k: str):
    # k like "single_diff_ratio_0.40"
    try:
        return float(k.split("_")[-1])
    except Exception:
        return None

def flatten_acc_results(acc_results):
    rows = []
    for method, cases in acc_results.items():
        for case, models in cases.items():
            for model, datasets in models.items():
                for dataset, params in datasets.items():
                    for param, diff_sources in params.items():
                        for diff_src, acc_dict in diff_sources.items():
                            if not isinstance(acc_dict, dict):
                                continue
                            if "single" not in acc_dict:
                                continue

                            baseline = float(acc_dict["single"])

                            # collect thresholded accuracies
                            t_acc = []
                            for k, v in acc_dict.items():
                                if k.startswith("single_diff_ratio_"):
                                    t = parse_threshold_key(k)
                                    if t is None:
                                        continue
                                    t_acc.append((t, float(v)))

                            if len(t_acc) == 0:
                                best_acc = None
                                best_t = None
                                gain = None
                            else:
                                best_t, best_acc = max(t_acc, key=lambda x: x[1])
                                gain = best_acc - baseline

                            diff_avg = acc_dict.get("diff_ratio_avg", None)
                            if diff_avg is not None:
                                diff_avg = float(diff_avg)

                            rows.append({
                                "method": method,
                                "case": case,
                                "model": model,
                                "dataset": dataset,
                                "param": str(param),
                                "diff_src": str(diff_src),
                                "baseline": baseline,
                                "best_acc": best_acc,
                                "best_t": best_t,
                                "gain": gain,
                                "diff_ratio_avg": diff_avg,
                                "acc_dict": acc_dict,  # keep raw for curve plots
                            })
    return rows

ROWS = flatten_acc_results(ACC_RESULTS_LOADED)
print("Num rows:", len(ROWS))
print("Example row:", ROWS[0].keys())

import numpy as np
import matplotlib.pyplot as plt

def extract_thresholds(acc_dict):
    """Return sorted thresholds present in acc_dict."""
    ts = []
    for k in acc_dict.keys():
        if k.startswith("single_diff_ratio_"):
            try:
                ts.append(float(k.split("_")[-1]))
            except Exception:
                pass
    return sorted(ts)

def curve_from_acc_dict(acc_dict):
    """
    Returns:
      baseline: float
      thresholds: list[float]
      accs: list[float] aligned with thresholds
      best_t, best_acc
    """
    baseline = float(acc_dict["single"])
    thresholds = extract_thresholds(acc_dict)
    accs = [float(acc_dict[f"single_diff_ratio_{t:.2f}"]) for t in thresholds]
    best_idx = int(np.argmax(accs)) if len(accs) else None
    best_t = thresholds[best_idx] if best_idx is not None else None
    best_acc = accs[best_idx] if best_idx is not None else None
    return baseline, thresholds, accs, best_t, best_acc

def get_group_curves(ACC_RESULTS, method, case, model, dataset, param):
    """
    Returns a dict:
      curves[diff_ratio_key] = {
        "baseline": float,
        "thresholds": [...],
        "accs": [...],
        "best_t": float,
        "best_acc": float,
        "diff_ratio_avg": float or None
      }
    """
    curves = {}
    block = ACC_RESULTS[method][case][model][dataset][str(param)]

    for diff_key, acc_dict in block.items():
        # skip empty or malformed entries
        if not isinstance(acc_dict, dict) or "single" not in acc_dict:
            continue

        baseline, thresholds, accs, best_t, best_acc = curve_from_acc_dict(acc_dict)
        diff_avg = acc_dict.get("diff_ratio_avg", None)
        if diff_avg is not None:
            diff_avg = float(diff_avg)

        curves[str(diff_key)] = {
            "baseline": baseline,
            "thresholds": thresholds,
            "accs": accs,
            "best_t": best_t,
            "best_acc": best_acc,
            "diff_ratio_avg": diff_avg,
        }
    return curves

def get_avg_diff_ratio_per_param(
    ACC_RESULTS,
    *,
    method,
    case,
    model,
    dataset,
):
    """
    Returns:
      dict[param] = avg_diff_ratio (float)
    """
    out = {}

    block = ACC_RESULTS[method][case][model][dataset]

    for param, diff_sources in block.items():
        # we want diff_ratio_key == param
        if param not in diff_sources:
            continue

        acc_dict = diff_sources[param]
        if "diff_ratio_avg" not in acc_dict:
            continue

        out[param] = float(acc_dict["diff_ratio_avg"])

    return out

import matplotlib.pyplot as plt
import numpy as np

def plot_grid_bar(all_avg_diffs, *, title="", save_path=None):
    """
    all_avg_diffs: list of (dataset_name, avg_diff_dict)
    """
    n = len(all_avg_diffs)
    if n == 0:
        return
    
    cols = (n + 1) // 2
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=150, squeeze=False)
    
    def _sort_key(x):
        try:
            return float(x)
        except Exception:
            return x

    for i, (dataset, avg_diff_dict) in enumerate(all_avg_diffs):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        
        params = sorted(avg_diff_dict.keys(), key=_sort_key)
        values = [avg_diff_dict[p] for p in params]
        x_pos = np.arange(len(params))
        
        bars = ax.bar(x_pos, values)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(params)
        ax.set_title(dataset)
        ax.set_ylabel("Avg Diff Ratio")
        
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_grid_bar_comparison(all_avg_diffs_cases, cases, *, title="", save_path=None):
    """
    all_avg_diffs_cases: dict[case] -> list of (dataset_name, avg_diff_dict)
    cases: list of case names to compare
    """
    # Assuming all cases have the same datasets
    first_case = cases[0]
    datasets_data = all_avg_diffs_cases[first_case]
    n = len(datasets_data)
    if n == 0:
        return
    
    cols = (n + 1) // 2
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), dpi=150, squeeze=False)
    
    def _sort_key(x):
        try:
            return float(x)
        except Exception:
            return x

    for i, (dataset, _) in enumerate(datasets_data):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        
        # Collect params from all cases to be safe, though they should be same
        all_params = set()
        for case in cases:
            avg_diff_dict = dict(all_avg_diffs_cases[case])[dataset]
            all_params.update(avg_diff_dict.keys())
        
        params = sorted(list(all_params), key=_sort_key)
        x_pos = np.arange(len(params))
        width = 0.8 / len(cases)
        
        for j, case in enumerate(cases):
            avg_diff_dict = dict(all_avg_diffs_cases[case])[dataset]
            values = [avg_diff_dict.get(p, 0.0) for p in params]
            offset = (j - (len(cases) - 1) / 2) * width
            bars = ax.bar(x_pos + offset, values, width, label=case)
            
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=6)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(params)
        ax.set_title(dataset)
        ax.set_ylabel("Avg Diff Ratio")
        ax.set_xlabel("Diff Ratio Param")
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_grid_lines(all_curves, *, title="", save_path=None):
    """
    all_curves: list of (dataset_name, param, curves_dict)
    Ordered by dataset (rows) and then param (cols).
    """
    if not all_curves:
        return
    
    datasets = []
    params = []
    for d, p, _ in all_curves:
        if d not in datasets: datasets.append(d)
        if p not in params: params.append(p)
    
    num_datasets = len(datasets)
    num_params = len(params)
    
    # Each row will have plots for two datasets.
    # Each dataset has 'num_params' plots.
    # So total columns = 2 * num_params
    # Total rows = ceil(num_datasets / 2)
    
    datasets_per_row = 2
    rows = (num_datasets + datasets_per_row - 1) // datasets_per_row
    cols = datasets_per_row * num_params
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150, squeeze=False)
    
    for i, (dataset, param, curves) in enumerate(all_curves):
        ds_idx = datasets.index(dataset)
        p_idx = params.index(param)
        
        # Calculate grid position
        # ds_row: which pair of datasets (row index)
        # ds_col_offset: 0 for the first dataset in the pair, 1 for the second
        ds_row, ds_col_offset = divmod(ds_idx, datasets_per_row)
        
        r = ds_row
        c = ds_col_offset * num_params + p_idx
        
        ax = axes[r, c]
        
        if not curves:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            continue
            
        any_key = next(iter(curves))
        baseline = curves[any_key]["baseline"]
        
        global_best = None
        for diff_key, curve_data in curves.items():
            if curve_data["best_acc"] is not None:
                cand = (curve_data["best_acc"], diff_key, curve_data["best_t"])
                if global_best is None or cand[0] > global_best[0]:
                    global_best = cand
        
        for diff_key, curve_data in sorted(curves.items(), key=lambda x: x[0]):
            ts = curve_data["thresholds"]
            ys = curve_data["accs"]
            if len(ts) == 0: continue
            ax.plot(ts, ys, marker="o", linewidth=1, markersize=3, label=f"diff={diff_key}")
            # ax.scatter([curve_data["best_t"]], [curve_data["best_acc"]], s=20) # replaced by star below
            
        row_title = f"{dataset} | param={param}"
        if global_best:
            best_acc, best_diff, best_t = global_best
            ax.scatter([best_t], [best_acc], color="red", marker="*", s=100, zorder=5, label="Best")
            ax.set_title(f"{row_title}\nBest: d={best_diff}, t={best_t:.2f}, acc={best_acc:.2f}", fontsize=9)
        else:
            ax.set_title(row_title, fontsize=9)
            
        ax.axhline(baseline, linestyle="--", linewidth=1, color="black", label="Baseline")
        
        # Add legend to the first plot of each dataset
        if p_idx == 0:
            ax.legend(fontsize=7, loc="lower right")
            
        ax.grid(True, linestyle="--", alpha=0.4)
        if r == rows - 1: ax.set_xlabel("Threshold")
        
        # In a multi-dataset-per-row layout, we show ylabel for the first plot of each dataset's group
        if p_idx == 0: ax.set_ylabel("Accuracy")
        
    # hide unused axes
    for i in range(num_datasets, rows * datasets_per_row):
        ds_row, ds_col_offset = divmod(i, datasets_per_row)
        for p_idx in range(num_params):
            r = ds_row
            c = ds_col_offset * num_params + p_idx
            axes[r, c].axis("off")
            
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_grid_baseline(all_baselines, *, title="", save_path=None):
    """
    all_baselines: list of (dataset_name, baseline_dict)
    baseline_dict: dict[param] -> baseline_acc
    """
    n = len(all_baselines)
    if n == 0:
        return

    datasets_per_row = 2
    rows = (n + datasets_per_row - 1) // datasets_per_row
    cols = datasets_per_row

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=150, squeeze=False)

    def _sort_key(x):
        try:
            return float(x)
        except Exception:
            return x

    for i, (dataset, baseline_dict) in enumerate(all_baselines):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        params = sorted(baseline_dict.keys(), key=_sort_key)
        values = [baseline_dict[p] for p in params]
        x_pos = np.arange(len(params))

        ax.plot(x_pos, values, marker="o", linestyle="-", linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(params)
        ax.set_title(dataset)
        ax.set_ylabel("Baseline Accuracy")
        ax.set_xlabel("Param")

        for x, y in zip(x_pos, values):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_grid_best_param_lines(all_best_curves, *, title="", save_path=None):
    """
    all_best_curves: list of (dataset_name, best_param, curves_dict)
    One plot per dataset.
    """
    n = len(all_best_curves)
    if n == 0:
        return

    datasets_per_row = 4
    rows = (n + datasets_per_row - 1) // datasets_per_row
    cols = datasets_per_row

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), dpi=150, squeeze=False)

    for i, (dataset, best_param, curves) in enumerate(all_best_curves):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        if not curves:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            continue

        any_key = next(iter(curves))
        baseline = curves[any_key]["baseline"]

        global_best = None
        for diff_key, curve_data in curves.items():
            if curve_data["best_acc"] is not None:
                cand = (curve_data["best_acc"], diff_key, curve_data["best_t"])
                if global_best is None or cand[0] > global_best[0]:
                    global_best = cand

        for diff_key, curve_data in sorted(curves.items(), key=lambda x: x[0]):
            ts = curve_data["thresholds"]
            ys = curve_data["accs"]
            if len(ts) == 0: continue
            ax.plot(ts, ys, marker="o", linewidth=1, markersize=3, label=f"diff={diff_key}")

        ax.axhline(baseline, linestyle="--", linewidth=1, color="black", label="Baseline")

        title_str = f"{dataset} (Best Param: {best_param})"
        if global_best:
            best_acc, best_diff, best_t = global_best
            ax.scatter([best_t], [best_acc], color="red", marker="*", s=100, zorder=5, label="Best")
            ax.set_title(f"{title_str}\nBest: d={best_diff}, t={best_t:.2f}, acc={best_acc:.2f}", fontsize=10)
        else:
            ax.set_title(title_str, fontsize=10)

        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Accuracy")

    # hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_single_bar(avg_diff_dict, *, title="", save_path=None):
    """
    avg_diff_dict: dict[param] -> value
    """
    plt.figure(figsize=(8, 6), dpi=150)
    
    def _sort_key(x):
        try:
            return float(x)
        except Exception:
            return x

    params = sorted(avg_diff_dict.keys(), key=_sort_key)
    values = [avg_diff_dict[p] for p in params]
    x_pos = np.arange(len(params))
    
    bars = plt.bar(x_pos, values)
    plt.xticks(x_pos, params)
    plt.title(title)
    plt.ylabel("Avg Diff Ratio")
    plt.xlabel("Param")
    
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_single_baseline(baseline_dict, *, title="", save_path=None):
    """
    baseline_dict: dict[param] -> baseline_acc
    """
    plt.figure(figsize=(8, 6), dpi=150)
    
    def _sort_key(x):
        try:
            return float(x)
        except Exception:
            return x

    params = sorted(baseline_dict.keys(), key=_sort_key)
    values = [baseline_dict[p] for p in params]
    x_pos = np.arange(len(params))
    
    plt.plot(x_pos, values, marker="o", linestyle="-", linewidth=2)
    plt.xticks(x_pos, params)
    plt.title(title)
    plt.ylabel("Baseline Accuracy")
    plt.xlabel("Param")
    
    for x, y in zip(x_pos, values):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_single_curves(curves, *, title="", save_path=None):
    """
    curves: curves_dict for one group
    """
    plt.figure(figsize=(8, 6), dpi=150)
    
    if not curves:
        plt.text(0.5, 0.5, "No Data", ha="center", va="center")
    else:
        any_key = next(iter(curves))
        baseline = curves[any_key]["baseline"]
        
        global_best = None
        for diff_key, curve_data in curves.items():
            if curve_data["best_acc"] is not None:
                cand = (curve_data["best_acc"], diff_key, curve_data["best_t"])
                if global_best is None or cand[0] > global_best[0]:
                    global_best = cand
        
        for diff_key, curve_data in sorted(curves.items(), key=lambda x: x[0]):
            ts = curve_data["thresholds"]
            ys = curve_data["accs"]
            if len(ts) == 0: continue
            plt.plot(ts, ys, marker="o", linewidth=1, markersize=4, label=f"diff={diff_key}")
            
        if global_best:
            best_acc, best_diff, best_t = global_best
            plt.scatter([best_t], [best_acc], color="red", marker="*", s=150, zorder=5, label="Best")
            plt.title(f"{title}\nBest: d={best_diff}, t={best_t:.2f}, acc={best_acc:.2f}")
        else:
            plt.title(title)
            
        plt.axhline(baseline, linestyle="--", linewidth=1, color="black", label="Baseline")
        plt.legend(fontsize=9, loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_single_bar_comparison(all_avg_diffs_cases, cases, *, title="", save_path=None):
    """
    all_avg_diffs_cases: dict[case] -> avg_diff_dict {param: value}
    cases: list of case names to compare
    """
    plt.figure(figsize=(10, 6), dpi=150)
    
    def _sort_key(x):
        try:
            return float(x)
        except Exception:
            return x

    # Collect all params
    all_params = set()
    for case in cases:
        all_params.update(all_avg_diffs_cases[case].keys())
    
    params = sorted(list(all_params), key=_sort_key)
    x_pos = np.arange(len(params))
    width = 0.8 / len(cases)
    
    for j, case in enumerate(cases):
        avg_diff_dict = all_avg_diffs_cases[case]
        values = [avg_diff_dict.get(p, 0.0) for p in params]
        offset = (j - (len(cases) - 1) / 2) * width
        bars = plt.bar(x_pos + offset, values, width, label=case)
        
        for b in bars:
            h = b.get_height()
            if h > 0:
                plt.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    
    plt.xticks(x_pos, params)
    plt.title(title)
    plt.ylabel("Avg Diff Ratio")
    plt.xlabel("Diff Ratio Param")
    plt.legend(fontsize=9)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def summarize_best(curves):
    """
    Returns a dict summary for the group.
    """
    if not curves:
        return None

    any_key = next(iter(curves))
    baseline = curves[any_key]["baseline"]

    best = None  # (best_acc, diff_key, best_t)
    for diff_key, c in curves.items():
        if c["best_acc"] is None:
            continue
        cand = (c["best_acc"], diff_key, c["best_t"])
        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        return None

    best_acc, best_diff, best_t = best
    return {
        "baseline": baseline,
        "best_acc": best_acc,
        "best_gain": best_acc - baseline,
        "best_diff_ratio_key": best_diff,
        "best_threshold": best_t,
    }


MODELS = [
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

CASES = ["clean", "adversarial_eps4_steps100", "adversarial_eps4_steps100_image_only"]

METHODS = ["zero_shot_uniform_single", "zero_shot_uniform_anchors", "zero_shot_gaussian_anchors"]

PARAMS_1 = ["03", "06", "12", "18"]
PARAMS_2 = ["4", "8", "12", "16"]

method = METHODS[2]

for model in MODELS:
    all_avg_diffs_cases = {}
    global_avg_diffs_cases = {}

    for case in CASES:
        # Create directory for method/case
        case_dir = os.path.join(OUTPUT_DIR, method, case)
        os.makedirs(case_dir, exist_ok=True)

        all_avg_diffs = []
        all_curves_data = []
        all_baselines = []
        all_best_curves_data = []

        for dataset in DATASETS:
            # gather avg diff ratio for grid bar plot
            avg_diff = get_avg_diff_ratio_per_param(
                ACC_RESULTS_LOADED,
                method=method,
                case=case,
                model=model,
                dataset=dataset,
            )
            all_avg_diffs.append((dataset, avg_diff))

            dataset_baselines = {}
            dataset_best_param = None
            dataset_best_acc = -1
            dataset_best_curves = None

            if method == "zeros_shot_gaussian_anchors":
                PARAMS = PARAMS_2
            else:
                PARAMS = PARAMS_1

            for param in PARAMS:
                curves = get_group_curves(ACC_RESULTS_LOADED, method, case, model, dataset, param)
                all_curves_data.append((dataset, param, curves))

                if curves:
                    any_key = next(iter(curves))
                    dataset_baselines[param] = curves[any_key]["baseline"]
                    
                    # Find best param for this dataset
                    summary = summarize_best(curves)
                    if summary and summary["best_acc"] > dataset_best_acc:
                        dataset_best_acc = summary["best_acc"]
                        dataset_best_param = param
                        dataset_best_curves = curves
            
            if dataset_baselines:
                all_baselines.append((dataset, dataset_baselines))
            
            if dataset_best_param is not None:
                all_best_curves_data.append((dataset, dataset_best_param, dataset_best_curves))

        all_avg_diffs_cases[case] = all_avg_diffs

        # --- Aggregate across datasets ---
        # 1. Global Avg Diff Ratio
        global_avg_diff = {}
        param_counts = {}
        for _, ds_diffs in all_avg_diffs:
            for p, val in ds_diffs.items():
                global_avg_diff[p] = global_avg_diff.get(p, 0.0) + val
                param_counts[p] = param_counts.get(p, 0) + 1
        for p in global_avg_diff:
            global_avg_diff[p] /= param_counts[p]
        
        global_avg_diffs_cases[case] = global_avg_diff

        # 2. Global Baseline Accuracy
        global_baselines = {}
        baseline_counts = {}
        for _, ds_baselines in all_baselines:
            for p, val in ds_baselines.items():
                global_baselines[p] = global_baselines.get(p, 0.0) + val
                baseline_counts[p] = baseline_counts.get(p, 0) + 1
        for p in global_baselines:
            global_baselines[p] /= baseline_counts[p]

        # 3. Global Curves per Param
        global_curves_all_params = {}
        for param in PARAMS:
            # Collect all curves for this param across datasets
            param_curves = [c for ds, p, c in all_curves_data if p == param]
            if not param_curves: continue
            
            # curves is dict[diff_src] -> {baseline, thresholds, accs, ...}
            aggregated_curves = {}
            diff_srcs = set()
            for curves in param_curves:
                diff_srcs.update(curves.keys())
            
            for dsrc in diff_srcs:
                # Find all curves for this dsrc
                dsrc_curves = [c[dsrc] for c in param_curves if dsrc in c]
                if not dsrc_curves: continue
                
                # Assume all have same thresholds for simplicity (usually true)
                thresholds = dsrc_curves[0]["thresholds"]
                
                avg_accs = []
                for i in range(len(thresholds)):
                    acc_sum = sum(c["accs"][i] for c in dsrc_curves)
                    avg_accs.append(acc_sum / len(dsrc_curves))
                
                avg_baseline = sum(c["baseline"] for c in dsrc_curves) / len(dsrc_curves)
                
                best_idx = int(np.argmax(avg_accs))
                aggregated_curves[dsrc] = {
                    "baseline": avg_baseline,
                    "thresholds": thresholds,
                    "accs": avg_accs,
                    "best_t": thresholds[best_idx],
                    "best_acc": avg_accs[best_idx],
                }
            global_curves_all_params[param] = aggregated_curves

        # --- Plot Global Averages ---
        global_dir = os.path.join(case_dir, "global_average")
        os.makedirs(global_dir, exist_ok=True)

        if global_avg_diff:
            plot_single_bar(global_avg_diff, title=f"Global Avg Diff Ratio | {method} | {case}", 
                            save_path=os.path.join(global_dir, "global_avg_diff.png"))
        
        if global_baselines:
            plot_single_baseline(global_baselines, title=f"Global Baseline Accuracy | {method} | {case}",
                                 save_path=os.path.join(global_dir, "global_baseline.png"))
        
        for param, curves in global_curves_all_params.items():
            plot_single_curves(curves, title=f"Global Avg Accuracy Curves | {method} | {case} | param={param}",
                               save_path=os.path.join(global_dir, f"global_curves_param_{param}.png"))

        # 1. Grid of bar plots (2 rows)
        bar_grid_title = f"Avg Diff Ratio Grid | {method} | {case} | {model}"
        bar_grid_save = os.path.join(case_dir, f"grid_bar_{model}.png")
        plot_grid_bar(all_avg_diffs, title=bar_grid_title, save_path=bar_grid_save)

        # 2. Grid of line curves
        line_grid_title = f"Accuracy Curves Grid | {method} | {case} | {model}"
        line_grid_save = os.path.join(case_dir, f"grid_lines_{model}.png")
        plot_grid_lines(all_curves_data, title=line_grid_title, save_path=line_grid_save)

        # 3. Grid of baseline accuracy
        baseline_grid_title = f"Baseline Accuracy vs Param | {method} | {case} | {model}"
        baseline_grid_save = os.path.join(case_dir, f"grid_baseline_{model}.png")
        plot_grid_baseline(all_baselines, title=baseline_grid_title, save_path=baseline_grid_save)

        # 4. Grid of best param line curves
        best_line_grid_title = f"Best Param Accuracy Curves Grid | {method} | {case} | {model}"
        best_line_grid_save = os.path.join(case_dir, f"grid_best_param_lines_{model}.png")
        plot_grid_best_param_lines(all_best_curves_data, title=best_line_grid_title, save_path=best_line_grid_save)

    # After all cases, plot the comparison bar grid
    comparison_dir = os.path.join(OUTPUT_DIR, method, "diff_ratio_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    comp_bar_grid_title = f"Avg Diff Ratio Comparison Grid | {method} | {model}"
    comp_bar_grid_save = os.path.join(comparison_dir, f"grid_bar_comparison_{model}.png")
    plot_grid_bar_comparison(all_avg_diffs_cases, CASES, title=comp_bar_grid_title, save_path=comp_bar_grid_save)

    # Plot global average comparison
    global_comp_dir = os.path.join(OUTPUT_DIR, method, "diff_ratio_global_average_comparison")
    os.makedirs(global_comp_dir, exist_ok=True)
    global_comp_title = f"Global Avg Diff Ratio Comparison | {method} | {model}"
    global_comp_save = os.path.join(global_comp_dir, f"global_bar_comparison_{model}.png")
    plot_single_bar_comparison(global_avg_diffs_cases, CASES, title=global_comp_title, save_path=global_comp_save)
