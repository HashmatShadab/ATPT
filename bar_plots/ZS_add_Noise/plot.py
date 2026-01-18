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

def plot_group_lines(curves, *, title="", save_path=None):
    """
    curves = output of get_group_curves(...)
    """
    if not curves:
        print("No curves to plot.")
        return

    # assume all share same baseline (they do for a fixed setting)
    any_key = next(iter(curves))
    baseline = curves[any_key]["baseline"]

    # find global best across diff_key and thresholds
    global_best = None  # (best_acc, diff_key, best_t)
    for diff_key, c in curves.items():
        if c["best_acc"] is None:
            continue
        cand = (c["best_acc"], diff_key, c["best_t"])
        if global_best is None or cand[0] > global_best[0]:
            global_best = cand

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    for diff_key, c in sorted(curves.items(), key=lambda x: x[0]):
        ts = c["thresholds"]
        ys = c["accs"]
        if len(ts) == 0:
            continue
        ax.plot(ts, ys, marker="o", linewidth=1, label=f"diff={diff_key}")

        # mark per-line best
        ax.scatter([c["best_t"]], [c["best_acc"]], s=40)

    # baseline
    ax.axhline(baseline, linestyle="--", linewidth=1)
    ax.text(0.01, baseline, f" baseline={baseline:.2f}", va="bottom")

    # global best annotation
    if global_best is not None:
        best_acc, best_diff, best_t = global_best
        ax.scatter([best_t], [best_acc], s=120, marker="*", zorder=5)
        ax.set_title(f"{title}\nBEST: diff={best_diff}, t={best_t:.2f}, acc={best_acc:.2f}")
    else:
        ax.set_title(title)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_group_heatmap(curves, *, title="", save_path=None):
    if not curves:
        print("No curves to plot.")
        return

    # unify threshold axis (your dict uses same thresholds everywhere)
    diff_keys = sorted(curves.keys(), key=lambda x: float(x) if x.replace('.','',1).isdigit() else x)
    thresholds = curves[diff_keys[0]]["thresholds"]

    mat = np.full((len(diff_keys), len(thresholds)), np.nan, dtype=float)

    # fill
    for i, diff_key in enumerate(diff_keys):
        c = curves[diff_key]
        for j, acc in enumerate(c["accs"]):
            mat[i, j] = acc

    # global best
    best_idx = np.nanargmax(mat)
    bi, bj = np.unravel_index(best_idx, mat.shape)
    best_diff = diff_keys[bi]
    best_t = thresholds[bj]
    best_acc = mat[bi, bj]

    fig, ax = plt.subplots(figsize=(1.2 * len(thresholds) + 3, 0.6 * len(diff_keys) + 2), dpi=150)
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(diff_keys)))
    ax.set_yticklabels(diff_keys)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Diff-ratio source (diff_ratio_key)")
    ax.set_title(f"{title}\nBEST: diff={best_diff}, t={best_t:.2f}, acc={best_acc:.2f}")

    # annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)

    # mark best cell
    ax.scatter([bj], [bi], s=120, marker="*", zorder=5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

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

def plot_avg_diff_ratio_bar(
    avg_diff_dict,
    *,
    title="",
    ylabel="Average Diff Ratio",
    save_path=None,
):
    """
    avg_diff_dict: dict[param] -> float
    """
    if not avg_diff_dict:
        print("No data to plot.")
        return

    # sort params numerically if possible
    def _sort_key(x):
        try:
            return float(x)
        except Exception:
            return x

    params = sorted(avg_diff_dict.keys(), key=_sort_key)
    values = [avg_diff_dict[p] for p in params]

    x = np.arange(len(params))

    fig, ax = plt.subplots(figsize=(6 + len(params), 4), dpi=150)

    bars = ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_xlabel("Param (eps / sigma)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # annotate values
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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

PARAMS = ["4", "8", "12"]

method = METHODS[0]

for case in CASES:
    # Create directory for method/case
    case_dir = os.path.join(OUTPUT_DIR, method, case)
    os.makedirs(case_dir, exist_ok=True)

    for model in MODELS:
        for dataset in DATASETS:
            # save bar plot once per dataset/case/model
            avg_diff = get_avg_diff_ratio_per_param(
                ACC_RESULTS_LOADED,
                method=method,
                case=case,
                model=model,
                dataset=dataset,
            )
            bar_title = f"Avg Diff Ratio vs Param\n{method} | {case} | {model} | {dataset}"
            bar_save_name = f"bar_{model}_{dataset}.png"
            plot_avg_diff_ratio_bar(
                avg_diff,
                title=bar_title,
                save_path=os.path.join(case_dir, bar_save_name)
            )

            for param in PARAMS:
                curves = get_group_curves(ACC_RESULTS_LOADED, method, case, model, dataset, param)
                title = f"{method} | {case} | {model} | {dataset} | pred_param={param}"
                
                line_save_name = f"lines_{model}_{dataset}_param{param}.png"
                plot_group_lines(curves, title=title, save_path=os.path.join(case_dir, line_save_name))
                
                # heatmap_save_name = f"heatmap_{model}_{dataset}_param{param}.png"
                # plot_group_heatmap(curves, title=title, save_path=os.path.join(case_dir, heatmap_save_name))
