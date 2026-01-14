import re
import glob
import os
import json
from pathlib import Path
import pandas as pd
# -----------------------
# CONFIG
# -----------------------
ROOT = "Final_Results_corrected_ca_tau"
MODEL = "**"
LOG_GLOB = os.path.join(ROOT, MODEL, "**", "*ratio.json")


def find_root_model_indices(parts):
    """
    Returns (root_idx, model_idx) such that:
      parts[root_idx] == ROOT and model_idx = root_idx + 1
    Works on absolute/relative paths.
    """
    # Try exact match
    for i in range(len(parts) - 1):
        if parts[i] == ROOT or parts[i].endswith(os.sep + ROOT) or parts[i].endswith("/" + ROOT):
            return i, i + 1

    raise ValueError(f"Could not locate ROOT in path parts: {parts}")


def parse_path_metadata_and_groupkey(log_path: str):
    """
    Expected:
      ROOT/MODEL/DATASET/IMAGE_TYPE/REST.../log_*log

    We extract:
      dataset: parts[model_idx + 1]
      image_type: parts[model_idx + 2]
      group_key: join(parts[model_idx + 2:])  (i.e., IMAGE_TYPE/REST.../logfile)
                => everything AFTER dataset is identical across datasets
    """
    p = Path(log_path)
    parts = p.parts

    root_idx, model_idx = find_root_model_indices(parts)

    # dataset is next after MODEL
    if parts[model_idx] == "ViT-L":
        dataset = parts[model_idx + 2] if model_idx + 2 < len(parts) else "UNKNOWN_DATASET"
        # image_type is next after dataset
        image_type = parts[model_idx + 3] if model_idx + 3 < len(parts) else "UNKNOWN_IMAGE_TYPE"
        # group key = everything after dataset (starts at IMAGE_TYPE)
        start = model_idx + 3
    else:
        dataset = parts[model_idx + 1] if model_idx + 1 < len(parts) else "UNKNOWN_DATASET"
        # image_type is next after dataset
        image_type = parts[model_idx + 2] if model_idx + 2 < len(parts) else "UNKNOWN_IMAGE_TYPE"
        # group key = everything after dataset (starts at IMAGE_TYPE)
        start = model_idx + 2

    group_key = "/".join(parts[start:-1]) if start < len(parts) else os.path.basename(log_path)

    return dataset, image_type, group_key, parts[model_idx]


def parse_json_file(log_path: str):
    dataset, image_type, group_key, model_name = parse_path_metadata_and_groupkey(log_path)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed reading {log_path}: {e}")
        return []

    avg_diff_ratio = data.get("avergae_diff_ratio")
    if avg_diff_ratio is None:
        print(f"[WARN] No 'avergae_diff_ratio' in {log_path}")
        return []

    rows = [{
        "method": "Counter Attack",
        "avg_diff_ratio": avg_diff_ratio,
        "model": model_name,
        "dataset": dataset,
        "image_type": image_type,
        "group_key": group_key,
        "log_file": os.path.basename(log_path),
        "log_path": log_path,
    }]

    return rows


def main():
    log_files = glob.glob(LOG_GLOB, recursive=True)
    if not log_files:
        raise SystemExit(f"No log files found: {LOG_GLOB}")

    all_rows = []
    for lp in log_files:
        all_rows.extend(parse_json_file(lp))

    if not all_rows:
        raise SystemExit("Parsed 0 entries from JSON files.")

    df = pd.DataFrame(all_rows)

    # We want a JSON structure:
    # {
    #   "model|image_type|group_key": [
    #      { "dataset": "...", "method": "...", "avg_diff_ratio": ... },
    #      ...
    #   ]
    # }

    grouped_json = {}

    # Sort for consistent output
    sort_cols = ["model", "image_type", "group_key", "dataset", "method"]
    df = df.sort_values(by=sort_cols)

    # Deduplicate: if multiple entries exist for the same exact configuration, keep the last one.
    df = df.drop_duplicates(subset=sort_cols, keep='last')

    # Group by model and save separate JSONs
    for model_name, model_group in df.groupby("model"):
        grouped_json = {}
        for (img_type, g_key), group in model_group.groupby(["image_type", "group_key"]):
            key = f"{img_type} | {g_key}"
            results = []
            for _, row in group.iterrows():
                results.append({
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "avg_diff_ratio": row["avg_diff_ratio"]
                })
            grouped_json[key] = results

        out_name = f"results_diff_ratio_{model_name}.json"
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(grouped_json, f, indent=4)
        print(f"Saved grouped results for model '{model_name}' to {out_name}")


if __name__ == "__main__":
    main()
