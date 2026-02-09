import argparse
import tarfile
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SRC_ROOT = Path(
    "/leonardo_work/EUHPC_R04_192/fmohamma/"
    "Adversarial_Robust_Clip/atpt_data"
)

DST_ROOT = Path(
    "/leonardo_work/EUHPC_R04_192/fmohamma/"
    "Adversarial_Robust_Clip/atpt_data_compressed"
)

# Model → optional sublevel mapping
MODELS = {
    "delta_clip_l14_224": None,
    "fare4": None,
    "RN50": None,
    "vit_l_14_datacomp_1b": None,
    "ViT-L": "14",
}

DATASETS = [
    "eurosat", "I", "DTD", "Aircraft", "R", "UCF101", "Caltech101",
    "Cars", "Pets", "V", "A", "K", "Flower102"
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compress_dataset(dataset_dir: Path, dst_tar: Path):
    """
    Compress a full dataset directory into a single .tar.gz archive.
    """
    print(f"    ↳ Creating archive: {dst_tar.name}")
    with tarfile.open(dst_tar, "w:gz") as tar:
        tar.add(dataset_dir, arcname=dataset_dir.name)
    print(f"    ✓ Finished: {dst_tar.name}")

def process_model(model_name: str, sublevel: str | None, dataset_name: str | None = None):
    """
    Process one model directory and compress each dataset inside it.
    If dataset_name is provided, only process that specific dataset.
    """
    print(f"\n[MODEL] {model_name}")

    src_model_dir = SRC_ROOT / model_name
    dst_model_dir = DST_ROOT / model_name

    if sublevel is not None:
        print(f"  ↳ Detected nested structure: /{model_name}/{sublevel}/")
        src_model_dir = src_model_dir / sublevel
        dst_model_dir = dst_model_dir / sublevel

    print(f"  ↳ Source directory: {src_model_dir}")
    print(f"  ↳ Output directory: {dst_model_dir}")

    dst_model_dir.mkdir(parents=True, exist_ok=True)

    if not src_model_dir.exists():
        print(f"  ⚠ WARNING: Source directory does not exist, skipping.")
        return

    if dataset_name:
        target_dir = src_model_dir / dataset_name
        if target_dir.exists() and target_dir.is_dir():
            datasets = [target_dir]
        else:
            print(f"  ⚠ WARNING: Dataset '{dataset_name}' not found in {src_model_dir}, skipping.")
            return
    else:
        datasets = [d for d in src_model_dir.iterdir() if d.is_dir()]

    print(f"  ↳ Found {len(datasets)} dataset(s)")

    for idx, dataset_dir in enumerate(sorted(datasets), start=1):
        print(f"\n  [DATASET {idx}/{len(datasets)}] {dataset_dir.name}")

        tar_path = dst_model_dir / f"{dataset_dir.name}.tar.gz"

        if tar_path.exists():
            print(f"    ⏭ Archive already exists, skipping")
            continue

        print(f"    ↳ Compressing: {dataset_dir}")
        compress_dataset(dataset_dir, tar_path)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Dataset-level compression script")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Name of the model to process (default: all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS + ["all"],
        default="all",
        help="Name of the specific dataset to process (default: all)"
    )
    args = parser.parse_args()

    print("==============================================")
    print(" Dataset-level compression started")
    print("==============================================")
    print(f"Source root:      {SRC_ROOT}")
    print(f"Destination root: {DST_ROOT}")

    if args.model == "all":
        models_to_process = MODELS.items()
    else:
        models_to_process = [(args.model, MODELS[args.model])]

    for model_name, sublevel in models_to_process:
        dataset_arg = None if args.dataset == "all" else args.dataset
        process_model(model_name, sublevel, dataset_arg)

    print("\n==============================================")
    print(" Compression completed successfully")
    print("==============================================")

if __name__ == "__main__":
    # Ensure all output is flushed immediately (useful for Slurm)
    import functools
    print = functools.partial(print, flush=True)
    main()
