import tarfile
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Root directory containing the original data
SRC_ROOT = Path(
    "/leonardo_work/EUHPC_R04_192/fmohamma/"
    "Adversarial_Robust_Clip/atpt_data"
)

# Root directory where compressed data will be written
# (original data is NOT modified)
DST_ROOT = Path(
    "/leonardo_work/EUHPC_R04_192/fmohamma/"
    "Adversarial_Robust_Clip/atpt_data_compressed"
)

# Model directories to process.
# Value = None     → datasets live directly under the model directory
# Value = "14"     → one extra nesting level (e.g., ViT-L/14)
MODELS = {
    "delta_clip_l14_224": None,
    "fare4": None,
    "RN50": None,
    "vit_l_14_datacomp_1b": None,
    "ViT-L": "14",   # special case: datasets under ViT-L/14/
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compress_dataset(dataset_dir: Path, dst_tar: Path):
    """
    Compress an entire dataset directory into a single .tar.gz file.

    Example:
        Cars/
          ├── clean/
          ├── adv_images_eps_4.0/
          └── adv_images_eps_8.0/

    → Cars.tar.gz (containing all of the above)

    Args:
        dataset_dir (Path): Path to the dataset directory
        dst_tar (Path): Output .tar.gz file path
    """
    # Open a gzip-compressed tar archive for writing
    with tarfile.open(dst_tar, "w:gz") as tar:
        # Add the full dataset directory.
        # arcname ensures that when extracted, the top-level folder
        # is named exactly like the dataset (e.g., Cars/)
        tar.add(dataset_dir, arcname=dataset_dir.name)

def process_model(model_name: str, sublevel: str | None):
    """
    Process a single model directory:
    - Iterate over all dataset folders
    - Create one compressed archive per dataset

    Args:
        model_name (str): Name of the model directory
        sublevel (str | None): Optional extra nesting (e.g., "14" for ViT-L/14)
    """
    # Source model directory
    src_model_dir = SRC_ROOT / model_name
    # Destination model directory
    dst_model_dir = DST_ROOT / model_name

    # Handle special nested structure (e.g., ViT-L/14)
    if sublevel is not None:
        src_model_dir = src_model_dir / sublevel
        dst_model_dir = dst_model_dir / sublevel

    # Create destination directory if it does not exist
    dst_model_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over datasets (Cars, DTD, UCF101, ...)
    for dataset_dir in src_model_dir.iterdir():
        # Skip anything that is not a directory
        if not dataset_dir.is_dir():
            continue

        # Output archive path (one archive per dataset)
        tar_path = dst_model_dir / f"{dataset_dir.name}.tar.gz"

        # Skip if archive already exists (safe for re-runs)
        if tar_path.exists():
            print(f"[SKIP] {tar_path}")
            continue

        print(f"[COMPRESS] {dataset_dir} → {tar_path}")
        compress_dataset(dataset_dir, tar_path)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """
    Main driver function:
    iterates over all models and compresses their datasets.
    """
    for model_name, sublevel in MODELS.items():
        print(f"\n=== Processing model: {model_name} ===")
        process_model(model_name, sublevel)

if __name__ == "__main__":
    main()
