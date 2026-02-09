import tarfile
from pathlib import Path
from datetime import datetime
import os
import sys
import time

# ================= CONFIG =================

SRC_ROOT = Path(
    "/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/"
    "robust_mllm/eval/LLaVA-V/checkpoints_adv_ft_multilingual_merge_lora_training_llava"
)

DST_BASE = Path(
    "/leonardo_work/EUHPC_R04_192/fmohamma/Adversarial_Robust_Clip/"
    "atpt_data_compressed"
)

DST_ROOT = DST_BASE / SRC_ROOT.name

# =========================================


def ts():
    """Timestamp for SLURM logs"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def compress_folder(folder: Path, idx: int, total: int):
    tar_path = DST_ROOT / f"{folder.name}.tar.gz"

    log(f"[{idx}/{total}] START folder: {folder.name}")

    if tar_path.exists():
        size_gb = tar_path.stat().st_size / (1024 ** 3)
        log(f"[{idx}/{total}] SKIP (already exists): {tar_path.name} | {size_gb:.2f} GB")
        return

    start_time = time.time()

    log(f"[{idx}/{total}] Creating archive: {tar_path}")
    log(f"[{idx}/{total}] Source path: {folder}")

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(folder, arcname=folder.name)

    except Exception as e:
        log(f"[{idx}/{total}] ERROR while compressing {folder.name}")
        log(str(e))
        raise

    elapsed = time.time() - start_time
    size_gb = tar_path.stat().st_size / (1024 ** 3)

    log(
        f"[{idx}/{total}] DONE folder: {folder.name} | "
        f"Size: {size_gb:.2f} GB | Time: {elapsed/60:.1f} min"
    )


def main():
    log("========== COMPRESSION JOB START ==========")

    log(f"Python PID: {os.getpid()}")
    log(f"Running on host: {os.uname().nodename}")
    log(f"Working directory: {Path.cwd()}")

    if not SRC_ROOT.exists():
        log(f"FATAL: Source directory does not exist: {SRC_ROOT}")
        sys.exit(1)

    log(f"Source root: {SRC_ROOT}")
    log(f"Destination root: {DST_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    log("Destination directory ready")

    folders = sorted([p for p in SRC_ROOT.iterdir() if p.is_dir()])
    total = len(folders)

    log(f"Found {total} folders to compress")

    if total == 0:
        log("Nothing to do. Exiting.")
        return

    for idx, folder in enumerate(folders, start=1):
        compress_folder(folder, idx, total)

    log("========== ALL FOLDERS PROCESSED ==========")


if __name__ == "__main__":
    main()
