#!/usr/bin/env python3
"""
get_results.py
--------------
Print, for every matching *.log file:

    <absolute path to log file>
    Final results - Original Acc@1: …, Acc@5: …, TTA Acc@1: …, Acc@5: …

The script looks for the *last* line in each file that contains the string
'Final results -' (it can have a timestamp / logger prefix).

Usage examples
--------------

# 1) Just run – uses defaults shown below
python get_results.py

# 2) Different root folder, keep default glob
python get_results.py --root /mnt/experiments

# 3) Supply several glob patterns
python get_results.py --glob 'fare4/*/Clean/**/Inference_Ensemble_n*/log_*.log' \
                      --glob 'other_expts/**/Inference_Ensemble_n*/log_*.log'
"""
from pathlib import Path
import argparse
import re
import sys
from typing import Optional        #  <-- add this

# ----------------------------------------------------------------------
#  Config - sensible defaults, all override-able on the CLI
# ----------------------------------------------------------------------
DEFAULT_ROOT  = Path(".")          # current working directory
DEFAULT_GLOBS = [
    # Matches every line you showed in the ls listing
    #   fare4/{dataset}/Clean/Counter_Attack/Eps_4_0_Steps_2_Alpha_1_0/
    #        tau_0_2_beta_2_0_weighted_pertrubation_True/No_TPT/
    #        Inference_Ensemble_n{…}/log_*.log
    "fare4/*/Clean/Counter_Attack/Eps_4_0_Steps_2_Alpha_1_0/"
    "tau_0_2_beta_2_0_weighted_pertrubation_True/No_TPT/"
    "Inference_Ensemble_n*/log_*.log"
]

FINAL_RE = re.compile(r"Final results\s*-\s*.*", re.I)


# ----------------------------------------------------------------------
def last_final_line(path: Path)  -> Optional[str]:   #  <-- change here
    """Return the final 'Final results - …' line in *path* (None if absent)."""
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for line in reversed(fh.readlines()):
                m = FINAL_RE.search(line)
                if m:
                    return m.group(0).strip()
    except (OSError, UnicodeDecodeError):
        pass
    return None


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract the last 'Final results - …' line from log files."
    )
    ap.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help=f"Top-level directory to search (default: {DEFAULT_ROOT.resolve()})"
    )
    ap.add_argument(
        "--glob", nargs="+", default=DEFAULT_GLOBS,
        help="One or more pathlib glob patterns *relative* to --root "
             f"(default: {DEFAULT_GLOBS[0]})"
    )
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] --root '{root}' is not a directory.")


    order = ["DTD", "Flower102", "Cars", "Aircraft", "Pets", "Caltech101", "UCF101", "eurosat"]

    total_files, files_with_results = 0, 0
    for pat in args.glob:
        for log_file in root.glob(pat):
            total_files += 1
            res = last_final_line(log_file)
            if res:
                files_with_results += 1
                print(f"{log_file.resolve()}\n{res}\n")

    # --- summary -------------------------------------------------------
    if total_files == 0:
        sys.stderr.write("[WARN] No log files matched the supplied pattern(s).\n")
    else:
        print("--- Summary -------------------------------------------------")
        print(f"Matched files          : {total_files}")
        print(f"Files with Final result: {files_with_results}")


if __name__ == "__main__":
    main()
