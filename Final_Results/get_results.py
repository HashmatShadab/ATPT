#!/usr/bin/env python3
"""
get_results.py  –  long-path–safe version
----------------------------------------
Exactly the same CLI and behaviour as before, but now works even when some
log-file paths are longer than Windows' legacy 260-char MAX_PATH limit.

Key additions
-------------
*  ensure_long_path()   → converts any `Path` to `\\?\…` form on Windows.
*  root_long            → we run `Path.glob()` / `rglob()` on this.
*  All I/O (`open`, `.exists()`, …) goes through long paths.
"""
from pathlib import Path
import argparse, re, sys, os
from typing import Optional

# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------
def ensure_long_path(p: Path) -> Path:
    """
    Prepend \\?\\ (or \\?\\UNC\\ for network shares) **only on Windows** so
    that Win32 APIs accept paths >260 chars.  On other OSes, returns p.
    """
    if os.name != "nt":                       # nothing to do on POSIX
        return p

    p = p.resolve()                           # absolute, no symlinks
    s = str(p)
    if s.startswith("\\\\?\\"):               # already long-form
        return p
    if s.startswith("\\\\"):                  # UNC share → \\?\UNC\server\share
        s = "\\\\?\\UNC\\" + s.lstrip("\\")
    else:                                     # local drive
        s = "\\\\?\\" + s
    return Path(s)


FINAL_RE = re.compile(r"(Final results\s*-\s*.*|Original: Clean Acc @1.*|Single TTA: Clean Acc @1.*|Vanilla TTA: Clean Acc @1.*|Vanilla Topk TTA: Clean Acc @1.*|Weighted TTA: Clean Acc @1.*|=> Acc\. on testset \[.*\]: Clean Acc @1.*)", re.I)

def last_final_line(path: Path) -> Optional[str]:
    """Return the last line containing 'Final results - …' or accuracy results (or None).

    Matches patterns like:
    - 'Final results - ...'
    - 'Original: Clean Acc @1 ...'
    - 'Single TTA: Clean Acc @1 ...'
    - 'Vanilla TTA: Clean Acc @1 ...'
    - 'Vanilla Topk TTA: Clean Acc @1 ...'
    - 'Weighted TTA: Clean Acc @1 ...'
    - '=> Acc. on testset [dataset]: Clean Acc @1 ...'
    """
    path = ensure_long_path(path)             # <— long-path magic
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
            # Find all lines that match the pattern
            matches = []
            for line in reversed(lines):
                m = FINAL_RE.search(line)
                if m:
                    matches.append(m.group(0).strip())

            # Define the order for the results
            def get_result_order(result):
                if "Final results - Original" in result:
                    return 0
                elif "Final results - Single TTA" in result:
                    return 1
                elif "Final results - Vanilla TTA" in result:
                    return 2
                elif "Final results - Vanilla Topk" in result:
                    return 3
                elif "Final results - Weighted TTA" in result:
                    return 4
                elif "Final results" in result:
                    return 5
                elif "Original: Clean Acc" in result:
                    return 6
                elif "Single TTA: Clean Acc" in result:
                    return 7
                elif "Vanilla TTA: Clean Acc" in result:
                    return 8
                elif "Vanilla Topk TTA: Clean Acc" in result:
                    return 9
                elif "Weighted TTA: Clean Acc" in result:
                    return 10
                else:
                    return 11

            # Sort the matches according to the defined order
            matches.sort(key=get_result_order)

            # Return all matches as a single string, separated by newlines
            if matches:
                return "\n".join(matches)
    except (OSError, UnicodeDecodeError):
        pass
    return None


# ----------------------------------------------------------------------
#  Defaults  (unchanged)
# ----------------------------------------------------------------------
DEFAULT_ROOT  = Path(".")
DEFAULT_GLOBS = [
    "fare4/*/Clean/Counter_Attack/Eps_4_0_Steps_2_Alpha_1_0/"
    "tau_0_2_beta_2_0_weighted_pertrubation_True/No_TPT/"
    "Inference_Ensemble_n*/log_*.log"
]
ORDER = ["DTD", "Flower102", "Cars", "Aircraft",
         "Pets", "Caltech101", "UCF101", "eurosat"]


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract the last results line from log files (Final results or accuracy metrics)."
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Top-level directory to search (default: {DEFAULT_ROOT.resolve()})")
    ap.add_argument("--glob", nargs="+", default=DEFAULT_GLOBS,
                    help="One or more pathlib glob patterns *relative* to --root")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"[ERROR] --root '{root}' is not a directory.")

    root_long = ensure_long_path(root)        # safe version for glob()

    total_files, files_with_results = 0, 0
    log_results: list[tuple[Path, str]] = []

    # ------------------------------------------------------------------ #
    #  Crawl the tree (long-path aware)
    # ------------------------------------------------------------------ #
    for pat in args.glob:
        for log_file in root_long.glob(pat):
            total_files += 1
            res = last_final_line(log_file)   # opens with long path
            if res:
                files_with_results += 1
                log_results.append((log_file, res))

    # Sort & bucket by dataset
    def order_index(p: Path) -> int:
        s = str(p)
        for i, dataset in enumerate(ORDER):
            if dataset in s:
                return i
        return len(ORDER)

    log_results.sort(key=lambda x: order_index(x[0]))
    dataset_results = {d: [] for d in ORDER}
    for log_file, res in log_results:
        for d in ORDER:
            if d in str(log_file):
                dataset_results[d].append((log_file, res))
                break

    # ------------------------------------------------------------------ #
    #  Print nicely (strip \\?\ for readability)
    # ------------------------------------------------------------------ #
    def pretty(p: Path) -> str:
        s = str(p)
        if os.name == "nt" and s.startswith("\\\\?\\"):
            return s[4:] if not s.startswith("\\\\?\\UNC") else "\\" + s[7:]
        return s

    for d in ORDER:
        if dataset_results[d]:
            for log_file, res in dataset_results[d]:
                print(f"{pretty(log_file.resolve())}\n{res}\n")
        else:
            print(f"{d}: results not available\n")

    # Summary
    if total_files == 0:
        sys.stderr.write("[WARN] No log files matched the supplied pattern(s).\n")
    else:
        print("--- Summary -------------------------------------------------")
        print(f"Matched files          : {total_files}")
        print(f"Files with Final result: {files_with_results}")


if __name__ == "__main__":
    main()
