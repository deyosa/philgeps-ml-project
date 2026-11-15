"""Run the pipeline from the cleaned dataset to the summary, in order.

Usage (from the repository root):

    python scripts/run_pipeline.py              # harmonise, then notebooks 01, 02, 03, 04
    python scripts/run_pipeline.py --from 03    # start at notebook 03 (e.g. after editing it)
    python scripts/run_pipeline.py --summary    # only refresh notebook 04

Notebook 00 (data cleaning) is not run here: it needs well over 16 GB of RAM, and its
output, data/raw/cleaned_data/PhilGeps_Cleaned.csv, is committed with Git LFS. Run it
by hand when the raw data changes, then run this script.

Each notebook is executed in place, so its saved outputs always match the latest run.
Notebook 04 is last, so the summary reflects everything before it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("harmonise", [sys.executable, str(ROOT / "scripts" / "harmonise_cleaned.py")]),
    ("01", "01_data_preparation_and_eda.ipynb"),
    ("02", "02_modeling.ipynb"),
    ("03", "03_diagnostics_and_interpretation.ipynb"),
    ("04", "04_project_summary.ipynb"),
]


def run(step: str, target) -> None:
    start = time.time()
    print(f"=== {step}", flush=True)
    if isinstance(target, list):
        cmd = target
    else:
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=-1", target]
    result = subprocess.run(cmd, cwd=ROOT / "notebooks" if not isinstance(target, list) else ROOT)
    if result.returncode != 0:
        sys.exit(f"Step {step} failed; later steps were not run.")
    print(f"    done in {time.time() - start:,.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", default="harmonise", choices=[s for s, _ in STEPS],
                        help="first step to run")
    parser.add_argument("--summary", action="store_true", help="only refresh notebook 04")
    args = parser.parse_args()
    start = "04" if args.summary else args.start
    names = [s for s, _ in STEPS]
    for step, target in STEPS[names.index(start):]:
        run(step, target)


if __name__ == "__main__":
    main()
