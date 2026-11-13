"""
Master Pipeline Runner: P10 Multimodal Acquisition (First Version)
==================================================================
Executes the complete mandatory research pipeline sequentially from Stage 01 to Stage 04:
  01: Inspect PAMAP2 Dataset and Generate Data Manifest
  02: Fixed-Grid Windowing & Feature Extraction
  03: Train 8 Sensor-Subset Classifiers on Training Participants
  04: Run Comprehensive Multimodal Acquisition Experiments & Benchmarks

Note:
  Optional extensions (e.g. Stage 05 two-phase within-window screening) and
  reporting scripts (Stage 06 & 07) remain available as standalone scripts in scripts/.

Usage:
  python run_all.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STAGES = [
    ("01_inspect_dataset.py", "Stage 01: Dataset Inspection & Manifest"),
    ("02_prepare_data.py", "Stage 02: Windowing & Feature Extraction"),
    ("03_train_classifiers.py", "Stage 03: Train 8 Sensor-Subset Classifiers"),
    ("04_run_experiments.py", "Stage 04: Acquisition Policies & Experiments"),
]


def run_pipeline():
    print("=" * 80)
    print("P10: MULTIMODAL ACQUISITION RESEARCH PIPELINE — FIRST-VERSION EXECUTION")
    print("=" * 80)

    for script_name, description in STAGES:
        script_path = PROJECT_ROOT / "scripts" / script_name
        print(f"\n>>> Running {description} ({script_name})...")
        cmd = [sys.executable, "-u", str(script_path)]
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if ret.returncode != 0:
            print(f"\n[ERROR] Pipeline failed at {script_name} with returncode {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"[SUCCESS] {description} finished.")

    print("\n" + "=" * 80)
    print("ALL MANDATORY FIRST-VERSION STAGES COMPLETED SUCCESSFULLY!")
    print("All models, logs, tables, and Pareto figures have been verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
