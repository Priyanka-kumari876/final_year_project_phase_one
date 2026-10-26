"""
Inspect PAMAP2 Dataset Script
=============================
Inspects raw PAMAP2 dataset files, produces data_manifest.json,
prints summary statistics, and generates visual plots of raw sensor signals.
"""

import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_subject_raw_data,
    inspect_dataset_directory,
    save_manifest,
    PROTOCOL_ACTIVITIES,
)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    manifest_path = PROJECT_ROOT / config["data"]["manifest_path"]
    fig_dir = PROJECT_ROOT / config["output"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 2: INSPECTING PAMAP2 DATASET DIRECTORY")
    print(f"Dataset Protocol Path: {raw_dir}")
    print("=" * 70)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    manifest = inspect_dataset_directory(raw_dir)
    save_manifest(manifest, manifest_path)

    print("\n--- Summary of Found Subjects ---")
    print(f"{'Subject':<12} {'Size (MB)':<12} {'Total Rows':<12} {'Protocol Rows':<15} {'HR Missing %':<15}")
    print("-" * 68)
    for sub, info in manifest["subjects"].items():
        print(f"{sub:<12} {info['file_size_mb']:<12.1f} {info['total_rows']:<12} {info['valid_protocol_rows']:<15} {info['missing_rates_pct']['heart_rate']:<15.1f}")
    print("-" * 68)
    print(f"Total Protocol Samples across all subjects: {manifest['total_protocol_samples']:,}")

    # Plot sample raw signals for subject 101
    sample_sub_file = raw_dir / "subject101.dat"
    if sample_sub_file.exists():
        print(f"\nPlotting sample raw signals from {sample_sub_file.name}...")
        df_sample = load_subject_raw_data(sample_sub_file, subject_id=101, drop_transient=False, protocol_only=False)

        # Take a slice of 300 seconds (30,000 samples)
        slice_df = df_sample.iloc[10000:40000].copy()
        time_rel = slice_df["timestamp"] - slice_df["timestamp"].iloc[0]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # Plot Hand Accel
        axes[0].plot(time_rel, slice_df["hand_acc_16g_x"], label="X (m/s²)", alpha=0.7, lw=0.8)
        axes[0].plot(time_rel, slice_df["hand_acc_16g_y"], label="Y (m/s²)", alpha=0.7, lw=0.8)
        axes[0].plot(time_rel, slice_df["hand_acc_16g_z"], label="Z (m/s²)", alpha=0.7, lw=0.8)
        axes[0].set_ylabel("Hand Acc (±16g)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("PAMAP2 Raw Signals Inspection (Subject 101)")

        # Plot Ankle Accel
        axes[1].plot(time_rel, slice_df["ankle_acc_16g_x"], label="X (m/s²)", alpha=0.7, lw=0.8)
        axes[1].plot(time_rel, slice_df["ankle_acc_16g_y"], label="Y (m/s²)", alpha=0.7, lw=0.8)
        axes[1].plot(time_rel, slice_df["ankle_acc_16g_z"], label="Z (m/s²)", alpha=0.7, lw=0.8)
        axes[1].set_ylabel("Ankle Acc (±16g)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Plot Heart Rate
        axes[2].plot(time_rel, slice_df["heart_rate"], color="crimson", lw=1.2, label="Heart Rate (bpm)")
        axes[2].set_ylabel("HR (bpm)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # Plot Activity ID
        axes[3].step(time_rel, slice_df["activity_id"], color="darkblue", lw=1.5, where="post", label="Activity ID")
        axes[3].set_ylabel("Activity ID")
        axes[3].set_xlabel("Elapsed Time (seconds)")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_out = fig_dir / "raw_signals_sample.png"
        plt.savefig(plot_out, dpi=150)
        plt.close()
        print(f"Sample raw signals plot saved to {plot_out}")

    print("\nPhase 2 Dataset Inspection Completed Successfully.")


if __name__ == "__main__":
    main()
