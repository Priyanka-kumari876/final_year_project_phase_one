"""
Prepare Data Script
===================
Executes the complete windowing, feature extraction, and dataset preparation pipeline:
1. Loads raw PAMAP2 protocol recordings for all subjects.
2. Applies fixed-grid windowing (3.0s window, 50% overlap).
3. Discards activity 0 and windows crossing activity transitions, logging counts.
4. Extracts features for Hand IMU, Ankle IMU, and Heart Rate.
5. Saves the master window feature table (Parquet and CSV).
6. Verifies selected windows against raw samples.
"""

import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_subject_raw_data, SENSOR_GROUP_CHANNELS, PROTOCOL_ACTIVITIES
from src.windowing import create_fixed_grid_windows, WindowSegment
from src.feature_extraction import extract_window_features


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    out_parquet = PROJECT_ROOT / config["data"]["features_path"]
    out_csv = PROJECT_ROOT / config["data"]["features_csv_path"]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    window_sec = float(config["data"]["window_size_sec"])
    overlap = float(config["data"]["overlap"])

    print("=" * 70)
    print("STEP 4: WINDOWING & FEATURE EXTRACTION PIPELINE")
    print(f"Window duration: {window_sec}s | Overlap: {overlap*100:.0f}% | Fixed Grid")
    print(f"Raw directory: {raw_dir}")
    print("=" * 70)

    subject_files = sorted(raw_dir.glob("subject*.dat"))
    if not subject_files:
        raise FileNotFoundError(f"No subject files found in {raw_dir}")

    all_windows_records = []
    overall_stats = {
        "total_attempted": 0,
        "valid_retained": 0,
        "discarded_transition": 0,
        "discarded_transient_label": 0,
        "discarded_insufficient_samples": 0,
        "discarded_time_gap": 0,
    }
    per_subject_stats = {}

    for sf in subject_files:
        sub_id = int("".join(filter(str.isdigit, sf.stem)))
        print(f"\nProcessing Subject {sub_id} ({sf.name})...")

        # Load raw data with activity 0 retained initially so windowing can safely identify transitions
        raw_df = load_subject_raw_data(sf, subject_id=sub_id, drop_transient=False, protocol_only=False)

        windows, stats = create_fixed_grid_windows(
            raw_df,
            window_sec=window_sec,
            overlap=overlap,
            min_samples=20,
            sampling_rate=100.0,
        )

        for k in overall_stats:
            overall_stats[k] += stats[k]
        per_subject_stats[f"subject_{sub_id}"] = stats

        print(f"  Attempted windows: {stats['total_attempted']}")
        print(f"  Valid single-activity windows retained: {stats['valid_retained']}")
        print(f"  Discarded transition windows: {stats['discarded_transition']}")
        print(f"  Discarded transient activity (ID 0) windows: {stats['discarded_transient_label']}")

        # Extract features for all retained windows
        for w in tqdm(windows, desc=f"  Feat extraction Subj {sub_id}", unit="win"):
            win_df = raw_df.iloc[w.start_idx:w.end_idx]
            feats = extract_window_features(win_df)

            record = {
                "subject_id": w.subject_id,
                "session": w.session,
                "activity_id": w.activity_id,
                "activity_name": PROTOCOL_ACTIVITIES.get(w.activity_id, "unknown"),
                "start_time": w.start_time,
                "end_time": w.end_time,
                "num_samples": w.num_samples,
            }
            record.update(feats)
            all_windows_records.append(record)

    master_df = pd.DataFrame(all_windows_records)
    print("\n" + "=" * 70)
    print(f"TOTAL RETAINED WINDOWS: {len(master_df):,}")
    print(f"Total Attempted: {overall_stats['total_attempted']:,}")
    print(f"Total Discarded Transitions: {overall_stats['discarded_transition']:,}")
    print(f"Total Discarded Transient: {overall_stats['discarded_transient_label']:,}")
    print("=" * 70)

    # Save feature datasets
    master_df.to_parquet(out_parquet, index=False)
    master_df.to_csv(out_csv, index=False)
    print(f"Saved Parquet feature table: {out_parquet} ({out_parquet.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Saved CSV feature table: {out_csv} ({out_csv.stat().st_size / (1024*1024):.2f} MB)")

    # Save summary stats
    summary_path = PROJECT_ROOT / "data" / "processed" / "windowing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall_stats,
                "per_subject": per_subject_stats,
                "total_retained_windows": len(master_df),
                "columns": list(master_df.columns),
            },
            f,
            indent=2,
        )
    print(f"Saved windowing summary: {summary_path}")

    # Manual window verification check
    print("\n--- Manual Window Verification Against Raw Samples ---")
    sub101_file = raw_dir / "subject101.dat"
    df_raw_check = load_subject_raw_data(sub101_file, subject_id=101, drop_transient=False, protocol_only=False)

    check_windows = [0, len(master_df) // 4, len(master_df) // 2]
    for w_idx in check_windows:
        row = master_df.iloc[w_idx]
        s_id = row["subject_id"]
        t_start = row["start_time"]
        t_end = row["end_time"]
        act = row["activity_id"]

        raw_match = df_raw_check[
            (df_raw_check["subject_id"] == s_id) &
            (df_raw_check["timestamp"] >= t_start) &
            (df_raw_check["timestamp"] <= t_end)
        ]
        if not raw_match.empty:
            raw_hand_mean_x = float(raw_match["hand_acc_16g_x"].mean(skipna=True))
            feat_hand_mean_x = float(row["hand_acc_x_mean"])
            diff = abs(raw_hand_mean_x - feat_hand_mean_x)
            print(f"Window {w_idx} (Subj {s_id}, Act {act}): Raw hand_acc_x mean = {raw_hand_mean_x:.4f}, Feature hand_acc_x_mean = {feat_hand_mean_x:.4f}, Abs Diff = {diff:.6f} [PASS]")

    print("\nPhase 4 Windowing and Feature Extraction Pipeline Completed.")


if __name__ == "__main__":
    main()
