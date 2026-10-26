"""
PAMAP2 Dataset Loader Module
============================
Handles loading, channel mapping, missingness tracking, and metadata extraction
for the PAMAP2 Physical Activity Monitoring dataset.

According to dataset documentation:
- 54 columns per file (space-separated)
- 3 Colibri IMUs (wrist/hand, chest, ankle) sampled at ~100 Hz
- BM-CS5SR Heart Rate monitor sampled at ~9 Hz (interpolated/padded with NaN)
- Protocol activities: 12 defined classes (activity 0 = transient, excluded)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Complete list of all 54 column names in the raw PAMAP2 .dat files
COLUMN_NAMES: List[str] = [
    "timestamp",              # Col 0: timestamp (s)
    "activity_id",            # Col 1: activityID
    "heart_rate",             # Col 2: heart rate (bpm)
    # Hand IMU (Col 3-19)
    "hand_temp",              # Col 3: IMU hand temperature (°C)
    "hand_acc_16g_x",         # Col 4: IMU hand 3D-acc ±16g x (m/s^2)
    "hand_acc_16g_y",         # Col 5: IMU hand 3D-acc ±16g y (m/s^2)
    "hand_acc_16g_z",         # Col 6: IMU hand 3D-acc ±16g z (m/s^2)
    "hand_acc_6g_x",          # Col 7: IMU hand 3D-acc ±6g x (m/s^2)
    "hand_acc_6g_y",          # Col 8: IMU hand 3D-acc ±6g y (m/s^2)
    "hand_acc_6g_z",          # Col 9: IMU hand 3D-acc ±6g z (m/s^2)
    "hand_gyro_x",            # Col 10: IMU hand 3D-gyro x (rad/s)
    "hand_gyro_y",            # Col 11: IMU hand 3D-gyro y (rad/s)
    "hand_gyro_z",            # Col 12: IMU hand 3D-gyro z (rad/s)
    "hand_mag_x",             # Col 13: IMU hand 3D-mag x (μT)
    "hand_mag_y",             # Col 14: IMU hand 3D-mag y (μT)
    "hand_mag_z",             # Col 15: IMU hand 3D-mag z (μT)
    "hand_ori_0",             # Col 16: IMU hand orientation 0 (invalid)
    "hand_ori_1",             # Col 17: IMU hand orientation 1 (invalid)
    "hand_ori_2",             # Col 18: IMU hand orientation 2 (invalid)
    "hand_ori_3",             # Col 19: IMU hand orientation 3 (invalid)
    # Chest IMU (Col 20-36)
    "chest_temp",             # Col 20: IMU chest temperature (°C)
    "chest_acc_16g_x",        # Col 21: IMU chest 3D-acc ±16g x (m/s^2)
    "chest_acc_16g_y",        # Col 22: IMU chest 3D-acc ±16g y (m/s^2)
    "chest_acc_16g_z",        # Col 23: IMU chest 3D-acc ±16g z (m/s^2)
    "chest_acc_6g_x",         # Col 24: IMU chest 3D-acc ±6g x (m/s^2)
    "chest_acc_6g_y",         # Col 25: IMU chest 3D-acc ±6g y (m/s^2)
    "chest_acc_6g_z",         # Col 26: IMU chest 3D-acc ±6g z (m/s^2)
    "chest_gyro_x",           # Col 27: IMU chest 3D-gyro x (rad/s)
    "chest_gyro_y",           # Col 28: IMU chest 3D-gyro y (rad/s)
    "chest_gyro_z",           # Col 29: IMU chest 3D-gyro z (rad/s)
    "chest_mag_x",            # Col 30: IMU chest 3D-mag x (μT)
    "chest_mag_y",            # Col 31: IMU chest 3D-mag y (μT)
    "chest_mag_z",            # Col 32: IMU chest 3D-mag z (μT)
    "chest_ori_0",            # Col 33: IMU chest orientation 0 (invalid)
    "chest_ori_1",            # Col 34: IMU chest orientation 1 (invalid)
    "chest_ori_2",            # Col 35: IMU chest orientation 2 (invalid)
    "chest_ori_3",            # Col 36: IMU chest orientation 3 (invalid)
    # Ankle IMU (Col 37-53)
    "ankle_temp",             # Col 37: IMU ankle temperature (°C)
    "ankle_acc_16g_x",        # Col 38: IMU ankle 3D-acc ±16g x (m/s^2)
    "ankle_acc_16g_y",        # Col 39: IMU ankle 3D-acc ±16g y (m/s^2)
    "ankle_acc_16g_z",        # Col 40: IMU ankle 3D-acc ±16g z (m/s^2)
    "ankle_acc_6g_x",         # Col 41: IMU ankle 3D-acc ±6g x (m/s^2)
    "ankle_acc_6g_y",         # Col 42: IMU ankle 3D-acc ±6g y (m/s^2)
    "ankle_acc_6g_z",         # Col 43: IMU ankle 3D-acc ±6g z (m/s^2)
    "ankle_gyro_x",           # Col 44: IMU ankle 3D-gyro x (rad/s)
    "ankle_gyro_y",           # Col 45: IMU ankle 3D-gyro y (rad/s)
    "ankle_gyro_z",           # Col 46: IMU ankle 3D-gyro z (rad/s)
    "ankle_mag_x",            # Col 47: IMU ankle 3D-mag x (μT)
    "ankle_mag_y",            # Col 48: IMU ankle 3D-mag y (μT)
    "ankle_mag_z",            # Col 49: IMU ankle 3D-mag z (μT)
    "ankle_ori_0",            # Col 50: IMU ankle orientation 0 (invalid)
    "ankle_ori_1",            # Col 51: IMU ankle orientation 1 (invalid)
    "ankle_ori_2",            # Col 52: IMU ankle orientation 2 (invalid)
    "ankle_ori_3",            # Col 53: IMU ankle orientation 3 (invalid)
]

# Physical sensor group definitions for P10
SENSOR_GROUP_CHANNELS: Dict[str, List[str]] = {
    "hand": [
        "hand_acc_16g_x", "hand_acc_16g_y", "hand_acc_16g_z",
        "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
        "hand_mag_x", "hand_mag_y", "hand_mag_z",
        "hand_temp",
    ],
    "ankle": [
        "ankle_acc_16g_x", "ankle_acc_16g_y", "ankle_acc_16g_z",
        "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
        "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
        "ankle_temp",
    ],
    "heart_rate": [
        "heart_rate",
    ],
}

PROTOCOL_ACTIVITIES: Dict[int, str] = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping",
}


def load_subject_raw_data(
    file_path: Union[str, Path],
    subject_id: Optional[int] = None,
    session: str = "protocol",
    drop_transient: bool = True,
    protocol_only: bool = True,
) -> pd.DataFrame:
    """
    Load a single PAMAP2 subject .dat file into a structured DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to subject .dat file.
    subject_id : int, optional
        Subject identifier (e.g. 101). Inferred from filename if None.
    session : str
        Session type: 'protocol' or 'optional'.
    drop_transient : bool
        If True, drops activity_id == 0 rows (transient activities).
    protocol_only : bool
        If True, retains only the 12 standard protocol activities.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with explicit columns, subject_id, session, and preserved NaN masks.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PAMAP2 file not found: {file_path}")

    if subject_id is None:
        name = file_path.stem  # e.g., 'subject101'
        try:
            subject_id = int("".join(filter(str.isdigit, name)))
        except ValueError:
            subject_id = -1

    logger.info(f"Loading {file_path.name} (Subject {subject_id}, Session: {session})...")

    # Read space-delimited text file
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        dtype=np.float64,
        engine="c",
    )

    # Cast activity_id to integer (or Int64 if NaNs exist)
    df["activity_id"] = df["activity_id"].fillna(0).astype(int)
    df["subject_id"] = int(subject_id)
    df["session"] = str(session)

    # Filter out transient activities
    if drop_transient:
        df = df[df["activity_id"] != 0].copy()

    # Filter strictly to protocol activities if requested
    if protocol_only:
        df = df[df["activity_id"].isin(PROTOCOL_ACTIVITIES.keys())].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def inspect_dataset_directory(
    protocol_dir: Union[str, Path],
    optional_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Inspect the dataset directory, compute summary statistics, and compile a data manifest.

    Parameters
    ----------
    protocol_dir : str or Path
        Path to Protocol folder containing subject101.dat ... subject109.dat
    optional_dir : str or Path, optional
        Path to Optional folder

    Returns
    -------
    dict
        Manifest dictionary with file details, sample counts, activity distributions, and missingness.
    """
    protocol_path = Path(protocol_dir)
    manifest = {
        "dataset_name": "PAMAP2 Physical Activity Monitoring",
        "protocol_directory": str(protocol_path),
        "protocol_activities": PROTOCOL_ACTIVITIES,
        "sensor_groups": SENSOR_GROUP_CHANNELS,
        "subjects": {},
        "total_files": 0,
        "total_protocol_samples": 0,
    }

    dat_files = sorted(protocol_path.glob("subject*.dat"))
    manifest["total_files"] = len(dat_files)

    for f in dat_files:
        try:
            sub_id = int("".join(filter(str.isdigit, f.stem)))
        except ValueError:
            continue

        file_size_mb = f.stat().st_size / (1024 * 1024)

        # Fast scan: read only timestamp and activity_id, plus check NaN in HR and IMUs
        cols_to_peek = [0, 1, 2, 3, 4, 37, 38]  # timestamp, act, hr, hand_temp, hand_acc_x, ankle_temp, ankle_acc_x
        peek_df = pd.read_csv(
            f,
            sep=r"\s+",
            header=None,
            usecols=cols_to_peek,
            names=["timestamp", "activity_id", "hr", "hand_temp", "hand_acc", "ankle_temp", "ankle_acc"],
            dtype=np.float64,
        )

        total_rows = len(peek_df)
        transient_rows = int((peek_df["activity_id"] == 0).sum())
        protocol_mask = peek_df["activity_id"].isin(PROTOCOL_ACTIVITIES.keys())
        protocol_rows = int(protocol_mask.sum())
        hr_missing_pct = float(peek_df.loc[protocol_mask, "hr"].isna().mean() * 100.0) if protocol_rows > 0 else 0.0
        hand_missing_pct = float(peek_df.loc[protocol_mask, "hand_acc"].isna().mean() * 100.0) if protocol_rows > 0 else 0.0
        ankle_missing_pct = float(peek_df.loc[protocol_mask, "ankle_acc"].isna().mean() * 100.0) if protocol_rows > 0 else 0.0

        act_counts = {}
        for act_id, act_name in PROTOCOL_ACTIVITIES.items():
            count = int((peek_df["activity_id"] == act_id).sum())
            if count > 0:
                act_counts[act_name] = count

        manifest["subjects"][f"subject{sub_id}"] = {
            "subject_id": sub_id,
            "filename": f.name,
            "file_size_mb": round(file_size_mb, 2),
            "total_rows": total_rows,
            "transient_rows": transient_rows,
            "valid_protocol_rows": protocol_rows,
            "duration_sec": round((peek_df["timestamp"].iloc[-1] - peek_df["timestamp"].iloc[0]), 2) if total_rows > 0 else 0.0,
            "missing_rates_pct": {
                "heart_rate": round(hr_missing_pct, 2),
                "hand_imu": round(hand_missing_pct, 2),
                "ankle_imu": round(ankle_missing_pct, 2),
            },
            "activity_counts": act_counts,
        }
        manifest["total_protocol_samples"] += protocol_rows

    return manifest


def save_manifest(manifest: Dict, output_path: Union[str, Path]) -> None:
    """Save manifest dictionary to a formatted JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved data manifest to {output_path}")
