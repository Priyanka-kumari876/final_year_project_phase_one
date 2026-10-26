"""
Feature Extraction Module
=========================
Extracts statistical, time-domain, and spectral features independently
for each physical sensor group:
1. Hand IMU
2. Ankle IMU
3. Heart Rate Monitor

Also computes group-level availability and missingness indicators.
"""

from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_spectral_energy(signal: np.ndarray) -> float:
    """Compute spectral energy (sum of squared FFT magnitude) for a 1D signal."""
    valid_mask = ~np.isnan(signal)
    if np.sum(valid_mask) < 4:
        return 0.0
    clean_sig = signal[valid_mask] - np.mean(signal[valid_mask])
    fft_vals = np.fft.rfft(clean_sig)
    energy = float(np.sum(np.abs(fft_vals) ** 2) / len(clean_sig))
    return energy


def extract_imu_features(window_df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """
    Extract features for a single IMU device (hand or ankle).

    Parameters
    ----------
    window_df : pd.DataFrame
        Window data containing the IMU columns.
    prefix : str
        Prefix for the IMU, e.g. 'hand' or 'ankle'.

    Returns
    -------
    dict
        Extracted features and group presence indicator.
    """
    features: Dict[str, float] = {}

    acc_cols = [f"{prefix}_acc_16g_x", f"{prefix}_acc_16g_y", f"{prefix}_acc_16g_z"]
    gyro_cols = [f"{prefix}_gyro_x", f"{prefix}_gyro_y", f"{prefix}_gyro_z"]
    mag_cols = [f"{prefix}_mag_x", f"{prefix}_mag_y", f"{prefix}_mag_z"]
    temp_col = f"{prefix}_temp"

    # Check overall group presence
    acc_x = window_df[acc_cols[0]].values
    valid_mask = ~np.isnan(acc_x)
    valid_count = int(np.sum(valid_mask))
    is_present = 1.0 if valid_count >= 5 else 0.0
    features[f"{prefix}_present"] = is_present

    if is_present == 0.0:
        # Fill group features with zeros if device is absent/NaN
        for axis in ["x", "y", "z"]:
            for stat in ["mean", "std", "min", "max", "rms", "energy"]:
                features[f"{prefix}_acc_{axis}_{stat}"] = 0.0
                features[f"{prefix}_gyro_{axis}_{stat}"] = 0.0
            for stat in ["mean", "std", "min", "max"]:
                features[f"{prefix}_mag_{axis}_{stat}"] = 0.0
        features[f"{prefix}_acc_sma"] = 0.0
        features[f"{prefix}_temp_mean"] = 0.0
        return features

    # Acceleration features (±16g)
    acc_arrays = []
    for col, axis in zip(acc_cols, ["x", "y", "z"]):
        arr = window_df[col].values
        v = arr[~np.isnan(arr)]
        if len(v) > 0:
            features[f"{prefix}_acc_{axis}_mean"] = float(np.mean(v))
            features[f"{prefix}_acc_{axis}_std"] = float(np.std(v))
            features[f"{prefix}_acc_{axis}_min"] = float(np.min(v))
            features[f"{prefix}_acc_{axis}_max"] = float(np.max(v))
            features[f"{prefix}_acc_{axis}_rms"] = float(np.sqrt(np.mean(v ** 2)))
            features[f"{prefix}_acc_{axis}_energy"] = compute_spectral_energy(v)
            acc_arrays.append(np.abs(v))
        else:
            for stat in ["mean", "std", "min", "max", "rms", "energy"]:
                features[f"{prefix}_acc_{axis}_{stat}"] = 0.0

    # Signal Magnitude Area (SMA) for acceleration
    if len(acc_arrays) == 3 and min(len(a) for a in acc_arrays) > 0:
        min_len = min(len(a) for a in acc_arrays)
        sma = float(np.mean(acc_arrays[0][:min_len] + acc_arrays[1][:min_len] + acc_arrays[2][:min_len]))
        features[f"{prefix}_acc_sma"] = sma
    else:
        features[f"{prefix}_acc_sma"] = 0.0

    # Gyroscope features
    for col, axis in zip(gyro_cols, ["x", "y", "z"]):
        arr = window_df[col].values
        v = arr[~np.isnan(arr)]
        if len(v) > 0:
            features[f"{prefix}_gyro_{axis}_mean"] = float(np.mean(v))
            features[f"{prefix}_gyro_{axis}_std"] = float(np.std(v))
            features[f"{prefix}_gyro_{axis}_min"] = float(np.min(v))
            features[f"{prefix}_gyro_{axis}_max"] = float(np.max(v))
            features[f"{prefix}_gyro_{axis}_rms"] = float(np.sqrt(np.mean(v ** 2)))
            features[f"{prefix}_gyro_{axis}_energy"] = compute_spectral_energy(v)
        else:
            for stat in ["mean", "std", "min", "max", "rms", "energy"]:
                features[f"{prefix}_gyro_{axis}_{stat}"] = 0.0

    # Magnetometer features
    for col, axis in zip(mag_cols, ["x", "y", "z"]):
        arr = window_df[col].values
        v = arr[~np.isnan(arr)]
        if len(v) > 0:
            features[f"{prefix}_mag_{axis}_mean"] = float(np.mean(v))
            features[f"{prefix}_mag_{axis}_std"] = float(np.std(v))
            features[f"{prefix}_mag_{axis}_min"] = float(np.min(v))
            features[f"{prefix}_mag_{axis}_max"] = float(np.max(v))
        else:
            for stat in ["mean", "std", "min", "max"]:
                features[f"{prefix}_mag_{axis}_{stat}"] = 0.0

    # Temperature
    t_arr = window_df[temp_col].values
    tv = t_arr[~np.isnan(t_arr)]
    features[f"{prefix}_temp_mean"] = float(np.mean(tv)) if len(tv) > 0 else 0.0

    return features


def extract_hr_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features from the Heart Rate channel.

    Parameters
    ----------
    window_df : pd.DataFrame
        Window containing the 'heart_rate' column.

    Returns
    -------
    dict
        Heart rate features and availability indicator.
    """
    features: Dict[str, float] = {}
    hr_arr = window_df["heart_rate"].values
    valid_hr = hr_arr[~np.isnan(hr_arr)]
    n_samples = len(hr_arr)
    valid_count = len(valid_hr)

    features["hr_present"] = 1.0 if valid_count > 0 else 0.0
    features["hr_valid_count"] = float(valid_count)
    features["hr_valid_fraction"] = float(valid_count / n_samples) if n_samples > 0 else 0.0

    if valid_count > 0:
        features["hr_mean"] = float(np.mean(valid_hr))
        features["hr_std"] = float(np.std(valid_hr)) if valid_count > 1 else 0.0
        features["hr_min"] = float(np.min(valid_hr))
        features["hr_max"] = float(np.max(valid_hr))
        features["hr_last"] = float(valid_hr[-1])
    else:
        features["hr_mean"] = 0.0
        features["hr_std"] = 0.0
        features["hr_min"] = 0.0
        features["hr_max"] = 0.0
        features["hr_last"] = 0.0

    return features


def extract_window_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract all features across the 3 physical sensor modalities for a single window.

    Parameters
    ----------
    window_df : pd.DataFrame
        Raw samples for the window.

    Returns
    -------
    dict
        Combined dictionary of all extracted features.
    """
    feat_hand = extract_imu_features(window_df, prefix="hand")
    feat_ankle = extract_imu_features(window_df, prefix="ankle")
    feat_hr = extract_hr_features(window_df)

    all_features = {}
    all_features.update(feat_hand)
    all_features.update(feat_ankle)
    all_features.update(feat_hr)
    return all_features


def get_feature_column_names() -> Dict[str, List[str]]:
    """
    Return dictionary mapping each sensor group to its list of feature column names.
    """
    dummy_df = pd.DataFrame(
        np.zeros((10, 54)),
        columns=[
            "timestamp", "activity_id", "heart_rate",
            "hand_temp", "hand_acc_16g_x", "hand_acc_16g_y", "hand_acc_16g_z",
            "hand_acc_6g_x", "hand_acc_6g_y", "hand_acc_6g_z",
            "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
            "hand_mag_x", "hand_mag_y", "hand_mag_z",
            "hand_ori_0", "hand_ori_1", "hand_ori_2", "hand_ori_3",
            "chest_temp", "chest_acc_16g_x", "chest_acc_16g_y", "chest_acc_16g_z",
            "chest_acc_6g_x", "chest_acc_6g_y", "chest_acc_6g_z",
            "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
            "chest_mag_x", "chest_mag_y", "chest_mag_z",
            "chest_ori_0", "chest_ori_1", "chest_ori_2", "chest_ori_3",
            "ankle_temp", "ankle_acc_16g_x", "ankle_acc_16g_y", "ankle_acc_16g_z",
            "ankle_acc_6g_x", "ankle_acc_6g_y", "ankle_acc_6g_z",
            "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
            "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
            "ankle_ori_0", "ankle_ori_1", "ankle_ori_2", "ankle_ori_3",
        ]
    )
    hand_feats = list(extract_imu_features(dummy_df, "hand").keys())
    ankle_feats = list(extract_imu_features(dummy_df, "ankle").keys())
    hr_feats = list(extract_hr_features(dummy_df).keys())

    return {
        "hand": hand_feats,
        "ankle": ankle_feats,
        "heart_rate": hr_feats,
    }
