
from typing import Dict, List, NamedTuple, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WindowSegment(NamedTuple):
    subject_id: int
    session: str
    activity_id: int
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    num_samples: int


def create_fixed_grid_windows(
    df: pd.DataFrame,
    window_sec: float = 3.0,
    overlap: float = 0.5,
    min_samples: int = 10,
    sampling_rate: float = 100.0,
) -> Tuple[List[WindowSegment], Dict[str, int]]:
    """
    Segment a continuous session DataFrame into fixed-grid windows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'timestamp', 'activity_id', 'subject_id', 'session'.
        Must be sorted by timestamp.
    window_sec : float
        Duration of each window in seconds (e.g. 3.0).
    overlap : float
        Fractional overlap between consecutive windows (e.g. 0.5 = 50%).
    min_samples : int
        Minimum number of samples required to form a valid window.
    sampling_rate : float
        Nominal sampling rate in Hz (100 Hz for PAMAP2 IMU).

    Returns
    -------
    windows : List[WindowSegment]
        List of valid, single-activity window segments.
    stats : Dict[str, int]
        Dictionary tracking window counts:
        - total_attempted
        - valid_retained
        - discarded_transition (contained multiple activities)
        - discarded_transient_label (activity 0)
        - discarded_insufficient_samples
        - discarded_time_gap (gap in timestamps > threshold)
    """
    if df.empty:
        return [], {
            "total_attempted": 0,
            "valid_retained": 0,
            "discarded_transition": 0,
            "discarded_transient_label": 0,
            "discarded_insufficient_samples": 0,
            "discarded_time_gap": 0,
        }

    hop_sec = window_sec * (1.0 - overlap)
    timestamps = df["timestamp"].values
    activities = df["activity_id"].values
    subject_id = int(df["subject_id"].iloc[0])
    session = str(df["session"].iloc[0])

    t_start_global = timestamps[0]
    t_end_global = timestamps[-1]

    # Generate fixed time grid
    grid_starts = np.arange(t_start_global, t_end_global - window_sec + 1e-6, hop_sec)

    windows: List[WindowSegment] = []
    stats = {
        "total_attempted": len(grid_starts),
        "valid_retained": 0,
        "discarded_transition": 0,
        "discarded_transient_label": 0,
        "discarded_insufficient_samples": 0,
        "discarded_time_gap": 0,
    }

    # Binary search indices for fast time grid slicing
    for w_start in grid_starts:
        w_end = w_start + window_sec
        idx_start = np.searchsorted(timestamps, w_start, side="left")
        idx_end = np.searchsorted(timestamps, w_end, side="right")

        n_samples = idx_end - idx_start
        if n_samples < min_samples:
            stats["discarded_insufficient_samples"] += 1
            continue

        # Check for substantial time gaps inside the window (e.g. sensor recording disconnect)
        # Expected spacing is ~0.01s; a gap > 0.5s indicates discontiguous recording
        sample_times = timestamps[idx_start:idx_end]
        if np.any(np.diff(sample_times) > 0.5):
            stats["discarded_time_gap"] += 1
            continue

        window_acts = activities[idx_start:idx_end]
        unique_acts = np.unique(window_acts)

        # Check if window contains transient activity 0
        if 0 in unique_acts:
            stats["discarded_transient_label"] += 1
            continue

        # Check if window crosses activity transitions
        if len(unique_acts) > 1:
            stats["discarded_transition"] += 1
            continue

        act_label = int(unique_acts[0])
        windows.append(
            WindowSegment(
                subject_id=subject_id,
                session=session,
                activity_id=act_label,
                start_time=float(w_start),
                end_time=float(w_end),
                start_idx=int(idx_start),
                end_idx=int(idx_end),
                num_samples=int(n_samples),
            )
        )
        stats["valid_retained"] += 1

    return windows, stats
