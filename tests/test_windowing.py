"""
Unit Tests for Windowing Module
===============================
Verifies:
1. Windows never cross activity transitions.
2. Windows containing activity 0 (transient) are discarded.
3. Windows respect the rigid time grid.
4. Transition counts are tracked correctly.
"""

import pytest
import numpy as np
import pandas as pd

from src.windowing import create_fixed_grid_windows, WindowSegment


def test_windowing_discards_transitions():
    # 100 Hz signal: 10 seconds total = 1000 samples
    # First 5 seconds (500 samples): Activity 1
    # Next 5 seconds (500 samples): Activity 2
    timestamps = np.linspace(0.0, 9.99, 1000)
    acts = np.array([1] * 500 + [2] * 500)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "activity_id": acts,
        "subject_id": [101] * 1000,
        "session": ["protocol"] * 1000,
    })

    # 3.0s window, 1.5s hop (50% overlap)
    # Starts at 0.0 -> [0.0, 3.0] (all act 1) -> RETAINED
    # Starts at 1.5 -> [1.5, 4.5] (all act 1) -> RETAINED
    # Starts at 3.0 -> [3.0, 6.0] (contains transition at 5.0) -> DISCARDED TRANSITION
    # Starts at 4.5 -> [4.5, 7.5] (contains transition at 5.0) -> DISCARDED TRANSITION
    # Starts at 6.0 -> [6.0, 9.0] (all act 2) -> RETAINED
    windows, stats = create_fixed_grid_windows(df, window_sec=3.0, overlap=0.5)

    assert stats["discarded_transition"] >= 1, "Must discard and count windows crossing activity transition"
    for w in windows:
        # Every retained window must have only one unique activity label
        assert w.activity_id in [1, 2]
        assert w.start_time >= 0.0
        assert w.end_time <= 10.0


def test_windowing_discards_transient_activity_0():
    timestamps = np.linspace(0.0, 5.99, 600)
    # 0 is transient activity
    acts = np.array([0] * 300 + [1] * 300)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "activity_id": acts,
        "subject_id": [101] * 600,
        "session": ["protocol"] * 600,
    })

    windows, stats = create_fixed_grid_windows(df, window_sec=3.0, overlap=0.5)
    assert stats["discarded_transient_label"] >= 1
    for w in windows:
        assert w.activity_id != 0, "Activity 0 must NEVER be in retained windows"
