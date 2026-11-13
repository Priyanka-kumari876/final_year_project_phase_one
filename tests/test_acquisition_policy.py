"""
Unit Tests for Acquisition Policy
=================================
Verifies:
1. Unavailable sensors cannot be selected under observed availability.
2. The selected set A_t is strictly a subset of available sensors U_t.
3. Hysteresis threshold behavior: waking a sleeping sensor requires higher gain than keeping an awake sensor on.
4. Determinism: Fixed random seeds reproduce identical selections.
"""

from typing import Set, Tuple
import pytest
import numpy as np
import pandas as pd

from src.cost_model import SensorCostModel
from src.gain_estimator import MarginalGainEstimator
from src.acquisition_policy import SwitchingAwarePolicy
from src.baselines import RandomFeasiblePolicy


class MockGainEstimator:
    """Mock gain estimator returning fixed predictable gains."""

    def __init__(self, fixed_gains: dict):
        self.fixed_gains = fixed_gains

    def predict_gain(
        self,
        prev_confidence: float,
        prev_entropy: float,
        current_subset: Tuple[str, ...],
        candidate_sensor: str,
    ) -> float:
        return self.fixed_gains.get(candidate_sensor, 0.0)


def test_unavailable_sensor_cannot_be_selected():
    cm = SensorCostModel()
    mock_estimator = MockGainEstimator({"hand": 100.0, "ankle": 100.0, "heart_rate": 100.0})
    policy = SwitchingAwarePolicy(cost_model=cm, gain_estimator=mock_estimator)

    # Ankle is in outage (only hand and heart_rate available)
    available: Set[str] = {"hand", "heart_rate"}
    selected = policy.select_sensors(
        available_sensors=available,
        prev_confidence=0.5,
        prev_entropy=1.0,
        prev_prediction=1,
    )

    assert "ankle" not in selected, "Unavailable sensor 'ankle' must NEVER be selected"
    assert selected.issubset(available), "Selected set must be a subset of available sensors"


def test_hysteresis_threshold_gap():
    # Set c_run = 1.0, c_wake = 5.0
    cm = SensorCostModel(
        running_costs={"hand": 1.0},
        default_wake_ratio=5.0,
        lambda_weight=1.0,
        mu_weight=1.0,
    )
    # Gain is 2.5:
    # If hand is awake: keep threshold is 1.0 * 1.0 = 1.0 -> 2.5 > 1.0 -> KEPT AWAKE!
    # If hand is asleep: wake threshold is 1.0 + 5.0 = 6.0 -> 2.5 < 6.0 -> REMAINS ASLEEP!
    mock_estimator = MockGainEstimator({"hand": 2.5})
    policy = SwitchingAwarePolicy(cost_model=cm, gain_estimator=mock_estimator)

    # Case 1: hand is initially asleep
    policy.reset()
    sel1 = policy.select_sensors({"hand"}, 0.5, 1.0, 1)
    assert "hand" not in sel1, "Gain of 2.5 is below wake threshold (6.0), sensor must stay asleep"

    # Case 2: hand was already active
    policy.active_sensors = {"hand"}
    sel2 = policy.select_sensors({"hand"}, 0.5, 1.0, 1)
    assert "hand" in sel2, "Gain of 2.5 is above keep threshold (1.0), sensor must be kept active"


def test_random_policy_deterministic_with_seed():
    cm = SensorCostModel()
    p1 = RandomFeasiblePolicy(cost_model=cm, p_acq=0.5, random_seed=999)
    p2 = RandomFeasiblePolicy(cost_model=cm, p_acq=0.5, random_seed=999)

    available = {"hand", "ankle", "heart_rate"}
    s1 = [p1.select_sensors(available, 0.5, 1.0, 1) for _ in range(20)]
    s2 = [p2.select_sensors(available, 0.5, 1.0, 1) for _ in range(20)]

    assert s1 == s2, "Identical seeds must produce identical random policy selections"


def test_switching_no_outage_policy():
    from src.baselines import SwitchingNoOutagePolicy

    cm = SensorCostModel()
    mock_estimator = MockGainEstimator({"hand": 100.0, "ankle": 100.0, "heart_rate": 100.0})
    policy = SwitchingNoOutagePolicy(cost_model=cm, gain_estimator=mock_estimator)

    # Ankle is in outage (only hand and heart_rate available)
    available = {"hand", "heart_rate"}
    selected = policy.select_sensors(
        available_sensors=available,
        prev_confidence=0.5,
        prev_entropy=1.0,
        prev_prediction=1,
    )

    # Policy should only return feasible sensors after physical intersection
    assert "ankle" not in selected, "Unavailable sensor 'ankle' must not be acquired"
    assert selected.issubset(available), "Selected set must be a subset of available sensors"
    assert "hand" in selected and "heart_rate" in selected

