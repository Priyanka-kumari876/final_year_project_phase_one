"""
Unit Tests for Cost Model
=========================
Verifies:
1. An awake sensor pays running cost but no new wake-up cost.
2. A sleeping sensor pays running cost plus wake-up cost when selected.
3. A sensor kept awake over consecutive windows pays the wake-up cost only once.
4. A recovered sensor pays wake-up cost under the declared outage-state convention.
5. Online cost matches recomputed cost from action logs.
"""

import pytest
from src.cost_model import SensorCostModel


def test_awake_sensor_pays_running_cost_only():
    cm = SensorCostModel(
        running_costs={"hand": 1.0, "ankle": 1.0, "heart_rate": 0.5},
        default_wake_ratio=5.0,
    )
    # Hand was already active in previous window
    prev_active = {"hand"}
    curr_active = {"hand"}
    c_run, c_wake, c_att, c_tot = cm.compute_step_cost(curr_active, prev_active)

    assert c_run == 1.0, "Running cost for hand must be 1.0"
    assert c_wake == 0.0, "Awake sensor must NOT pay any wake-up cost"
    assert c_tot == 1.0


def test_sleeping_sensor_pays_running_plus_wake():
    cm = SensorCostModel(
        running_costs={"hand": 1.0, "ankle": 1.0, "heart_rate": 0.5},
        default_wake_ratio=5.0,
    )
    # Hand was asleep in previous window
    prev_active = set()
    curr_active = {"hand"}
    c_run, c_wake, c_att, c_tot = cm.compute_step_cost(curr_active, prev_active)

    assert c_run == 1.0
    assert c_wake == 5.0, "Wake-up cost must be ratio (5.0) * running cost (1.0) = 5.0"
    assert c_tot == 6.0


def test_sensor_kept_awake_pays_wake_cost_only_once():
    cm = SensorCostModel(
        running_costs={"hand": 1.0},
        default_wake_ratio=4.0,
    )
    # Window 1: wakes from sleep
    w1_run, w1_wake, _, w1_tot = cm.compute_step_cost({"hand"}, set())
    assert w1_run == 1.0 and w1_wake == 4.0 and w1_tot == 5.0

    # Window 2: kept awake
    w2_run, w2_wake, _, w2_tot = cm.compute_step_cost({"hand"}, {"hand"})
    assert w2_run == 1.0 and w2_wake == 0.0 and w2_tot == 1.0

    # Window 3: kept awake
    w3_run, w3_wake, _, w3_tot = cm.compute_step_cost({"hand"}, {"hand"})
    assert w3_run == 1.0 and w3_wake == 0.0 and w3_tot == 1.0

    total_cost_3_windows = w1_tot + w2_tot + w3_tot
    assert total_cost_3_windows == 7.0, "Total cost should be 5 + 1 + 1 = 7.0"


def test_recovered_sensor_pays_wake_cost():
    cm = SensorCostModel(
        running_costs={"ankle": 1.0},
        default_wake_ratio=5.0,
    )
    # Ankle was in outage (unavailable), surviving prev was empty
    surviving_prev = set()  # forced off during outage
    curr_active = {"ankle"}  # selected after recovery
    c_run, c_wake, _, c_tot = cm.compute_step_cost(curr_active, surviving_prev)

    assert c_run == 1.0
    assert c_wake == 5.0, "Recovered sensor must incur wake-up cost when re-activated"
    assert c_tot == 6.0


def test_recomputed_cost_matches_online():
    cm = SensorCostModel(default_wake_ratio=3.0)
    history = [
        ({"hand"}, set()),
        ({"hand", "ankle"}, {"hand"}),
        ({"hand", "ankle"}, {"hand", "ankle"}),
        (set(), {"hand", "ankle"}),
        ({"ankle"}, set()),
    ]
    online_totals = []
    recomputed_totals = []

    for curr, prev in history:
        _, _, _, tot = cm.compute_step_cost(curr, prev)
        online_totals.append(tot)

        # Recompute from scratch
        newly_woken = curr - prev
        rec_run = sum(cm.running_costs[s] for s in curr)
        rec_wake = sum(cm.wake_up_costs[s] for s in newly_woken)
        recomputed_totals.append(rec_run + rec_wake)

    assert online_totals == recomputed_totals
