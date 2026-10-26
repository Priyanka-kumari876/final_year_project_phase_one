"""
Unit Tests for Outage Simulation
================================
Verifies:
1. Paired masks generate identical sequences under the same seed.
2. Distinct seeds produce distinct outage realizations.
3. Availability sequences contain valid boolean states.
"""

import pytest
import numpy as np

from src.outage_simulator import MarkovOutageSimulator, get_predefined_scenarios


def test_paired_outage_masks_are_identical():
    sim = MarkovOutageSimulator(p_fail=0.05, p_recover=0.20, random_seed=42)
    masks_a = sim.generate_availability_sequence(100, seed=12345)
    masks_b = sim.generate_availability_sequence(100, seed=12345)

    for sensor in masks_a:
        assert np.array_equal(masks_a[sensor], masks_b[sensor]), (
            f"Paired masks for {sensor} must be strictly identical under identical seed"
        )


def test_distinct_seeds_differ():
    sim = MarkovOutageSimulator(p_fail=0.10, p_recover=0.30)
    m1 = sim.generate_availability_sequence(200, seed=111)
    m2 = sim.generate_availability_sequence(200, seed=222)

    # They should not be identical
    is_diff = False
    for s in m1:
        if not np.array_equal(m1[s], m2[s]):
            is_diff = True
            break
    assert is_diff, "Different seeds must produce different outage patterns"


def test_no_outage_scenario_always_available():
    scenarios = get_predefined_scenarios()
    no_out = scenarios["no_outage"]
    sim = MarkovOutageSimulator(p_fail=no_out["p_fail"], p_recover=no_out["p_recover"])
    m = sim.generate_availability_sequence(100)

    for s in m:
        assert np.all(m[s]), f"In no_outage scenario, {s} must remain 100% available"
