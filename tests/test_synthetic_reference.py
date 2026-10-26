"""
Unit Tests for Synthetic Exact DP Reference Model
=================================================
Verifies:
1. Dynamic programming value iteration converges to a stable policy.
2. The analytical hysteresis gap is strictly positive (wake threshold > keep threshold).
"""

import pytest
from src.synthetic_reference import SyntheticHMM, DPSolver


def test_dp_solver_convergence_and_hysteresis():
    hmm = SyntheticHMM(p_stay=0.88, c_run=(1.0, 1.0), c_wake=(5.0, 5.0), discount=0.92)
    solver = DPSolver(hmm, num_belief_points=31)
    solver.solve(max_iter=100, tol=1e-3)

    analysis = solver.analyze_hysteresis()

    assert analysis["hysteresis_confirmed"] is True
    assert analysis["hysteresis_gap"] > 0.0, "Wake threshold must be strictly higher than keep threshold"
    assert analysis["fraction_active_when_awake"] > analysis["fraction_active_when_asleep"]
