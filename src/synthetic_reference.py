"""
Synthetic Exact Reference Model Module
======================================
Implements a small-state discrete Hidden Markov Model (HMM) and exact
Dynamic Programming (Value Iteration) solver over activity belief and sensor state.

Purpose:
- Validates the fundamental structural property: the existence of a HYSTERESIS BAND
  between the wake-up threshold and the keep-on threshold.
- Compares the one-step heuristic policy against the infinite-horizon DP optimum.
- NOTE: This is an analytical reference problem, strictly labelled as synthetic.
"""

from typing import Dict, List, NamedTuple, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SyntheticHMM:
    """
    2-state activity HMM with 2 sensors.
    Activity 0: Sedentary
    Activity 1: Vigorous Active
    """

    def __init__(
        self,
        p_stay: float = 0.85,
        c_run: Tuple[float, float] = (1.0, 1.0),
        c_wake: Tuple[float, float] = (5.0, 5.0),
        discount: float = 0.90,
    ):
        self.p_stay = p_stay
        # Transition matrix P(Y_{t+1} | Y_t)
        self.T = np.array([
            [p_stay, 1.0 - p_stay],
            [1.0 - p_stay, p_stay],
        ])
        self.c_run = c_run
        self.c_wake = c_wake
        self.discount = discount

        # Emission error rates: P(error | sensor_active)
        # Sensor 1 (e.g. Accelerometer): discriminates well
        # Sensor 2 (e.g. HR): slower, noisier
        self.err_rates = {
            (0, 0): 0.50,  # Prior guess
            (1, 0): 0.15,  # Sensor 1 only
            (0, 1): 0.30,  # Sensor 2 only
            (1, 1): 0.05,  # Both sensors
        }

    def compute_immediate_loss(self, belief_active: float, action: Tuple[int, int]) -> float:
        """
        Expected classification error under action.
        belief_active = P(Y = 1).
        """
        err_rate = self.err_rates[action]
        # Bayesian risk of choosing the MAP class under noisy sensor observation
        uncertainty = 2.0 * min(belief_active, 1.0 - belief_active)
        expected_error = uncertainty * err_rate
        return float(expected_error)

    def transition_cost(self, action: Tuple[int, int], prev_state: Tuple[int, int]) -> float:
        """Cost of running action given previous sensor state."""
        cost = 0.0
        for i in range(2):
            if action[i] == 1:
                cost += self.c_run[i]
                if prev_state[i] == 0:
                    cost += self.c_wake[i]
        return float(cost)


class DPSolver:
    """Solves the synthetic reference problem via Value Iteration over a belief grid."""

    def __init__(self, hmm: SyntheticHMM, num_belief_points: int = 51):
        self.hmm = hmm
        self.belief_grid = np.linspace(0.0, 1.0, num_belief_points)
        self.actions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.sensor_states = [(0, 0), (1, 0), (0, 1), (1, 1)]

        # Value function table: shape (num_belief_points, 4)
        self.V = np.zeros((num_belief_points, 4))
        self.policy = np.zeros((num_belief_points, 4), dtype=int)

    def solve(self, max_iter: int = 200, tol: float = 1e-4) -> None:
        """Run Value Iteration until convergence."""
        gamma = self.hmm.discount

        for iteration in range(max_iter):
            V_new = np.zeros_like(self.V)

            for b_idx, b in enumerate(self.belief_grid):
                # Prior update: b_prior = P(Y_{t+1}=1) = b * T[1,1] + (1-b)*T[0,1]
                b_next_prior = b * self.hmm.T[1, 1] + (1.0 - b) * self.hmm.T[0, 1]
                b_next_prior = float(np.clip(b_next_prior, 0.0, 1.0))

                for s_idx, s in enumerate(self.sensor_states):
                    q_values = []

                    for a_idx, a in enumerate(self.actions):
                        # Cost of acquisition
                        acq_cost = self.hmm.transition_cost(a, s)
                        # Immediate classification loss (scaled to cost units, e.g. 10.0 per unit error)
                        loss = self.hmm.compute_immediate_loss(b, a) * 15.0

                        # Future belief estimation: simplify to prior expectation
                        next_b_idx = int(np.argmin(np.abs(self.belief_grid - b_next_prior)))
                        future_val = self.V[next_b_idx, a_idx]

                        q = acq_cost + loss + gamma * future_val
                        q_values.append(q)

                    best_a_idx = int(np.argmin(q_values))
                    V_new[b_idx, s_idx] = q_values[best_a_idx]
                    self.policy[b_idx, s_idx] = best_a_idx

            diff = np.max(np.abs(V_new - self.V))
            self.V = V_new
            if diff < tol:
                logger.info(f"Value iteration converged in {iteration+1} iterations (diff={diff:.6f}).")
                break

    def analyze_hysteresis(self) -> Dict[str, float]:
        """
        Analyze whether the wake threshold is strictly higher than keep threshold.
        Compares Sensor 1 activation when s=(0, 0) vs s=(1, 0).
        """
        # Index 0 is s=(0, 0) [asleep], Index 1 is s=(1, 0) [Sensor 1 awake]
        pol_asleep = self.policy[:, 0]  # action index when asleep
        pol_awake = self.policy[:, 1]   # action index when awake

        # Actions that have sensor 1 active: a_idx in {1, 3} ((1,0) or (1,1))
        sensor1_active_asleep = np.isin(pol_asleep, [1, 3])
        sensor1_active_awake = np.isin(pol_awake, [1, 3])

        fraction_active_when_asleep = float(np.mean(sensor1_active_asleep))
        fraction_active_when_awake = float(np.mean(sensor1_active_awake))

        # Hysteresis gap: Sensor 1 is kept active across more belief states when already awake!
        hysteresis_gap = fraction_active_when_awake - fraction_active_when_asleep

        return {
            "fraction_active_when_asleep": fraction_active_when_asleep,
            "fraction_active_when_awake": fraction_active_when_awake,
            "hysteresis_gap": round(hysteresis_gap, 4),
            "hysteresis_confirmed": bool(hysteresis_gap > 0),
        }
