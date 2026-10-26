r"""
Cost Model Module
=================
Tracks and computes acquisition costs for multimodal sensor systems:
1. Running Cost: Paid for every active sensor in each time window.
2. Wake-up Cost: Paid ONLY when a sensor transitions from OFF (sleeping) to ON.
3. Attempt Cost: Incurred in hidden-availability stress tests when an attempted read fails.

Mathematical Formulation:
Total Cost for window t:
    Cost(t) = sum_{j in A_t} c_run,j + sum_{j in A_t \ A_{t-1}} c_wake,j + sum_{j in F_t} c_attempt,j
"""

from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class SensorCostModel:
    """Computes and tracks running, wake-up, and attempt costs for sensor groups."""

    def __init__(
        self,
        running_costs: Optional[Dict[str, float]] = None,
        wake_ratios: Optional[Dict[str, float]] = None,
        default_wake_ratio: float = 5.0,
        attempt_cost: float = 0.2,
        lambda_weight: float = 1.0,
        mu_weight: float = 1.0,
    ):
        # Default running costs in experimental cost units
        self.running_costs = running_costs or {
            "hand": 1.0,
            "ankle": 1.0,
            "heart_rate": 0.5,
        }
        self.default_wake_ratio = default_wake_ratio
        self.wake_ratios = wake_ratios or {
            sensor: default_wake_ratio for sensor in self.running_costs
        }
        self.wake_up_costs = {
            s: self.running_costs[s] * self.wake_ratios.get(s, self.default_wake_ratio)
            for s in self.running_costs
        }
        self.attempt_cost = attempt_cost
        self.lambda_weight = lambda_weight
        self.mu_weight = mu_weight

    def update_wake_ratio(self, ratio: float) -> None:
        """Update wake-to-running cost ratio across all sensors."""
        self.default_wake_ratio = ratio
        for s in self.running_costs:
            self.wake_ratios[s] = ratio
            self.wake_up_costs[s] = self.running_costs[s] * ratio

    def compute_step_cost(
        self,
        current_active: Set[str],
        previous_active: Set[str],
        failed_attempts: Optional[Set[str]] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Compute cost components for a single window transition.

        Parameters
        ----------
        current_active : Set[str]
            Sensors selected and active in the current window.
        previous_active : Set[str]
            Sensors that were active in the immediately preceding window.
        failed_attempts : Set[str], optional
            Sensors where read was attempted but failed (hidden availability).

        Returns
        -------
        running_cost : float
            Sum of running costs for current_active sensors.
        wake_up_cost : float
            Sum of wake-up costs for newly activated sensors (current_active - previous_active).
        attempt_cost : float
            Cost of failed read attempts.
        total_modeled_cost : float
            Total unweighted modeled cost (running + wake_up + attempt).
        """
        # Running cost: 1 unit per selected sensor per window
        c_run = sum(self.running_costs.get(s, 1.0) for s in current_active)

        # Wake-up cost: charged ONLY when transitioning from sleeping to awake
        newly_woken = current_active - previous_active
        c_wake = sum(self.wake_up_costs.get(s, 0.0) for s in newly_woken)

        # Attempt cost (for hidden outage reads)
        c_att = 0.0
        if failed_attempts:
            c_att = len(failed_attempts) * self.attempt_cost

        c_total = c_run + c_wake + c_att
        return float(c_run), float(c_wake), float(c_att), float(c_total)

    def get_keep_threshold(self, sensor: str) -> float:
        """Cost threshold to keep an already awake sensor on."""
        return self.lambda_weight * self.running_costs.get(sensor, 1.0)

    def get_wake_threshold(self, sensor: str) -> float:
        """Cost threshold to wake a sleeping sensor."""
        return (
            self.lambda_weight * self.running_costs.get(sensor, 1.0)
            + self.mu_weight * self.wake_up_costs.get(sensor, 0.0)
        )
