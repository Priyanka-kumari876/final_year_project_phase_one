"""
Acquisition Policies Module
===========================
Implements the proposed Switching-Aware Multimodal Acquisition Controller
with persistent sensor state and hysteresis, along with policy interfaces.

Decision Rule:
Select candidate sensor j for window t if:
    predicted_gain(history, S, j) > lambda * c_run,j + mu * c_wake,j * I[j was not awake]

Hysteresis Principle:
- If sensor j is already awake: threshold is lambda * c_run,j
- If sensor j is asleep: threshold is lambda * c_run,j + mu * c_wake,j
"""

from typing import Dict, List, Optional, Set, Tuple
import logging
import numpy as np

from src.cost_model import SensorCostModel
from src.gain_estimator import MarginalGainEstimator
from src.subset_classifiers import ALL_MODALITIES

logger = logging.getLogger(__name__)


class BaseAcquisitionPolicy:
    """Abstract base class for active multimodal acquisition policies."""

    def __init__(self, name: str, cost_model: SensorCostModel):
        self.name = name
        self.cost_model = cost_model
        self.active_sensors: Set[str] = set()

    def reset(self) -> None:
        """Reset internal memory and persistent state at the start of a session."""
        self.active_sensors = set()

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        """
        Choose sensor subset A_t for window t before observing target window features.

        Parameters
        ----------
        available_sensors : Set[str]
            Declared available sensors U_t for the current window.
        prev_confidence : float
            Classifier confidence from window t-1.
        prev_entropy : float
            Prediction entropy from window t-1.
        prev_prediction : int
            Predicted activity from window t-1.

        Returns
        -------
        Set[str]
            Subset of sensors A_t to activate, guaranteed to be a subset of available_sensors.
        """
        raise NotImplementedError


class SwitchingAwarePolicy(BaseAcquisitionPolicy):
    """
    Proposed switching-aware controller featuring persistent sensor state,
    hysteresis thresholds, and declared outage handling.
    """

    def __init__(
        self,
        cost_model: SensorCostModel,
        gain_estimator: MarginalGainEstimator,
        name: str = "Proposed (Switching-Aware)",
        lambda_mult: float = 1.0,
        mu_mult: float = 1.0,
    ):
        super().__init__(name=name, cost_model=cost_model)
        self.gain_estimator = gain_estimator
        self.lambda_mult = lambda_mult
        self.mu_mult = mu_mult

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        # If any previously active sensor suffered an outage, it is forced off
        surviving_active = self.active_sensors.intersection(available_sensors)

        selected_subset: Set[str] = set()

        for sensor in ALL_MODALITIES:
            # Cannot select unavailable sensor
            if sensor not in available_sensors:
                continue

            was_awake = sensor in surviving_active

            # Marginal gain prediction
            current_tuple = tuple(sorted(list(selected_subset)))
            pred_gain = self.gain_estimator.predict_gain(
                prev_confidence=prev_confidence,
                prev_entropy=prev_entropy,
                current_subset=current_tuple,
                candidate_sensor=sensor,
            )

            # Hysteresis decision rule
            c_run = self.cost_model.running_costs.get(sensor, 1.0)
            c_wake = self.cost_model.wake_up_costs.get(sensor, 0.0)

            if was_awake:
                threshold = self.lambda_mult * self.cost_model.lambda_weight * c_run
            else:
                threshold = (
                    self.lambda_mult * self.cost_model.lambda_weight * c_run
                    + self.mu_mult * self.cost_model.mu_weight * c_wake
                )

            if pred_gain > threshold:
                selected_subset.add(sensor)

        # Enforce that selected subset is strictly a subset of available sensors
        feasible_selected = selected_subset.intersection(available_sensors)
        self.active_sensors = set(feasible_selected)
        return feasible_selected
