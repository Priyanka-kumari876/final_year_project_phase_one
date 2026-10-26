"""
Baseline Policies Module
========================
Implements comparison baselines and mechanism ablations:
1. AlwaysAll: Always request all available sensors.
2. BestFixedSubset: Fixed sensor combination tuned on validation set.
3. RandomFeasible: Random feasible subset matching target duty cycle.
4. MyopicGain: State-agnostic acquisition ignoring wake-up costs in decision.
5. HistoryConfidence: Confidence-triggered sensor activation.
6. HistoryConfidenceKeepAlive: Confidence trigger with a K-window keep-alive timer.
7. SwitchingNoOutageReasoning: Ablation that ignores declared availability.
8. ProposedNoWakeCost: Proposed policy ablation with mu = 0 (no wake cost hysteresis).
"""

from typing import Dict, List, Optional, Set, Tuple
import logging
import numpy as np

from src.cost_model import SensorCostModel
from src.gain_estimator import MarginalGainEstimator
from src.acquisition_policy import BaseAcquisitionPolicy
from src.subset_classifiers import ALL_MODALITIES

logger = logging.getLogger(__name__)


class AlwaysAllPolicy(BaseAcquisitionPolicy):
    """Always selects all declared available sensors."""

    def __init__(self, cost_model: SensorCostModel):
        super().__init__(name="Always All Sensors", cost_model=cost_model)

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        self.active_sensors = set(available_sensors)
        return set(available_sensors)


class BestFixedSubsetPolicy(BaseAcquisitionPolicy):
    """Selects a fixed pre-configured subset of sensors, filtered by availability."""

    def __init__(self, target_subset: Set[str], cost_model: SensorCostModel, name: Optional[str] = None):
        policy_name = name or f"Fixed ({'+'.join(sorted(target_subset)) or 'Empty'})"
        super().__init__(name=policy_name, cost_model=cost_model)
        self.target_subset = set(target_subset)

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        chosen = self.target_subset.intersection(available_sensors)
        self.active_sensors = set(chosen)
        return set(chosen)


class RandomFeasiblePolicy(BaseAcquisitionPolicy):
    """Randomly selects feasible sensors with probability p_acq per sensor."""

    def __init__(
        self,
        cost_model: SensorCostModel,
        p_acq: float = 0.5,
        random_seed: int = 42,
        name: str = "Random Feasible",
    ):
        super().__init__(name=name, cost_model=cost_model)
        self.p_acq = p_acq
        self.rng = np.random.RandomState(random_seed)

    def reset(self) -> None:
        super().reset()

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        chosen = set()
        for s in sorted(list(available_sensors)):
            if self.rng.uniform(0.0, 1.0) < self.p_acq:
                chosen.add(s)
        self.active_sensors = set(chosen)
        return chosen


class MyopicGainPolicy(BaseAcquisitionPolicy):
    """
    Myopic Information Gain per cost:
    Selects sensor j whenever predicted gain > lambda * c_run,j.
    Ignores persistent sensor state (no wake-up cost considered in decision).
    """

    def __init__(
        self,
        cost_model: SensorCostModel,
        gain_estimator: MarginalGainEstimator,
        name: str = "Myopic Gain (State-Agnostic)",
        lambda_mult: float = 1.0,
    ):
        super().__init__(name=name, cost_model=cost_model)
        self.gain_estimator = gain_estimator
        self.lambda_mult = lambda_mult

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        selected: Set[str] = set()
        for sensor in ALL_MODALITIES:
            if sensor not in available_sensors:
                continue
            pred_gain = self.gain_estimator.predict_gain(
                prev_confidence=prev_confidence,
                prev_entropy=prev_entropy,
                current_subset=tuple(sorted(list(selected))),
                candidate_sensor=sensor,
            )
            # Threshold only considers running cost, ignoring whether it was awake or asleep
            threshold = self.lambda_mult * self.cost_model.lambda_weight * self.cost_model.running_costs.get(sensor, 1.0)
            if pred_gain > threshold:
                selected.add(sensor)

        chosen = selected.intersection(available_sensors)
        self.active_sensors = set(chosen)
        return chosen


class HistoryConfidencePolicy(BaseAcquisitionPolicy):
    """
    Confidence-Triggered Acquisition:
    When prediction confidence is below threshold, wakes extra sensors.
    """

    def __init__(
        self,
        cost_model: SensorCostModel,
        conf_threshold: float = 0.70,
        base_subset: Optional[Set[str]] = None,
        name: str = "History Confidence Trigger",
    ):
        super().__init__(name=name, cost_model=cost_model)
        self.conf_threshold = conf_threshold
        self.base_subset = base_subset or {"hand"}

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        if prev_confidence < self.conf_threshold:
            # Wake all available sensors
            chosen = set(available_sensors)
        else:
            chosen = self.base_subset.intersection(available_sensors)
        self.active_sensors = set(chosen)
        return chosen


class HistoryConfidenceKeepAlivePolicy(BaseAcquisitionPolicy):
    """
    Confidence-Triggered Acquisition with Keep-Alive Timeout:
    Once extra sensors are activated, keeps them awake for K consecutive windows
    to prevent costly high-frequency on/off thrashing.
    """

    def __init__(
        self,
        cost_model: SensorCostModel,
        conf_threshold: float = 0.70,
        keep_alive_windows: int = 4,
        base_subset: Optional[Set[str]] = None,
        name: str = "Confidence + Keep-Alive",
    ):
        super().__init__(name=name, cost_model=cost_model)
        self.conf_threshold = conf_threshold
        self.keep_alive_windows = keep_alive_windows
        self.base_subset = base_subset or {"hand"}
        self.sensor_timers: Dict[str, int] = {s: 0 for s in ALL_MODALITIES}

    def reset(self) -> None:
        super().reset()
        self.sensor_timers = {s: 0 for s in ALL_MODALITIES}

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        # Decrement keep-alive timers
        for s in list(self.sensor_timers.keys()):
            if self.sensor_timers[s] > 0:
                self.sensor_timers[s] -= 1

        trigger_active = prev_confidence < self.conf_threshold
        chosen = set(self.base_subset)

        if trigger_active:
            for s in ALL_MODALITIES:
                self.sensor_timers[s] = self.keep_alive_windows
                chosen.add(s)
        else:
            # Keep any sensor whose timer is still active
            for s, t_left in self.sensor_timers.items():
                if t_left > 0:
                    chosen.add(s)

        feasible = chosen.intersection(available_sensors)
        # Clear timers for sensors that experienced an outage
        for s in ALL_MODALITIES:
            if s not in available_sensors:
                self.sensor_timers[s] = 0

        self.active_sensors = set(feasible)
        return feasible


class ProposedNoWakeCostPolicy(BaseAcquisitionPolicy):
    """
    Ablation of Proposed Policy with mu = 0:
    Uses marginal gain predictions but charges/weights mu = 0 for wake costs
    when making decisions, removing the hysteresis gap.
    """

    def __init__(
        self,
        cost_model: SensorCostModel,
        gain_estimator: MarginalGainEstimator,
        lambda_mult: float = 1.0,
        name: str = "Ablation (No Wake Cost / mu=0)",
    ):
        super().__init__(name=name, cost_model=cost_model)
        self.gain_estimator = gain_estimator
        self.lambda_mult = lambda_mult

    def select_sensors(
        self,
        available_sensors: Set[str],
        prev_confidence: float,
        prev_entropy: float,
        prev_prediction: int,
    ) -> Set[str]:
        selected: Set[str] = set()
        for sensor in ALL_MODALITIES:
            if sensor not in available_sensors:
                continue

            pred_gain = self.gain_estimator.predict_gain(
                prev_confidence=prev_confidence,
                prev_entropy=prev_entropy,
                current_subset=tuple(sorted(list(selected))),
                candidate_sensor=sensor,
            )
            # mu = 0 so no wake cost in threshold
            threshold = self.lambda_mult * self.cost_model.lambda_weight * self.cost_model.running_costs.get(sensor, 1.0)
            if pred_gain > threshold:
                selected.add(sensor)

        chosen = selected.intersection(available_sensors)
        self.active_sensors = set(chosen)
        return chosen
