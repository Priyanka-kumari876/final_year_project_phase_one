"""
Chronological Simulator Module
==============================
Executes sequential multimodal acquisition experiments over continuous window streams.

Strict Event Order:
1. End of window t-1: Completed history (confidence, entropy, previous active set) is ready.
2. Start of window t: Declared availability flags U_t are observed (or hidden in stress test).
3. Sensor Selection: Policy selects feasible subset A_t ⊆ U_t.
4. Cost Accounting:
   - Running cost charged for each sensor in A_t.
   - Wake-up cost charged for sensors in A_t that were NOT awake in window t-1.
   - (In hidden outage mode, failed attempts charge attempt costs).
5. Data Acquisition: Features of window t are revealed ONLY for selected sensors A_t.
6. Prediction: Subset classifier f_{A_t} predicts activity y_t.
7. Logging: State, actions, costs, predictions, and errors are saved into an action log.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import logging
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, log_loss

from src.cost_model import SensorCostModel
from src.subset_classifiers import (
    SubsetClassifierSuite,
    SUBSET_DEFINITIONS,
    ALL_MODALITIES,
)
from src.acquisition_policy import BaseAcquisitionPolicy

logger = logging.getLogger(__name__)


def map_modality_set_to_subset_id(mods: Set[str]) -> int:
    """Map a set of active modality names to the corresponding subset ID (0 to 7)."""
    clean_mods = tuple(sorted(list(mods)))
    for sid, def_mods in SUBSET_DEFINITIONS.items():
        if tuple(sorted(list(def_mods))) == clean_mods:
            return sid
    return 0


class SimulationResult:
    """Stores logs and aggregated performance metrics from a simulation run."""

    def __init__(
        self,
        policy_name: str,
        action_log: pd.DataFrame,
        metrics: Dict[str, float],
        duty_cycles: Dict[str, float],
        wake_rate_per_1000: float,
    ):
        self.policy_name = policy_name
        self.action_log = action_log
        self.metrics = metrics
        self.duty_cycles = duty_cycles
        self.wake_rate_per_1000 = wake_rate_per_1000


def run_chronological_simulation(
    policy: BaseAcquisitionPolicy,
    suite: SubsetClassifierSuite,
    cost_model: SensorCostModel,
    windows_df: pd.DataFrame,
    availability_masks: Dict[str, np.ndarray],
    hidden_availability: bool = False,
    label_col: str = "activity_id",
    prob_cache: Optional[Dict[int, np.ndarray]] = None,
) -> SimulationResult:
    """
    Run chronological acquisition simulation on a sequence of windows.
    """
    policy.reset()
    n_windows = len(windows_df)
    action_records = []

    # Precompute subset probabilities in batch if not already cached (500x speedup)
    if prob_cache is None:
        prob_cache = {s: suite.predict_proba(s, windows_df) for s in range(8)}

    sub_ids = windows_df["subject_id"].values
    start_times = windows_df["start_time"].values
    end_times = windows_df["end_time"].values

    # Initial state before first window
    prev_conf = 1.0 / len(suite.classes_)  # default prior confidence
    prev_entropy = float(np.log(len(suite.classes_)))
    prev_pred = int(suite.classes_[0])
    prev_active: Set[str] = set()

    y_true_all = windows_df[label_col].values
    y_pred_all = np.zeros(n_windows, dtype=int)
    prob_all = np.zeros((n_windows, len(suite.classes_)), dtype=np.float64)

    t_start_sim = time.perf_counter()

    for t in range(n_windows):
        t_decision_start = time.perf_counter()

        # True availability at window t
        U_t = {s for s in ALL_MODALITIES if availability_masks[s][t]}

        # If a previously active sensor suffered an outage, it is forced off
        surviving_prev = prev_active.intersection(U_t)

        failed_attempts: Set[str] = set()

        if not hidden_availability:
            # Main experiment: Policy observes declared availability flags U_t
            A_t = policy.select_sensors(
                available_sensors=U_t,
                prev_confidence=prev_conf,
                prev_entropy=prev_entropy,
                prev_prediction=prev_pred,
            )
            # Enforce feasibility
            A_t = A_t.intersection(U_t)
        else:
            # Hidden outage stress test: Policy attempts to acquire sensors without knowing U_t
            A_attempt = policy.select_sensors(
                available_sensors=set(ALL_MODALITIES),
                prev_confidence=prev_conf,
                prev_entropy=prev_entropy,
                prev_prediction=prev_pred,
            )
            # Revealed outcome
            A_t = A_attempt.intersection(U_t)
            failed_attempts = A_attempt - U_t

        decision_latency_ms = (time.perf_counter() - t_decision_start) * 1000.0

        # Cost accounting
        c_run, c_wake, c_att, c_total = cost_model.compute_step_cost(
            current_active=A_t,
            previous_active=surviving_prev,
            failed_attempts=failed_attempts,
        )

        # Classification using selected subset model (fast cache lookup)
        subset_id = map_modality_set_to_subset_id(A_t)
        probs = prob_cache[subset_id][t]
        y_hat = int(suite.classes_[np.argmax(probs)])

        y_true_t = int(y_true_all[t])
        y_pred_all[t] = y_hat
        prob_all[t] = probs

        # Metrics for window t
        conf_t = float(np.max(probs))
        entropy_t = float(-np.sum(probs * np.log(np.clip(probs, 1e-7, 1.0))))
        class_idx = np.where(suite.classes_ == y_true_t)[0]
        prob_true = float(probs[class_idx[0]]) if len(class_idx) > 0 else 1e-7
        loss_t = float(-np.log(np.clip(prob_true, 1e-7, 1.0)))

        newly_woken = A_t - surviving_prev

        action_records.append({
            "window_idx": t,
            "subject_id": int(sub_ids[t]),
            "start_time": float(start_times[t]),
            "end_time": float(end_times[t]),
            "y_true": y_true_t,
            "y_pred": y_hat,
            "confidence": conf_t,
            "entropy": entropy_t,
            "loss": loss_t,
            "subset_id": subset_id,
            "selected_sensors": "+".join(sorted(A_t)) or "none",
            "num_sensors": len(A_t),
            "newly_woken": "+".join(sorted(newly_woken)) or "none",
            "num_woken": len(newly_woken),
            "running_cost": c_run,
            "wake_up_cost": c_wake,
            "attempt_cost": c_att,
            "total_cost": c_total,
            "decision_latency_ms": decision_latency_ms,
        })

        # Update historical state for window t+1
        prev_conf = conf_t
        prev_entropy = entropy_t
        prev_pred = y_hat
        prev_active = set(A_t)

    action_df = pd.DataFrame(action_records)

    # Calculate overall metrics
    macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_all, y_pred_all))
    acc = float(accuracy_score(y_true_all, y_pred_all))
    clf_err = float(1.0 - acc)

    clipped_prob = np.clip(prob_all, 1e-7, 1.0 - 1e-7)
    clipped_prob /= clipped_prob.sum(axis=1, keepdims=True)
    mean_log_loss = float(log_loss(y_true_all, clipped_prob, labels=suite.classes_))

    tot_run_cost = float(action_df["running_cost"].sum())
    tot_wake_cost = float(action_df["wake_up_cost"].sum())
    tot_att_cost = float(action_df["attempt_cost"].sum())
    tot_modeled_cost = float(action_df["total_cost"].sum())
    mean_cost_per_window = float(tot_modeled_cost / n_windows) if n_windows > 0 else 0.0

    tot_wake_ups = int(action_df["num_woken"].sum())
    wake_rate_per_1000 = float((tot_wake_ups / n_windows) * 1000.0) if n_windows > 0 else 0.0

    # Duty cycles
    duty_cycles: Dict[str, float] = {}
    for s in ALL_MODALITIES:
        active_count = sum(s in row["selected_sensors"].split("+") for _, row in action_df.iterrows())
        duty_cycles[f"duty_cycle_{s}"] = float(active_count / n_windows) if n_windows > 0 else 0.0

    metrics = {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "classification_error": round(clf_err, 4),
        "mean_log_loss": round(mean_log_loss, 4),
        "total_running_cost": round(tot_run_cost, 2),
        "total_wake_up_cost": round(tot_wake_cost, 2),
        "total_attempt_cost": round(tot_att_cost, 2),
        "total_modeled_cost": round(tot_modeled_cost, 2),
        "mean_cost_per_window": round(mean_cost_per_window, 4),
        "wake_ups_per_1000_windows": round(wake_rate_per_1000, 2),
        "mean_latency_ms": round(float(action_df["decision_latency_ms"].mean()), 3),
    }
    metrics.update(duty_cycles)

    return SimulationResult(
        policy_name=policy.name,
        action_log=action_df,
        metrics=metrics,
        duty_cycles=duty_cycles,
        wake_rate_per_1000=wake_rate_per_1000,
    )
