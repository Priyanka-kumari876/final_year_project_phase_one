"""
Sensor Outage Simulator Module
==============================
Generates reproducible, controlled sensor outage processes using
a two-state Markov chain:
    Available (1) --p_fail--> Unavailable (0)
    Unavailable (0) --p_recover--> Available (1)

Strict Experimental Rules:
1. Paired masks: Exactly identical availability realizations are applied to all
   policies compared under the same experimental seed.
2. Observed vs Hidden availability:
   - Observed: Availability flags U_t are revealed at the start of window t before
     acquisition. Unavailable sensors cannot be selected.
   - Hidden: Availability is hidden until read is attempted. Failed reads return no data
     and pay an attempt cost.
3. Outage Recovery Convention:
   - A sensor recovering from an outage returns in a SLEEPING state.
   - Re-activating a recovered sensor incurs the full wake-up cost.
"""

from typing import Dict, List, Optional, Set, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALL_SENSORS: List[str] = ["hand", "ankle", "heart_rate"]


class MarkovOutageSimulator:
    """Generates 2-state Markov availability sequences for sensor modalities."""

    def __init__(
        self,
        p_fail: float = 0.05,
        p_recover: float = 0.20,
        sensors: Optional[List[str]] = None,
        random_seed: int = 42,
    ):
        self.p_fail = float(p_fail)
        self.p_recover = float(p_recover)
        self.sensors = sensors or ALL_SENSORS
        self.random_seed = int(random_seed)

    def generate_availability_sequence(
        self,
        num_windows: int,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate paired boolean availability masks across all windows for each sensor.

        Parameters
        ----------
        num_windows : int
            Number of consecutive time windows in the session.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping sensor name -> 1D boolean array of shape (num_windows,),
            where True = Available, False = In Outage.
        """
        rng = np.random.RandomState(self.random_seed if seed is None else seed)
        availability: Dict[str, np.ndarray] = {}

        for sensor in self.sensors:
            mask = np.ones(num_windows, dtype=bool)
            is_available = True

            for t in range(num_windows):
                if is_available:
                    # Transition to unavailable with probability p_fail
                    if rng.uniform(0.0, 1.0) < self.p_fail:
                        is_available = False
                else:
                    # Transition to available with probability p_recover
                    if rng.uniform(0.0, 1.0) < self.p_recover:
                        is_available = True
                mask[t] = is_available

            availability[sensor] = mask

        return availability


def get_predefined_scenarios() -> Dict[str, Dict[str, float]]:
    """Return dictionary of standard research outage scenarios."""
    return {
        "no_outage": {"p_fail": 0.0, "p_recover": 1.0},
        "rare_independent": {"p_fail": 0.02, "p_recover": 0.80},
        "bursty_short": {"p_fail": 0.05, "p_recover": 0.40},
        "bursty_long": {"p_fail": 0.05, "p_recover": 0.10},
    }
