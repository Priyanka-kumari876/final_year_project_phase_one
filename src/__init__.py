"""
P10 Multimodal Acquisition Source Package.
=========================================
Core scientific modules for dynamic multimodal acquisition with sensor wake-up costs and outages:

- data_loader: PAMAP2 raw ingestion and channel mapping
- preprocessing: Missing value imputation, signal cleaning, and participant splitting
- windowing: Fixed-grid sliding windowing and transition boundary protection
- feature_extraction: Time, frequency (FFT), and physiological feature extraction
- subset_classifiers: 8 isolated power-set sensor-subset classifiers
- gain_estimator: State-conditioned marginal information gain estimation
- cost_model: Running, wake-up switching, and attempt cost calculations
- acquisition_policy: Switching-aware hysteresis acquisition controller
- outage_simulator: 2-state discrete-time Gilbert-Elliott Markov outage simulator
- baselines: Standard benchmark policies (Always-All, Single-Sensor, Myopic Greedy, Heuristics)
- simulator: Chronological forward simulation engine with probability caching
- synthetic_reference: Analytical 2-state HMM & Dynamic Programming Value Iteration solver
- plotting: Publication-grade visualization suite (Pareto frontier, ratio sweeps, outage bars)
"""

from . import data_loader
from . import preprocessing
from . import windowing
from . import feature_extraction
from . import subset_classifiers
from . import gain_estimator
from . import cost_model
from . import acquisition_policy
from . import outage_simulator
from . import baselines
from . import simulator
from . import synthetic_reference
from . import plotting

__all__ = [
    "data_loader",
    "preprocessing",
    "windowing",
    "feature_extraction",
    "subset_classifiers",
    "gain_estimator",
    "cost_model",
    "acquisition_policy",
    "outage_simulator",
    "baselines",
    "simulator",
    "synthetic_reference",
    "plotting",
]
