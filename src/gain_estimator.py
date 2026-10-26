"""
Marginal Gain Estimator Module
==============================
Estimates the expected information gain (loss reduction) from adding candidate
sensor j to current subset S for the next time window:
    gain(S, j) = log_loss(S) - log_loss(S ∪ {j})

Strict Information Boundary:
- Decision-time inputs to the gain estimator use ONLY information available
  before window t starts:
    1. Previous window's classifier confidence and entropy
    2. Previous active sensor subset
    3. Candidate sensor identifier and current subset composition
    4. Sensor availability flags
    5. Recent acquisition history
- The target window's sensor values and ground-truth label are NEVER exposed
  to the estimator during inference.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.subset_classifiers import (
    SubsetClassifierSuite,
    SUBSET_DEFINITIONS,
    SUBSET_NAMES,
    ALL_MODALITIES,
)

logger = logging.getLogger(__name__)


def build_gain_training_dataset(
    suite: SubsetClassifierSuite,
    train_df: pd.DataFrame,
    label_col: str = "activity_id",
) -> pd.DataFrame:
    """
    Construct training targets for marginal gain estimation.

    Parameters
    ----------
    suite : SubsetClassifierSuite
        Trained suite of all 8 subset classifiers.
    train_df : pd.DataFrame
        Chronologically sorted training windows (grouped by subject_id).
    label_col : str
        Target activity column.

    Returns
    -------
    pd.DataFrame
        Dataset where each row represents a (window t, subset S, candidate j) decision state
        with pre-window historical features and target realized gain.
    """
    records = []
    classes = suite.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Precompute probabilities for all subsets across train_df
    prob_cache: Dict[int, np.ndarray] = {}
    for s_id in range(8):
        prob_cache[s_id] = suite.predict_proba(s_id, train_df)

    # Compute cross-entropy loss for each sample under each subset
    # loss(y, p) = -log(p[y])
    loss_cache: Dict[int, np.ndarray] = {}
    y_vals = train_df[label_col].values
    for s_id in range(8):
        probs = prob_cache[s_id]
        losses = np.zeros(len(train_df), dtype=np.float64)
        for i, (y, p) in enumerate(zip(y_vals, probs)):
            c_idx = class_to_idx.get(y, -1)
            if c_idx >= 0:
                prob_y = np.clip(p[c_idx], 1e-7, 1.0)
                losses[i] = -np.log(prob_y)
            else:
                losses[i] = -np.log(1e-7)
        loss_cache[s_id] = losses

    # Map subsets to integer IDs for quick union lookup
    def get_subset_id(mods: Tuple[str, ...]) -> int:
        clean_mods = tuple(sorted(mods))
        for sid, def_mods in SUBSET_DEFINITIONS.items():
            if tuple(sorted(def_mods)) == clean_mods:
                return sid
        return 0

    # Group by subject to preserve chronological order and avoid cross-subject history
    for sub_id, sub_group in train_df.groupby("subject_id", sort=False):
        sub_indices = sub_group.index.values
        n_windows = len(sub_indices)
        if n_windows < 2:
            continue

        for step in range(1, n_windows):
            prev_idx = sub_indices[step - 1]
            curr_idx = sub_indices[step]

            # Features available before window t (derived from t-1)
            # Default previous reference model: Subset 7 (all sensors) or previous subset
            prev_prob = prob_cache[7][prev_idx]
            prev_conf = float(np.max(prev_prob))
            prev_entropy = float(-np.sum(prev_prob * np.log(np.clip(prev_prob, 1e-7, 1.0))))

            # Evaluate candidate sensors j for each subset S
            for s_id, current_mods in SUBSET_DEFINITIONS.items():
                current_set = set(current_mods)
                loss_S = loss_cache[s_id][curr_idx]

                for j in ALL_MODALITIES:
                    if j in current_set:
                        continue  # Already in subset

                    # Target subset S ∪ {j}
                    union_mods = tuple(sorted(list(current_set.union({j}))))
                    union_s_id = get_subset_id(union_mods)
                    loss_union = loss_cache[union_s_id][curr_idx]

                    # Ground truth target: gain(S, j) = loss(S) - loss(S ∪ {j})
                    realized_gain = float(loss_S - loss_union)

                    rec = {
                        "prev_confidence": prev_conf,
                        "prev_entropy": prev_entropy,
                        "current_subset_size": len(current_mods),
                        "has_hand": 1.0 if "hand" in current_set else 0.0,
                        "has_ankle": 1.0 if "ankle" in current_set else 0.0,
                        "has_heart_rate": 1.0 if "heart_rate" in current_set else 0.0,
                        "cand_is_hand": 1.0 if j == "hand" else 0.0,
                        "cand_is_ankle": 1.0 if j == "ankle" else 0.0,
                        "cand_is_hr": 1.0 if j == "heart_rate" else 0.0,
                        "realized_gain": realized_gain,
                    }
                    records.append(rec)

    return pd.DataFrame(records)


class MarginalGainEstimator:
    """Estimates expected loss reduction for candidate sensor j given subset S and history."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.feature_cols = [
            "prev_confidence",
            "prev_entropy",
            "current_subset_size",
            "has_hand",
            "has_ankle",
            "has_heart_rate",
            "cand_is_hand",
            "cand_is_ankle",
            "cand_is_hr",
        ]
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=self.alpha, random_state=self.random_state)),
        ])
        self.is_fitted = False

    def fit(self, training_data: pd.DataFrame) -> "MarginalGainEstimator":
        X = training_data[self.feature_cols].values
        y = training_data["realized_gain"].values
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Fitted MarginalGainEstimator on {len(training_data)} decision instances.")
        return self

    def predict_gain(
        self,
        prev_confidence: float,
        prev_entropy: float,
        current_subset: Tuple[str, ...],
        candidate_sensor: str,
    ) -> float:
        """
        Predict expected gain using only pre-window decision state.
        """
        if not self.is_fitted:
            raise RuntimeError("MarginalGainEstimator is not fitted.")

        current_set = set(current_subset)
        feature_vector = np.array([[
            prev_confidence,
            prev_entropy,
            float(len(current_subset)),
            1.0 if "hand" in current_set else 0.0,
            1.0 if "ankle" in current_set else 0.0,
            1.0 if "heart_rate" in current_set else 0.0,
            1.0 if candidate_sensor == "hand" else 0.0,
            1.0 if candidate_sensor == "ankle" else 0.0,
            1.0 if candidate_sensor == "heart_rate" else 0.0,
        ]], dtype=np.float64)

        pred = float(self.model.predict(feature_vector)[0])
        return pred

    def save(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Saved MarginalGainEstimator to {file_path}")

    @staticmethod
    def load(file_path: Union[str, Path]) -> "MarginalGainEstimator":
        return joblib.load(file_path)
