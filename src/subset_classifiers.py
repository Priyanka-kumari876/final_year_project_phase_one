"""
Subset Classifiers Module
=========================
Trains and manages 8 compact classifiers corresponding to all subsets
of the 3 physical sensor modalities:
  0: Empty subset (predicts training class priors)
  1: {Hand}
  2: {Ankle}
  3: {Heart Rate}
  4: {Hand, Ankle}
  5: {Hand, Heart Rate}
  6: {Ankle, Heart Rate}
  7: {Hand, Ankle, Heart Rate}

Strict Leakage Rules:
- Only features belonging to the selected sensor groups in subset S are used.
- Imputers and scalers are fitted exclusively on training data.
- Empty subset model outputs empirical class priors computed on training split.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import logging
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Canonical subset definitions (0 to 7)
ALL_MODALITIES: List[str] = ["hand", "ankle", "heart_rate"]

SUBSET_DEFINITIONS: Dict[int, Tuple[str, ...]] = {
    0: (),
    1: ("hand",),
    2: ("ankle",),
    3: ("heart_rate",),
    4: ("hand", "ankle"),
    5: ("hand", "heart_rate"),
    6: ("ankle", "heart_rate"),
    7: ("hand", "ankle", "heart_rate"),
}

SUBSET_NAMES: Dict[int, str] = {
    0: "Empty (Priors)",
    1: "Hand Only",
    2: "Ankle Only",
    3: "HR Only",
    4: "Hand + Ankle",
    5: "Hand + HR",
    6: "Ankle + HR",
    7: "All Sensors (Hand+Ankle+HR)",
}


class EmptySubsetClassifier:
    """Classifier for Subset 0 that always predicts the training class priors."""

    def __init__(self):
        self.classes_: np.ndarray = np.array([])
        self.priors_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, y: np.ndarray, classes: np.ndarray) -> "EmptySubsetClassifier":
        self.classes_ = np.array(classes)
        n_total = len(y)
        counts = [np.sum(y == c) for c in self.classes_]
        self.priors_ = np.array([c / n_total for c in counts], dtype=np.float64)
        # Avoid exact zero probabilities for log_loss stability
        self.priors_ = np.clip(self.priors_, 1e-7, 1.0)
        self.priors_ /= np.sum(self.priors_)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EmptySubsetClassifier is not fitted.")
        n_samples = len(X)
        return np.tile(self.priors_, (n_samples, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)
        best_idx = np.argmax(prob, axis=1)
        return self.classes_[best_idx]


class SubsetClassifierSuite:
    """Suite containing all 8 trained subset models with feature mappings."""

    def __init__(
        self,
        feature_columns_by_group: Dict[str, List[str]],
        classifier_type: str = "logistic_regression",
        random_state: int = 42,
    ):
        self.feature_columns_by_group = feature_columns_by_group
        self.classifier_type = classifier_type
        self.random_state = random_state

        self.models: Dict[int, Union[LogisticRegression, EmptySubsetClassifier]] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.impute_medians: Dict[int, Dict[str, float]] = {}
        self.feature_subsets: Dict[int, List[str]] = {}
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def _get_features_for_subset(self, subset_id: int) -> List[str]:
        modalities = SUBSET_DEFINITIONS[subset_id]
        cols = []
        for mod in modalities:
            cols.extend(self.feature_columns_by_group.get(mod, []))
        return sorted(cols)

    def fit(
        self,
        train_df: pd.DataFrame,
        label_col: str = "activity_id",
        protocol_classes: Optional[List[int]] = None,
    ) -> "SubsetClassifierSuite":
        """
        Train all 8 subset classifiers strictly on training participant data.
        """
        if train_df.empty:
            raise ValueError("Training data cannot be empty.")

        y_train = train_df[label_col].values
        if protocol_classes is not None:
            self.classes_ = np.array(sorted(protocol_classes))
        else:
            self.classes_ = np.array(sorted(np.unique(y_train)))

        logger.info(f"Training 8 subset classifiers on {len(train_df)} samples across classes: {self.classes_}")

        for s_id in range(8):
            feat_cols = self._get_features_for_subset(s_id)
            self.feature_subsets[s_id] = feat_cols

            if s_id == 0:
                # Subset 0: Empty subset
                model = EmptySubsetClassifier()
                model.fit(y_train, self.classes_)
                self.models[s_id] = model
                continue

            # Compute medians strictly on train
            X_raw = train_df[feat_cols].copy()
            medians = {}
            for col in feat_cols:
                m = float(X_raw[col].median(skipna=True))
                if np.isnan(m):
                    m = 0.0
                medians[col] = m
                X_raw[col] = X_raw[col].fillna(m)
            self.impute_medians[s_id] = medians

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw.values)
            self.scalers[s_id] = scaler

            if self.classifier_type == "logistic_regression":
                clf = LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    random_state=self.random_state,
                    solver="lbfgs",
                )
            else:
                raise ValueError(f"Unknown classifier type: {self.classifier_type}")

            clf.fit(X_scaled, y_train)
            self.models[s_id] = clf
            logger.info(f"Trained Subset {s_id} ({SUBSET_NAMES[s_id]}) with {len(feat_cols)} features.")

        self.is_fitted = True
        return self

    def predict_proba(self, subset_id: int, df: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for a given subset model."""
        if not self.is_fitted:
            raise RuntimeError("SubsetClassifierSuite is not fitted.")

        if subset_id == 0:
            return self.models[0].predict_proba(np.zeros((len(df), 1)))

        feat_cols = self.feature_subsets[subset_id]
        X = df[feat_cols].copy()
        for col in feat_cols:
            X[col] = X[col].fillna(self.impute_medians[subset_id].get(col, 0.0))

        X_scaled = self.scalers[subset_id].transform(X.values)
        model = self.models[subset_id]
        raw_prob = model.predict_proba(X_scaled)

        # Align probabilities to canonical classes_ order
        if np.array_equal(model.classes_, self.classes_):
            return raw_prob

        full_prob = np.zeros((len(df), len(self.classes_)), dtype=np.float64)
        for idx, cls in enumerate(model.classes_):
            target_idx = np.where(self.classes_ == cls)[0]
            if len(target_idx) > 0:
                full_prob[:, target_idx[0]] = raw_prob[:, idx]
        return full_prob

    def predict(self, subset_id: int, df: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(subset_id, df)
        best_indices = np.argmax(prob, axis=1)
        return self.classes_[best_indices]

    def evaluate_subset(
        self, subset_id: int, df: pd.DataFrame, label_col: str = "activity_id"
    ) -> Dict[str, Union[float, Dict[int, float]]]:
        """
        Evaluate a subset model on a dataset.
        Returns log_loss, macro_f1, balanced_accuracy, classification_error, and per_class_recall.
        """
        y_true = df[label_col].values
        prob = self.predict_proba(subset_id, df)
        preds = self.predict(subset_id, df)

        # Compute log loss (clip to avoid inf)
        prob_clipped = np.clip(prob, 1e-7, 1.0 - 1e-7)
        prob_clipped /= prob_clipped.sum(axis=1, keepdims=True)
        ll = float(log_loss(y_true, prob_clipped, labels=self.classes_))

        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        balanced_acc = float(balanced_accuracy_score(y_true, preds))
        acc = float(accuracy_score(y_true, preds))
        clf_err = float(1.0 - acc)

        # Per-class recall
        recalls = recall_score(y_true, preds, labels=self.classes_, average=None, zero_division=0)
        per_class_recall = {int(cls): float(r) for cls, r in zip(self.classes_, recalls)}

        return {
            "subset_id": subset_id,
            "subset_name": SUBSET_NAMES[subset_id],
            "num_sensors": len(SUBSET_DEFINITIONS[subset_id]),
            "sensors": list(SUBSET_DEFINITIONS[subset_id]),
            "log_loss": round(ll, 4),
            "macro_f1": round(macro_f1, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "classification_error": round(clf_err, 4),
            "per_class_recall": per_class_recall,
        }

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save the fitted model suite to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        suite_file = out_path / "subset_classifiers.joblib"
        joblib.dump(self, suite_file)
        logger.info(f"Saved SubsetClassifierSuite to {suite_file}")

    @staticmethod
    def load(file_path: Union[str, Path]) -> "SubsetClassifierSuite":
        return joblib.load(file_path)
