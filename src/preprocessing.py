"""
Preprocessing and Participant Splitting Module
==============================================
Provides strict, leakage-safe participant-level partitioning,
feature scaling, imputation, and class prior estimation.

Strict Leakage Rules:
1. Partitions are formed strictly by subject ID.
2. No row or window from a subject in the test set may ever appear in train or validation.
3. Preprocessing parameters (StandardScaler mean/std, imputation medians, class priors)
   are fitted EXCLUSIVELY on training participants.
4. Test set is strictly held out until final evaluation.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ParticipantSplitter:
    """Manages leakage-safe train/val/test splits by subject ID."""

    def __init__(
        self,
        train_subjects: List[int],
        val_subjects: List[int],
        test_subjects: List[int],
    ):
        self.train_subjects = sorted(list(set(train_subjects)))
        self.val_subjects = sorted(list(set(val_subjects)))
        self.test_subjects = sorted(list(set(test_subjects)))
        self._validate_disjoint()

    def _validate_disjoint(self):
        s_train = set(self.train_subjects)
        s_val = set(self.val_subjects)
        s_test = set(self.test_subjects)

        inter_train_val = s_train.intersection(s_val)
        inter_train_test = s_train.intersection(s_test)
        inter_val_test = s_val.intersection(s_test)

        if inter_train_val or inter_train_test or inter_val_test:
            raise ValueError(
                f"Data leakage detected in participant splits! Overlapping subjects: "
                f"Train/Val={inter_train_val}, Train/Test={inter_train_test}, Val/Test={inter_val_test}"
            )

    def split_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a window feature DataFrame into train, val, and test subsets."""
        if "subject_id" not in df.columns:
            raise KeyError("DataFrame must contain 'subject_id' column for participant splitting.")

        train_df = df[df["subject_id"].isin(self.train_subjects)].copy()
        val_df = df[df["subject_id"].isin(self.val_subjects)].copy()
        test_df = df[df["subject_id"].isin(self.test_subjects)].copy()

        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        logger.info(
            f"Participant split created: Train={len(train_df)} windows ({self.train_subjects}), "
            f"Val={len(val_df)} windows ({self.val_subjects}), Test={len(test_df)} windows ({self.test_subjects})"
        )
        return train_df, val_df, test_df

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            "train_subjects": self.train_subjects,
            "val_subjects": self.val_subjects,
            "test_subjects": self.test_subjects,
        }


class FeaturePreprocessor:
    """
    Fits scalers, imputation values, and class priors strictly on training data
    and applies them consistently across splits.
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        self.scaler: Optional[StandardScaler] = None
        self.impute_values: Dict[str, float] = {}
        self.class_priors: Dict[int, float] = {}
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, label_col: str = "activity_id") -> "FeaturePreprocessor":
        """Fit scaler, imputation, and class priors strictly on training data."""
        if train_df.empty:
            raise ValueError("Cannot fit FeaturePreprocessor on empty training DataFrame.")

        # Compute median imputation values for numeric features
        X_train_raw = train_df[self.feature_cols].copy()
        for col in self.feature_cols:
            median_val = float(X_train_raw[col].median(skipna=True))
            if np.isnan(median_val):
                median_val = 0.0
            self.impute_values[col] = median_val
            X_train_raw[col] = X_train_raw[col].fillna(median_val)

        # Fit StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_raw.values)

        # Compute class priors strictly on training labels
        labels = train_df[label_col].values
        unique_classes, counts = np.unique(labels, return_counts=True)
        self.classes_ = unique_classes
        total_samples = len(labels)
        self.class_priors = {
            int(cls): float(count / total_samples)
            for cls, count in zip(unique_classes, counts)
        }

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Impute and scale feature columns using fitted parameters."""
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transforming data.")

        X = df[self.feature_cols].copy()
        for col in self.feature_cols:
            X[col] = X[col].fillna(self.impute_values.get(col, 0.0))

        X_scaled = self.scaler.transform(X.values)
        return X_scaled

    def get_class_priors_vector(self, class_order: Optional[List[int]] = None) -> np.ndarray:
        """Return probability vector matching class priors."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted first.")
        if class_order is None:
            class_order = list(self.classes_)
        return np.array([self.class_priors.get(c, 0.0) for c in class_order], dtype=np.float64)
