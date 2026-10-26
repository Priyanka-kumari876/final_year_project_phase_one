"""
Unit Tests for Subset Classifiers
=================================
Verifies:
1. Every subset classifier accesses only its permitted sensor group features.
2. The empty subset classifier predicts empirical class priors and does not rely on sensor features.
3. Probability predictions are valid distributions (sum to 1.0, non-negative).
"""

import pytest
import numpy as np
import pandas as pd

from src.subset_classifiers import (
    SubsetClassifierSuite,
    EmptySubsetClassifier,
    SUBSET_DEFINITIONS,
    SUBSET_NAMES,
)


def test_empty_subset_classifier():
    y_train = np.array([1, 1, 1, 2, 2])
    classes = np.array([1, 2])
    clf = EmptySubsetClassifier()
    clf.fit(y_train, classes)

    # Prior for class 1 should be 3/5 = 0.6, class 2 should be 2/5 = 0.4
    dummy_X = np.zeros((3, 10))
    probs = clf.predict_proba(dummy_X)

    assert probs.shape == (3, 2)
    assert np.allclose(probs[:, 0], 0.6, atol=1e-3)
    assert np.allclose(probs[:, 1], 0.4, atol=1e-3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_subset_models_feature_isolation():
    group_feats = {
        "hand": ["hand_acc_x_mean", "hand_acc_y_mean"],
        "ankle": ["ankle_acc_x_mean", "ankle_acc_y_mean"],
        "heart_rate": ["hr_mean", "hr_valid_fraction"],
    }
    suite = SubsetClassifierSuite(feature_columns_by_group=group_feats)

    # Synthetic train df
    np.random.seed(42)
    train_df = pd.DataFrame({
        "activity_id": [1, 2, 1, 2, 1, 2] * 5,
        "hand_acc_x_mean": np.random.randn(30),
        "hand_acc_y_mean": np.random.randn(30),
        "ankle_acc_x_mean": np.random.randn(30),
        "ankle_acc_y_mean": np.random.randn(30),
        "hr_mean": np.random.randn(30),
        "hr_valid_fraction": np.random.randn(30),
    })

    suite.fit(train_df, label_col="activity_id")

    # Subset 1 (Hand only): must ONLY use hand features
    assert suite.feature_subsets[1] == sorted(group_feats["hand"])
    # Subset 2 (Ankle only): must ONLY use ankle features
    assert suite.feature_subsets[2] == sorted(group_feats["ankle"])
    # Subset 3 (HR only): must ONLY use hr features
    assert suite.feature_subsets[3] == sorted(group_feats["heart_rate"])
    # Subset 4 (Hand + Ankle)
    assert suite.feature_subsets[4] == sorted(group_feats["hand"] + group_feats["ankle"])
    # Subset 0 (Empty)
    assert suite.feature_subsets[0] == []
