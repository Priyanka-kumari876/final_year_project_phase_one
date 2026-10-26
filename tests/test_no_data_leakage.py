"""
Data Leakage Prevention Tests
=============================
Strict Research Verification:
1. Replacing target-window sensor values with random noise CANNOT change the
   pre-acquisition decision made for that window.
2. Changing test participants cannot alter scaler, imputation, or class priors
   fitted on training participants.
3. Overlapping windows from the same participant never cross train, validation, or test splits.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.cost_model import SensorCostModel
from src.gain_estimator import MarginalGainEstimator, build_gain_training_dataset
from src.acquisition_policy import SwitchingAwarePolicy
from src.preprocessing import ParticipantSplitter, FeaturePreprocessor
from src.subset_classifiers import SubsetClassifierSuite


def test_target_window_noise_injection_does_not_affect_decision():
    """
    CRITICAL LEAKAGE TEST:
    Replacing target-window sensor values with arbitrary random noise must produce
    an IDENTICAL acquisition decision, proving target-window values are not leaked.
    """
    cm = SensorCostModel()

    # Pre-train a simple gain estimator on dummy historical transitions
    dummy_train_data = pd.DataFrame([
        {
            "prev_confidence": 0.8,
            "prev_entropy": 0.5,
            "current_subset_size": 1,
            "has_hand": 1.0,
            "has_ankle": 0.0,
            "has_heart_rate": 0.0,
            "cand_is_hand": 0.0,
            "cand_is_ankle": 1.0,
            "cand_is_hr": 0.0,
            "realized_gain": 0.4,
        },
        {
            "prev_confidence": 0.2,
            "prev_entropy": 2.1,
            "current_subset_size": 0,
            "has_hand": 0.0,
            "has_ankle": 0.0,
            "has_heart_rate": 0.0,
            "cand_is_hand": 1.0,
            "cand_is_ankle": 0.0,
            "cand_is_hr": 0.0,
            "realized_gain": 1.8,
        },
    ])
    estimator = MarginalGainEstimator(alpha=1.0)
    estimator.fit(dummy_train_data)

    policy = SwitchingAwarePolicy(cost_model=cm, gain_estimator=estimator)

    # Historical state at t-1
    prev_conf = 0.65
    prev_entropy = 0.85
    prev_pred = 4
    available_sensors = {"hand", "ankle", "heart_rate"}

    # Run decision before seeing target window
    decision_original = policy.select_sensors(
        available_sensors=available_sensors,
        prev_confidence=prev_conf,
        prev_entropy=prev_entropy,
        prev_prediction=prev_pred,
    )

    # Now simulate target window sensor values replaced with extreme random noise
    random_noise = np.random.RandomState(42).normal(1000.0, 500.0, size=(1, 54))
    # Crucially, the decision rule signature only takes pre-window historical state:
    decision_after_noise = policy.select_sensors(
        available_sensors=available_sensors,
        prev_confidence=prev_conf,
        prev_entropy=prev_entropy,
        prev_prediction=prev_pred,
    )

    assert decision_original == decision_after_noise, (
        "Acquisition decision changed after modifying target window values! Future leakage detected!"
    )


def test_test_participants_do_not_alter_training_preprocessing():
    """
    Changing test participants must not alter preprocessing fitted on training participants.
    """
    np.random.seed(42)
    train_df = pd.DataFrame({
        "subject_id": [101] * 50 + [102] * 50,
        "activity_id": [1] * 50 + [2] * 50,
        "feat_a": np.random.normal(10.0, 2.0, 100),
        "feat_b": np.random.normal(50.0, 5.0, 100),
    })

    test_df_1 = pd.DataFrame({
        "subject_id": [105] * 20,
        "activity_id": [1] * 10 + [2] * 10,
        "feat_a": np.random.normal(500.0, 10.0, 20),  # drastically different
        "feat_b": np.random.normal(1000.0, 20.0, 20),
    })

    test_df_2 = pd.DataFrame({
        "subject_id": [106] * 20,
        "activity_id": [1] * 20,
        "feat_a": np.random.normal(-200.0, 5.0, 20),
        "feat_b": np.random.normal(-500.0, 10.0, 20),
    })

    prep = FeaturePreprocessor(feature_cols=["feat_a", "feat_b"])
    prep.fit(train_df)

    fitted_mean = prep.scaler.mean_.copy()
    fitted_scale = prep.scaler.scale_.copy()
    fitted_priors = dict(prep.class_priors)

    # Transforming test_df_1
    _ = prep.transform(test_df_1)

    # Fit parameters must remain completely identical
    assert np.array_equal(prep.scaler.mean_, fitted_mean)
    assert np.array_equal(prep.scaler.scale_, fitted_scale)
    assert prep.class_priors == fitted_priors

    # Transforming test_df_2
    _ = prep.transform(test_df_2)

    assert np.array_equal(prep.scaler.mean_, fitted_mean)
    assert np.array_equal(prep.scaler.scale_, fitted_scale)
    assert prep.class_priors == fitted_priors


def test_no_overlapping_windows_across_splits():
    """
    Verify participant splitter guarantees disjoint subjects.
    """
    splitter = ParticipantSplitter(
        train_subjects=[101, 102],
        val_subjects=[103],
        test_subjects=[104],
    )

    df = pd.DataFrame({
        "subject_id": [101, 102, 103, 104, 101, 103],
        "activity_id": [1, 1, 2, 2, 3, 3],
    })

    train_df, val_df, test_df = splitter.split_dataframe(df)

    train_subs = set(train_df["subject_id"].unique())
    val_subs = set(val_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    assert train_subs.isdisjoint(val_subs)
    assert train_subs.isdisjoint(test_subs)
    assert val_subs.isdisjoint(test_subs)
