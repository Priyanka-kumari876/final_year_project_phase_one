# 05. Execution and Reproducibility Guide

## 1. Environment & Dependencies

The project is built and verified on **Python 3.10+** (tested on Python 3.12).

### Required Packages (`requirements.txt`)
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `pyarrow >= 12.0.0` (for fast Parquet reading and writing)
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `pyyaml >= 6.0`
- `joblib >= 1.3.0`
- `pytest >= 7.3.0`
- `tqdm >= 4.65.0`
- `python-pptx >= 1.0.0`

Install all dependencies in one command:
```powershell
pip install -r requirements.txt
```

---

## 2. Dataset Setup & Location

The project expects the official **PAMAP2 Physical Activity Monitoring** dataset extracted in the project root:
```
pamap2+physical+activity+monitoring/
└── PAMAP2_Dataset/
    └── PAMAP2_Dataset/
        └── Protocol/
            ├── subject101.dat
            ├── subject102.dat
            ├── subject103.dat
            ├── subject104.dat
            ├── subject105.dat
            ├── subject106.dat
            ├── subject107.dat
            ├── subject108.dat
            └── subject109.dat
```
Raw files contain 54 space-separated columns sampled at 100 Hz.

---

## 3. Configuration Management

All hyperparameters, subject partitions, cost matrices, and simulation flags are centralized in:
[`configs/default.yaml`](file:///c:/Users/princ/Desktop/New%20folder/final_year_project/configs/default.yaml)

Key configurable parameters:
- `data.window_size_sec`: 3.0 seconds
- `data.overlap`: 0.5 (50% overlap)
- `splits.default.train`: `[101, 102, 103, 104, 107, 108]`
- `splits.default.val`: `[105]`
- `splits.default.test`: `[106, 109]`
- `costs.running_costs`: `{hand: 1.0, ankle: 1.0, heart_rate: 0.5}`
- `costs.default_wake_ratio`: 5.0
- `models.random_seed`: 42

---

## 4. Single-Command Master Pipeline Execution

To execute the entire First-Version research workflow from raw data ingestion to final benchmark figures:
```powershell
cd final_year_project
python run_all.py
```

### Stage-by-Stage Breakdown
If you prefer running individual stages manually, follow this strict execution sequence:

1. **Stage 01: Inspect Dataset and Generate Inventory**
   ```powershell
   python scripts/01_inspect_dataset.py
   ```
   *Outputs*: `data/processed/data_manifest.json`, `results/figures/raw_signals_sample.png`.

2. **Stage 02: Fixed-Grid Windowing & Feature Extraction**
   ```powershell
   python scripts/02_prepare_data.py
   ```
   *Outputs*: `data/processed/windows_features.parquet`, `data/processed/windows_features.csv`, `data/processed/windowing_summary.json`.

3. **Stage 03: Train 8 Sensor-Subset Classifiers**
   ```powershell
   python scripts/03_train_classifiers.py
   ```
   *Outputs*: `results/models/subset_classifiers.joblib`, `results/tables/subset_classifier_performance.csv`, `results/tables/subset_per_class_recall_test.csv`.

4. **Stage 04: Acquisition Policies, Sweeps, Rotations & DP Solver**
   ```powershell
   python scripts/04_run_experiments.py
   ```
   *Outputs*: All 7 benchmark tables, 4 publication plots, and 10 chronological action logs.

---

## 5. Automated Verification & Testing

Execute the comprehensive unit and integration test suite:
```powershell
pytest tests -v
```

### Verified Test Cases (20 Passed)
- `tests/test_acquisition_policy.py`:
  - `test_unavailable_sensor_cannot_be_selected`: Asserts $A_t \subseteq U_t$.
  - `test_hysteresis_threshold_gap`: Verifies wake threshold $>$ keep threshold.
  - `test_random_policy_deterministic_with_seed`: Ensures reproducibility.
  - `test_switching_no_outage_policy`: Verifies outage-blind ablation logic.
- `tests/test_cost_model.py`:
  - `test_awake_sensor_pays_running_cost_only`: Asserts zero wake cost for active sensors.
  - `test_sleeping_sensor_pays_running_plus_wake`: Asserts $c_{\text{run}} + c_{\text{wake}}$ on start-up.
  - `test_sensor_kept_awake_pays_wake_cost_only_once`: Confirms no double billing.
  - `test_recovered_sensor_pays_wake_cost`: Confirms wake cost on post-outage activation.
  - `test_recomputed_cost_matches_online`: Re-verifies costs against saved logs.
- `tests/test_no_data_leakage.py`:
  - `test_target_window_noise_injection_does_not_affect_decision`: Target window values are not leaked.
  - `test_test_participants_do_not_alter_training_preprocessing`: Train-only scalers.
  - `test_no_overlapping_windows_across_splits`: Disjoint participant sets.
- `tests/test_outages.py`:
  - `test_paired_outage_masks_are_identical`: Fair evaluation under identical masks.
  - `test_distinct_seeds_differ`: Seed variance check.
  - `test_no_outage_scenario_always_available`: 100% availability check.
- `tests/test_subset_classifiers.py`:
  - `test_empty_subset_classifier`: Predicts training priors with zero features.
  - `test_subset_models_feature_isolation`: Strict feature isolation per subset.
- `tests/test_synthetic_reference.py`:
  - `test_dp_solver_convergence_and_hysteresis`: Verifies Bellman convergence.
- `tests/test_windowing.py`:
  - `test_windowing_discards_transitions`: Boundary transition rejection.
  - `test_windowing_discards_transient_activity_0`: Label 0 rejection.

---

## 6. Output Artifact Locations

### Trained Models (`results/models/`)
- `subset_classifiers.joblib`: 8 trained scikit-learn models, scalers, and imputation dicts.
- `gain_estimator.joblib`: Fitted Ridge marginal gain regressor.

### Benchmark Tables (`results/tables/`)
- `main_policy_comparison.csv`: Primary 10-policy comparison metrics.
- `held_out_participant_performance.csv`: Individual test participant breakdowns (Subject 106 and 109).
- `policy_per_class_recall_test.csv`: 12-class recall scores for all policies.
- `wake_ratio_sweep.csv`: Cost ratio sensitivity sweep ($\{0, 1, 5, 10\}$).
- `outage_comparison.csv`: Outage performance across 4 Markov regimes.
- `participant_rotations_comparison.csv`: 3-fold cross-participant rotation results.
- `subset_classifier_performance.csv`: Subsets 0 to 7 on train, val, and test splits.
- `subset_per_class_recall_test.csv`: Subset recall by activity class.
- `synthetic_reference_results.json`: Dynamic programming hysteresis gap proof metrics.

### Visualizations (`results/figures/`)
- `raw_signals_sample.png`: Sample 4-channel raw PAMAP2 signals.
- `pareto_frontier.png`: Accuracy vs Total Modeled Cost identifying the Pareto frontier.
- `wake_ratio_sweep.png`: Switching rate and cost burden curves across cost ratios.
- `outage_comparison.png`: Outage robustness bar chart across 4 regimes.
- `synthetic_hysteresis.png`: Analytical hysteresis band visualization from DP solver.

### Action Logs (`results/logs/`)
- Contains per-window chronological decision logs for all evaluated policies (`action_log_proposed_switching-aware.csv`, etc.).
