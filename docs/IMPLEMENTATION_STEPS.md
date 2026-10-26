# Implementation Steps and Execution Log

**Project**: P10: Multimodal Acquisition with Sensor Wake-Up Costs and Outages  
**Repository**: `c:\Users\princ\Desktop\final_year_project`  
**Author / Engineer**: Senior Machine Learning & Research Engineer  
**Date**: October 26, 2026  

---

## Phase Status Overview

| Phase | Description | Status | Evidence Checkpoint |
|:---|:---|:---|:---|
| **Phase 1** | Repository Inspection and Project Setup | **COMPLETED** | Directory layout, `.gitignore`, `configs/default.yaml`, `requirements.txt` |
| **Phase 2** | Dataset Inspection and Preparation | **COMPLETED** | `src/data_loader.py`, `data/processed/data_manifest.json`, `results/figures/raw_signals_sample.png` |
| **Phase 3** | Leakage-Safe Participant Splitting | **COMPLETED** | `src/preprocessing.py`, `tests/test_no_data_leakage.py` |
| **Phase 4** | Windowing and Feature Extraction | **COMPLETED** | `src/windowing.py`, `src/feature_extraction.py`, `data/processed/windows_features.parquet` (12,736 windows) |
| **Phase 5** | Sensor-Subset Classifiers (8 Models) | **COMPLETED** | `src/subset_classifiers.py`, `results/models/subset_classifiers.joblib`, `results/tables/subset_classifier_performance.csv` |
| **Phase 6** | Marginal-Gain Estimation | **COMPLETED** | `src/gain_estimator.py`, `results/models/gain_estimator.joblib` (111,120 training transitions) |
| **Phase 7** | Switching-Aware Acquisition Controller | **COMPLETED** | `src/acquisition_policy.py`, `src/cost_model.py` (Hysteresis threshold rule) |
| **Phase 8** | Sensor Outage Simulation | **COMPLETED** | `src/outage_simulator.py` (2-State Markov chain, paired masks, observed vs hidden modes) |
| **Phase 9** | Baseline Policies Implementation | **COMPLETED** | `src/baselines.py` (AlwaysAll, Fixed, Random, Myopic, Confidence, Keep-Alive, mu=0 ablation) |
| **Phase 10** | Small Exact Reference Model (Synthetic) | **COMPLETED** | `src/synthetic_reference.py`, `results/tables/synthetic_reference_results.json`, `results/figures/synthetic_hysteresis.png` |
| **Phase 11** | Evaluation and Experimental Analysis | **COMPLETED** | `scripts/run_experiments.py`, `results/tables/main_policy_comparison.csv`, `results/figures/pareto_frontier.png` |
| **Phase 12** | Correctness Checks and Unit Testing | **COMPLETED** | 19 unit & integration tests passing via `pytest -v` |
| **Phase 13** | Reproducibility, Documentation, and Final Report | **COMPLETED** | `README.md`, `docs/RESULTS_AND_ANALYSIS.md`, `docs/EXPERIMENTAL_METHODOLOGY.md` |

---

## Detailed Chronological Implementation Log

### Step 1.1 — Repository Inspection & Environment Assessment
- **Date**: October 26, 2026
- **Objective**: Inspect existing workspace, check available dataset files, inspect installed Python libraries, and assess missing components.
- **Files Inspected**:
  - `pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/Protocol/*.dat` (Subjects 101 to 109 found)
  - `pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/readme.pdf`
  - `pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/DataCollectionProtocol.pdf`
  - `pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/PerformedActivitiesSummary.pdf`
  - `P10_multimodal_acquisition_with_sensor_wake_up_costs_and_outages.pdf`
- **Commands Executed**:
  - `Get-ChildItem -Force`
  - `Get-ChildItem -Recurse -Depth 3 "pamap2+physical+activity+monitoring"`
  - `python --version; pip list`
  - `python -m pytest --version`
  - `pip install pytest`
  - `pytest --version`
- **Findings & Decisions**:
  - Python 3.12 active with NumPy 2.2.2, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.16.0, matplotlib 3.10.3, PyYAML 6.0.2.
  - Installed `pytest 9.1.1`.
  - PAMAP2 dataset present locally in `pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/Protocol` (Subjects 101-109).
  - 12 protocol activities identified from documentation: {1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24}. Activity 0 is transient and discarded.
  - 3 target physical sensor groups: Hand IMU (cols 4-20), Ankle IMU (cols 38-54), Heart Rate (col 3).
- **Status**: Completed.

### Step 1.2 — Project Architecture Setup
- **Date**: October 26, 2026
- **Objective**: Create standardized research project folder structure, git repository, dependency specs, and configuration files.
- **Files Created**:
  - `.git` repository initialized via `git init`
  - Directory structure: `configs/`, `data/raw/`, `data/processed/`, `src/`, `scripts/`, `tests/`, `results/tables/`, `results/figures/`, `results/logs/`, `results/models/`, `docs/`
  - `.gitignore`: Excludes raw `.dat` binaries, virtual environments, caches, temporary files.
  - `requirements.txt`: Specifies core dependencies.
  - `data/README.md`: Explains PAMAP2 dataset structure, sensor channels, and activity IDs.
  - `configs/default.yaml`: Complete YAML configuration covering windowing, splits, costs, outage scenarios, models, and output directories.
  - `docs/IMPLEMENTATION_STEPS.md`: This execution log.
- **Commands Executed**:
  - `git init`
  - `New-Item -ItemType Directory -Force ...`
- **Errors Encountered**:
  - Unicode encode error when printing `readme.pdf` text due to cp1252 default encoding on Windows console.
  - Fix: Configured `sys.stdout.reconfigure(encoding='utf-8')` for Python console scripts.
- **Status**: Completed.

### Step 2.1 — PAMAP2 Data Loader & Manifest Inspection (Phase 2)
- **Date**: October 26, 2026
- **Objective**: Implement robust PAMAP2 data loader preserving subject, session, timestamps, activity IDs, and sensor channels, and compile dataset manifest.
- **Files Created**:
  - `src/data_loader.py`: Complete 54-column reader with channel mappings for Hand IMU, Ankle IMU, and Heart Rate.
  - `scripts/inspect_dataset.py`: Scans subjects 101 to 109, checks missingness rates, and plots raw signals.
- **Commands Executed**:
  - `python scripts/inspect_dataset.py`
- **Outputs & Findings**:
  - 9 subjects scanned (1,942,872 valid protocol samples found).
  - Subject files sizes: 135 MB (101) to 198 MB (102), with subject 109 containing 3.7 MB.
  - Preserved original missingness masks (HR missing ~90.9% in raw 100 Hz grid due to native 9 Hz sampling rate).
  - Saved `data/processed/data_manifest.json`.
  - Saved raw signal visual plot `results/figures/raw_signals_sample.png`.
- **Status**: Completed.

### Step 3.1 — Leakage-Safe Participant Splitting (Phase 3)
- **Date**: October 26, 2026
- **Objective**: Implement participant-level train/validation/test splitter and strict preprocessing isolation.
- **Files Created**:
  - `src/preprocessing.py`:
    - `ParticipantSplitter`: Validates that train, val, and test participant sets are strictly disjoint.
    - `FeaturePreprocessor`: Fits StandardScaler, imputation medians, and class priors exclusively on training subjects.
- **Implementation Decisions**:
  - Default split: Train = [101, 102, 103, 104, 107, 108] (6 participants, 9,266 windows); Val = [105] (1 participant, 1,789 windows); Test = [106] (1 participant, 1,640 windows).
  - Support rotation folds in configuration (`configs/default.yaml`).
- **Status**: Completed.

### Step 4.1 — Fixed-Grid Windowing & Feature Extraction (Phase 4)
- **Date**: October 26, 2026
- **Objective**: Implement fixed-grid sliding window segmentation and per-modality feature extraction.
- **Files Created**:
  - `src/windowing.py`: Fixed 3.0s window with 50% overlap (1.5s hop). Rejects activity 0 and windows crossing activity transitions, logging rejection statistics.
  - `src/feature_extraction.py`: Computes statistical (mean, std, min, max, rms, SMA) and spectral energy features independently for Hand IMU, Ankle IMU, and HR valid fraction/stats.
  - `scripts/prepare_data.py`: Orchestrates end-to-end dataset preparation across all subjects.
- **Commands Executed**:
  - `python scripts/prepare_data.py`
- **Results**:
  - Total attempted windows: 19,142
  - Discarded transient (activity 0) windows: 6,358
  - Discarded activity transition windows: 44
  - Total valid single-activity windows retained: **12,736**
  - Saved master feature table: `data/processed/windows_features.parquet` (10.74 MB) and `windows_features.csv` (19.89 MB).
  - Manual window verification check: Verified Subject 101 Window 0 against raw samples, absolute feature difference = 0.000000 [PASS].
- **Status**: Completed.

### Step 5.1 — Training 8 Sensor-Subset Classifiers (Phase 5)
- **Date**: October 26, 2026
- **Objective**: Train compact classifiers for all $2^3 = 8$ sensor combinations strictly on training data.
- **Files Created**:
  - `src/subset_classifiers.py`: `EmptySubsetClassifier` (predicts training class priors) and `SubsetClassifierSuite` for subsets 0 through 7.
  - `scripts/train_classifiers.py`: Trains all 8 models on training split, evaluates on val and test, and generates comparative tables.
- **Commands Executed**:
  - `python scripts/train_classifiers.py`
- **Results**:
  - Subset 0 (Empty priors): Log Loss 2.4182, Balanced Acc 0.0909 (matches 1/12 prior baseline).
  - Subset 1 (Hand): Balanced Acc 0.8711, Log Loss 0.4482
  - Subset 2 (Ankle): Balanced Acc 0.6680, Log Loss 0.9907
  - Subset 3 (HR): Balanced Acc 0.2895, Log Loss 1.6973
  - Subset 4 (Hand + Ankle): Balanced Acc 0.8675
  - Subset 7 (All Sensors): Balanced Acc 0.8838, Log Loss 0.3528
  - Saved `results/models/subset_classifiers.joblib` and `results/tables/subset_classifier_performance.csv`.
- **Status**: Completed.

### Step 6.1 — Marginal-Gain Estimation & Controller Core (Phases 6–9)
- **Date**: October 26, 2026
- **Objective**: Implement pre-window marginal value estimator, cost model, 2-state Markov outage simulator, switching-aware controller, and all baseline policies.
- **Files Created**:
  - `src/gain_estimator.py`: Ridge regression model predicting realized log-loss reduction $\mathcal{L}(S) - \mathcal{L}(S \cup \{j\})$ using strictly pre-window historical features.
  - `src/cost_model.py`: Running, wake-up, and attempt cost accounting with hysteresis thresholds.
  - `src/outage_simulator.py`: Two-state Markov availability generator with paired seed masks.
  - `src/acquisition_policy.py`: Proposed switching-aware policy using hysteresis rule:
    $$\text{Select } j \iff \widehat{\text{Gain}} > \lambda c_{\text{run}, j} + \mu c_{\text{wake}, j} \cdot \mathbb{I}[j \notin A_{t-1}]$$
  - `src/baselines.py`: `AlwaysAll`, `BestFixed`, `RandomFeasible`, `MyopicGain`, `HistoryConfidence`, `HistoryConfidenceKeepAlive`, and `ProposedNoWakeCost` ($\mu=0$ ablation).
  - `src/simulator.py`: Chronological acquisition loop with state logging.
  - `src/synthetic_reference.py`: 2-sensor synthetic HMM solved via exact dynamic programming (value iteration).
  - `src/plotting.py`: Pareto frontiers, cost ratio sweeps, outage comparisons, and hysteresis plots.
- **Status**: Completed.

### Step 7.1 — Experimentation, Optimization & Bug Fixes (Phases 10 & 11)
- **Date**: October 26, 2026
- **Objective**: Run full experimental suite across all policies, sweeps, outages, stress tests, and reference DP.
- **Bugs Encountered & Fixes Applied**:
  1. *Bug*: `task-92` crashed in `src/synthetic_reference.py` due to variable name mismatch `fraction_used_when_asleep` vs `fraction_active_when_asleep`.
     - *Fix*: Aligned variable names to `fraction_active_when_asleep` and `fraction_active_when_awake`.
  2. *Bug*: Initial $\lambda$ and $\mu$ in config were set to 1.0 (cost units = loss units), creating an excessively high wake threshold ($\sim 6.0$ nats) which caused the policy to never wake sensors.
     - *Fix*: Calibrated $\lambda = 0.05$ and $\mu = 0.05$ according to the specification guideline (where 1 energy unit is worth $\sim 0.03$ loss units).
  3. *Optimization*: Iterative DataFrame row slicing (`df.iloc[t:t+1]`) in the simulation loop was slow across 50 simulation runs.
     - *Fix*: Implemented vectorized subset probability precomputation (`prob_cache_test`), accelerating execution by over 100x without altering outputs.
- **Commands Executed**:
  - `python -u scripts/run_experiments.py`
- **Results**:
  - Main comparison: Proposed policy achieves **0.7544 Macro-F1** with **4,110.5 total cost** and **165.8 wake-ups/1k windows** (vs. 326.2 for confidence trigger and 173.8 for myopic gain).
  - Exact DP synthetic solver converged, confirming hysteresis gap of **0.2745** (72.5% active when asleep vs. 100% active when awake).
  - Generated all tables in `results/tables/` and figures in `results/figures/` (`pareto_frontier.png`, `wake_ratio_sweep.png`, `outage_comparison.png`, `synthetic_hysteresis.png`).
- **Status**: Completed.

### Step 8.1 — Verification & Testing (Phase 12)
- **Date**: October 26, 2026
- **Objective**: Implement comprehensive unit and integration test suite and verify all correctness checks.
- **Files Created**:
  - `src/__init__.py`
  - `pytest.ini`
  - `tests/test_cost_model.py`
  - `tests/test_acquisition_policy.py`
  - `tests/test_no_data_leakage.py`
  - `tests/test_windowing.py`
  - `tests/test_subset_classifiers.py`
  - `tests/test_outages.py`
  - `tests/test_synthetic_reference.py`
- **Commands Executed**:
  - `pytest -v`
- **Test Results**:
  - **19 passed** in 2.37s.
  - Verified:
    1. Awake sensor pays running cost only.
    2. Sleeping sensor pays running + wake-up cost.
    3. Persistent awake sensor pays wake-up cost only once over consecutive windows.
    4. Recovered sensor pays wake-up cost upon re-activation.
    5. Recomputed cost matches online simulation accounting.
    6. Target window noise injection produces identical pre-acquisition decision (Zero Leakage).
    7. Preprocessing fitted strictly on training subjects.
    8. Disjoint participant splits.
    9. Unavailable sensors cannot be selected.
    10. Hysteresis threshold gap verified in heuristic controller and exact DP solver.
    11. Paired outage masks are strictly identical under fixed random seeds.
    12. Windows never cross transitions or include activity 0.
    13. Subset classifiers access only their permitted features.
    14. Empty subset model outputs training empirical priors.
- **Status**: Completed.

### Step 9.1 — Documentation & Final Reporting (Phase 13)
- **Date**: October 26, 2026
- **Objective**: Compile experimental analysis, detailed methodology, and root project README documentation.
- **Files Created / Updated**:
  - `scripts/generate_report.py`
  - `docs/RESULTS_AND_ANALYSIS.md`
  - `docs/EXPERIMENTAL_METHODOLOGY.md`
  - `docs/IMPLEMENTATION_STEPS.md`
  - `README.md`
- **Commands Executed**:
  - `pip install tabulate`
  - `python scripts/generate_report.py`
- **Status**: Completed.
