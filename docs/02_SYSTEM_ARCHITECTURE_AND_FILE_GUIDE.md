# 02. System Architecture and File Guide

## 1. Overall System Architecture

The P10 system operates as an end-to-end causal pipeline dividing physical sensor acquisition and machine learning inference into sequential, leakage-protected stages:

```
[Raw PAMAP2 .dat Files]
         │
         ▼
[01_inspect_dataset.py] ──> data_manifest.json + raw_signals_sample.png
         │
         ▼
[02_prepare_data.py]    ──> Fixed-Grid Windowing (3.0s, 50% overlap)
         │                  Transition Removal & Feature Extraction
         ▼
[windows_features.parquet] (12,736 windows × 117 columns across 9 subjects)
         │
         ▼
[03_train_classifiers.py] ──> Trains 8 Subset Classifiers on Training Participants
         │                    Evaluates Subsets 0 to 7 (subset_classifier_performance.csv)
         ▼
[04_run_experiments.py]   ──> Sequential Acquisition Simulator (simulator.py)
         │                    ├─ Marginal Gain Estimator (gain_estimator.py)
         │                    ├─ Switching-Aware Controller (acquisition_policy.py)
         │                    ├─ 6 Baselines + 2 Ablations (baselines.py)
         │                    ├─ 4 Markov Outage Regimes (outage_simulator.py)
         │                    ├─ Cost Sweeps & Participant Rotations
         │                    └─ Exact Dynamic Programming Solver (synthetic_reference.py)
         ▼
[results/]                ──> Tables, Figures, Action Logs, and Model Artifacts
```

---

## 2. Chronological Online Event Flow

For each time window $t = 1, 2, \dots, T$:
1. **Historical Context Generation ($t-1$)**: The controller receives past classifier confidence $p(y_{t-1} \mid x_{t-1})$, entropy $H(t-1)$, and the previous active set $A_{t-1}$.
2. **Availability Declaration ($U_t$)**: The environment reveals which physical sensor groups are functioning. Any sensor experiencing an outage is forced OFF.
3. **Sensor Selection ($A_t \subseteq U_t$)**: The policy evaluates expected information gain versus marginal cost for candidate sensors and selects subset $A_t$ before window $t$ is recorded.
4. **Cost Accounting**:
   - Running cost ($c_{\text{run}}$) is charged for every sensor in $A_t$.
   - Wake-up cost ($c_{\text{wake}}$) is charged strictly for newly awakened sensors ($A_t \setminus A_{t-1}$).
5. **Physical Measurement**: Raw channels from active sensors $A_t$ are observed and window features computed.
6. **Classification & Logging**: The pre-trained subset classifier $f_{A_t}$ outputs class probabilities $\hat{y}_t$, updating memory for window $t+1$.

---

## 3. Clean First-Version Directory Layout

```
final_year_project/
├── configs/
│   └── default.yaml             # Canonical configuration (splits, costs, outages, parameters)
├── data/
│   ├── processed/
│   │   ├── data_manifest.json   # Dataset inventory across subjects
│   │   ├── windowing_summary.json # Discard and window statistics
│   │   ├── windows_features.csv # Exported window feature table (CSV)
│   │   └── windows_features.parquet # Fast binary window feature storage
│   └── README.md
├── docs/
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_SYSTEM_ARCHITECTURE_AND_FILE_GUIDE.md
│   ├── 03_METHODOLOGY.md
│   ├── 04_RESULTS_AND_ANALYSIS.md
│   ├── 05_EXECUTION_AND_REPRODUCIBILITY.md
│   └── P10_Final_Project_Presentation.pptx # Final college project PowerPoint
├── pamap2+physical+activity+monitoring/ # Original UCI PAMAP2 protocol recordings
├── results/
│   ├── figures/                 # 5 canonical publication plots
│   ├── logs/                    # 10 full chronological action logs
│   ├── models/                  # Trained classifier suite & gain estimator
│   └── tables/                  # 9 canonical CSV/JSON performance tables
├── scripts/
│   ├── 01_inspect_dataset.py    # Stage 01: Raw data inspection & manifest
│   ├── 02_prepare_data.py       # Stage 02: Fixed-grid windowing & feature extraction
│   ├── 03_train_classifiers.py  # Stage 03: Subset classifier suite training
│   └── 04_run_experiments.py    # Stage 04: Benchmarks, sweeps, rotations & DP solver
├── src/
│   ├── __init__.py              # Package initialization
│   ├── acquisition_policy.py    # Proposed switching-aware policy with hysteresis
│   ├── baselines.py             # 6 comparison baselines and 2 mechanism ablations
│   ├── cost_model.py            # Running, wake-up, and cumulative cost engine
│   ├── data_loader.py           # PAMAP2 raw ingestion and protocol activity mapping
│   ├── feature_extraction.py    # Statistical, temporal, and frequency feature extractors
│   ├── gain_estimator.py        # Ridge-based state-conditioned marginal gain estimator
│   ├── outage_simulator.py      # Gilbert-Elliott 2-state Markov outage simulator
│   ├── plotting.py              # Publication-grade plotting routines
│   ├── preprocessing.py         # Leakage-safe participant splitting and scaling
│   ├── simulator.py             # Chronological forward simulator & probability cache
│   ├── subset_classifiers.py    # 8-classifier power-set architecture
│   ├── synthetic_reference.py   # 2-sensor HMM and Bellman Value Iteration DP solver
│   └── windowing.py             # Fixed-grid sliding window generator
├── tests/                       # 7 automated test suites (20 tests, 100% passing)
├── pytest.ini                   # Pytest test discovery configuration
├── requirements.txt             # Clean pip dependencies
├── run_all.py                   # Master single-command execution pipeline
└── README.md                    # Root project guide
```

---

## 4. Responsibility of Core Source Modules (`src/`)

| Module | Core Responsibility | Key Classes / Functions |
| :--- | :--- | :--- |
| `data_loader.py` | Parses raw 54-column PAMAP2 `.dat` files; filters protocol activities; maps physical channels. | `load_subject_raw_data`, `inspect_dataset_directory` |
| `windowing.py` | Imposes fixed time grid; discards activity-transition windows; filters transient label 0. | `create_fixed_grid_windows`, `WindowSegment` |
| `feature_extraction.py`| Computes mean, std, min, max, RMS, SMA, and frequency energy for IMUs; HR sample stats. | `extract_window_features`, `get_feature_column_names` |
| `preprocessing.py` | Strictly splits data by subject ID; fits scalers/imputers exclusively on training split. | `ParticipantSplitter`, `FeaturePreprocessor` |
| `subset_classifiers.py`| Trains and manages the 8 logistic regression models for subsets 0–7; handles class priors. | `EmptySubsetClassifier`, `SubsetClassifierSuite` |
| `gain_estimator.py` | Builds historical gain dataset; trains Ridge regressor predicting EIG from pre-window state. | `MarginalGainEstimator`, `build_gain_training_dataset` |
| `cost_model.py` | Tracks active states; charges $c_{\text{run}}$ per active window and $c_{\text{wake}}$ on transitions. | `SensorCostModel` |
| `acquisition_policy.py`| Implements one-step switching-aware decision rule with stateful hysteresis thresholding. | `BaseAcquisitionPolicy`, `SwitchingAwarePolicy` |
| `outage_simulator.py` | Simulates bursty sensor outages via 2-state Gilbert-Elliott Markov chain with fixed seeds. | `MarkovOutageSimulator`, `get_predefined_scenarios` |
| `baselines.py` | Implements Always-All, Best Fixed, Random, Myopic, Confidence, Keep-Alive, and 2 ablations. | `AlwaysAllPolicy`, `MyopicGainPolicy`, `SwitchingNoOutagePolicy` |
| `simulator.py` | Coordinates chronological simulation; precomputes subset probability cache; logs actions. | `run_chronological_simulation`, `SimulationResult` |
| `synthetic_reference.py`| Solves 2-sensor HMM DP via Bellman Value Iteration; proves existence of hysteresis gap. | `SyntheticHMM`, `DPSolver` |
| `plotting.py` | Generates publication plots (Pareto frontier, wake-ratio sweep, outage comparison, hysteresis).| `plot_pareto_frontier`, `plot_synthetic_hysteresis` |

---

## 5. Script Execution Pipeline (`scripts/`)

- **`01_inspect_dataset.py`**: Scans the raw PAMAP2 dataset directory, verifies channel counts, generates `data_manifest.json`, and outputs `raw_signals_sample.png`.
- **`02_prepare_data.py`**: Reads raw recordings, slices 3.0s windows with 50% overlap, removes transition and transient windows, computes statistical features, and writes `windows_features.parquet` and `windows_features.csv`.
- **`03_train_classifiers.py`**: Partitions data by training subjects, fits all 8 subset classifiers, evaluates on validation and test sets, and writes `subset_classifier_performance.csv` and `subset_per_class_recall_test.csv`.
- **`04_run_experiments.py`**: Trains the gain estimator, benchmarks the proposed policy against all 6 baselines and 2 ablations, sweeps cost ratios, tests 4 outage regimes, evaluates 3 participant rotations, solves the exact DP model, and outputs all canonical tables, action logs, and plots.
- **`run_all.py`**: Master execution entry point running Stages 01 to 04 sequentially with complete status reporting.
