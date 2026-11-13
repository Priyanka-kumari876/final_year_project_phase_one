# 03. Technical Methodology

## 1. Dataset & Sensor Groups

The system uses the benchmark **PAMAP2 Physical Activity Monitoring Dataset** collected from 9 participants wearing three 3D Inertial Measurement Units (IMUs) and a wireless heart-rate monitor.

In accordance with physical device grouping rules, individual channels from the same hardware package share power rails and are charged as unified physical modalities:

| Modality Index | Sensor Group | Hardware Channels | Sampling Rate | Baseline Running Cost ($c_{\text{run}}$) |
| :---: | :--- | :--- | :---: | :---: |
| **1** | **Hand IMU** | 3D Accel $\pm16g$, 3D Gyro, 3D Mag, Temperature (16 ch) | 100 Hz | 1.0 cost unit |
| **2** | **Ankle IMU** | 3D Accel $\pm16g$, 3D Gyro, 3D Mag, Temperature (16 ch) | 100 Hz | 1.0 cost unit |
| **3** | **Heart Rate (HR)** | Chest strap ECG receiver (1 ch) | ~9 Hz | 0.5 cost units |

*Physical Grouping Rule*: Accelerometer, gyroscope, and magnetometer channels on the same IMU device cannot be woken independently without custom multi-rail hardware. They are treated as an integrated physical unit.

---

## 2. Activity Classes

The study evaluates the **12 standardized protocol activities**:
- `1`: Lying down
- `2`: Sitting
- `3`: Standing
- `4`: Walking
- `5`: Running
- `6`: Cycling
- `7`: Nordic walking
- `12`: Ascending stairs
- `13`: Descending stairs
- `16`: Vacuum cleaning
- `17`: Ironing
- `24`: Rope jumping

**Activity Label 0 Discard Rule**: Transient, unscripted activity periods (label `0`) occurring between structured tasks are strictly discarded to avoid ambiguous supervised targets.

---

## 3. Data Preprocessing, Windowing & Feature Extraction

### Windowing Strategy
- **Window Length**: 3.0 seconds ($N = 300$ samples per IMU at 100 Hz).
- **Overlap**: 50% ($1.5$ seconds stride).
- **Fixed-Grid Temporal Alignment**: Windows are generated along a strict chronological time grid inside each participant session without dynamically adjusting boundaries.
- **Transition Removal**: Windows overlapping an activity state transition boundary are rejected and logged (44 transition windows discarded across all subjects).

### Feature Extraction (117 Total Columns)
For each IMU group:
- **Time-Domain Statistics**: Mean, standard deviation, minimum, maximum, root-mean-square (RMS).
- **Kinematic Descriptors**: Signal Magnitude Area (SMA):
  $$\text{SMA} = \frac{1}{N} \sum_{k=1}^N \left( |a_{x,k}| + |a_{y,k}| + |a_{z,k}| \right)$$
- **Frequency-Domain Statistics**: Spectral energy computed via Fast Fourier Transform (FFT).

For the Heart-Rate Monitor:
- Mean heart rate, valid sample count, and window valid fraction ($\frac{N_{\text{valid}}}{N_{\text{expected}}}$) to handle low-rate bursty transmission.

---

## 4. Participant-Based Splitting (Zero Leakage)

To guarantee generalization and prevent cross-window data leakage, all splits are partitioned **strictly by participant ID**:
- **Default Partition**:
  - **Training Split (6 subjects)**: `[101, 102, 103, 104, 107, 108]` $\to$ 9,266 windows
  - **Validation Split (1 subject)**: `[105]` $\to$ 1,789 windows
  - **Held-Out Test Split (2 subjects)**: `[106, 109]` $\to$ 1,681 windows
- **Rotation Folds**:
  - `fold_0`: Train `[101, 102, 103, 104, 107, 108]`, Val `[105]`, Test `[106, 109]`
  - `fold_1`: Train `[103, 104, 105, 106, 107, 108]`, Val `[101]`, Test `[102, 109]`
  - `fold_2`: Train `[101, 102, 105, 106, 107, 108]`, Val `[103]`, Test `[104, 109]`

**Strict Rule**: Median imputation values and `StandardScaler` mean/variance parameters are computed exclusively on the training participants and applied downstream.

---

## 5. Sensor Subsets & Modular Classifiers

With 3 physical sensor modalities, exactly $2^3 = 8$ subsets are formed:

| Subset ID | Sensor Combination | Modalities Included | Features | Classifier Role |
| :---: | :--- | :--- | :---: | :--- |
| **0** | Empty Subset | None | 0 | Outputs empirical class priors $\pi(y)$ from training data |
| **1** | Hand Only | Hand | 56 | Predicts exclusively using hand movement dynamics |
| **2** | Ankle Only | Ankle | 56 | Predicts exclusively using lower-body kinematics |
| **3** | HR Only | Heart Rate | 4 | Predicts exclusively from cardiovascular response |
| **4** | Hand + Ankle | Hand, Ankle | 112 | Dual-extremity motion coordination |
| **5** | Hand + HR | Hand, Heart Rate | 60 | Upper-body motion plus physiological arousal |
| **6** | Ankle + HR | Ankle, Heart Rate | 60 | Lower-body locomotion plus metabolic exertion |
| **7** | All Sensors | Hand, Ankle, HR | 116 | Full multi-modal feature set (upper accuracy bound) |

Each classifier is an independent, compact **Multinomial Logistic Regression** model ($L_2$ regularization, L-BFGS solver) trained exclusively on its subset's features.

---

## 6. Information Gain Estimation

To decide whether waking sensor $j$ is worthwhile, the system predicts the expected reduction in cross-entropy loss:
$$\text{gain}(S, j) = \text{Loss}(S) - \text{Loss}(S \cup \{j\})$$

A **Ridge Regression Estimator** is trained using only historical features available before window $t$ starts:
1. Classifier confidence from previous window: $\max_k p(y_{t-1} = k \mid x_{t-1})$.
2. Prediction entropy: $H(t-1) = - \sum_k p_k \log p_k$.
3. Current candidate subset composition flags (one-hot indicator).
4. Candidate sensor identity.

*Leakage Prevention*: Target window sensor values and true activity labels are strictly prohibited during inference.

---

## 7. Cost Model & Switching Dynamics

For window $t$, total modeled cost is:
$$\text{Cost}(t) = \sum_{j \in A_t} c_{\text{run}, j} + \sum_{j \in A_t \setminus A_{t-1}} c_{\text{wake}, j}$$

- **Running Cost ($c_{\text{run}}$)**: Incurred for every window sensor $j$ remains active.
- **Wake-Up Cost ($c_{\text{wake}}$)**: Incurred **only** when sensor $j$ transitions from $0 \to 1$ (asleep $\to$ awake).
- **Default Wake Ratio**: $\frac{c_{\text{wake}}}{c_{\text{run}}} = 5.0$ (swept across $\{0, 1, 5, 10\}$).
- **Cost Semantics**: All costs are expressed in *modeled cost units*, not physical millijoules.

---

## 8. Proposed Switching-Aware Policy & Hysteresis Principle

Let $S$ be the currently selected sensor set and $j$ be a candidate sensor available at window $t$ ($j \in U_t$). The controller applies the decision rule:

$$\text{Select } j \iff \widehat{\text{gain}}(S, j) > \lambda c_{\text{run}, j} + \mu c_{\text{wake}, j} \cdot \mathbb{I}[j \notin A_{t-1}]$$

### Emergence of the Hysteresis Gap
- **If sensor $j$ was already awake in window $t-1$** ($\mathbb{I} = 0$):
  $$\theta_{\text{keep}} = \lambda c_{\text{run}, j}$$
- **If sensor $j$ was sleeping in window $t-1$** ($\mathbb{I} = 1$):
  $$\theta_{\text{wake}} = \lambda c_{\text{run}, j} + \mu c_{\text{wake}, j}$$

When $\mu c_{\text{wake}, j} > 0$, the wake-up threshold is **strictly higher** than the retention threshold:
$$\Delta_{\text{hyst}} = \theta_{\text{wake}} - \theta_{\text{keep}} = \mu c_{\text{wake}, j} > 0$$

This gap prevents high-frequency chattering by keeping moderately informative sensors awake across short activity transitions.

---

## 9. Observed-Availability Outage Simulation

Sensor channel reliability is modeled via a 2-state discrete-time **Gilbert-Elliott Markov Chain**:
- State 0: **Available** (good channel)
- State 1: **Unavailable** (outage burst)
- Transition Probabilities: $p_{\text{fail}} = P(0 \to 1)$, $p_{\text{recover}} = P(1 \to 0)$.

Four evaluation regimes:
1. **No Outage**: $p_{\text{fail}} = 0.0, p_{\text{recover}} = 1.0$ (100% reliable).
2. **Rare Independent**: $p_{\text{fail}} = 0.02, p_{\text{recover}} = 0.80$.
3. **Bursty Short**: $p_{\text{fail}} = 0.05, p_{\text{recover}} = 0.40$ (mean outage length = 2.5 windows).
4. **Bursty Long**: $p_{\text{fail}} = 0.05, p_{\text{recover}} = 0.10$ (mean outage length = 10.0 windows).

*State Convention at Outage*: Availability flags $U_t$ are observed prior to acquisition. Any active sensor experiencing an outage is forced OFF. Upon recovery, it starts in the sleeping state and incurs a wake-up cost when selected again.

---

## 10. Baseline Policies & Mechanism Ablations

### Baselines (6 Methods)
1. **Always All Sensors**: Activates all available sensors $U_t$ at every window (upper bound on accuracy).
2. **Best Fixed Subset (Hand + Ankle)**: Statically queries the best-performing multi-sensor subset determined on validation data.
3. **Best Fixed Subset (Hand Only)**: Statically queries the single most informative sensor.
4. **Random Feasible**: Randomly samples available sensors with probability $p = 0.6$, matching average acquisition rate.
5. **Myopic Gain (State-Agnostic)**: Evaluates $\widehat{\text{gain}} > \lambda c_{\text{run}}$, completely ignoring previous state $A_{t-1}$ and wake-up costs ($\mu = 0$).
6. **History Confidence Trigger**: Activates all sensors when previous confidence drops below threshold $\tau = 0.75$; otherwise uses Hand Only.
7. **Confidence + Keep-Alive**: Activates all sensors on low confidence and holds them awake for a minimum of $K = 4$ windows.

### Mechanism Ablations (2 Methods)
1. **Switching without Outage Reasoning**: Plans acquisition assuming 100% sensor availability, only learning of outages when physical acquisition fails.
2. **Proposed Ablation without Wake Cost ($\mu = 0$)**: Uses gain estimation but disables the wake-up penalty in the decision threshold, removing the hysteresis gap.
