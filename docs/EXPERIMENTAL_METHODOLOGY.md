# Experimental Methodology: Multimodal Acquisition with Sensor Wake-Up Costs and Outages

## 1. Problem Formulation & Research Question

In wearable Human Activity Recognition (HAR), edge devices are equipped with multiple physical sensors (e.g., inertial measurement units on wrist and ankle, photoplethysmography or chest heart rate monitors). While continuous reading from all available sensors maximizes predictive accuracy, it imposes severe battery drain. Conversely, relying on minimal sensors risks misclassifying structurally similar activities (e.g., ascending vs. descending stairs, sitting vs. standing).

An essential but frequently overlooked physical reality of embedded hardware is that **sensors are stateful physical devices**:
- An active sensor incurs a steady-state **running cost** ($c_{\text{run}}$) for each sampling window.
- A sleeping sensor powered down to preserve energy cannot be read instantaneously; powering up its internal oscillators, sampling circuitry, and wireless transceivers incurs a **wake-up cost** ($c_{\text{wake}}$).
- Hardware or communication failures cause **outages**, rendering a sensor temporarily unavailable. When it recovers, its hardware state returns to sleeping.

### Central Research Question
> **Can an acquisition policy that maintains persistent sensor state, leverages hysteresis thresholds, and accounts for sensor wake-up costs and declared outages achieve a superior accuracy-cost trade-off compared to state-agnostic and myopic policies?**

---

## 2. Mathematical Formulation

Consider a chronological stream of time windows indexed by $t = 1, 2, \dots, T$, and a set of $M$ physical sensor modalities $j \in \{1, \dots, M\}$ (specifically Hand IMU, Ankle IMU, and Heart-Rate monitor).

### Decision Protocol
1. **At the conclusion of window $t-1$**: The controller observes only historical observations up to $t-1$, including classifier confidence $\hat{p}_{t-1}$, prediction entropy $H_{t-1}$, and previous active set $A_{t-1}$.
2. **At the start of window $t$**: The declared hardware availability signal $U_t \subseteq \{1, \dots, M\}$ is observed. An unavailable sensor is forced off ($A_t \subseteq U_t$).
3. **Sensor Selection**: The policy selects active subset $A_t \subseteq U_t$.
4. **Hardware Cost Accounting**:
   $$\text{Cost}(t) = \sum_{j \in A_t} c_{\text{run}, j} + \sum_{j \in A_t \setminus A_{t-1}} c_{\text{wake}, j}$$
   If an attempted read fails under hidden availability, an attempt penalty $c_{\text{attempt}}$ is incurred.
5. **Data Acquisition**: Sensor features $x_t[A_t]$ for window $t$ are sampled *only* for modalities in $A_t$. Unselected or unavailable channels remain masked.
6. **Activity Prediction**: Pre-trained subset classifier $f_{A_t}$ predicts class probabilities $\hat{p}_t = f_{A_t}(x_t[A_t])$.

### Global Objective Function
The experimental objective evaluates the trade-off across $T$ windows:
$$J = \frac{1}{T} \sum_{t=1}^T \mathcal{L}(y_t, f_{A_t}(x_t[A_t])) + \lambda \left( \frac{1}{T} \sum_{t=1}^T \sum_{j \in A_t} c_{\text{run}, j} \right) + \mu \left( \frac{1}{T} \sum_{t=1}^T \sum_{j \in A_t \setminus A_{t-1}} c_{\text{wake}, j} \right)$$
where:
- $\mathcal{L}$ is the multi-class cross-entropy (log-loss).
- $\lambda$ is the trade-off multiplier converting running cost units to loss units.
- $\mu$ is the trade-off multiplier for sensor wake-up costs.

---

## 3. The Hysteresis Mechanism

When $\mu \cdot c_{\text{wake}, j} > 0$, deciding whether to acquire sensor $j$ depends fundamentally on whether sensor $j$ is currently awake ($j \in A_{t-1}$) or asleep ($j \notin A_{t-1}$):

$$\text{Decision Rule: Select sensor } j \iff \widehat{\text{Gain}}(S, j) > \lambda c_{\text{run}, j} + \mu c_{\text{wake}, j} \cdot \mathbb{I}[j \notin A_{t-1}]$$

- **Keep-On Threshold** (if $j \in A_{t-1}$):
  $$\theta_{\text{keep}} = \lambda c_{\text{run}, j}$$
- **Wake-Up Threshold** (if $j \notin A_{t-1}$):
  $$\theta_{\text{wake}} = \lambda c_{\text{run}, j} + \mu c_{\text{wake}, j}$$

The interval $[\theta_{\text{keep}}, \theta_{\text{wake}}]$ forms the **Hysteresis Band**. If expected information gain falls within this band, the controller keeps an already awake sensor on, avoiding the heavy wake-up penalty that would be incurred if it were shut down and reactivated shortly after.

---

## 4. Participant Splitting & Zero Data Leakage

Evaluating adaptive policies on time-series data requires rigorous protection against data leakage:
1. **Subject-Disjoint Splits**: Partitions are formed strictly by subject ID.
   - Default: Training on Subjects 101, 102, 103, 104, 107, 108 (9,266 windows).
   - Validation on Subject 105 (1,789 windows).
   - Held-Out Test on Subject 106 (1,640 windows).
2. **Preprocessing Isolation**: All normalization statistics ($\mu, \sigma$), imputation medians, and class priors are fitted *exclusively* on training participants.
3. **Pre-Window Decision Boundary**: Acquisition decisions for window $t$ use only state derived up to $t-1$. Verified via unit tests injecting extreme random noise into window $t$, confirming identical decisions.
4. **Boundary Integrity**: Window boundaries are segmented on a rigid time grid. Windows crossing activity transitions or containing label 0 are discarded and tracked.

---

## 5. Marginal Gain Estimation

The sensor value estimator predicts the expected log-loss reduction from adding candidate $j$ to current subset $S$:
$$\text{gain}(S, j) = \mathcal{L}(S) - \mathcal{L}(S \cup \{j\})$$

Training targets are generated using the 8 pre-trained subset classifiers evaluated on training participants. A lightweight Ridge regression model learns to map pre-window state features (previous prediction entropy, maximum confidence, current subset composition, candidate sensor identity) to predicted gain. At test time, ground-truth labels and target-window raw features are strictly hidden.

---

## 6. Outage Simulation Regimes

Two availability regimes are evaluated:
1. **Observed Availability (Main Benchmark)**:
   Sensor availability is modeled via a two-state ergodic Markov chain:
   - $\text{Available} \xrightarrow{p_{\text{fail}}} \text{Unavailable}$
   - $\text{Unavailable} \xrightarrow{p_{\text{recover}}} \text{Available}$
   Availability flags $U_t$ are observed at the start of window $t$ before acquisition. Unavailable sensors cannot be chosen. Upon recovery, the sensor returns in a *sleeping* state.
2. **Hidden Availability (Stress Test)**:
   Availability is hidden until a physical read is attempted. Failed attempts yield no valid observations and incur an attempt cost penalty ($c_{\text{attempt}} = 0.2$).

---

## 7. Baseline Comparisons & Ablations

All methods are evaluated on identical test sequences with paired random seeds:
1. **Always All Sensors**: Upper bound on predictive performance.
2. **Best Fixed Subset**: Static selection (e.g. Hand + Ankle) tuned on validation data.
3. **Random Feasible**: Uniformly samples available sensors matching target duty cycle.
4. **Myopic Gain per Cost**: Evaluates $\widehat{\text{Gain}} > \lambda c_{\text{run}}$, ignoring previous sensor state ($\mu = 0$ threshold).
5. **Confidence Trigger**: Wakes all sensors when confidence drops below threshold $\theta$.
6. **Confidence + Keep-Alive**: Confidence trigger equipped with a $K$-window keep-alive timer.
7. **Switching Ablation ($\mu = 0$)**: Evaluates proposed model without wake-up cost awareness.
8. **Synthetic DP Reference**: Infinite-horizon exact dynamic programming solver over discrete activity belief state.
