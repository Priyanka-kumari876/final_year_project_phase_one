# 01. Project Overview

## 1. Project Title
**P10 — Multimodal Acquisition with Sensor Wake-Up Costs and Outages**  
*A Switching-Aware Dynamic Information-Theoretic Framework for Wearable Physical Activity Recognition*

---

## 2. Problem Statement
Wearable systems for Human Activity Recognition (HAR) commonly deploy multimodal sensor suites—such as inertial measurement units (IMUs) mounted on extremities and physiological sensors like heart-rate monitors. While querying every sensor modality continuously yields rich observational data, it quickly depletes device battery reserves and wastes wireless network bandwidth.

Conversely, aggressive duty-cycling or naively minimizing sensor activations leads to severe predictive confusion between kinematically similar activities (e.g., sitting vs. standing, or walking vs. ascending stairs).

A fundamental physical complication in embedded sensing is that **sensor cost is state-dependent**:
1. **Steady-State Running Cost ($c_{\text{run}}$)**: An already active sensor consumes standard running energy per time window.
2. **Transient Wake-Up Cost ($c_{\text{wake}}$)**: A sleeping sensor requires significant start-up overhead—including analog rail stabilization, clock initialization, bus calibration, and firmware boot—before it can deliver valid samples.
3. **Hardware Outages**: Wireless body-area sensor links or battery failures occasionally make sensors temporarily unavailable.

Consequently, treating each time window independently causes **sensor chattering** (rapid, alternating on/off switching) that rapidly wastes energy on repeated wake-up penalties without improving classification accuracy.

---

## 3. Research Motivation
Traditional Cost-Sensitive Machine Learning and Feature Selection assume static feature sets or feature acquisition where obtaining a measurement incurs a constant marginal cost regardless of temporal context. In real-world embedded wearables, switching a sensor from **OFF to ON** incurs an energy penalty substantially greater than keeping an active sensor powered ($c_{\text{wake}} \gg c_{\text{run}}$). 

To make battery-powered wearable monitoring sustainable, an intelligent acquisition controller must maintain **persistent memory** of its previous sensor state and account for bursty hardware outages.

---

## 4. Main Research Question
> **Can a switching-aware multimodal acquisition policy achieve a superior prediction-cost trade-off compared to policies that ignore sensor state, especially when activities persist across consecutive windows and sensors experience communication outages?**

---

## 5. First-Version Project Scope
In strict accordance with the authoritative B.Tech P10 project guide, the completed **First Version** focuses on:
1. **Dataset**: PAMAP2 Physical Activity Monitoring Benchmark.
2. **Physical Sensor Modalities (3 groups)**:
   - Hand IMU (3D Accel $\pm16g$, Gyro, Mag, Temp)
   - Ankle IMU (3D Accel $\pm16g$, Gyro, Mag, Temp)
   - Heart-Rate Monitor (chest strap, ~9 Hz)
3. **Activity Classes**: Exactly the 12 protocol activities; transient activity label 0 is strictly discarded.
4. **Subsets**: All $2^3 = 8$ sensor combinations (including the empty subset predicting empirical class priors).
5. **Acquisition Policy**: One-step switching-aware decision rule incorporating expected information gain, running costs, wake-up costs, and declared sensor availability.
6. **Outage Simulation**: Reproducible 2-state discrete-time Gilbert-Elliott Markov outage process under observed availability.
7. **Baselines & Ablations**: 6 standard baselines and 2 dedicated mechanism ablations.
8. **Mathematical Structural Analysis**: Exact 2-sensor Dynamic Programming (DP) Bellman solver proving the emergence of the hysteresis band.

---

## 6. Primary Research Contributions
1. **Stateful Hysteresis Acquisition Policy**: Formulated a causal, lightweight decision rule where the wake-up threshold ($\theta_{\text{wake}} = \lambda c_{\text{run}} + \mu c_{\text{wake}}$) is strictly higher than the retention threshold ($\theta_{\text{keep}} = \lambda c_{\text{run}}$), preventing high-frequency chattering.
2. **Theoretical Proof via Exact DP**: Implemented a discrete-state Hidden Markov Model solved via infinite-horizon Value Iteration, formally proving an analytical hysteresis gap of $\Delta_{\text{hyst}} = 0.2745$.
3. **Empirical Validation on Held-Out Participants**:
   - Preserves **97.5%** of the full-sensing ceiling Macro-F1 (0.7939 vs. 0.8016).
   - Reduces total modeled energy cost by **84.6%** compared to continuous all-sensor acquisition.
   - Suppresses sensor wake-up frequency by **73.9%** (163.0 vs. 624.5 wake-ups per 1,000 windows) compared to state-agnostic myopic acquisition.
4. **Rigorous Leakage-Free Design**: All windowing, scaling, imputation, and threshold fitting strictly separate training, validation, and testing participants.
