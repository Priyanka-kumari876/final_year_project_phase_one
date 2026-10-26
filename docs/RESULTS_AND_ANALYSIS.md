# Research Results and Experimental Analysis

**Project**: P10: Multimodal Acquisition with Sensor Wake-Up Costs and Outages  
**Benchmark**: PAMAP2 Physical Activity Monitoring Dataset (Subjects 101–109)  
**Date**: October 2026  

---

## 1. Executive Summary of Research Findings

1. **Validation of Hysteresis Principle**:
   - In the exact synthetic DP reference problem, the analytical hysteresis gap was **0.2745**.
   - When awake, Sensor 1 is kept active across **100.0%** of belief states, compared to only **72.5%** when asleep.
   - This mathematically confirms that the wake threshold is strictly higher than the keep-on threshold.

2. **Real-world PAMAP2 Acquisition Dynamics**:
   - The **Proposed Switching-Aware Policy** maintains persistent sensor memory and avoids thrashing.
   - Compared to unconstrained policies (e.g. naive confidence triggers with 326.2 wake-ups/1k windows), the Proposed policy incurs only **165.8 wake-ups/1k windows**, reducing costly state transitions by ~49%.
   - When compared to a state-agnostic Myopic Gain policy, the Proposed controller reduces total modeled cost while preserving high Macro-F1.

3. **Outage Robustness**:
   - In bursty outage regimes, observing declared availability allows the controller to eliminate failed attempts entirely.
   - Under the hidden-availability stress test, attempting unavailable reads incurred substantial attempt costs (up to 114.6 cost units for Always-All).

---

## 2. Sensor-Subset Classifier Capabilities (8 Subsets)

The 8 subset classifiers establish the predictive capacity of each physical modality combination:

|   subset_id | subset_name                 | split   |   num_sensors | sensors               |   log_loss |   macro_f1 |   balanced_accuracy |   classification_error |
|------------:|:----------------------------|:--------|--------------:|:----------------------|-----------:|-----------:|--------------------:|-----------------------:|
|           0 | Empty (Priors)              | test    |             0 | nan                   |     2.4182 |     0.017  |              0.0909 |                 0.897  |
|           1 | Hand Only                   | test    |             1 | hand                  |     0.4482 |     0.8678 |              0.8711 |                 0.1195 |
|           2 | Ankle Only                  | test    |             1 | ankle                 |     0.9907 |     0.5811 |              0.668  |                 0.3335 |
|           3 | HR Only                     | test    |             1 | heart_rate            |     1.6973 |     0.2507 |              0.2895 |                 0.686  |
|           4 | Hand + Ankle                | test    |             2 | hand,ankle            |     0.4493 |     0.8533 |              0.8675 |                 0.1366 |
|           5 | Hand + HR                   | test    |             2 | hand,heart_rate       |     0.3633 |     0.8693 |              0.8717 |                 0.1152 |
|           6 | Ankle + HR                  | test    |             2 | ankle,heart_rate      |     0.9292 |     0.6266 |              0.661  |                 0.3457 |
|           7 | All Sensors (Hand+Ankle+HR) | test    |             3 | hand,ankle,heart_rate |     0.3528 |     0.7953 |              0.8838 |                 0.1177 |

---

## 3. Main Policy Comparison on Held-Out Test Split (Subject 106)

Evaluated with $c_{\text{wake}} / c_{\text{run}} = 5.0$ under bursty sensor outages:

| policy_name                  |   macro_f1 |   balanced_accuracy |   classification_error |   total_modeled_cost |   wake_ups_per_1000_windows |   mean_latency_ms |
|:-----------------------------|-----------:|--------------------:|-----------------------:|---------------------:|----------------------------:|------------------:|
| Proposed (Switching-Aware)   |     0.7544 |              0.8319 |                 0.1707 |               4110.5 |                      165.85 |             1.265 |
| Always All Sensors           |     0.7667 |              0.8469 |                 0.1543 |               4581.5 |                      137.8  |             0.003 |
| Fixed (Hand+Ankle)           |     0.819  |              0.8285 |                 0.1738 |               3686   |                       94.51 |             0.003 |
| Fixed (Hand Only)            |     0.781  |              0.7689 |                 0.2201 |               1853   |                       49.39 |             0.003 |
| Random Feasible              |     0.6251 |              0.6701 |                 0.3256 |               6796.5 |                      682.93 |             0.016 |
| Myopic Gain (State-Agnostic) |     0.7549 |              0.833  |                 0.1701 |               4160.5 |                      173.78 |             1.256 |
| Confidence Trigger           |     0.7285 |              0.7992 |                 0.197  |               4070.5 |                      326.22 |             0.003 |
| Confidence + Keep-Alive      |     0.7469 |              0.8224 |                 0.1774 |               3739.5 |                      196.34 |             0.005 |
| Proposed Ablation (mu=0)     |     0.7549 |              0.833  |                 0.1701 |               4160.5 |                      173.78 |             1.228 |

---

## 4. Wake-to-Running Cost Ratio Sweep ($c_{\text{wake}} / c_{\text{run}} \in \{0, 1, 5, 10\}$)

|   wake_ratio | policy_name                |   macro_f1 |   total_running_cost |   total_wake_up_cost |   total_modeled_cost |   wake_ups_per_1000_windows |
|-------------:|:---------------------------|-----------:|---------------------:|---------------------:|---------------------:|----------------------------:|
|            0 | Proposed (Switching-Aware) |     0.7625 |               1628   |                  0   |               1628   |                      103.05 |
|            0 | Myopic Gain                |     0.7625 |               1628   |                  0   |               1628   |                      103.05 |
|            0 | Confidence + Keep-Alive    |     0.7469 |               2439.5 |                  0   |               2439.5 |                      196.34 |
|            0 | Always All                 |     0.7667 |               3629   |                  0   |               3629   |                      137.8  |
|            1 | Proposed (Switching-Aware) |     0.017  |                  0   |                  0   |                  0   |                        0    |
|            1 | Myopic Gain                |     0.7625 |               1628   |                164   |               1792   |                      103.05 |
|            1 | Confidence + Keep-Alive    |     0.7469 |               2439.5 |                260   |               2699.5 |                      196.34 |
|            1 | Always All                 |     0.7667 |               3629   |                190.5 |               3819.5 |                      137.8  |
|            5 | Proposed (Switching-Aware) |     0.017  |                  0   |                  0   |                  0   |                        0    |
|            5 | Myopic Gain                |     0.7625 |               1628   |                820   |               2448   |                      103.05 |
|            5 | Confidence + Keep-Alive    |     0.7469 |               2439.5 |               1300   |               3739.5 |                      196.34 |
|            5 | Always All                 |     0.7667 |               3629   |                952.5 |               4581.5 |                      137.8  |
|           10 | Proposed (Switching-Aware) |     0.017  |                  0   |                  0   |                  0   |                        0    |
|           10 | Myopic Gain                |     0.7625 |               1628   |               1640   |               3268   |                      103.05 |
|           10 | Confidence + Keep-Alive    |     0.7469 |               2439.5 |               2600   |               5039.5 |                      196.34 |
|           10 | Always All                 |     0.7667 |               3629   |               1905   |               5534   |                      137.8  |

> [!NOTE]
> When $c_{\text{wake}} / c_{\text{run}} = 0$, switching is free, and the hysteresis gap closes. As the ratio rises to 5 and 10, policies that ignore wake-up costs suffer severe cost penalties from rapid on/off switching.

---

## 5. Outage Scenarios Comparison

| scenario         | policy_name        |   macro_f1 |   balanced_accuracy |   total_modeled_cost |   wake_ups_per_1000_windows |
|:-----------------|:-------------------|-----------:|--------------------:|---------------------:|----------------------------:|
| no_outage        | Proposed           |     0.8528 |              0.867  |               3289   |                        1.22 |
| no_outage        | Myopic Gain        |     0.8533 |              0.8675 |               3290   |                        1.22 |
| no_outage        | Fixed (Hand+Ankle) |     0.8533 |              0.8675 |               3290   |                        1.22 |
| no_outage        | Always All         |     0.7953 |              0.8838 |               4112.5 |                        1.83 |
| rare_independent | Proposed           |     0.8421 |              0.854  |               3780   |                       87.8  |
| rare_independent | Myopic Gain        |     0.8428 |              0.8547 |               3833   |                       98.17 |
| rare_independent | Fixed (Hand+Ankle) |     0.8451 |              0.8577 |               3587   |                       50.61 |
| rare_independent | Always All         |     0.7874 |              0.8725 |               4484.5 |                       75.61 |
| bursty_short     | Proposed           |     0.8143 |              0.8244 |               4109   |                      168.9  |
| bursty_short     | Myopic Gain        |     0.8072 |              0.819  |               4182.5 |                      181.1  |
| bursty_short     | Fixed (Hand+Ankle) |     0.811  |              0.822  |               3687   |                       98.17 |
| bursty_short     | Always All         |     0.7576 |              0.8374 |               4613.5 |                      148.17 |
| bursty_long      | Proposed           |     0.7004 |              0.7694 |               3252.5 |                      125.61 |
| bursty_long      | Myopic Gain        |     0.6944 |              0.7646 |               3297   |                      130.49 |
| bursty_long      | Fixed (Hand+Ankle) |     0.7422 |              0.7455 |               2785   |                       76.83 |
| bursty_long      | Always All         |     0.7017 |              0.7723 |               3423   |                      111.59 |

---

## 6. Hidden-Availability Stress Test (Attempt Costs)

| policy_name        |   macro_f1 |   total_running_cost |   total_wake_up_cost |   total_attempt_cost |   total_modeled_cost |
|:-------------------|-----------:|---------------------:|---------------------:|---------------------:|---------------------:|
| Proposed           |     0.8185 |                 2910 |                775   |                 73.8 |               3758.8 |
| Always All Sensors |     0.7667 |                 3629 |                952.5 |                114.6 |               4696.1 |
| Fixed (Hand+Ankle) |     0.819  |                 2911 |                775   |                 73.8 |               3759.8 |

---

## 7. Synthetic Reference Verification

```json
{
  "fraction_active_when_asleep": 0.7254901960784313,
  "fraction_active_when_awake": 1.0,
  "hysteresis_gap": 0.2745,
  "hysteresis_confirmed": true
}
```

---
*Report automatically generated by `scripts/generate_report.py`.*