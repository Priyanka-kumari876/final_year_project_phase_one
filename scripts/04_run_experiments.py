"""
Comprehensive Experiments Runner Script
=======================================
Executes the full suite of research experiments:
1. Trains the Marginal Gain Estimator on training split history.
2. Compares Proposed Switching-Aware policy against all baselines on held-out test split.
3. Sweeps wake-to-running cost ratios in {0, 1, 5, 10}.
4. Evaluates robustness under 4 paired Markov outage regimes.
5. Runs hidden-availability stress test with attempt costs.
6. Solves exact synthetic DP reference model and proves hysteresis existence.
7. Generates Pareto curves, tables, logs, and publication plots.
"""

import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import ParticipantSplitter
from src.subset_classifiers import SubsetClassifierSuite
from src.gain_estimator import MarginalGainEstimator, build_gain_training_dataset
from src.cost_model import SensorCostModel
from src.outage_simulator import MarkovOutageSimulator, get_predefined_scenarios
from src.acquisition_policy import SwitchingAwarePolicy
from src.baselines import (
    AlwaysAllPolicy,
    BestFixedSubsetPolicy,
    RandomFeasiblePolicy,
    MyopicGainPolicy,
    HistoryConfidencePolicy,
    HistoryConfidenceKeepAlivePolicy,
    ProposedNoWakeCostPolicy,
)
from src.simulator import run_chronological_simulation
from src.synthetic_reference import SyntheticHMM, DPSolver
from src.plotting import (
    plot_pareto_frontier,
    plot_wake_ratio_sweep,
    plot_outage_comparison,
    plot_synthetic_hysteresis,
)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    features_path = PROJECT_ROOT / config["data"]["features_path"]
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    tables_dir = PROJECT_ROOT / config["output"]["tables_dir"]
    fig_dir = PROJECT_ROOT / config["output"]["figures_dir"]
    logs_dir = PROJECT_ROOT / config["output"]["logs_dir"]

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 6-11: COMPREHENSIVE MULTIMODAL ACQUISITION EXPERIMENT SUITE")
    print("=" * 80)

    # 1. Load Data and Split
    df = pd.read_parquet(features_path)
    split_cfg = config["splits"]["default"]
    splitter = ParticipantSplitter(
        train_subjects=split_cfg["train"],
        val_subjects=split_cfg["val"],
        test_subjects=split_cfg["test"],
    )
    train_df, val_df, test_df = splitter.split_dataframe(df)

    # 2. Load Subset Classifiers Suite
    suite_path = models_dir / "subset_classifiers.joblib"
    if not suite_path.exists():
        raise FileNotFoundError(f"Trained suite not found at {suite_path}. Run train_classifiers.py first.")
    suite = SubsetClassifierSuite.load(suite_path)
    print(f"Loaded SubsetClassifierSuite with classes: {suite.classes_}")

    # 3. Train Marginal Gain Estimator
    print("\n--- Training Marginal Gain Estimator ---")
    gain_est_path = models_dir / "gain_estimator.joblib"
    gain_train_df = build_gain_training_dataset(suite, train_df, label_col="activity_id")
    print(f"Built gain training dataset with {len(gain_train_df):,} decision transitions.")

    gain_estimator = MarginalGainEstimator(alpha=1.0, random_state=config["models"]["random_seed"])
    gain_estimator.fit(gain_train_df)
    gain_estimator.save(gain_est_path)
    print("Marginal Gain Estimator successfully trained and saved.")

    # Base cost model
    base_cost_model = SensorCostModel(
        running_costs=config["costs"]["running_costs"],
        default_wake_ratio=config["costs"]["default_wake_ratio"],
        attempt_cost=config["costs"]["attempt_cost"],
        lambda_weight=config["costs"]["lambda_weight"],
        mu_weight=config["costs"]["mu_weight"],
    )

    # Generate paired outage masks for the test split
    outage_sim = MarkovOutageSimulator(p_fail=0.05, p_recover=0.40, random_seed=42)
    paired_test_masks = outage_sim.generate_availability_sequence(len(test_df), seed=42)

    # Precompute subset probabilities for test_df once for massive speedup
    print("Precomputing subset prediction cache for test split...")
    prob_cache_test = {s: suite.predict_proba(s, test_df) for s in range(8)}

    # -------------------------------------------------------------
    # EXPERIMENT 1: MAIN POLICY COMPARISON ON HELD-OUT TEST SPLIT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: MAIN POLICY COMPARISON (TEST SET: SUBJECT 106)")
    print(f"Cost settings: c_wake/c_run = {base_cost_model.default_wake_ratio:.1f} | Outage: Bursty Short")
    print("=" * 80)

    policies = [
        SwitchingAwarePolicy(base_cost_model, gain_estimator, name="Proposed (Switching-Aware)"),
        AlwaysAllPolicy(base_cost_model),
        BestFixedSubsetPolicy({"hand", "ankle"}, base_cost_model, name="Fixed (Hand+Ankle)"),
        BestFixedSubsetPolicy({"hand"}, base_cost_model, name="Fixed (Hand Only)"),
        RandomFeasiblePolicy(base_cost_model, p_acq=0.6, random_seed=42, name="Random Feasible"),
        MyopicGainPolicy(base_cost_model, gain_estimator, name="Myopic Gain (State-Agnostic)"),
        HistoryConfidencePolicy(base_cost_model, conf_threshold=0.75, name="Confidence Trigger"),
        HistoryConfidenceKeepAlivePolicy(base_cost_model, conf_threshold=0.75, keep_alive_windows=4, name="Confidence + Keep-Alive"),
        ProposedNoWakeCostPolicy(base_cost_model, gain_estimator, name="Proposed Ablation (mu=0)"),
    ]

    exp1_records = []
    print(f"{'Policy Name':<32} {'Macro-F1':<10} {'Bal Acc':<10} {'Clf Err':<10} {'Tot Cost':<12} {'Wake-ups/1k':<12}")
    print("-" * 88)

    for pol in policies:
        res = run_chronological_simulation(
            policy=pol,
            suite=suite,
            cost_model=base_cost_model,
            windows_df=test_df,
            availability_masks=paired_test_masks,
            hidden_availability=False,
            prob_cache=prob_cache_test,
        )
        # Save sample action log
        clean_name = pol.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        log_file = logs_dir / f"action_log_{clean_name}.csv"
        res.action_log.to_csv(log_file, index=False)

        m = res.metrics
        exp1_records.append({
            "policy_name": pol.name,
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "accuracy": m["accuracy"],
            "classification_error": m["classification_error"],
            "mean_log_loss": m["mean_log_loss"],
            "total_running_cost": m["total_running_cost"],
            "total_wake_up_cost": m["total_wake_up_cost"],
            "total_modeled_cost": m["total_modeled_cost"],
            "mean_cost_per_window": m["mean_cost_per_window"],
            "wake_ups_per_1000_windows": m["wake_ups_per_1000_windows"],
            "duty_cycle_hand": m.get("duty_cycle_hand", 0.0),
            "duty_cycle_ankle": m.get("duty_cycle_ankle", 0.0),
            "duty_cycle_heart_rate": m.get("duty_cycle_heart_rate", 0.0),
            "mean_latency_ms": m["mean_latency_ms"],
        })
        print(f"{pol.name:<32} {m['macro_f1']:<10.4f} {m['balanced_accuracy']:<10.4f} {m['classification_error']:<10.4f} {m['total_modeled_cost']:<12.1f} {m['wake_ups_per_1000_windows']:<12.1f}")

    df_exp1 = pd.DataFrame(exp1_records)
    exp1_table_path = tables_dir / "main_policy_comparison.csv"
    df_exp1.to_csv(exp1_table_path, index=False)
    print("=" * 88)
    print(f"Saved Main Comparison Table: {exp1_table_path}")

    # Plot Pareto Frontier
    plot_pareto_frontier(
        df_exp1,
        output_path=fig_dir / "pareto_frontier.png",
        title="Accuracy vs Cost Trade-off across Acquisition Policies",
    )

    # -------------------------------------------------------------
    # EXPERIMENT 2: WAKE-TO-RUNNING COST RATIO SWEEP (0, 1, 5, 10)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: WAKE-TO-RUNNING COST RATIO SWEEP (0, 1, 5, 10)")
    print("=" * 80)

    sweep_ratios = [0.0, 1.0, 5.0, 10.0]
    sweep_policies_templates = [
        ("Proposed (Switching-Aware)", lambda cm: SwitchingAwarePolicy(cm, gain_estimator, name="Proposed")),
        ("Myopic Gain", lambda cm: MyopicGainPolicy(cm, gain_estimator, name="Myopic Gain")),
        ("Confidence + Keep-Alive", lambda cm: HistoryConfidenceKeepAlivePolicy(cm, conf_threshold=0.75, name="Confidence + Keep-Alive")),
        ("Always All", lambda cm: AlwaysAllPolicy(cm)),
    ]

    exp2_records = []
    for r in sweep_ratios:
        cm = SensorCostModel(
            running_costs=config["costs"]["running_costs"],
            default_wake_ratio=r,
            attempt_cost=config["costs"]["attempt_cost"],
        )
        cm.update_wake_ratio(r)

        for name, factory in sweep_policies_templates:
            p = factory(cm)
            res = run_chronological_simulation(
                p, suite, cm, test_df, paired_test_masks, prob_cache=prob_cache_test
            )
            m = res.metrics
            exp2_records.append({
                "wake_ratio": r,
                "policy_name": name,
                "macro_f1": m["macro_f1"],
                "total_running_cost": m["total_running_cost"],
                "total_wake_up_cost": m["total_wake_up_cost"],
                "total_modeled_cost": m["total_modeled_cost"],
                "wake_ups_per_1000_windows": m["wake_ups_per_1000_windows"],
            })

    df_exp2 = pd.DataFrame(exp2_records)
    exp2_table_path = tables_dir / "wake_ratio_sweep.csv"
    df_exp2.to_csv(exp2_table_path, index=False)
    print(f"Saved Wake Ratio Sweep Table: {exp2_table_path}")
    plot_wake_ratio_sweep(df_exp2, output_path=fig_dir / "wake_ratio_sweep.png")

    # -------------------------------------------------------------
    # EXPERIMENT 3: OUTAGE SCENARIOS COMPARISON
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: OUTAGE REGIME COMPARISON")
    print("=" * 80)

    scenarios = get_predefined_scenarios()
    eval_outage_policies = [
        ("Proposed", SwitchingAwarePolicy(base_cost_model, gain_estimator, name="Proposed")),
        ("Myopic Gain", MyopicGainPolicy(base_cost_model, gain_estimator, name="Myopic Gain")),
        ("Fixed (Hand+Ankle)", BestFixedSubsetPolicy({"hand", "ankle"}, base_cost_model, name="Fixed (Hand+Ankle)")),
        ("Always All", AlwaysAllPolicy(base_cost_model)),
    ]

    exp3_records = []
    for sc_name, sc_params in scenarios.items():
        sim_out = MarkovOutageSimulator(p_fail=sc_params["p_fail"], p_recover=sc_params["p_recover"], random_seed=123)
        sc_masks = sim_out.generate_availability_sequence(len(test_df), seed=123)

        for pol_name, pol in eval_outage_policies:
            res = run_chronological_simulation(
                pol, suite, base_cost_model, test_df, sc_masks, prob_cache=prob_cache_test
            )
            m = res.metrics
            exp3_records.append({
                "scenario": sc_name,
                "policy_name": pol_name,
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "total_modeled_cost": m["total_modeled_cost"],
                "wake_ups_per_1000_windows": m["wake_ups_per_1000_windows"],
            })

    df_exp3 = pd.DataFrame(exp3_records)
    exp3_table_path = tables_dir / "outage_comparison.csv"
    df_exp3.to_csv(exp3_table_path, index=False)
    print(f"Saved Outage Comparison Table: {exp3_table_path}")
    plot_outage_comparison(df_exp3, output_path=fig_dir / "outage_comparison.png")

    # -------------------------------------------------------------
    # EXPERIMENT 4: HIDDEN AVAILABILITY STRESS TEST
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: HIDDEN AVAILABILITY STRESS TEST (ATTEMPT COSTS)")
    print("=" * 80)

    hidden_policies = [
        SwitchingAwarePolicy(base_cost_model, gain_estimator, name="Proposed"),
        AlwaysAllPolicy(base_cost_model),
        BestFixedSubsetPolicy({"hand", "ankle"}, base_cost_model, name="Fixed (Hand+Ankle)"),
    ]

    exp4_records = []
    for pol in hidden_policies:
        res = run_chronological_simulation(
            pol,
            suite,
            base_cost_model,
            test_df,
            paired_test_masks,
            hidden_availability=True,
            prob_cache=prob_cache_test,
        )
        m = res.metrics
        exp4_records.append({
            "policy_name": pol.name,
            "macro_f1": m["macro_f1"],
            "total_running_cost": m["total_running_cost"],
            "total_wake_up_cost": m["total_wake_up_cost"],
            "total_attempt_cost": m["total_attempt_cost"],
            "total_modeled_cost": m["total_modeled_cost"],
        })
        print(f"{pol.name:<24} Macro-F1: {m['macro_f1']:.4f} | Attempt Cost: {m['total_attempt_cost']:.1f} | Tot Cost: {m['total_modeled_cost']:.1f}")

    df_exp4 = pd.DataFrame(exp4_records)
    exp4_table_path = tables_dir / "hidden_outage_stress_test.csv"
    df_exp4.to_csv(exp4_table_path, index=False)
    print(f"Saved Hidden Outage Stress Test Table: {exp4_table_path}")

    # -------------------------------------------------------------
    # EXPERIMENT 5: EXACT SYNTHETIC REFERENCE SOLVER
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: EXACT SYNTHETIC REFERENCE SOLVER (DP & HYSTERESIS)")
    print("=" * 80)

    synth_hmm = SyntheticHMM(p_stay=0.88, c_run=(1.0, 1.0), c_wake=(5.0, 5.0), discount=0.92)
    dp_solver = DPSolver(synth_hmm, num_belief_points=51)
    dp_solver.solve()
    hyst_analysis = dp_solver.analyze_hysteresis()

    print(f"DP Reference Analysis Results:")
    print(f"  Fraction active when previously asleep: {hyst_analysis['fraction_active_when_asleep']*100:.1f}%")
    print(f"  Fraction active when previously awake:  {hyst_analysis['fraction_active_when_awake']*100:.1f}%")
    print(f"  Hysteresis Gap: {hyst_analysis['hysteresis_gap']:.4f}")
    print(f"  Hysteresis Confirmed: {hyst_analysis['hysteresis_confirmed']}")

    synth_out_path = tables_dir / "synthetic_reference_results.json"
    with open(synth_out_path, "w", encoding="utf-8") as f:
        json.dump(hyst_analysis, f, indent=2)
    print(f"Saved Synthetic Analysis: {synth_out_path}")

    # Plot synthetic hysteresis curve
    plot_synthetic_hysteresis(
        belief_grid=dp_solver.belief_grid,
        policy_asleep=dp_solver.policy[:, 0],
        policy_awake=dp_solver.policy[:, 1],
        output_path=fig_dir / "synthetic_hysteresis.png",
    )

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS SUCCESSFULLY EXECUTED AND ARTIFACTS GENERATED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
