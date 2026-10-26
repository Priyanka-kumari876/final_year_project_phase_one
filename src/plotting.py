"""
Plotting and Visualization Module
=================================
Produces publication-grade figures for research analysis:
1. Accuracy-Cost Pareto Frontier (Macro-F1 vs Total Modeled Cost)
2. Wake-to-Running Cost Ratio Sweeps (c_wake / c_run in {0, 1, 5, 10})
3. Outage Robustness Comparison (Macro-F1 across outage regimes)
4. Hysteresis Threshold Demonstration (from exact DP reference)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Use standard clean matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    cost_col: str = "total_modeled_cost",
    acc_col: str = "macro_f1",
    hue_col: str = "policy_name",
    title: str = "Macro-F1 vs Total Modeled Cost (Pareto Frontier)",
) -> None:
    """
    Plot Macro-F1 against Total Modeled Cost across all evaluated policies.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    policies = results_df[hue_col].unique()
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
    palette = sns.color_palette("tab10", len(policies))

    for idx, pol in enumerate(policies):
        subset = results_df[results_df[hue_col] == pol]
        ax.scatter(
            subset[cost_col],
            subset[acc_col],
            label=pol,
            s=90,
            marker=markers[idx % len(markers)],
            color=palette[idx],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=4,
        )

    ax.set_xlabel("Total Modeled Cost (Experimental Units)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Pareto Frontier plot to {output_path}")


def plot_wake_ratio_sweep(
    sweep_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> None:
    """
    Plot total wake-up cost and wake-up rate across wake-to-running ratios (0, 1, 5, 10).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    policies = sweep_df["policy_name"].unique()
    for pol in policies:
        sub = sweep_df[sweep_df["policy_name"] == pol].sort_values("wake_ratio")
        ax1.plot(sub["wake_ratio"], sub["total_wake_up_cost"], marker="o", lw=2, label=pol)
        ax2.plot(sub["wake_ratio"], sub["wake_ups_per_1000_windows"], marker="s", lw=2, label=pol)

    ax1.set_xlabel("Wake / Run Cost Ratio (c_wake / c_run)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Wake-Up Cost", fontsize=11, fontweight="bold")
    ax1.set_title("Wake-Up Cost Burden vs Ratio", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Wake / Run Cost Ratio (c_wake / c_run)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Wake-ups per 1,000 Windows", fontsize=11, fontweight="bold")
    ax2.set_title("Switching Frequency vs Ratio", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Wake Ratio sweep plot to {output_path}")


def plot_outage_comparison(
    outage_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> None:
    """
    Plot Macro-F1 across different outage scenarios (No Outage, Rare, Bursty Short, Bursty Long).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    scenarios = outage_df["scenario"].unique()
    policies = outage_df["policy_name"].unique()

    x = np.arange(len(scenarios))
    width = 0.8 / len(policies)

    for i, pol in enumerate(policies):
        sub = outage_df[outage_df["policy_name"] == pol]
        f1_vals = [float(sub[sub["scenario"] == sc]["macro_f1"].iloc[0]) if len(sub[sub["scenario"] == sc]) > 0 else 0.0 for sc in scenarios]
        ax.bar(x + i * width - 0.4 + width / 2, f1_vals, width, label=pol, alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Robustness Across Outage Conditions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Outage comparison plot to {output_path}")


def plot_synthetic_hysteresis(
    belief_grid: np.ndarray,
    policy_asleep: np.ndarray,
    policy_awake: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """
    Plot optimal action from DP solver when Sensor 1 is asleep vs awake,
    showing the hysteresis band visually.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Binary: is sensor 1 active in optimal action? (action 1 or 3)
    s1_asleep = np.isin(policy_asleep, [1, 3]).astype(int)
    s1_awake = np.isin(policy_awake, [1, 3]).astype(int)

    ax.step(belief_grid, s1_asleep, label="Sensor 1 Selected when Previously ASLEEP (Wake Threshold)", color="crimson", lw=2.2, where="mid")
    ax.step(belief_grid, s1_awake, label="Sensor 1 Kept Active when Previously AWAKE (Keep Threshold)", color="navy", lw=2.2, linestyle="--", where="mid")

    ax.fill_between(belief_grid, s1_asleep, s1_awake, color="purple", alpha=0.2, label="Hysteresis Band (Memory Value)")

    ax.set_xlabel("Belief State P(Activity = Active)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Optimal Action (Sensor 1 Active)", fontsize=11, fontweight="bold")
    ax.set_title("Exact Dynamic Programming Reference: Emergence of Hysteresis Band", fontsize=12, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OFF", "ON"])
    ax.legend(loc="lower center", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Synthetic Hysteresis plot to {output_path}")
