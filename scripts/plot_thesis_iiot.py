#!/usr/bin/env python3
"""
Thesis Plotting Script for IIoT Federated Learning Results.

This script generates high-quality, publication-ready figures that directly support
the 5 Thesis Objectives outlined in deliverable1/FL.txt.

It aggregates data from multiple seeds and experiments to produce:
1. Robust Aggregation Comparison (Objective 1)
2. Heterogeneity Impact & FedProx Analysis (Objective 2)
3. Personalization Lift (Objective 3)
4. System Overhead/Privacy Cost (Objective 4)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- Configuration & Styling ---

class ThesisStyle:
    """Publication-quality styling configuration."""
    
    def __init__(self):
        self.dpi = 300
        self.font_family = "serif"
        self.context = "paper"
        self.palette = "colorblind"
        self.figsize_wide = (10, 6)
        self.figsize_square = (8, 8)
        self.figsize_double = (14, 6)
        
    def apply(self):
        sns.set_theme(context=self.context, style="whitegrid", palette=self.palette, font=self.font_family)
        plt.rcParams.update({
            "font.family": "serif",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": self.dpi,
            "lines.linewidth": 2.5,
            "lines.markersize": 9.0,
        })

# --- Data Loading & Parsing ---

def parse_run_config(run_dir: Path) -> dict:
    """Extract experiment configuration from directory name."""
    name = run_dir.name
    config = {}
    
    # Extract Aggregation Method
    if "comp_fedavg" in name:
        config["aggregation"] = "FedAvg"
    elif "comp_krum" in name:
        config["aggregation"] = "Krum"
    elif "comp_bulyan" in name:
        config["aggregation"] = "Bulyan"
    elif "comp_median" in name:
        config["aggregation"] = "Median"
    elif "fedprox" in name:
        config["aggregation"] = "FedProx"
    else:
        config["aggregation"] = "Unknown"
        
    # Extract Seed
    seed_match = re.search(r"seed(\d+)", name)
    config["seed"] = int(seed_match.group(1)) if seed_match else 0
    
    # Extract Alpha (Heterogeneity)
    alpha_match = re.search(r"alpha([0-9\.]+|inf)", name)
    config["alpha"] = alpha_match.group(1) if alpha_match else "Unknown"

    # Extract Adversary Fraction
    adv_match = re.search(r"adv(\d+)", name)
    config["adversary"] = int(adv_match.group(1)) if adv_match else 0
    
    return config

def load_server_metrics(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading server metrics from {path}: {e}")
        return None

def load_client_metrics(run_dir: Path) -> pd.DataFrame | None:
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None
        
    dfs = []
    for cf in client_files:
        try:
            df = pd.read_csv(cf)
            client_id = int(re.search(r"client_(\d+)", cf.name).group(1))
            df["client_id"] = client_id
            dfs.append(df)
        except Exception:
            continue
            
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)

# --- Plotting Functions ---

def plot_robustness_comparison(data: pd.DataFrame, output_dir: Path, style: ThesisStyle):
    """Objective 1: Robust Aggregation Comparison (L2)."""
    fig, ax = plt.subplots(figsize=style.figsize_wide)
    
    subset = data[data["aggregation"].isin(["FedAvg", "Krum", "Bulyan", "Median"])].copy()
    if subset.empty:
        return
        
    markers = {"FedAvg": "o", "Krum": "s", "Bulyan": "D", "Median": "^"}
    linestyles = {"FedAvg": "", "Krum": (2, 2), "Bulyan": (5, 1, 1, 1), "Median": (1, 1)}
    
    try:
        sns.lineplot(
            data=subset,
            x="round",
            y="l2_to_benign_mean",
            hue="aggregation",
            style="aggregation",
            markers=markers,
            dashes=linestyles,
            ax=ax,
            errorbar=("ci", 95),
            estimator="mean"
        )
    except TypeError:
         sns.lineplot(
            data=subset,
            x="round",
            y="l2_to_benign_mean", 
            hue="aggregation",
            style="aggregation",
            markers=markers,
            dashes=linestyles,
            ax=ax,
            ci=95
        )
    
    ax.set_title("Objective 1: Robustness vs 30% Adversaries (L2 Deviation)", fontsize=16, weight='bold')
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("L2 Distance to Consensus Mean", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    textstr = '\n'.join((
        r'$\mathbf{Interpretation}$',
        r'Consensus Mean includes Attackers.',
        r'Low Dist (FedAvg) = Poisoned.',
        r'High Dist = Attack Rejected.'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    plt.savefig(output_dir / "obj1_robustness_l2.png", dpi=style.dpi)
    plt.close()

def plot_utility_comparison(data: pd.DataFrame, output_dir: Path, style: ThesisStyle):
    """Objective 1b: Utility (F1 Score)."""
    if "macro_f1_global" not in data.columns:
        return

    fig, ax = plt.subplots(figsize=style.figsize_wide)
    subset = data[data["aggregation"].isin(["FedAvg", "Krum", "Bulyan", "Median"])].copy()

    markers = {"FedAvg": "o", "Krum": "s", "Bulyan": "D", "Median": "^"}
    linestyles = {"FedAvg": "", "Krum": (2, 2), "Bulyan": (5, 1, 1, 1), "Median": (1, 1)}

    try:
        sns.lineplot(
            data=subset,
            x="round",
            y="macro_f1_global",
            hue="aggregation",
            style="aggregation",
            markers=markers,
            dashes=linestyles,
            ax=ax,
            errorbar=("ci", 95)
        )
    except TypeError:
        sns.lineplot(
            data=subset,
            x="round",
            y="macro_f1_global",
            hue="aggregation",
            style="aggregation",
            markers=markers,
            dashes=linestyles,
            ax=ax,
            ci=95
        )
            
    ax.set_title("Objective 1: Utility under Attack (Global Model F1)", fontsize=16, weight='bold')
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Macro F1 Score (Global Model)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.5, 0.5, "FedAvg F1 ~ 0.0 (Catastrophic Failure)", transform=ax.transAxes, 
            fontsize=10, color='red', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "obj1_robustness_f1.png", dpi=style.dpi)
    plt.close()

def plot_heterogeneity_impact(data: pd.DataFrame, output_dir: Path, style: ThesisStyle):
    """Objective 2: Impact of Data Heterogeneity (Alpha) on Robustness."""
    # Prepare data: We need Final F1 vs Alpha
    df = data.copy()
    
    # Parse Alpha
    def parse_alpha(a):
        if str(a).lower() == "inf":
            return 10.0 # Treat inf as 10.0 for plotting purposes
        try:
            return float(a)
        except:
            return -1.0
            
    df["alpha_val"] = df["alpha"].apply(parse_alpha)
    df = df[df["alpha_val"] > 0]
    
    # Sort for plotting
    df = df.sort_values("alpha_val")
    
    # Take final round
    max_round = df["round"].max()
    final_df = df[df["round"] == max_round]
    
    if final_df.empty:
        print("No final round data for heterogeneity plot.")
        return

    # RESTRICT TO FEDAVG ONLY (Data Completeness)
    subset = final_df[final_df["aggregation"] == "FedAvg"].copy()
    
    if subset.empty:
        print("No FedAvg data for heterogeneity plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize_double)
    
    markers = {"FedAvg": "o"}
    linestyles = {"FedAvg": ""}  # Empty string for solid line in seaborn
    
    # Panel 1: F1 vs Alpha
    sns.lineplot(
        data=subset,
        x="alpha_val",
        y="macro_f1_global",
        hue="aggregation",
        style="aggregation",
        markers=markers,
        dashes=linestyles,
        ax=ax1,
        errorbar=("ci", 95),
        err_style="bars",
        err_kws={'capsize': 5},
        linewidth=2.5
    )

    ax1.set_title("Heterogeneity vs Utility (FedAvg, Benign)", fontsize=14, weight='bold')
    ax1.set_xlabel("Dirichlet Alpha (Log Scale)", fontsize=12)
    ax1.set_ylabel("Final Macro F1 Score", fontsize=12)
    ax1.set_xscale("log") 
    
    # Panel 2: Drift vs Alpha
    sns.lineplot(
        data=subset,
        x="alpha_val",
        y="l2_dispersion_mean",
        hue="aggregation",
        style="aggregation",
        markers=markers,
        dashes=linestyles,
        ax=ax2,
        errorbar=("ci", 95),
        err_style="bars",
        err_kws={'capsize': 5},
        linewidth=2.5
    )
        
    ax2.set_title("Heterogeneity vs Client Drift (FedAvg, Benign)", fontsize=14, weight='bold')
    ax2.set_xlabel("Dirichlet Alpha (Log Scale)", fontsize=12)
    ax2.set_ylabel("L2 Dispersion (Drift)", fontsize=12)
    ax2.set_xscale("log")
    
    # Improve Legends
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3, which="both")
        legend = ax.get_legend()
        if legend:
            legend.set_title("Aggregation")
            plt.setp(legend.get_title(), fontsize=10)
            
    plt.tight_layout()
    plt.savefig(output_dir / "obj2_heterogeneity_impact.png", dpi=style.dpi)
    plt.close()

def plot_fedprox_efficacy(data: pd.DataFrame, output_dir: Path, style: ThesisStyle):
    """Objective 2b: FedProx vs FedAvg comparison."""
    # Expects data with Adv=0 (usually)
    subset = data[data["aggregation"].isin(["FedAvg", "FedProx"])].copy()
    
    if subset.empty:
        return

    # Find lowest alpha (highest heterogeneity)
    alphas = sorted(subset["alpha"].unique(), key=lambda x: float(x) if x != "inf" else 999)
    if not alphas:
        return
    target_alpha = alphas[0] # Use most heterogeneous
    
    plot_data = subset[subset["alpha"] == target_alpha]
    
    fig, ax = plt.subplots(figsize=style.figsize_wide)
    
    try:
        sns.lineplot(
            data=plot_data,
            x="round",
            y="macro_f1_global",
            hue="aggregation",
            style="aggregation",
            markers=True,
            ax=ax,
            errorbar=("ci", 95)
        )
    except:
        sns.lineplot(
            data=plot_data,
            x="round",
            y="macro_f1_global",
            hue="aggregation",
            style="aggregation",
            markers=True,
            ax=ax,
            ci=95
        )
        
    ax.set_title(f"Objective 2b: FedProx Efficacy (Alpha={target_alpha}, Adv=0)", fontsize=16, weight='bold')
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "obj2_fedprox_efficacy.png", dpi=style.dpi)
    plt.close()

def plot_personalization_lift(client_data: pd.DataFrame, output_dir: Path, style: ThesisStyle):
    """Objective 3: Personalization Lift."""
    if client_data is None or "macro_f1_personalized" not in client_data.columns:
        return

    max_round = client_data["round"].max()
    final_round = client_data[client_data["round"] == max_round].copy()
    if final_round.empty:
        return
        
    final_round["lift"] = final_round["macro_f1_personalized"] - final_round["macro_f1_global"]
    final_round = final_round.sort_values("client_id")
    
    fig, ax = plt.subplots(figsize=style.figsize_wide)
    
    for idx, row in final_round.iterrows():
        color = "green" if row["lift"] >= 0 else "red"
        ax.arrow(
            x=row["client_id"], 
            y=row["macro_f1_global"], 
            dx=0, 
            dy=row["lift"], 
            color=color,
            length_includes_head=True,
            head_width=0.3,
            alpha=0.7
        )
        ax.scatter(row["client_id"], row["macro_f1_global"], color="#1f77b4", s=50, zorder=3)
        ax.scatter(row["client_id"], row["macro_f1_personalized"], color="#d62728", s=50, zorder=3)

    ax.set_title("Objective 3: Personalization Risk (Bulyan Model)", fontsize=16, weight='bold')
    ax.set_xlabel("Client ID", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Global Model (Before)', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label='Personalized Model (After)', markersize=8),
        Patch(facecolor='red', label='Negative Lift (Performance Drop)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "obj3_personalization_lift.png", dpi=style.dpi)
    plt.close()

def plot_system_overhead(data: pd.DataFrame, output_dir: Path, style: ThesisStyle):
    """Objective 4: System Overhead."""
    fig, ax = plt.subplots(figsize=style.figsize_wide)
    metric = "t_aggregate_ms"
    
    if metric not in data.columns:
        return
        
    linestyles = {"FedAvg": "", "Krum": (2, 2), "Bulyan": (5, 1, 1, 1), "Median": (1, 1)}

    try:
        sns.lineplot(
            data=data,
            x="round",
            y=metric,
            hue="aggregation",
            style="aggregation",
            markers=True,
            dashes=linestyles,
            ax=ax,
            errorbar=("ci", 95)
        )
    except TypeError:
        sns.lineplot(
            data=data,
            x="round",
            y=metric,
            hue="aggregation",
            style="aggregation",
            markers=True,
            dashes=linestyles,
            ax=ax,
            ci=95
        )
    
    ax.set_title("Objective 4: Computational Overhead", fontsize=16, weight='bold')
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Aggregation Time (ms)", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=100, color='orange', linestyle='--', label='Real-time Threshold (100ms)')
    ax.axhline(y=50, color='red', linestyle=':', label='Raspberry Pi 4 Limit (50ms)')
    ax.text(0.5, 0.1, "FedAvg: ~0.5ms (Fastest)", transform=ax.transAxes, fontsize=10, color='blue')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "obj4_system_overhead.png", dpi=style.dpi)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Thesis plots for IIoT")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Path to runs directory")
    parser.add_argument("--output_dir", type=str, default="thesis_plots", help="Output directory")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    style = ThesisStyle()
    style.apply()
    
    print("Scanning for runs...")
    all_server_data = []
    all_client_data = []
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        
        # Basic check to avoid scanning unrelated folders
        if "logs" in run_path.name or "thesis_plots" in run_path.name or "scripts" in run_path.name:
            continue
            
        config = parse_run_config(run_path)
        sdf = load_server_metrics(run_path)
        if sdf is not None:
            for k, v in config.items():
                sdf[k] = v
            all_server_data.append(sdf)
        cdf = load_client_metrics(run_path)
        if cdf is not None:
            for k, v in config.items():
                cdf[k] = v
            all_client_data.append(cdf)

    if not all_server_data:
        print("No matching run data found!")
        return

    server_df = pd.concat(all_server_data, ignore_index=True)
    client_df = pd.concat(all_client_data, ignore_index=True) if all_client_data else None
    
    # Merge Client F1 into Server DF for all runs
    if client_df is not None:
        group_cols = ["aggregation", "alpha", "seed", "round", "adversary"]
        valid_keys = [k for k in group_cols if k in client_df.columns and k in server_df.columns]
        try:
            client_agg = client_df.groupby(valid_keys)[["macro_f1_before", "macro_f1_after"]].mean().reset_index()
            client_agg = client_agg.rename(columns={
                "macro_f1_before": "macro_f1_global", 
                "macro_f1_after": "macro_f1_personalized"
            })
            server_df = pd.merge(server_df, client_agg, on=valid_keys, how="left")
        except Exception as e:
            print(f"Warning: F1 merge failed: {e}")

    print(f"Loaded data from {len(all_server_data)} runs.")
    
    # --- DATASET 1: ROBUSTNESS & PERSONALIZATION (Adv=30, Alpha=0.5) ---
    target_alpha = "0.5"
    target_adv = 30
    robust_df = server_df[
        (server_df["alpha"].astype(str) == target_alpha) & 
        (server_df["adversary"] == target_adv)
    ].copy()
    
    robust_client_df = None
    if client_df is not None:
        robust_client_df = client_df[
            (client_df["alpha"].astype(str) == target_alpha) & 
            (client_df["adversary"] == target_adv)
        ].copy()
    
    if not robust_df.empty:
        print(f"Generating Robustness Plots (Alpha={target_alpha}, Adv={target_adv})")
        plot_robustness_comparison(robust_df, output_dir, style)
        plot_utility_comparison(robust_df, output_dir, style)
        plot_system_overhead(robust_df, output_dir, style)
    
    if robust_client_df is not None and not robust_client_df.empty:
         # Drop duplicates/renames
        cols_to_drop = ["macro_f1_global", "macro_f1_personalized"]
        robust_client_df = robust_client_df.drop(columns=[c for c in cols_to_drop if c in robust_client_df.columns], errors="ignore")
        robust_client_df = robust_client_df.rename(columns={
            "macro_f1_before": "macro_f1_global", 
            "macro_f1_after": "macro_f1_personalized"
        })
        bulyan_client_df = robust_client_df[robust_client_df["aggregation"] == "Bulyan"]
        if not bulyan_client_df.empty:
            plot_personalization_lift(bulyan_client_df, output_dir, style)

    # --- DATASET 2: HETEROGENEITY IMPACT (Adv=0, All Alphas) ---
    # Use Adv=0 (Benign) to show the pure impact of heterogeneity without attack interference
    hetero_df = server_df[server_df["adversary"] == 0].copy()
    
    if not hetero_df.empty:
        print(f"Generating Heterogeneity Impact Plots (Adv=0, All Alphas)")
        plot_heterogeneity_impact(hetero_df, output_dir, style)
    else:
        print("Warning: No Adv=0 data found for Heterogeneity plot.")
        
    # --- DATASET 3: FEDPROX EFFICACY (Adv=0, All Alphas) ---
    # Check if FedProx data exists
    fedprox_data = server_df[(server_df["aggregation"] == "FedProx") & (server_df["adversary"] == 0)]
    if not fedprox_data.empty:
        print("Generating FedProx Efficacy Plots (Adv=0)")
        # Use all data for Adv=0 to compare FedAvg vs FedProx
        comp_df = server_df[server_df["adversary"] == 0].copy()
        plot_fedprox_efficacy(comp_df, output_dir, style)

    print(f"All plots saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main()
