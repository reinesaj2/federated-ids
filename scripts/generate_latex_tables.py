#!/usr/bin/env python3
"""
Generate LaTeX Tables for Thesis

Creates publication-ready LaTeX tables summarizing experimental results
with mean ± 95% confidence intervals across random seeds.

Supports all 5 comparison dimensions:
1. Aggregation methods
2. Data heterogeneity
3. Attack resilience
4. Privacy-utility tradeoff
5. Personalization benefit
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_server_macro_f1_from_clients(run_dir: Path, round_num: int) -> Optional[float]:
    """Compute server macro-F1 by averaging client scores for a round."""
    client_f1_scores = []

    for client_csv in run_dir.glob("client_*_metrics.csv"):
        try:
            client_df = pd.read_csv(client_csv)
            round_data = client_df[client_df["round"] == round_num]

            if not round_data.empty:
                f1_value = None
                if "macro_f1_after" in round_data.columns:
                    f1_value = round_data["macro_f1_after"].iloc[0]
                elif "macro_f1_argmax" in round_data.columns:
                    f1_value = round_data["macro_f1_argmax"].iloc[0]

                if f1_value is not None and not pd.isna(f1_value):
                    client_f1_scores.append(float(f1_value))
        except Exception:
            continue

    if not client_f1_scores:
        return None

    return float(np.mean(client_f1_scores))


def load_experiment_results(runs_dir: Path, dataset: str) -> pd.DataFrame:
    """Load all experiment results from runs directory."""
    all_data = []

    for pattern in ["comp_*", "d2_*"]:
        for run_dir in runs_dir.glob(pattern):
            config = {}
            config_file = run_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

            if config.get("dataset") != dataset:
                continue

            metrics_file = run_dir / "metrics.csv"
            if not metrics_file.exists():
                continue

            try:
                df = pd.read_csv(metrics_file)
                if df.empty:
                    continue

                final_round = df.iloc[-1]["round"]
                macro_f1 = compute_server_macro_f1_from_clients(run_dir, final_round)

                if macro_f1 is None:
                    continue

                result = {
                    "run_dir": str(run_dir.name),
                    "aggregation": config.get("aggregation", "unknown"),
                    "alpha": config.get("alpha", 1.0),
                    "adversary_fraction": config.get("adversary_fraction", 0.0),
                    "dp_enabled": config.get("dp_enabled", False),
                    "dp_noise_multiplier": config.get("dp_noise_multiplier", 0.0),
                    "personalization_epochs": config.get("personalization_epochs", 0),
                    "seed": config.get("seed", 42),
                    "dataset": config.get("dataset", "unknown"),
                    "macro_f1": macro_f1,
                    "final_round": final_round,
                }
                all_data.append(result)

            except Exception as e:
                logger.warning(f"Error loading {run_dir.name}: {e}")
                continue

    return pd.DataFrame(all_data)


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute mean and 95% confidence interval."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))

    if len(values) == 1:
        return mean, mean, mean

    se = stats.sem(values)
    margin = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

    return mean, mean - margin, mean + margin


def format_ci(mean: float, ci_lower: float, ci_upper: float) -> str:
    """Format value with confidence interval for LaTeX."""
    if np.isnan(mean):
        return "--"

    margin = (ci_upper - ci_lower) / 2
    return f"{mean:.3f} $\\pm$ {margin:.3f}"


def generate_aggregation_table(df: pd.DataFrame, dataset: str) -> str:
    """Generate LaTeX table for aggregation comparison."""
    baseline_df = df[(df["alpha"] == 1.0) & (df["adversary_fraction"] == 0.0) & (~df["dp_enabled"]) & (df["personalization_epochs"] == 0)]

    if baseline_df.empty:
        return "% No aggregation data available\n"

    rows = []
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_df = baseline_df[baseline_df["aggregation"] == agg]
        if agg_df.empty:
            continue

        scores = agg_df["macro_f1"].values
        mean, ci_lower, ci_upper = compute_confidence_interval(scores)

        rows.append(
            {
                "Method": agg.upper() if agg == "fedavg" else agg.capitalize(),
                "Macro-F1": format_ci(mean, ci_lower, ci_upper),
                "n": len(scores),
            }
        )

    if not rows:
        return "% No aggregation results\n"

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Aggregation Method Comparison ({dataset.upper()})}}" f"\\label{{tab:agg_{dataset}}}\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "Method & Macro-F1 & Seeds \\\\\n"
    latex += "\\midrule\n"

    for row in rows:
        latex += f"{row['Method']} & {row['Macro-F1']} & {row['n']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_heterogeneity_table(df: pd.DataFrame, dataset: str) -> str:
    """Generate LaTeX table for heterogeneity comparison."""
    hetero_df = df[
        (df["aggregation"] == "fedavg") & (df["adversary_fraction"] == 0.0) & (~df["dp_enabled"]) & (df["personalization_epochs"] == 0)
    ]

    if hetero_df.empty:
        return "% No heterogeneity data available\n"

    rows = []
    for alpha in sorted(hetero_df["alpha"].unique()):
        alpha_df = hetero_df[hetero_df["alpha"] == alpha]
        scores = alpha_df["macro_f1"].values
        mean, ci_lower, ci_upper = compute_confidence_interval(scores)

        alpha_label = "IID" if alpha == 1.0 else f"$\\alpha={alpha}$"
        rows.append(
            {
                "Alpha": alpha_label,
                "Macro-F1": format_ci(mean, ci_lower, ci_upper),
                "n": len(scores),
            }
        )

    if not rows:
        return "% No heterogeneity results\n"

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Data Heterogeneity Impact ({dataset.upper()})}}" f"\\label{{tab:hetero_{dataset}}}\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "Distribution & Macro-F1 & Seeds \\\\\n"
    latex += "\\midrule\n"

    for row in rows:
        latex += f"{row['Alpha']} & {row['Macro-F1']} & {row['n']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_attack_table(df: pd.DataFrame, dataset: str) -> str:
    """Generate LaTeX table for attack resilience."""
    rows = []

    for agg in ["fedavg", "krum", "bulyan", "median"]:
        for adv_frac in [0.0, 0.1, 0.3]:
            attack_df = df[
                (df["aggregation"] == agg)
                & (df["adversary_fraction"] == adv_frac)
                & (df["alpha"] == 1.0)
                & (~df["dp_enabled"])
                & (df["personalization_epochs"] == 0)
            ]

            if attack_df.empty:
                continue

            scores = attack_df["macro_f1"].values
            mean, ci_lower, ci_upper = compute_confidence_interval(scores)

            agg_label = agg.upper() if agg == "fedavg" else agg.capitalize()
            rows.append(
                {
                    "Method": agg_label,
                    "Adv%": f"{int(adv_frac * 100)}\\%",
                    "Macro-F1": format_ci(mean, ci_lower, ci_upper),
                    "n": len(scores),
                }
            )

    if not rows:
        return "% No attack data available\n"

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Attack Resilience ({dataset.upper()})}}" f"\\label{{tab:attack_{dataset}}}\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "Method & Adversaries & Macro-F1 & Seeds \\\\\n"
    latex += "\\midrule\n"

    for row in rows:
        latex += f"{row['Method']} & {row['Adv%']} & " f"{row['Macro-F1']} & {row['n']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_privacy_table(df: pd.DataFrame, dataset: str) -> str:
    """Generate LaTeX table for privacy-utility tradeoff."""
    rows = []

    baseline_df = df[
        (df["aggregation"] == "fedavg")
        & (df["adversary_fraction"] == 0.0)
        & (~df["dp_enabled"])
        & (df["alpha"] == 1.0)
        & (df["personalization_epochs"] == 0)
    ]

    if not baseline_df.empty:
        scores = baseline_df["macro_f1"].values
        mean, ci_lower, ci_upper = compute_confidence_interval(scores)
        rows.append(
            {
                "DP": "None",
                "Noise": "--",
                "Macro-F1": format_ci(mean, ci_lower, ci_upper),
                "n": len(scores),
            }
        )

    dp_noises = df[df["dp_enabled"]]["dp_noise_multiplier"].unique()
    for noise in sorted(dp_noises):
        dp_df = df[
            (df["aggregation"] == "fedavg")
            & (df["dp_enabled"])
            & (df["dp_noise_multiplier"] == noise)
            & (df["alpha"] == 1.0)
            & (df["adversary_fraction"] == 0.0)
            & (df["personalization_epochs"] == 0)
        ]

        if dp_df.empty:
            continue

        scores = dp_df["macro_f1"].values
        mean, ci_lower, ci_upper = compute_confidence_interval(scores)

        rows.append(
            {
                "DP": "Enabled",
                "Noise": f"${noise:.1f}$",
                "Macro-F1": format_ci(mean, ci_lower, ci_upper),
                "n": len(scores),
            }
        )

    if not rows:
        return "% No privacy data available\n"

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Privacy-Utility Tradeoff ({dataset.upper()})}}" f"\\label{{tab:privacy_{dataset}}}\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "DP & Noise $\\sigma$ & Macro-F1 & Seeds \\\\\n"
    latex += "\\midrule\n"

    for row in rows:
        latex += f"{row['DP']} & {row['Noise']} & " f"{row['Macro-F1']} & {row['n']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_personalization_table(df: pd.DataFrame, dataset: str) -> str:
    """Generate LaTeX table for personalization benefit."""
    rows = []

    for epochs in sorted(df["personalization_epochs"].unique()):
        pers_df = df[
            (df["aggregation"] == "fedavg")
            & (df["personalization_epochs"] == epochs)
            & (df["alpha"] == 1.0)
            & (df["adversary_fraction"] == 0.0)
            & (~df["dp_enabled"])
        ]

        if pers_df.empty:
            continue

        scores = pers_df["macro_f1"].values
        mean, ci_lower, ci_upper = compute_confidence_interval(scores)

        epochs_label = "Global Only" if epochs == 0 else f"{epochs}"
        rows.append(
            {
                "Epochs": epochs_label,
                "Macro-F1": format_ci(mean, ci_lower, ci_upper),
                "n": len(scores),
            }
        )

    if not rows:
        return "% No personalization data available\n"

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Personalization Benefit ({dataset.upper()})}}" f"\\label{{tab:pers_{dataset}}}\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "Personalization & Macro-F1 & Seeds \\\\\n"
    latex += "\\midrule\n"

    for row in rows:
        latex += f"{row['Epochs']} & {row['Macro-F1']} & {row['n']} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for thesis")
    parser.add_argument(
        "--dimension",
        type=str,
        choices=[
            "aggregation",
            "heterogeneity",
            "attack",
            "privacy",
            "personalization",
            "all",
        ],
        default="all",
        help="Which dimension to generate table for",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cic", "unsw", "both"],
        default="both",
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("runs"),
        help="Directory with experiment runs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tables"),
        help="Output directory for LaTeX tables",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["cic", "unsw"] if args.dataset == "both" else [args.dataset]
    dimensions = (
        [
            "aggregation",
            "heterogeneity",
            "attack",
            "privacy",
            "personalization",
        ]
        if args.dimension == "all"
        else [args.dimension]
    )

    generators = {
        "aggregation": generate_aggregation_table,
        "heterogeneity": generate_heterogeneity_table,
        "attack": generate_attack_table,
        "privacy": generate_privacy_table,
        "personalization": generate_personalization_table,
    }

    for dataset in datasets:
        logger.info(f"Loading {dataset.upper()} experiment results...")
        df = load_experiment_results(args.runs_dir, dataset)

        if df.empty:
            logger.warning(f"No data found for {dataset}")
            continue

        logger.info(f"Loaded {len(df)} experiments for {dataset.upper()}")

        for dim in dimensions:
            logger.info(f"Generating {dim} table for {dataset.upper()}...")
            latex_table = generators[dim](df, dataset)

            output_file = args.output_dir / f"table_{dim}_{dataset}.tex"
            output_file.write_text(latex_table)
            logger.info(f"Wrote {output_file}")

    logger.info("Table generation complete!")


if __name__ == "__main__":
    main()
