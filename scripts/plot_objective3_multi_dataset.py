#!/usr/bin/env python3
"""
OBJECTIVE 3: Multi-Dataset Validation Plots
Cross-dataset comparison between Edge-IIoT and CIC-IDS2017 datasets.

Generates publication-quality visualizations (300 DPI) for thesis comparing:
- Side-by-side F1 performance across datasets
- Per-class F1 scores (attack type detection rates)
- Confusion matrix summaries
- Training convergence comparison
- Statistical significance indicators

Author: Thesis Research
"""

import ast
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/Users/abrahamreines/Documents/Thesis")
DATA_DIR = BASE_DIR / "cluster-experiments" / "cluster-runs"
OUTPUT_DIR = BASE_DIR / "cluster-experiments" / "thesis_plots" / "objective3"

COLORBLIND_PALETTE = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#CC78BC",  # Purple
    "#ECE133",  # Yellow
    "#56B4E9",  # Light Blue
    "#D55E00",  # Vermillion
    "#F0E442",  # Pale Yellow
]

DATASET_COLORS = {
    "Edge-IIoT": "#0173B2",
    "CIC-IDS2017": "#DE8F05",
}

AGGREGATOR_MARKERS = {
    "fedavg": "o",
    "bulyan": "s",
    "krum": "^",
    "median": "D",
}


@dataclass
class PlotConfig:
    """Publication-ready plot configuration."""

    dpi: int = 300
    figsize_single: tuple = (8, 6)
    figsize_wide: tuple = (12, 6)
    figsize_tall: tuple = (10, 12)
    figsize_square: tuple = (10, 10)
    font_family: str = "sans-serif"
    font_size: int = 11
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    tick_size: int = 10
    linewidth: float = 2.0
    markersize: float = 8.0
    alpha: float = 0.8
    ci_alpha: float = 0.2
    grid_alpha: float = 0.3

    def apply(self):
        """Apply matplotlib style settings."""
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "legend.fontsize": self.legend_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "figure.dpi": self.dpi,
                "savefig.dpi": self.dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.1,
            }
        )
        sns.set_style("whitegrid")


CONFIG = PlotConfig()


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================


def parse_dict_string(s: str) -> dict:
    """Parse JSON-like string to dictionary."""
    if pd.isna(s) or s == "":
        return {}
    try:
        return json.loads(s.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return {}


def identify_dataset_from_path(path: Path) -> str:
    """Identify dataset from experiment directory name."""
    name = path.name.lower()
    if "edge-iiotset" in name or "iiot" in name:
        return "Edge-IIoT"
    elif "datasetcic" in name or "_cic" in name:
        return "CIC-IDS2017"
    elif "unsw" in name:
        return "UNSW-NB15"
    return "Unknown"


def extract_config_from_dirname(dirname: str) -> dict:
    """Extract experiment configuration from directory name."""
    config = {
        "aggregator": None,
        "alpha": None,
        "adv": None,
        "dp": None,
        "pers": None,
        "mu": None,
        "seed": None,
        "dataset": None,
    }

    # Extract aggregator
    agg_match = re.search(r"comp_(\w+)_alpha", dirname)
    if agg_match:
        config["aggregator"] = agg_match.group(1)

    # Extract alpha
    alpha_match = re.search(r"alpha([\d.]+)", dirname)
    if alpha_match:
        config["alpha"] = float(alpha_match.group(1))

    # Extract adversary fraction
    adv_match = re.search(r"adv(\d+)", dirname)
    if adv_match:
        config["adv"] = int(adv_match.group(1))

    # Extract dp
    dp_match = re.search(r"dp(\d+)", dirname)
    if dp_match:
        config["dp"] = int(dp_match.group(1))

    # Extract personalization epochs
    pers_match = re.search(r"pers(\d+)", dirname)
    if pers_match:
        config["pers"] = int(pers_match.group(1))

    # Extract mu (fedprox)
    mu_match = re.search(r"mu([\d.]+)", dirname)
    if mu_match:
        config["mu"] = float(mu_match.group(1))

    # Extract seed
    seed_match = re.search(r"seed(\d+)", dirname)
    if seed_match:
        config["seed"] = int(seed_match.group(1))

    # Identify dataset
    config["dataset"] = identify_dataset_from_path(Path(dirname))

    return config


def load_experiment_data(exp_dir: Path) -> dict:
    """Load all data from an experiment directory."""
    data = {
        "config": None,
        "server_metrics": None,
        "client_metrics": [],
        "path": exp_dir,
    }

    # Load config
    config_path = exp_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data["config"] = json.load(f)
        except Exception:
            pass

    # Load server metrics (with error handling for malformed CSVs)
    metrics_path = exp_dir / "metrics.csv"
    if metrics_path.exists():
        try:
            data["server_metrics"] = pd.read_csv(metrics_path, on_bad_lines="skip")
        except Exception:
            try:
                data["server_metrics"] = pd.read_csv(
                    metrics_path, on_bad_lines="skip", engine="python"
                )
            except Exception:
                pass

    # Load client metrics
    for client_file in sorted(exp_dir.glob("client_*_metrics.csv")):
        try:
            df = pd.read_csv(client_file, on_bad_lines="skip")
            client_id = int(re.search(r"client_(\d+)", client_file.name).group(1))
            data["client_metrics"].append({"client_id": client_id, "data": df})
        except Exception:
            try:
                df = pd.read_csv(client_file, on_bad_lines="skip", engine="python")
                client_id = int(re.search(r"client_(\d+)", client_file.name).group(1))
                data["client_metrics"].append({"client_id": client_id, "data": df})
            except Exception:
                pass

    return data


def find_matching_experiments(
    data_dir: Path,
    alpha: float = 0.5,
    adv_fractions: list = None,
    aggregators: list = None,
    seeds: list = None,
) -> dict:
    """
    Find matching experiments between IIoT and CIC datasets.

    Returns dict mapping configuration key to {iiot: path, cic: path}
    """
    if adv_fractions is None:
        adv_fractions = [0, 30]
    if aggregators is None:
        aggregators = ["fedavg", "bulyan"]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    experiments = {}

    for exp_path in data_dir.iterdir():
        if not exp_path.is_dir():
            continue

        dirname = exp_path.name
        config = extract_config_from_dirname(dirname)

        # Filter by criteria
        if config["alpha"] != alpha:
            continue
        if config["adv"] not in adv_fractions:
            continue
        if config["aggregator"] not in aggregators:
            continue
        if config["seed"] not in seeds:
            continue
        if config["dataset"] == "Unknown":
            continue

        # Create configuration key (without dataset and seed for matching)
        config_key = (
            f"{config['aggregator']}_alpha{config['alpha']}_"
            f"adv{config['adv']}_mu{config['mu']}_seed{config['seed']}"
        )

        if config_key not in experiments:
            experiments[config_key] = {"Edge-IIoT": None, "CIC-IDS2017": None}

        experiments[config_key][config["dataset"]] = exp_path

    # Filter to only experiments with both datasets
    matched = {
        k: v
        for k, v in experiments.items()
        if v["Edge-IIoT"] is not None and v["CIC-IDS2017"] is not None
    }

    return matched


def aggregate_client_metrics(client_data_list: list, final_round: int = None) -> dict:
    """Aggregate metrics across all clients for final round."""
    if not client_data_list:
        return {}

    all_f1_per_class = []
    all_macro_f1 = []
    all_conf_matrices = []
    class_names = None

    for client in client_data_list:
        df = client["data"]
        if df.empty:
            continue

        # Get final round data
        if final_round is not None:
            final_data = df[df["round"] == final_round]
        else:
            final_data = df[df["round"] == df["round"].max()]

        if final_data.empty:
            continue

        row = final_data.iloc[-1]

        # Extract macro F1
        if "macro_f1_after" in df.columns and pd.notna(row.get("macro_f1_after")):
            all_macro_f1.append(row["macro_f1_after"])

        # Extract per-class F1
        if "f1_per_class_after" in df.columns:
            f1_dict = parse_dict_string(str(row.get("f1_per_class_after", "{}")))
            if f1_dict:
                all_f1_per_class.append(f1_dict)

        # Extract confusion matrix
        if "confusion_matrix_normalized" in df.columns:
            cm = parse_dict_string(str(row.get("confusion_matrix_normalized", "[]")))
            if cm:
                all_conf_matrices.append(np.array(cm))

        # Extract class names
        if class_names is None and "confusion_matrix_class_names" in df.columns:
            names = parse_dict_string(
                str(row.get("confusion_matrix_class_names", "[]"))
            )
            if names:
                class_names = names

    result = {
        "macro_f1_mean": np.mean(all_macro_f1) if all_macro_f1 else None,
        "macro_f1_std": np.std(all_macro_f1) if all_macro_f1 else None,
        "macro_f1_all": all_macro_f1,
        "f1_per_class": all_f1_per_class,
        "confusion_matrices": all_conf_matrices,
        "class_names": class_names,
    }

    return result


def get_convergence_data(client_data_list: list) -> pd.DataFrame:
    """Extract round-by-round convergence data across clients."""
    if not client_data_list:
        return pd.DataFrame()

    all_rows = []
    for client in client_data_list:
        df = client["data"]
        if df.empty:
            continue

        for _, row in df.iterrows():
            all_rows.append(
                {
                    "round": row["round"],
                    "client_id": client["client_id"],
                    "macro_f1": row.get("macro_f1_after", np.nan),
                    "loss": row.get("loss_after", np.nan),
                    "accuracy": row.get("acc_after", np.nan),
                }
            )

    return pd.DataFrame(all_rows)


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================


def compute_confidence_interval(data: list, confidence: float = 0.95) -> tuple:
    """Compute confidence interval using t-distribution."""
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data)

    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

    return mean - ci, mean, mean + ci


def perform_paired_ttest(group1: list, group2: list) -> tuple:
    """Perform paired t-test and return p-value and effect size."""
    if len(group1) < 2 or len(group2) < 2:
        return None, None

    if len(group1) != len(group2):
        # Use independent t-test if paired not possible
        stat, pval = stats.ttest_ind(group1, group2)
    else:
        stat, pval = stats.ttest_rel(group1, group2)

    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1) + (len(group2) - 1) * np.var(group2))
        / (len(group1) + len(group2) - 2)
    )
    if pooled_std > 0:
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
    else:
        effect_size = 0

    return pval, effect_size


def significance_stars(pval: float) -> str:
    """Convert p-value to significance stars."""
    if pval is None:
        return ""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return ""


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_sidebyside_f1_comparison(
    matched_experiments: dict,
    output_path: Path,
    title: str = "Cross-Dataset F1 Score Comparison",
):
    """
    Plot side-by-side grouped bar chart comparing F1 scores across datasets.

    Groups by aggregator + adversary condition, with bars for each dataset.
    """
    CONFIG.apply()

    # Collect data
    plot_data = []

    for config_key, paths in matched_experiments.items():
        # Parse config
        parts = config_key.split("_")
        aggregator = parts[0]
        adv = int(re.search(r"adv(\d+)", config_key).group(1))

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            client_agg = aggregate_client_metrics(exp_data["client_metrics"])

            if client_agg.get("macro_f1_all"):
                for f1_val in client_agg["macro_f1_all"]:
                    plot_data.append(
                        {
                            "Aggregator": aggregator.upper(),
                            "Adversary %": f"{adv}%",
                            "Dataset": dataset_name,
                            "Macro F1": f1_val,
                            "config_key": config_key,
                        }
                    )

    if not plot_data:
        print("No data found for F1 comparison plot")
        return

    df = pd.DataFrame(plot_data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=CONFIG.figsize_wide)

    # Group by aggregator + adversary
    df["Group"] = df["Aggregator"] + "\n" + df["Adversary %"] + " Adv"

    # Order groups
    group_order = sorted(df["Group"].unique())

    # Plot
    sns.barplot(
        data=df,
        x="Group",
        y="Macro F1",
        hue="Dataset",
        palette=DATASET_COLORS,
        ax=ax,
        capsize=0.05,
        errwidth=1.5,
        alpha=CONFIG.alpha,
    )

    ax.set_xlabel("Configuration", fontsize=CONFIG.label_size)
    ax.set_ylabel("Macro F1 Score", fontsize=CONFIG.label_size)
    ax.set_title(title, fontsize=CONFIG.title_size, fontweight="bold")
    ax.set_ylim(0, 1.05)

    # Add significance annotations
    groups = df.groupby("Group")
    for i, group_name in enumerate(group_order):
        if group_name not in groups.groups:
            continue

        group_data = groups.get_group(group_name)
        iiot_vals = group_data[group_data["Dataset"] == "Edge-IIoT"]["Macro F1"].values
        cic_vals = group_data[group_data["Dataset"] == "CIC-IDS2017"]["Macro F1"].values

        if len(iiot_vals) > 1 and len(cic_vals) > 1:
            pval, _ = perform_paired_ttest(list(iiot_vals), list(cic_vals))
            stars = significance_stars(pval)

            if stars:
                max_y = max(group_data["Macro F1"].max(), 0.9) + 0.03
                ax.annotate(
                    stars,
                    xy=(i, max_y),
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

    ax.legend(title="Dataset", loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=CONFIG.grid_alpha)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_class_f1_heatmap(
    matched_experiments: dict,
    output_path: Path,
    aggregator: str = "fedavg",
    adv: int = 0,
):
    """
    Plot heatmap showing per-class F1 scores for each dataset.
    """
    CONFIG.apply()

    # Filter experiments
    target_key_pattern = f"{aggregator}_alpha.*_adv{adv}_"

    iiot_f1_per_class = []
    cic_f1_per_class = []
    iiot_class_names = None
    cic_class_names = None

    for config_key, paths in matched_experiments.items():
        if not re.match(target_key_pattern, config_key):
            continue

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            client_agg = aggregate_client_metrics(exp_data["client_metrics"])

            if client_agg.get("f1_per_class"):
                for f1_dict in client_agg["f1_per_class"]:
                    if dataset_name == "Edge-IIoT":
                        iiot_f1_per_class.append(f1_dict)
                        if iiot_class_names is None and client_agg.get("class_names"):
                            iiot_class_names = client_agg["class_names"]
                    else:
                        cic_f1_per_class.append(f1_dict)
                        if cic_class_names is None and client_agg.get("class_names"):
                            cic_class_names = client_agg["class_names"]

    if not iiot_f1_per_class and not cic_f1_per_class:
        print(f"No per-class F1 data found for {aggregator} adv{adv}")
        return

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Process IIoT data
    if iiot_f1_per_class:
        iiot_df = pd.DataFrame(iiot_f1_per_class)
        iiot_means = iiot_df.mean().sort_values(ascending=False)

        if iiot_class_names:
            new_index = []
            for i in iiot_means.index:
                if i.isdigit() and int(i) < len(iiot_class_names):
                    new_index.append(iiot_class_names[int(i)])
                else:
                    new_index.append(i)
            iiot_means.index = new_index

        # Create heatmap data
        heatmap_data = iiot_means.values.reshape(1, -1)

        sns.heatmap(
            heatmap_data,
            ax=axes[0],
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=iiot_means.index,
            yticklabels=["F1"],
            cbar_kws={"label": "F1 Score"},
        )
        axes[0].set_title(
            f"Edge-IIoT: Per-Class F1 ({aggregator.upper()}, {adv}% Adversary)",
            fontsize=CONFIG.title_size,
            fontweight="bold",
        )
        axes[0].tick_params(axis="x", rotation=45)

    # Process CIC data
    if cic_f1_per_class:
        cic_df = pd.DataFrame(cic_f1_per_class)
        cic_means = cic_df.mean().sort_values(ascending=False)

        if cic_class_names:
            new_index = []
            for i in cic_means.index:
                if i.isdigit() and int(i) < len(cic_class_names):
                    new_index.append(cic_class_names[int(i)])
                else:
                    new_index.append(i)
            cic_means.index = new_index

        heatmap_data = cic_means.values.reshape(1, -1)

        sns.heatmap(
            heatmap_data,
            ax=axes[1],
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=cic_means.index,
            yticklabels=["F1"],
            cbar_kws={"label": "F1 Score"},
        )
        axes[1].set_title(
            f"CIC-IDS2017: Per-Class F1 ({aggregator.upper()}, {adv}% Adversary)",
            fontsize=CONFIG.title_size,
            fontweight="bold",
        )
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_class_f1_grouped_bars(
    matched_experiments: dict,
    output_path: Path,
    aggregator: str = "fedavg",
    adv: int = 0,
    top_n_classes: int = 10,
):
    """
    Plot grouped bar chart comparing per-class F1 between datasets.
    """
    CONFIG.apply()

    target_key_pattern = f"{aggregator}_alpha.*_adv{adv}_"

    iiot_f1_per_class = []
    cic_f1_per_class = []
    iiot_class_names = None
    cic_class_names = None

    for config_key, paths in matched_experiments.items():
        if not re.match(target_key_pattern, config_key):
            continue

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            client_agg = aggregate_client_metrics(exp_data["client_metrics"])

            if client_agg.get("f1_per_class"):
                for f1_dict in client_agg["f1_per_class"]:
                    if dataset_name == "Edge-IIoT":
                        iiot_f1_per_class.append(f1_dict)
                        if client_agg.get("class_names"):
                            iiot_class_names = client_agg["class_names"]
                    else:
                        cic_f1_per_class.append(f1_dict)
                        if client_agg.get("class_names"):
                            cic_class_names = client_agg["class_names"]

    if not iiot_f1_per_class and not cic_f1_per_class:
        print(f"No per-class F1 data for {aggregator} adv{adv}")
        return

    # Prepare data for plotting
    plot_data = []

    # Process IIoT
    if iiot_f1_per_class:
        iiot_df = pd.DataFrame(iiot_f1_per_class)
        for col in iiot_df.columns:
            class_name = col
            if iiot_class_names and col.isdigit():
                idx = int(col)
                if idx < len(iiot_class_names):
                    class_name = iiot_class_names[idx]
            for val in iiot_df[col].dropna():
                plot_data.append(
                    {
                        "Dataset": "Edge-IIoT",
                        "Class": class_name,
                        "F1 Score": float(val),
                    }
                )

    # Process CIC
    if cic_f1_per_class:
        cic_df = pd.DataFrame(cic_f1_per_class)
        for col in cic_df.columns:
            class_name = col
            if cic_class_names and col.isdigit():
                idx = int(col)
                if idx < len(cic_class_names):
                    class_name = cic_class_names[idx]
            for val in cic_df[col].dropna():
                plot_data.append(
                    {
                        "Dataset": "CIC-IDS2017",
                        "Class": class_name,
                        "F1 Score": float(val),
                    }
                )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    # Select top N classes by mean F1
    class_means = df.groupby("Class")["F1 Score"].mean().sort_values(ascending=False)
    top_classes = class_means.head(top_n_classes).index.tolist()
    df_filtered = df[df["Class"].isin(top_classes)]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(
        data=df_filtered,
        x="Class",
        y="F1 Score",
        hue="Dataset",
        palette=DATASET_COLORS,
        ax=ax,
        capsize=0.05,
        errwidth=1.5,
        alpha=CONFIG.alpha,
    )

    ax.set_xlabel("Attack Class", fontsize=CONFIG.label_size)
    ax.set_ylabel("F1 Score", fontsize=CONFIG.label_size)
    ax.set_title(
        f"Per-Class F1 Comparison: {aggregator.upper()}, {adv}% Adversary",
        fontsize=CONFIG.title_size,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Dataset", loc="upper right")
    ax.grid(axis="y", alpha=CONFIG.grid_alpha)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_confusion_matrix_comparison(
    matched_experiments: dict,
    output_path: Path,
    aggregator: str = "fedavg",
    adv: int = 0,
):
    """
    Plot side-by-side confusion matrix summaries for both datasets.
    """
    CONFIG.apply()

    target_key_pattern = f"{aggregator}_alpha.*_adv{adv}_"

    iiot_cms = []
    cic_cms = []
    iiot_class_names = None
    cic_class_names = None

    for config_key, paths in matched_experiments.items():
        if not re.match(target_key_pattern, config_key):
            continue

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            client_agg = aggregate_client_metrics(exp_data["client_metrics"])

            if client_agg.get("confusion_matrices"):
                if dataset_name == "Edge-IIoT":
                    iiot_cms.extend(client_agg["confusion_matrices"])
                    if client_agg.get("class_names"):
                        iiot_class_names = client_agg["class_names"]
                else:
                    cic_cms.extend(client_agg["confusion_matrices"])
                    if client_agg.get("class_names"):
                        cic_class_names = client_agg["class_names"]

    if not iiot_cms and not cic_cms:
        print(f"No confusion matrices found for {aggregator} adv{adv}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot IIoT confusion matrix
    if iiot_cms:
        # Filter to same shape and average
        shapes = [cm.shape for cm in iiot_cms]
        most_common_shape = max(set(shapes), key=shapes.count)
        iiot_cms_filtered = [cm for cm in iiot_cms if cm.shape == most_common_shape]

        if iiot_cms_filtered:
            iiot_avg_cm = np.mean(np.stack(iiot_cms_filtered), axis=0)
        else:
            iiot_avg_cm = iiot_cms[0]

        # Shorten class names for display
        if iiot_class_names:
            short_names = [
                n[:12] + ".." if len(n) > 14 else n for n in iiot_class_names
            ]
        else:
            short_names = [str(i) for i in range(iiot_avg_cm.shape[0])]

        sns.heatmap(
            iiot_avg_cm,
            ax=axes[0],
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=short_names,
            yticklabels=short_names,
            cbar_kws={"label": "Normalized Count"},
        )
        axes[0].set_title(
            f"Edge-IIoT Confusion Matrix\n({aggregator.upper()}, {adv}% Adversary)",
            fontsize=CONFIG.title_size,
            fontweight="bold",
        )
        axes[0].set_xlabel("Predicted", fontsize=CONFIG.label_size)
        axes[0].set_ylabel("True", fontsize=CONFIG.label_size)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

    # Plot CIC confusion matrix
    if cic_cms:
        # Filter to same shape and average
        shapes = [cm.shape for cm in cic_cms]
        most_common_shape = max(set(shapes), key=shapes.count)
        cic_cms_filtered = [cm for cm in cic_cms if cm.shape == most_common_shape]

        if cic_cms_filtered:
            cic_avg_cm = np.mean(np.stack(cic_cms_filtered), axis=0)
        else:
            cic_avg_cm = cic_cms[0]

        if cic_class_names:
            short_names = [n[:12] + ".." if len(n) > 14 else n for n in cic_class_names]
        else:
            short_names = [str(i) for i in range(cic_avg_cm.shape[0])]

        sns.heatmap(
            cic_avg_cm,
            ax=axes[1],
            annot=True,
            fmt=".2f",
            cmap="Oranges",
            vmin=0,
            vmax=1,
            xticklabels=short_names,
            yticklabels=short_names,
            cbar_kws={"label": "Normalized Count"},
        )
        axes[1].set_title(
            f"CIC-IDS2017 Confusion Matrix\n({aggregator.upper()}, {adv}% Adversary)",
            fontsize=CONFIG.title_size,
            fontweight="bold",
        )
        axes[1].set_xlabel("Predicted", fontsize=CONFIG.label_size)
        axes[1].set_ylabel("True", fontsize=CONFIG.label_size)
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_convergence_comparison(
    matched_experiments: dict,
    output_path: Path,
    aggregator: str = "fedavg",
    adv: int = 0,
):
    """
    Plot training convergence curves comparing both datasets.
    """
    CONFIG.apply()

    target_key_pattern = f"{aggregator}_alpha.*_adv{adv}_"

    iiot_convergence = []
    cic_convergence = []

    for config_key, paths in matched_experiments.items():
        if not re.match(target_key_pattern, config_key):
            continue

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            conv_df = get_convergence_data(exp_data["client_metrics"])

            if not conv_df.empty:
                # Aggregate per round across clients
                round_agg = (
                    conv_df.groupby("round")["macro_f1"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                round_agg["dataset"] = dataset_name

                if dataset_name == "Edge-IIoT":
                    iiot_convergence.append(round_agg)
                else:
                    cic_convergence.append(round_agg)

    if not iiot_convergence and not cic_convergence:
        print(f"No convergence data for {aggregator} adv{adv}")
        return

    fig, ax = plt.subplots(figsize=CONFIG.figsize_wide)

    # Plot IIoT
    if iiot_convergence:
        iiot_all = pd.concat(iiot_convergence)
        iiot_grouped = (
            iiot_all.groupby("round")["mean"].agg(["mean", "std"]).reset_index()
        )

        ax.plot(
            iiot_grouped["round"],
            iiot_grouped["mean"],
            color=DATASET_COLORS["Edge-IIoT"],
            linewidth=CONFIG.linewidth,
            marker="o",
            markersize=CONFIG.markersize,
            label="Edge-IIoT",
        )

        ax.fill_between(
            iiot_grouped["round"],
            iiot_grouped["mean"] - iiot_grouped["std"],
            iiot_grouped["mean"] + iiot_grouped["std"],
            color=DATASET_COLORS["Edge-IIoT"],
            alpha=CONFIG.ci_alpha,
        )

    # Plot CIC
    if cic_convergence:
        cic_all = pd.concat(cic_convergence)
        cic_grouped = (
            cic_all.groupby("round")["mean"].agg(["mean", "std"]).reset_index()
        )

        ax.plot(
            cic_grouped["round"],
            cic_grouped["mean"],
            color=DATASET_COLORS["CIC-IDS2017"],
            linewidth=CONFIG.linewidth,
            marker="s",
            markersize=CONFIG.markersize,
            label="CIC-IDS2017",
        )

        ax.fill_between(
            cic_grouped["round"],
            cic_grouped["mean"] - cic_grouped["std"],
            cic_grouped["mean"] + cic_grouped["std"],
            color=DATASET_COLORS["CIC-IDS2017"],
            alpha=CONFIG.ci_alpha,
        )

    ax.set_xlabel("Communication Round", fontsize=CONFIG.label_size)
    ax.set_ylabel("Macro F1 Score", fontsize=CONFIG.label_size)
    ax.set_title(
        f"Training Convergence: {aggregator.upper()}, {adv}% Adversary",
        fontsize=CONFIG.title_size,
        fontweight="bold",
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=CONFIG.grid_alpha)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_attack_impact_comparison(
    matched_experiments: dict,
    output_path: Path,
):
    """
    Plot the impact of adversarial attacks on both datasets.
    Shows degradation from 0% to 30% adversary.
    """
    CONFIG.apply()

    # Collect data
    plot_data = []

    for config_key, paths in matched_experiments.items():
        parts = config_key.split("_")
        aggregator = parts[0]
        adv = int(re.search(r"adv(\d+)", config_key).group(1))

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            client_agg = aggregate_client_metrics(exp_data["client_metrics"])

            if client_agg.get("macro_f1_mean") is not None:
                plot_data.append(
                    {
                        "Aggregator": aggregator.upper(),
                        "Adversary %": adv,
                        "Dataset": dataset_name,
                        "Macro F1": client_agg["macro_f1_mean"],
                        "F1 Std": client_agg.get("macro_f1_std", 0),
                    }
                )

    if not plot_data:
        print("No data for attack impact comparison")
        return

    df = pd.DataFrame(plot_data)

    # Calculate degradation
    degradation_data = []
    for agg in df["Aggregator"].unique():
        for dataset in df["Dataset"].unique():
            subset = df[(df["Aggregator"] == agg) & (df["Dataset"] == dataset)]
            if len(subset) >= 2:
                f1_0 = subset[subset["Adversary %"] == 0]["Macro F1"].mean()
                f1_30 = subset[subset["Adversary %"] == 30]["Macro F1"].mean()
                if f1_0 > 0:
                    degradation = ((f1_0 - f1_30) / f1_0) * 100
                    degradation_data.append(
                        {
                            "Aggregator": agg,
                            "Dataset": dataset,
                            "Degradation %": degradation,
                            "F1 at 0%": f1_0,
                            "F1 at 30%": f1_30,
                        }
                    )

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: F1 at different adversary levels
    sns.barplot(
        data=df,
        x="Aggregator",
        y="Macro F1",
        hue="Dataset",
        ax=axes[0],
        palette=DATASET_COLORS,
        alpha=CONFIG.alpha,
    )
    axes[0].set_title(
        "F1 Score by Adversary Level",
        fontsize=CONFIG.title_size,
        fontweight="bold",
    )
    axes[0].set_ylabel("Macro F1 Score", fontsize=CONFIG.label_size)
    axes[0].legend(title="Dataset")
    axes[0].grid(axis="y", alpha=CONFIG.grid_alpha)
    axes[0].set_ylim(0, 1.05)

    # Right: Performance degradation
    if degradation_data:
        deg_df = pd.DataFrame(degradation_data)
        sns.barplot(
            data=deg_df,
            x="Aggregator",
            y="Degradation %",
            hue="Dataset",
            ax=axes[1],
            palette=DATASET_COLORS,
            alpha=CONFIG.alpha,
        )
        axes[1].set_title(
            "F1 Degradation (0% to 30% Adversary)",
            fontsize=CONFIG.title_size,
            fontweight="bold",
        )
        axes[1].set_ylabel("Degradation (%)", fontsize=CONFIG.label_size)
        axes[1].legend(title="Dataset")
        axes[1].grid(axis="y", alpha=CONFIG.grid_alpha)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comprehensive_summary(
    matched_experiments: dict,
    output_path: Path,
):
    """
    Create a comprehensive 2x2 summary plot for the thesis.
    """
    CONFIG.apply()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect all data
    all_data = []
    for config_key, paths in matched_experiments.items():
        aggregator = config_key.split("_")[0]
        adv = int(re.search(r"adv(\d+)", config_key).group(1))
        seed = int(re.search(r"seed(\d+)", config_key).group(1))

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            client_agg = aggregate_client_metrics(exp_data["client_metrics"])
            conv_df = get_convergence_data(exp_data["client_metrics"])

            if client_agg.get("macro_f1_mean") is not None:
                all_data.append(
                    {
                        "aggregator": aggregator.upper(),
                        "adv": adv,
                        "dataset": dataset_name,
                        "seed": seed,
                        "macro_f1": client_agg["macro_f1_mean"],
                        "convergence": conv_df if not conv_df.empty else None,
                    }
                )

    if not all_data:
        print("No data for comprehensive summary")
        return

    df = pd.DataFrame(all_data)

    # ---- Plot 1: Overall F1 comparison by dataset ----
    ax1 = axes[0, 0]
    summary_df = (
        df.groupby(["dataset", "adv"])["macro_f1"].agg(["mean", "std"]).reset_index()
    )

    x_pos = np.arange(2)
    width = 0.35

    for i, adv_level in enumerate([0, 30]):
        adv_data = summary_df[summary_df["adv"] == adv_level]
        iiot_val = adv_data[adv_data["dataset"] == "Edge-IIoT"]["mean"].values
        cic_val = adv_data[adv_data["dataset"] == "CIC-IDS2017"]["mean"].values

        iiot_val = iiot_val[0] if len(iiot_val) > 0 else 0
        cic_val = cic_val[0] if len(cic_val) > 0 else 0

        ax1.bar(
            x_pos + i * width,
            [iiot_val, cic_val],
            width,
            label=f"{adv_level}% Adversary",
            alpha=CONFIG.alpha,
        )

    ax1.set_ylabel("Macro F1 Score")
    ax1.set_title("(A) Overall Performance by Dataset", fontweight="bold")
    ax1.set_xticks(x_pos + width / 2)
    ax1.set_xticklabels(["Edge-IIoT", "CIC-IDS2017"])
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=CONFIG.grid_alpha)

    # ---- Plot 2: Aggregator comparison ----
    ax2 = axes[0, 1]
    df["config"] = df["aggregator"] + "\n" + df["adv"].astype(str) + "% Adv"

    sns.barplot(
        data=df,
        x="config",
        y="macro_f1",
        hue="dataset",
        ax=ax2,
        palette=DATASET_COLORS,
        alpha=CONFIG.alpha,
        capsize=0.05,
    )
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_xlabel("Configuration")
    ax2.set_title("(B) Performance by Aggregator", fontweight="bold")
    ax2.legend(title="Dataset", loc="lower right")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=CONFIG.grid_alpha)

    # ---- Plot 3: Dataset difference distribution ----
    ax3 = axes[1, 0]

    # Calculate paired differences
    differences = []
    for seed in df["seed"].unique():
        for agg in df["aggregator"].unique():
            for adv in df["adv"].unique():
                subset = df[
                    (df["seed"] == seed)
                    & (df["aggregator"] == agg)
                    & (df["adv"] == adv)
                ]
                iiot_val = subset[subset["dataset"] == "Edge-IIoT"]["macro_f1"].values
                cic_val = subset[subset["dataset"] == "CIC-IDS2017"]["macro_f1"].values
                if len(iiot_val) > 0 and len(cic_val) > 0:
                    differences.append(
                        {
                            "Difference (IIoT - CIC)": iiot_val[0] - cic_val[0],
                            "Adversary": f"{adv}%",
                        }
                    )

    if differences:
        diff_df = pd.DataFrame(differences)
        sns.boxplot(
            data=diff_df,
            x="Adversary",
            y="Difference (IIoT - CIC)",
            ax=ax3,
            palette=["#0173B2", "#DE8F05"],
        )
        ax3.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax3.set_title("(C) Performance Difference Distribution", fontweight="bold")
        ax3.set_ylabel("F1 Difference (Edge-IIoT - CIC)")
        ax3.grid(axis="y", alpha=CONFIG.grid_alpha)

    # ---- Plot 4: Statistical summary table ----
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary table
    table_data = []
    for adv in [0, 30]:
        iiot_vals = df[(df["dataset"] == "Edge-IIoT") & (df["adv"] == adv)][
            "macro_f1"
        ].values
        cic_vals = df[(df["dataset"] == "CIC-IDS2017") & (df["adv"] == adv)][
            "macro_f1"
        ].values

        if len(iiot_vals) > 0 and len(cic_vals) > 0:
            pval, effect = perform_paired_ttest(list(iiot_vals), list(cic_vals))
            table_data.append(
                [
                    f"{adv}%",
                    f"{np.mean(iiot_vals):.3f} +/- {np.std(iiot_vals):.3f}",
                    f"{np.mean(cic_vals):.3f} +/- {np.std(cic_vals):.3f}",
                    f"{pval:.4f}" if pval else "N/A",
                    significance_stars(pval),
                ]
            )

    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=[
                "Adversary",
                "Edge-IIoT F1",
                "CIC-IDS2017 F1",
                "p-value",
                "Sig.",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax4.set_title("(D) Statistical Comparison", fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_summary_statistics(matched_experiments: dict, output_path: Path):
    """Generate a CSV with summary statistics for the thesis."""

    rows = []

    for config_key, paths in matched_experiments.items():
        aggregator = config_key.split("_")[0]
        adv = int(re.search(r"adv(\d+)", config_key).group(1))
        seed = int(re.search(r"seed(\d+)", config_key).group(1))

        for dataset_name, exp_path in paths.items():
            if exp_path is None:
                continue

            exp_data = load_experiment_data(exp_path)
            client_agg = aggregate_client_metrics(exp_data["client_metrics"])

            rows.append(
                {
                    "aggregator": aggregator,
                    "alpha": 0.5,
                    "adversary_pct": adv,
                    "seed": seed,
                    "dataset": dataset_name,
                    "macro_f1_mean": client_agg.get("macro_f1_mean"),
                    "macro_f1_std": client_agg.get("macro_f1_std"),
                    "n_clients": len(exp_data["client_metrics"]),
                    "experiment_path": str(exp_path),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved summary statistics: {output_path}")

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Generate all Objective 3 thesis plots."""
    print("=" * 60)
    print("OBJECTIVE 3: Multi-Dataset Validation Plots")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find matching experiments
    print("\nSearching for matching experiments...")
    matched = find_matching_experiments(
        DATA_DIR,
        alpha=0.5,
        adv_fractions=[0, 30],
        aggregators=["fedavg", "bulyan"],
        seeds=[42, 43, 44, 45, 46],
    )

    print(f"Found {len(matched)} matched experiment pairs")

    if not matched:
        print("ERROR: No matching experiments found!")
        print("Looking for alternative configurations...")

        # Try with different seeds or aggregators
        matched = find_matching_experiments(
            DATA_DIR,
            alpha=0.5,
            adv_fractions=[0, 30],
            aggregators=["fedavg", "bulyan", "krum", "median"],
            seeds=list(range(42, 62)),
        )
        print(f"Found {len(matched)} matched experiments with expanded search")

    if not matched:
        print("No matched experiments found. Exiting.")
        return

    # Print found matches
    print("\nMatched experiments:")
    for key, paths in list(matched.items())[:5]:
        print(f"  {key}:")
        for ds, p in paths.items():
            print(f"    {ds}: {p.name if p else 'None'}")

    # Generate summary statistics
    print("\n" + "-" * 40)
    print("Generating summary statistics...")
    summary_df = generate_summary_statistics(
        matched, OUTPUT_DIR / "summary_statistics.csv"
    )
    print(f"Total experiments: {len(summary_df)}")

    # Plot 1: Side-by-side F1 comparison
    print("\n" + "-" * 40)
    print("Generating side-by-side F1 comparison...")
    plot_sidebyside_f1_comparison(
        matched,
        OUTPUT_DIR / "fig_objective3_f1_comparison.png",
        title="Cross-Dataset F1 Score Comparison (alpha=0.5)",
    )

    # Plot 2: Per-class F1 heatmaps
    print("\n" + "-" * 40)
    print("Generating per-class F1 heatmaps...")
    for agg in ["fedavg", "bulyan"]:
        for adv in [0, 30]:
            plot_per_class_f1_heatmap(
                matched,
                OUTPUT_DIR / f"fig_objective3_perclass_f1_heatmap_{agg}_adv{adv}.png",
                aggregator=agg,
                adv=adv,
            )

    # Plot 3: Per-class F1 grouped bars
    print("\n" + "-" * 40)
    print("Generating per-class F1 grouped bars...")
    for agg in ["fedavg", "bulyan"]:
        for adv in [0, 30]:
            plot_per_class_f1_grouped_bars(
                matched,
                OUTPUT_DIR / f"fig_objective3_perclass_f1_bars_{agg}_adv{adv}.png",
                aggregator=agg,
                adv=adv,
            )

    # Plot 4: Confusion matrix comparison
    print("\n" + "-" * 40)
    print("Generating confusion matrix comparisons...")
    for agg in ["fedavg", "bulyan"]:
        for adv in [0, 30]:
            plot_confusion_matrix_comparison(
                matched,
                OUTPUT_DIR / f"fig_objective3_confusion_matrix_{agg}_adv{adv}.png",
                aggregator=agg,
                adv=adv,
            )

    # Plot 5: Convergence comparison
    print("\n" + "-" * 40)
    print("Generating convergence comparisons...")
    for agg in ["fedavg", "bulyan"]:
        for adv in [0, 30]:
            plot_convergence_comparison(
                matched,
                OUTPUT_DIR / f"fig_objective3_convergence_{agg}_adv{adv}.png",
                aggregator=agg,
                adv=adv,
            )

    # Plot 6: Attack impact comparison
    print("\n" + "-" * 40)
    print("Generating attack impact comparison...")
    plot_attack_impact_comparison(
        matched,
        OUTPUT_DIR / "fig_objective3_attack_impact.png",
    )

    # Plot 7: Comprehensive summary
    print("\n" + "-" * 40)
    print("Generating comprehensive summary...")
    plot_comprehensive_summary(
        matched,
        OUTPUT_DIR / "fig_objective3_comprehensive_summary.png",
    )

    print("\n" + "=" * 60)
    print("OBJECTIVE 3 PLOTS COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
