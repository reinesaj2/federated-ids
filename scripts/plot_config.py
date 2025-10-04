#!/usr/bin/env python3
"""
Plot Configuration Layer for Federated Learning Visualizations

Provides configurable styling, palettes, and layout options for thesis-ready plots.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


# Colorblind-safe palettes for various client counts
PALETTES = {
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "colorblind": ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#ECE133", "#56B4E9"],
    "vibrant": ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988"],
    "muted": ["#88CCEE", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933"],
    "dark": ["#222255", "#225555", "#225522", "#666633", "#663333", "#555555"],
}


@dataclass
class PlotStyle:
    """Styling configuration for plots."""

    palette: str = "colorblind"
    theme: str = "whitegrid"
    font_family: str = "sans-serif"
    font_size: int = 11
    title_size: int = 14
    label_size: int = 11
    legend_size: int = 9
    dpi: int = 150
    figsize: Tuple[int, int] = (15, 10)
    alpha: float = 0.8
    linewidth: float = 2.0
    markersize: float = 8.0

    def apply(self):
        """Apply style to matplotlib."""
        sns.set_style(self.theme)
        plt.rcParams.update({
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "legend.fontsize": self.legend_size,
            "figure.dpi": self.dpi,
        })

    def get_colors(self, n: int) -> List[str]:
        """Get n colors from selected palette, extending if needed."""
        base_colors = PALETTES.get(self.palette, PALETTES["colorblind"])

        if n <= len(base_colors):
            return base_colors[:n]

        # Extend palette by cycling through with alpha variations
        extended = base_colors.copy()
        while len(extended) < n:
            extended.extend(base_colors)

        return extended[:n]


@dataclass
class LayoutConfig:
    """Layout configuration for subplot grids."""

    rows: Optional[int] = None
    cols: Optional[int] = None
    legend_position: str = "best"
    legend_outside: bool = False
    shared_legend: bool = False
    tight_layout: bool = True

    def compute_grid(self, num_plots: int) -> Tuple[int, int]:
        """Compute optimal grid dimensions for num_plots."""
        if self.rows and self.cols:
            return self.rows, self.cols

        if self.rows:
            cols = (num_plots + self.rows - 1) // self.rows
            return self.rows, cols

        if self.cols:
            rows = (num_plots + self.cols - 1) // self.cols
            return rows, self.cols

        # Auto-compute: prefer 2 rows for 2-6 plots, 3 rows for 7-9 plots
        if num_plots <= 2:
            return 1, num_plots
        elif num_plots <= 6:
            return 2, (num_plots + 1) // 2
        else:
            return 3, (num_plots + 2) // 3

    def setup_legend(self, fig, axes):
        """Setup legend according to configuration."""
        if self.shared_legend:
            handles, labels = [], []
            for ax in axes.flat if hasattr(axes, 'flat') else [axes]:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles.extend(h)
                    labels.extend(l)

            # Deduplicate while preserving order
            unique = {}
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = h

            if unique:
                if self.legend_outside:
                    fig.legend(unique.values(), unique.keys(),
                             loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    fig.legend(unique.values(), unique.keys(),
                             loc='upper right')

                # Remove individual legends
                for ax in axes.flat if hasattr(axes, 'flat') else [axes]:
                    legend = ax.get_legend()
                    if legend:
                        legend.remove()


@dataclass
class MetricDetector:
    """Auto-detect available metrics in dataframe."""

    server_metric_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "timing": ["t_aggregate_ms", "aggregation_time_ms", "t_round_ms"],
        "robustness": ["l2_to_benign_mean", "cos_to_benign_mean"],
        "norms": ["update_norm_mean", "update_norm_std"],
        "dispersion": ["pairwise_cosine_mean", "l2_dispersion_mean"],
    })

    client_metric_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "loss": ["loss_after", "local_loss"],
        "accuracy": ["acc_after", "local_accuracy"],
        "norms": ["weight_norm_after", "weight_norm"],
        "f1_comparison": ["macro_f1_argmax", "f1_bin_tau"],
        "threshold": ["tau_bin", "threshold_tau"],
        "fpr": ["benign_fpr_bin_tau", "fpr_after"],
    })

    def detect_available(self, df, metric_type: str = "server") -> Dict[str, Optional[str]]:
        """Detect which metrics are available in dataframe."""
        metric_map = self.server_metric_map if metric_type == "server" else self.client_metric_map
        available = {}

        for category, column_options in metric_map.items():
            found = None
            for col in column_options:
                if col in df.columns and not df[col].isna().all():
                    found = col
                    break
            available[category] = found

        return available

    def count_available_plots(self, available_metrics: Dict[str, Optional[str]]) -> int:
        """Count how many plots can be generated from available metrics."""
        return sum(1 for v in available_metrics.values() if v is not None)


@dataclass
class SmoothingConfig:
    """Configuration for data smoothing."""

    enabled: bool = False
    window_size: int = 3
    method: str = "rolling_mean"

    def apply(self, data):
        """Apply smoothing to data series."""
        import pandas as pd
        import numpy as np

        if not self.enabled or len(data) < self.window_size:
            return data

        if self.method == "rolling_mean":
            return pd.Series(data).rolling(window=self.window_size, center=True, min_periods=1).mean()
        elif self.method == "ewma":
            return pd.Series(data).ewm(span=self.window_size).mean()
        else:
            return data


@dataclass
class ConfidenceIntervalConfig:
    """Configuration for confidence interval visualization."""

    enabled: bool = False
    confidence: float = 0.95
    method: str = "t_distribution"
    alpha: float = 0.2

    def compute(self, data_groups: List):
        """Compute confidence intervals from multiple runs."""
        import numpy as np
        from scipy import stats

        if not self.enabled or not data_groups:
            return None

        data_array = np.array(data_groups)

        if len(data_array) < 2:
            return None

        mean = np.mean(data_array, axis=0)
        se = stats.sem(data_array, axis=0)

        if self.method == "t_distribution":
            ci = se * stats.t.ppf((1 + self.confidence) / 2, len(data_array) - 1)
        else:
            ci = se * 1.96  # 95% normal approximation

        return {
            "mean": mean,
            "lower": mean - ci,
            "upper": mean + ci,
        }


def create_default_config(title: str = "Federated Learning Metrics") -> Dict:
    """Create default configuration dict."""
    return {
        "style": PlotStyle(),
        "layout": LayoutConfig(),
        "detector": MetricDetector(),
        "smoothing": SmoothingConfig(),
        "ci": ConfidenceIntervalConfig(),
        "title": title,
    }
