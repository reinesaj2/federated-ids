"""
Layout Configuration

Provides utilities for managing subplot grids and figure layouts.
"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray


@dataclass
class LayoutConfig:
    """Configuration for subplot grid layouts."""

    rows: int | None = None
    cols: int | None = None
    legend_position: str = "best"
    legend_outside: bool = False
    shared_legend: bool = False
    tight_layout: bool = True

    def compute_grid(self, num_plots: int) -> tuple[int, int]:
        """
        Compute optimal grid dimensions for given number of plots.

        Args:
            num_plots: Number of subplots needed

        Returns:
            Tuple of (rows, cols)
        """
        if self.rows and self.cols:
            return self.rows, self.cols

        if self.rows:
            cols = (num_plots + self.rows - 1) // self.rows
            return self.rows, cols

        if self.cols:
            rows = (num_plots + self.cols - 1) // self.cols
            return rows, self.cols

        if num_plots <= 2:
            return 1, num_plots
        elif num_plots <= 6:
            return 2, (num_plots + 1) // 2
        else:
            return 3, (num_plots + 2) // 3

    def setup_legend(self, fig: Figure, axes: ndarray | plt.Axes) -> None:
        """
        Setup shared legend according to configuration.

        Args:
            fig: Matplotlib figure
            axes: Array of axes or single axes
        """
        if not self.shared_legend:
            return

        handles, labels = [], []
        axes_iter = axes.flat if hasattr(axes, "flat") else [axes]

        for ax in axes_iter:
            h, lb = ax.get_legend_handles_labels()
            if h:
                handles.extend(h)
                labels.extend(lb)

        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels, strict=False):
            if label not in unique:
                unique[label] = handle

        if not unique:
            return

        if self.legend_outside:
            fig.legend(
                unique.values(),
                unique.keys(),
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
        else:
            fig.legend(unique.values(), unique.keys(), loc="upper right")

        for ax in axes_iter:
            legend = ax.get_legend()
            if legend:
                legend.remove()
