"""
Plotting Primitives

Base plotting functions used across all figure types.
These are composable building blocks for more complex visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from plots.config.style import ThesisStyle


def plot_bar_comparison(
    ax: Axes,
    categories: list[str],
    values: list[float],
    errors: list[float] | None = None,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    ylabel: str = "",
    title: str = "",
    ylim: tuple[float, float] | None = None,
    annotate_counts: list[int] | None = None,
) -> None:
    """
    Create a bar chart with optional error bars.

    Args:
        ax: Matplotlib axes to plot on
        categories: X-axis category labels
        values: Bar heights
        errors: Error bar values (optional)
        colors: Bar colors (optional, uses style default)
        labels: Labels for legend (optional)
        ylabel: Y-axis label
        title: Plot title
        ylim: Y-axis limits (optional)
        annotate_counts: Sample counts to annotate above bars (optional)
    """
    style = ThesisStyle()
    if colors is None:
        colors = style.get_colors(len(categories))

    x = np.arange(len(categories))
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        color=colors,
        edgecolor="black",
        capsize=5,
        alpha=style.alpha,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")

    if ylim:
        ax.set_ylim(ylim)

    if annotate_counts:
        for i, (bar, cnt) in enumerate(zip(bars, annotate_counts)):
            height = values[i] + (errors[i] if errors else 0) + 0.02
            ax.annotate(
                f"n={cnt}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                fontsize=8,
            )
    if labels:
        for bar, label in zip(bars, labels):
            bar.set_label(label)
        ax.legend(loc="best")


def plot_grouped_bars(
    ax: Axes,
    categories: list[str],
    group_values: dict[str, list[float]],
    group_colors: dict[str, str] | None = None,
    ylabel: str = "",
    title: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Create grouped bar chart.

    Args:
        ax: Matplotlib axes
        categories: X-axis category labels
        group_values: Dict mapping group name -> values per category
        group_colors: Dict mapping group name -> color
        ylabel: Y-axis label
        title: Plot title
        ylim: Y-axis limits
    """
    style = ThesisStyle()
    groups = list(group_values.keys())
    n_groups = len(groups)
    n_categories = len(categories)

    if group_colors is None:
        colors = style.get_colors(n_groups)
        group_colors = dict(zip(groups, colors))

    width = 0.8 / n_groups
    x = np.arange(n_categories)

    for i, group in enumerate(groups):
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(
            x + offset,
            group_values[group],
            width,
            label=group,
            color=group_colors[group],
            alpha=style.alpha,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")

    if ylim:
        ax.set_ylim(ylim)


def plot_line_with_ci(
    ax: Axes,
    x_values: list | np.ndarray,
    y_values: list | np.ndarray,
    ci_lower: list | np.ndarray | None = None,
    ci_upper: list | np.ndarray | None = None,
    color: str | None = None,
    label: str = "",
    marker: str = "o",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    xscale: str = "linear",
) -> None:
    """
    Plot line with optional confidence interval shading.

    Args:
        ax: Matplotlib axes
        x_values: X coordinates
        y_values: Y coordinates (means)
        ci_lower: Lower CI bound (optional)
        ci_upper: Upper CI bound (optional)
        color: Line color
        label: Legend label
        marker: Marker style
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        xscale: X-axis scale ('linear', 'log')
    """
    style = ThesisStyle()
    if color is None:
        color = style.get_colors(1)[0]

    ax.plot(
        x_values,
        y_values,
        marker=marker,
        color=color,
        label=label,
        linewidth=style.linewidth,
        markersize=style.markersize,
    )

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            x_values,
            ci_lower,
            ci_upper,
            color=color,
            alpha=0.2,
        )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    if xscale != "linear":
        ax.set_xscale(xscale)
    if label:
        ax.legend(loc="best")


def plot_boxplot(
    ax: Axes,
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str] | None = None,
    ylabel: str = "",
    title: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Create boxplot visualization.

    Args:
        ax: Matplotlib axes
        data: List of arrays, one per box
        labels: Box labels
        colors: Box fill colors
        ylabel: Y-axis label
        title: Plot title
        ylim: Y-axis limits
    """
    style = ThesisStyle()
    if colors is None:
        colors = style.get_colors(len(data))

    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    if ylim:
        ax.set_ylim(ylim)


def plot_heatmap(
    ax: Axes,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    cmap: str = "RdYlGn",
    fmt: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = True,
) -> None:
    """
    Create heatmap visualization.

    Args:
        ax: Matplotlib axes
        data: 2D array of values
        row_labels: Row labels
        col_labels: Column labels
        title: Plot title
        cmap: Colormap name
        fmt: Annotation format string
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        annot: Whether to annotate cells with values
    """
    sns.heatmap(
        data,
        ax=ax,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.5,
    )
    ax.set_title(title, fontweight="bold")


def save_figure(
    fig: Figure,
    output_path: Path,
    formats: list[str] | None = None,
    close_after: bool = True,
) -> list[Path]:
    """
    Save figure to disk in specified formats.

    Args:
        fig: Matplotlib figure
        output_path: Base output path (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg')
        close_after: Whether to close figure after saving

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        path = output_path.parent / f"{output_path.name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)

    if close_after:
        plt.close(fig)

    return saved_paths


def save_individual_subplot(
    fig: Figure,
    ax: Axes,
    output_path: Path,
) -> Path | None:
    """
    Save individual subplot to disk.

    Args:
        fig: Parent figure
        ax: Axes to save
        output_path: Output file path

    Returns:
        Saved file path or None if failed
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        extent = ax.get_tightbbox(fig.canvas.get_renderer())
        if extent:
            fig.savefig(
                output_path,
                bbox_inches=extent.expanded(1.3, 1.3),
                dpi=300,
            )
            return output_path
    except Exception:
        pass
    return None
