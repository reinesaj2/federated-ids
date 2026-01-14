import os
import tempfile

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(tempfile.gettempdir()))

import matplotlib.pyplot as plt

from plots.figures.primitives import plot_bar_comparison, plot_heatmap, plot_line_with_ci, save_figure


def test_plot_bar_comparison_adds_legend_for_labels():
    categories = ["baseline", "robust"]
    values = [0.6, 0.8]
    labels = ["FedAvg", "Median"]

    fig, ax = plt.subplots()
    plot_bar_comparison(ax, categories, values, labels=labels)
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()] if legend else []
    assert legend_texts == labels
    plt.close(fig)


def test_plot_line_with_ci_adds_line_and_band():
    x_values = [1, 2, 3]
    y_values = [0.2, 0.4, 0.6]
    ci_lower = [0.1, 0.3, 0.5]
    ci_upper = [0.3, 0.5, 0.7]

    fig, ax = plt.subplots()
    plot_line_with_ci(ax, x_values, y_values, ci_lower=ci_lower, ci_upper=ci_upper)
    assert (len(ax.lines), len(ax.collections)) == (1, 1)
    plt.close(fig)


def test_plot_heatmap_creates_mesh():
    data = np.array([[0.1, 0.2], [0.3, 0.4]])
    row_labels = ["row1", "row2"]
    col_labels = ["col1", "col2"]

    fig, ax = plt.subplots()
    plot_heatmap(ax, data, row_labels, col_labels, title="Heatmap")
    assert len(ax.collections) == 1
    plt.close(fig)


def test_save_figure_writes_requested_formats(tmp_path):
    fig, _ = plt.subplots()
    output_path = tmp_path / "sample_plot"
    saved = save_figure(fig, output_path, formats=["png"])
    assert [(path.name, path.exists()) for path in saved] == [("sample_plot.png", True)]
