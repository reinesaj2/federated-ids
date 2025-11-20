from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

plot_config_mod = importlib.import_module("plot_config")
ConfidenceIntervalConfig = plot_config_mod.ConfidenceIntervalConfig
PlotStyle = plot_config_mod.PlotStyle
create_default_config = plot_config_mod.create_default_config

plot_metrics_server_mod = importlib.import_module("plot_metrics_server")
_render_server_dispersion = plot_metrics_server_mod._render_server_dispersion
_render_server_norms = plot_metrics_server_mod._render_server_norms
_render_server_robustness = plot_metrics_server_mod._render_server_robustness
_render_server_timing = plot_metrics_server_mod._render_server_timing
plot_server_metrics = plot_metrics_server_mod.plot_server_metrics


def _default_server_args(style: PlotStyle):
    colors = style.get_colors(6)
    ci = ConfidenceIntervalConfig()
    return colors, ci


def test_render_server_timing_renders_line():
    df = pd.DataFrame({"round": [1, 2, 3], "t_aggregate_ms": [100, 110, 105]})
    available = {"timing": "t_aggregate_ms"}
    style = PlotStyle()
    colors, ci = _default_server_args(style)

    fig, ax = plt.subplots()
    try:
        rendered = _render_server_timing(ax, df, style, available, colors, ci)
    finally:
        plt.close(fig)

    assert rendered is True
    assert len(ax.get_lines()) == 1


def test_render_server_timing_with_ci_adds_band():
    df = pd.DataFrame(
        {
            "round": [1, 1, 2, 2],
            "seed": [0, 1, 0, 1],
            "t_aggregate_ms": [100, 110, 105, 115],
        }
    )
    available = {"timing": "t_aggregate_ms"}
    style = PlotStyle()
    colors, ci = _default_server_args(style)
    ci.enabled = True

    fig, ax = plt.subplots()
    try:
        rendered = _render_server_timing(ax, df, style, available, colors, ci)
    finally:
        plt.close(fig)

    assert rendered is True
    assert any(isinstance(coll, mcollections.PolyCollection) for coll in ax.collections)


def test_render_server_robustness_plots_series():
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "l2_to_benign_mean": [0.1, 0.2, 0.15],
            "cos_to_benign_mean": [0.9, 0.92, 0.94],
        }
    )
    available = {"robustness": "l2_to_benign_mean"}
    style = PlotStyle()
    colors, ci = _default_server_args(style)

    fig, ax = plt.subplots()
    try:
        rendered = _render_server_robustness(ax, df, style, available, colors, ci)
    finally:
        plt.close(fig)

    assert rendered is True
    assert len(ax.get_lines()) == 2


def test_render_server_norms_skips_empty_series():
    df = pd.DataFrame({"round": [1, 2, 3], "update_norm_mean": [float("nan")] * 3})
    available = {"norms": "update_norm_mean"}
    style = PlotStyle()
    colors, ci = _default_server_args(style)

    fig, ax = plt.subplots()
    try:
        rendered = _render_server_norms(ax, df, style, available, colors, ci)
    finally:
        plt.close(fig)

    assert rendered is False


def test_render_server_dispersion_handles_multiple_series():
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "pairwise_cosine_mean": [0.7, 0.75, 0.8],
            "l2_dispersion_mean": [0.2, 0.25, 0.3],
        }
    )
    available = {"dispersion": "pairwise_cosine_mean"}
    style = PlotStyle()
    colors, ci = _default_server_args(style)

    fig, ax = plt.subplots()
    try:
        rendered = _render_server_dispersion(ax, df, style, available, colors, ci)
    finally:
        plt.close(fig)

    assert rendered is True
    assert len(ax.get_lines()) == 2


def test_plot_server_metrics_emits_caption(tmp_path):
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "t_aggregate_ms": [100, 110, 120],
            "l2_to_benign_mean": [0.25, 0.2, 0.15],
        }
    )
    metrics_path = tmp_path / "metrics.csv"
    df.to_csv(metrics_path, index=False)

    output_path = tmp_path / "server_plot.png"
    caption_path = tmp_path / "server_caption.md"

    config = create_default_config()
    config["caption"] = {"enabled": True, "path": caption_path}

    plot_server_metrics(str(metrics_path), str(output_path), config)

    assert caption_path.exists()
    contents = caption_path.read_text()
    assert "Aggregation Time" in contents
