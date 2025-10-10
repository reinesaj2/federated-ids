from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from plot_config import ConfidenceIntervalConfig, PlotStyle, create_default_config
from plot_metrics_client import (
    plot_client_metrics,
    _render_client_accuracy,
    _render_client_f1_overlay,
    _render_client_fpr,
    _render_client_loss,
    _render_client_norms,
    _render_client_tau,
)
from plot_metrics_utils import render_mu_scatter


def test_render_client_loss_draws_series():
    df = pd.DataFrame({"round": [1, 2, 3], "loss_after": [0.9, 0.8, 0.75]})
    style = PlotStyle()

    fig, ax = plt.subplots()
    try:
        rendered = _render_client_loss(
            ax,
            df,
            style,
            label="A",
            color="#123456",
            available={"loss": "loss_after"},
        )
    finally:
        plt.close(fig)

    assert rendered is True
    assert len(ax.get_lines()) == 1


def test_render_client_accuracy_skips_when_missing():
    df = pd.DataFrame({"round": [1, 2, 3]})
    style = PlotStyle()

    fig, ax = plt.subplots()
    try:
        rendered = _render_client_accuracy(
            ax,
            df,
            style,
            label="A",
            color="#654321",
            available={"accuracy": None},
        )
    finally:
        plt.close(fig)

    assert rendered is False


def test_render_client_f1_overlay_plots_both_series():
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "macro_f1_argmax": [0.7, 0.72, 0.74],
            "f1_bin_tau": [0.6, 0.61, 0.65],
        }
    )
    style = PlotStyle()

    fig, ax = plt.subplots()
    try:
        rendered = _render_client_f1_overlay(
            ax,
            df,
            style,
            label="B",
            color="#abcdef",
            available={"f1_comparison": "macro_f1_argmax"},
        )
    finally:
        plt.close(fig)

    assert rendered is True
    assert len(ax.get_lines()) == 2


def test_render_client_tau_draws_threshold_line():
    df = pd.DataFrame({"round": [1, 2, 3], "tau_bin": [0.5, 0.55, 0.6]})
    style = PlotStyle()

    fig, ax = plt.subplots()
    try:
        rendered = _render_client_tau(
            ax,
            df,
            style,
            label="C",
            color="#00ff00",
            available={"threshold": "tau_bin"},
        )
    finally:
        plt.close(fig)

    assert rendered is True
    assert len(ax.get_lines()) == 1


def test_render_client_fpr_handles_missing_series():
    df = pd.DataFrame({"round": [1, 2, 3], "benign_fpr_bin_tau": [float("nan")] * 3})
    style = PlotStyle()

    fig, ax = plt.subplots()
    try:
        rendered = _render_client_fpr(
            ax,
            df,
            style,
            label="C",
            color="#ff0000",
            available={"fpr": "benign_fpr_bin_tau"},
        )
    finally:
        plt.close(fig)

    assert rendered is False


def test_render_client_norms_overlays_weight_and_grad():
    df = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "weight_norm_after": [1.0, 1.1, 1.2],
            "grad_norm_l2": [0.5, 0.55, 0.6],
        }
    )
    style = PlotStyle()

    fig, ax = plt.subplots()
    try:
        rendered = _render_client_norms(
            ax,
            df,
            style,
            label="D",
            color="#112233",
            available={"norms": "weight_norm_after", "grad_norms": "grad_norm_l2"},
        )
    finally:
        plt.close(fig)

    assert rendered is True
    assert len(ax.get_lines()) == 2


def test_render_mu_scatter_plots_points_and_mean():
    records = [
        {"mu": 0.0, "metric": 0.7, "client": "A"},
        {"mu": 0.0, "metric": 0.72, "client": "B"},
        {"mu": 0.1, "metric": 0.75, "client": "A"},
        {"mu": 0.1, "metric": 0.78, "client": "B"},
    ]
    style = PlotStyle()
    ci_config = ConfidenceIntervalConfig(enabled=True)

    fig, ax = plt.subplots()
    try:
        rendered = render_mu_scatter(ax, records, style, ci_config)
    finally:
        plt.close(fig)

    assert rendered is True
    assert any(isinstance(coll, mcollections.PathCollection) for coll in ax.collections)
    assert len(ax.get_lines()) >= 1


def test_plot_client_metrics_writes_caption(tmp_path):
    client_a = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "loss_after": [0.9, 0.8, 0.7],
            "acc_after": [0.6, 0.65, 0.7],
            "macro_f1_argmax": [0.5, 0.55, 0.6],
            "mu": [0.0, 0.0, 0.0],
        }
    )
    client_b = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "loss_after": [0.85, 0.75, 0.68],
            "acc_after": [0.62, 0.67, 0.72],
            "macro_f1_argmax": [0.52, 0.58, 0.63],
            "mu": [0.1, 0.1, 0.1],
        }
    )

    client_a_path = tmp_path / "client_0_metrics.csv"
    client_b_path = tmp_path / "client_1_metrics.csv"
    client_a.to_csv(client_a_path, index=False)
    client_b.to_csv(client_b_path, index=False)

    output_path = tmp_path / "clients_plot.png"
    caption_path = tmp_path / "clients_caption.md"

    config = create_default_config()
    config["caption"] = {"enabled": True, "path": caption_path}

    plot_client_metrics(
        [str(client_a_path), str(client_b_path)],
        str(output_path),
        config,
    )

    assert caption_path.exists()
    assert "Client" in caption_path.read_text()
