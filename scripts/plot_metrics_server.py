from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from plot_config import (
    ConfidenceIntervalConfig,
    LayoutConfig,
    MetricDetector,
    PlotStyle,
    create_default_config,
)
from plot_metrics_utils import (
    apply_axis_policy,
    prepare_ci_matrix,
    summarize_final_metric,
    write_caption_table,
)


def _render_server_timing(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    available: dict[str, str | None],
    colors: list[str],
    ci_config: ConfidenceIntervalConfig,
) -> bool:
    column = available.get("timing")
    if not column or column not in df.columns:
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    if rounds is None or rounds.isna().all():
        return False

    palette = colors or style.get_colors(4)
    color = palette[0]

    ci_rendered = False
    ci_stats = None
    if ci_config and ci_config.enabled:
        ci_input = prepare_ci_matrix(df, column)
        if ci_input:
            rounds_ci, matrix = ci_input
            ci_values = ci_config.compute([row for row in matrix])
            if ci_values:
                mean = ci_values["mean"]
                lower = ci_values["lower"]
                upper = ci_values["upper"]
                ax.plot(
                    rounds_ci,
                    mean,
                    "o-",
                    color=color,
                    label="Aggregation Time (mean)",
                    linewidth=style.linewidth,
                    markersize=style.markersize,
                    alpha=style.alpha,
                )
                ax.fill_between(
                    rounds_ci,
                    lower,
                    upper,
                    color=color,
                    alpha=ci_config.alpha,
                    label=f"{int(ci_config.confidence * 100)}% CI",
                )
                box = ax.boxplot(
                    matrix.T,
                    positions=rounds_ci,
                    widths=0.2,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in box["boxes"]:
                    patch.set(facecolor=color, alpha=ci_config.alpha * 0.6)
                for median in box["medians"]:
                    median.set(color=color, linewidth=style.linewidth)
                ci_rendered = True
                ci_stats = pd.Series(mean, index=rounds_ci)

    time_series = pd.to_numeric(df[column], errors="coerce")
    if ci_rendered:
        time_series = ci_stats
    if time_series is None or time_series.isna().all():
        return False

    if not ci_rendered:
        ax.plot(
            rounds,
            time_series,
            "o-",
            color=color,
            label="Aggregation Time (ms)",
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Aggregation Time per Round")
    return True


def _render_server_robustness(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    available: dict[str, str | None],
    colors: list[str],
    ci_config: ConfidenceIntervalConfig,
) -> bool:
    column = available.get("robustness")
    if not column or column not in df.columns:
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    l2_series = pd.to_numeric(df[column], errors="coerce")
    cosine_series = None
    if "cos_to_benign_mean" in df.columns:
        cosine_series = pd.to_numeric(df["cos_to_benign_mean"], errors="coerce")

    if (l2_series is None or l2_series.isna().all()) and (cosine_series is None or cosine_series.isna().all()):
        return False

    palette = colors or style.get_colors(4)
    rendered = False
    if l2_series is not None and not l2_series.isna().all():
        ax.plot(
            rounds,
            l2_series,
            "o-",
            color=palette[1],
            label="L2 to Benign Mean",
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
        )
        rendered = True

    if cosine_series is not None and not cosine_series.isna().all():
        ax.plot(
            rounds,
            cosine_series,
            "s-",
            color=palette[2],
            label="Cosine Similarity",
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
        )
        rendered = True

    if not rendered:
        return False

    ax.set_xlabel("Round")
    ax.set_ylabel("Metric Value")
    ax.set_title("Robustness Metrics")
    return True


def _render_server_norms(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    available: dict[str, str | None],
    colors: list[str],
    ci_config: ConfidenceIntervalConfig,
) -> bool:
    column = available.get("norms")
    if not column or column not in df.columns:
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    norm_series = pd.to_numeric(df[column], errors="coerce")
    if norm_series.isna().all():
        return False

    palette = colors or style.get_colors(4)
    ax.plot(
        rounds,
        norm_series,
        "o-",
        color=palette[3],
        label="Mean Update Norm",
        linewidth=style.linewidth,
        markersize=style.markersize,
        alpha=style.alpha,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Norm Value")
    ax.set_title("Update Norms")
    return True


def _render_server_dispersion(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    available: dict[str, str | None],
    colors: list[str],
    ci_config: ConfidenceIntervalConfig,
) -> bool:
    palette = colors or style.get_colors(4)
    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    rendered = False

    if "pairwise_cosine_mean" in df.columns:
        cosine_series = pd.to_numeric(df["pairwise_cosine_mean"], errors="coerce")
        if not cosine_series.isna().all():
            ax.plot(
                rounds,
                cosine_series,
                "o-",
                color=palette[0],
                label="Pairwise Cosine Mean",
                linewidth=style.linewidth,
                markersize=style.markersize,
                alpha=style.alpha,
            )
            rendered = True

    if "l2_dispersion_mean" in df.columns:
        dispersion_series = pd.to_numeric(df["l2_dispersion_mean"], errors="coerce")
        if not dispersion_series.isna().all():
            ax.plot(
                rounds,
                dispersion_series,
                "s-",
                color=palette[1],
                label="L2 Dispersion Mean",
                linewidth=style.linewidth,
                markersize=style.markersize,
                alpha=style.alpha,
            )
            rendered = True

    if not rendered:
        return False

    ax.set_xlabel("Round")
    ax.set_ylabel("Value")
    ax.set_title("Pairwise Dispersion")
    return True


def plot_server_metrics(metrics_path: str, output_path: str, config: dict | None = None) -> None:
    if config is None:
        config = create_default_config()

    style = config.get("style", PlotStyle())
    layout = config.get("layout", LayoutConfig())
    detector = config.get("detector", MetricDetector())
    title = config.get("title", "Federated Learning Server Metrics")

    style.apply()

    df = pd.read_csv(metrics_path)
    if df.empty or "round" not in df.columns:
        print(f"Skipping server metrics plot for {metrics_path} (empty/missing round)")
        return

    available = detector.detect_available(df, metric_type="server")
    num_plots = detector.count_available_plots(available)
    if num_plots == 0:
        print(f"No server metrics columns available to plot for {metrics_path}")
        return

    rows, cols = layout.compute_grid(num_plots)
    fig, axes = plt.subplots(rows, cols, figsize=style.figsize)
    fig.suptitle(title, fontsize=style.title_size)

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flat

    ax_idx = 0
    ci_config = config.get("ci", ConfidenceIntervalConfig())
    axes_policy = config.get("axes", {})
    colors = style.get_colors(6)

    renderers = [
        ("timing", lambda axis: _render_server_timing(axis, df, style, available, colors, ci_config)),
        ("robustness", lambda axis: _render_server_robustness(axis, df, style, available, colors, ci_config)),
        ("norms", lambda axis: _render_server_norms(axis, df, style, available, colors, ci_config)),
        ("dispersion", lambda axis: _render_server_dispersion(axis, df, style, available, colors, ci_config)),
    ]

    for key, renderer in renderers:
        if ax_idx >= len(axes):
            break
        axis = axes[ax_idx]
        if renderer(axis):
            apply_axis_policy(axis, axes_policy.get(key))
            axis.grid(True, alpha=0.3)
            axis.legend()
            ax_idx += 1

    for i in range(ax_idx, len(axes)):
        axes[i].axis("off")

    if layout.tight_layout:
        plt.tight_layout()

    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight")
    plt.close()

    caption_cfg = config.get("caption", {})
    if caption_cfg.get("enabled"):
        rows: list[dict] = []

        for label, column in (
            ("Aggregation Time (ms)", available.get("timing")),
            ("L2 to Benign Mean", "l2_to_benign_mean" if "l2_to_benign_mean" in df.columns else None),
            ("Cosine Similarity", "cos_to_benign_mean" if "cos_to_benign_mean" in df.columns else None),
            ("Mean Update Norm", available.get("norms")),
            ("Pairwise Cosine Mean", "pairwise_cosine_mean" if "pairwise_cosine_mean" in df.columns else None),
            ("L2 Dispersion Mean", "l2_dispersion_mean" if "l2_dispersion_mean" in df.columns else None),
        ):
            summary = summarize_final_metric(df, column, ci_config)
            if summary:
                rows.append({"metric": label, **summary})

        if rows:
            caption_path = caption_cfg.get("path")
            if caption_path:
                caption_path = Path(caption_path)
            else:
                caption_path = Path(output_path).with_suffix(".md")
            fmt = caption_cfg.get("format", "markdown")
            write_caption_table(caption_path, rows, fmt, title="Server Metrics Summary")
