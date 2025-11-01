from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    first_present,
    render_mu_scatter,
    summarize_final_metric,
    write_caption_table,
)


def _render_client_loss(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    label: str,
    color: str,
    available: dict[str, str | None],
) -> bool:
    column = available.get("loss")
    if not column or column not in df.columns:
        return False

    series = first_present(df, [column])
    if series is None or series.isna().all():
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    ax.plot(
        rounds,
        series,
        "o-",
        color=color,
        label=f"Client {label}",
        linewidth=style.linewidth,
        markersize=style.markersize,
        alpha=style.alpha,
    )
    ax.set_title("Client Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    return True


def _render_client_accuracy(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    label: str,
    color: str,
    available: dict[str, str | None],
) -> bool:
    column = available.get("accuracy")
    if not column or column not in df.columns:
        return False

    series = first_present(df, [column])
    if series is None or series.isna().all():
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    ax.plot(
        rounds,
        series,
        "o-",
        color=color,
        label=f"Client {label}",
        linewidth=style.linewidth,
        markersize=style.markersize,
        alpha=style.alpha,
    )
    ax.set_title("Client Accuracy")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    return True


def _render_client_norms(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    label: str,
    color: str,
    available: dict[str, str | None],
) -> bool:
    weight_column = available.get("norms")
    grad_column = available.get("grad_norms")

    weight_series = first_present(df, [weight_column]) if weight_column else None
    grad_series = first_present(df, [grad_column]) if grad_column else None

    has_weight = weight_series is not None and not weight_series.isna().all()
    has_grad = grad_series is not None and not grad_series.isna().all()
    if not (has_weight or has_grad):
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    if has_weight:
        ax.plot(
            rounds,
            weight_series,
            "o-",
            color=color,
            label=f"Client {label} (weights)",
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
        )
    if has_grad:
        ax.plot(
            rounds,
            grad_series,
            "s--",
            color=color,
            label=f"Client {label} (grads)",
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=max(style.alpha - 0.1, 0.2),
        )

    ax.set_title("Client Norms")
    ax.set_xlabel("Round")
    ax.set_ylabel("Norm")
    return True


def _render_client_f1_overlay(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    label: str,
    color: str,
    available: dict[str, str | None],
) -> bool:
    column = available.get("f1_comparison")
    if not column or column not in df.columns:
        return False

    argmax_series = first_present(df, [column])
    tau_series = first_present(df, ["f1_bin_tau"])

    if (argmax_series is None or argmax_series.isna().all()) and (tau_series is None or tau_series.isna().all()):
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    if argmax_series is not None and not argmax_series.isna().all():
        ax.plot(
            rounds,
            argmax_series,
            "o-",
            color=color,
            label=f"Client {label} (argmax)",
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
        )

    if tau_series is not None and not tau_series.isna().all():
        ax.plot(
            rounds,
            tau_series,
            "s--",
            color=color,
            label=f"Client {label} (τ)",
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
        )

    ax.set_title("Macro F1 Comparison")
    ax.set_xlabel("Round")
    ax.set_ylabel("Macro F1")
    return True


def _render_client_tau(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    label: str,
    color: str,
    available: dict[str, str | None],
) -> bool:
    column = available.get("threshold")
    if not column or column not in df.columns:
        return False

    tau_series = first_present(df, [column])
    if tau_series is None or tau_series.isna().all():
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    ax.plot(
        rounds,
        tau_series,
        "o-",
        color=color,
        label=f"Client {label}",
        linewidth=style.linewidth,
        markersize=style.markersize,
        alpha=style.alpha,
    )
    ax.set_title("Decision Threshold τ")
    ax.set_xlabel("Round")
    ax.set_ylabel("τ")
    return True


def _render_client_fpr(
    ax,
    df: pd.DataFrame,
    style: PlotStyle,
    label: str,
    color: str,
    available: dict[str, str | None],
) -> bool:
    column = available.get("fpr")
    if not column or column not in df.columns:
        return False

    series = first_present(df, [column])
    if series is None or series.isna().all():
        return False

    rounds = pd.to_numeric(df.get("round"), errors="coerce")
    ax.plot(
        rounds,
        series,
        "o-",
        color=color,
        label=f"Client {label}",
        linewidth=style.linewidth,
        markersize=style.markersize,
        alpha=style.alpha,
    )
    ax.set_title("Benign FPR")
    ax.set_xlabel("Round")
    ax.set_ylabel("FPR")
    return True


def plot_client_metrics(
    client_metrics_paths: list[str],
    output_path: str,
    config: dict | None = None,
) -> None:
    if not client_metrics_paths:
        print("No client metrics found; skipping client plots")
        return

    if config is None:
        config = create_default_config()

    style = config.get("style", PlotStyle())
    layout = config.get("layout", LayoutConfig())
    title = config.get("title", "Federated Learning Client Metrics")

    style.apply()

    fig, axes = plt.subplots(2, 3, figsize=style.figsize)
    fig.suptitle(title, fontsize=style.title_size)

    colors = style.get_colors(len(client_metrics_paths) or 1)
    detector = config.get("detector", MetricDetector())
    ci_config = config.get("ci", ConfidenceIntervalConfig())
    axes_policy = config.get("axes", {})

    axis_map = {
        "loss": axes[0, 0],
        "accuracy": axes[0, 1],
        "norms": axes[1, 0],
        "f1_comparison": axes[0, 2],
        "threshold": axes[1, 1],
        "fpr": axes[1, 2],
    }

    plotted_flags = {name: False for name in axis_map}
    mu_records: list[dict[str, float | str]] = []
    client_summary: dict[str, dict] = {}

    render_plan = [
        ("loss", _render_client_loss),
        ("accuracy", _render_client_accuracy),
        ("norms", _render_client_norms),
        ("f1_comparison", _render_client_f1_overlay),
        ("threshold", _render_client_tau),
        ("fpr", _render_client_fpr),
    ]

    for idx, client_path in enumerate(client_metrics_paths):
        try:
            df = pd.read_csv(client_path)
        except Exception as exc:
            print(f"Failed to read {client_path}: {exc}")
            continue

        if df.empty or "round" not in df.columns:
            print(f"Skipping {client_path} (empty or missing round column)")
            continue

        color = colors[idx % len(colors)]
        client_id = _client_label(Path(client_path), df)
        available = detector.detect_available(df, metric_type="client")

        summary_entry = client_summary.setdefault("Client " + client_id, {"client": client_id})

        for key, renderer in render_plan:
            axis = axis_map[key]
            rendered = renderer(axis, df, style, client_id, color, available)
            plotted_flags[key] = plotted_flags[key] or rendered

        if "mu" in df.columns and available.get("f1_comparison"):
            mu_series = pd.to_numeric(df["mu"], errors="coerce")
            metric_series = pd.to_numeric(df[available["f1_comparison"]], errors="coerce")
            valid = ~(mu_series.isna() | metric_series.isna())
            for mu_value, metric_value in zip(mu_series[valid], metric_series[valid], strict=False):
                mu_records.append(
                    {
                        "mu": float(mu_value),
                        "metric": float(metric_value),
                        "client": client_id,
                    }
                )

        metric_mappings = {
            "loss_final": available.get("loss"),
            "accuracy_final": available.get("accuracy"),
            "norm_final": available.get("norms"),
            "macro_f1_final": available.get("f1_comparison"),
            "tau_final": available.get("threshold"),
            "fpr_final": available.get("fpr"),
        }

        for summary_key, column in metric_mappings.items():
            summary = summarize_final_metric(df, column, ci_config)
            if summary:
                summary_entry[summary_key] = summary.get("value")
                if "ci_lower" in summary:
                    summary_entry[f"{summary_key}_ci_lower"] = summary["ci_lower"]
                    summary_entry[f"{summary_key}_ci_upper"] = summary["ci_upper"]

    if mu_records:
        scatter_axis = axis_map["f1_comparison"]
        scatter_rendered = render_mu_scatter(scatter_axis, mu_records, style, ci_config)
        plotted_flags["f1_comparison"] = plotted_flags["f1_comparison"] or scatter_rendered

    for key, axis in axis_map.items():
        if not plotted_flags[key]:
            axis.axis("off")
        else:
            axis.legend()
            axis.grid(True, alpha=0.3)
            apply_axis_policy(axis, axes_policy.get(key))

    if layout.tight_layout:
        plt.tight_layout()

    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight")
    plt.close()

    caption_cfg = config.get("caption", {})
    if caption_cfg.get("enabled"):
        rows = list(client_summary.values())
        caption_path = caption_cfg.get("path")
        if caption_path:
            caption_path = Path(caption_path)
        else:
            caption_path = Path(output_path).with_suffix(".md")
        fmt = caption_cfg.get("format", "markdown")

        if rows:
            write_caption_table(caption_path, rows, fmt, title="Client Metrics Summary")

        if mu_records:
            grouped = {}
            for record in mu_records:
                grouped.setdefault(record["mu"], []).append(record["metric"])

            mu_rows = []
            for mu_value, values in sorted(grouped.items()):
                mean = float(np.mean(values))
                row = {"mu": mu_value, "macro_f1_mean": mean}
                if ci_config and ci_config.enabled and len(values) >= 2:
                    from scipy import stats

                    se = stats.sem(values)
                    margin = se * stats.t.ppf((1 + ci_config.confidence) / 2, len(values) - 1)
                    row["ci_lower"] = mean - margin
                    row["ci_upper"] = mean + margin
                mu_rows.append(row)

            if mu_rows:
                write_caption_table(
                    caption_path,
                    mu_rows,
                    fmt,
                    title="μ Scatter Summary",
                    append=bool(rows),
                )


def _client_label(path: Path, df: pd.DataFrame) -> str:
    column = df.get("client_id")
    if column is not None and not column.empty and pd.notna(column.iloc[0]):
        return str(column.iloc[0])

    stem = Path(path).stem
    if "client_" in stem:
        return stem.split("client_")[-1]
    return stem
