from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from plot_config import ConfidenceIntervalConfig, PlotStyle


def first_present(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series | None:
    for name in columns:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def compute_confidence_interval(data: np.ndarray | Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Compute mean and confidence interval using t-distribution.

    Args:
        data: Array of values (must have length >= 1). NaN values should be
              removed before calling this function.
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Raises:
        ValueError: If data is empty

    Note:
        For n=1, returns (mean, mean, mean) with no confidence range.
        For n>=2, computes proper confidence interval using t-distribution.
    """
    from scipy import stats

    data_array = np.asarray(data)

    if len(data_array) == 0:
        raise ValueError("Cannot compute confidence interval for empty data")

    if len(data_array) == 1:
        mean = float(data_array[0])
        return mean, mean, mean

    mean = float(np.mean(data_array))
    se = stats.sem(data_array)
    margin = se * stats.t.ppf((1 + confidence) / 2, len(data_array) - 1)
    return mean, mean - margin, mean + margin


def apply_axis_policy(ax, policy):
    """Apply axis scale/label policy if provided."""

    if not policy:
        return

    if isinstance(policy, dict):
        x_scale = policy.get("x_scale")
        y_scale = policy.get("y_scale")
        x_label = policy.get("x_label")
        y_label = policy.get("y_label")
    else:  # Allow simple namespace objects
        x_scale = getattr(policy, "x_scale", None)
        y_scale = getattr(policy, "y_scale", None)
        x_label = getattr(policy, "x_label", None)
        y_label = getattr(policy, "y_label", None)

    if x_scale and x_scale != "linear":
        ax.set_xscale(x_scale)
    if y_scale and y_scale != "linear":
        ax.set_yscale(y_scale)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def prepare_ci_matrix(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return rounds and matrix (seed x round) for CI computations."""

    if "seed" not in df.columns:
        return None

    pivot = df.pivot_table(index="round", columns="seed", values=column).sort_index().dropna(how="all")
    if pivot.empty:
        return None

    matrix = pivot.to_numpy().T
    valid_rows = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid_rows]
    if matrix.shape[0] < 2:
        return None

    rounds = pivot.index.to_numpy()
    return rounds, matrix


def format_numeric(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_caption_table(
    path: Path | str,
    rows: Sequence[dict],
    fmt: str = "markdown",
    *,
    title: str | None = None,
    append: bool = False,
) -> None:
    if not rows:
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    headers = list(rows[0].keys())
    with destination.open(mode, encoding="utf-8") as handle:
        if fmt == "markdown":
            if title:
                handle.write(f"### {title}\n\n")
            handle.write("| " + " | ".join(headers) + " |\n")
            handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in rows:
                handle.write("| " + " | ".join(format_numeric(row.get(header)) for header in headers) + " |\n")
            handle.write("\n")
        elif fmt == "latex":
            if title:
                handle.write(f"% {title}\n")
            handle.write("\\begin{tabular}{" + "c" * len(headers) + "}\\toprule\n")
            handle.write(" & ".join(headers) + " \\ " + r"\midrule" + "\n")
            for row in rows:
                handle.write(" & ".join(format_numeric(row.get(header)) for header in headers) + " \\ \n")
            handle.write(r"\bottomrule\end{tabular}" + "\n")
        else:
            for row in rows:
                handle.write(",".join(format_numeric(row.get(header)) for header in headers) + "\n")


def summarize_final_metric(
    df: pd.DataFrame,
    column: str | None,
    ci_config: ConfidenceIntervalConfig | None = None,
) -> dict | None:
    if not column or column not in df.columns:
        return None

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None

    summary: dict[str, float] = {"value": float(series.iloc[-1])}

    if ci_config and ci_config.enabled and "seed" in df.columns:
        last_round = df["round"].max()
        selection = pd.to_numeric(df.loc[df["round"] == last_round, column], errors="coerce").dropna()
        if len(selection) >= 2:
            from scipy import stats

            mean = float(selection.mean())
            se = stats.sem(selection)
            margin = se * stats.t.ppf((1 + ci_config.confidence) / 2, len(selection) - 1)
            summary.update(
                {
                    "value": mean,
                    "ci_lower": mean - margin,
                    "ci_upper": mean + margin,
                }
            )

    return summary


def render_mu_scatter(
    ax,
    records: Iterable[dict[str, float | str]],
    style: PlotStyle,
    ci_config: ConfidenceIntervalConfig | None = None,
) -> bool:
    records = list(records)
    if not records:
        return False

    mu_values = np.array([rec["mu"] for rec in records], dtype=float)
    metric_values = np.array([rec["metric"] for rec in records], dtype=float)
    clients = [rec["client"] for rec in records]

    valid = ~np.isnan(mu_values) & ~np.isnan(metric_values)
    if not valid.any():
        return False

    mu_values = mu_values[valid]
    metric_values = metric_values[valid]
    clients = [clients[i] for i, flag in enumerate(valid) if flag]

    unique_clients = sorted(set(clients))
    client_colors = {client: color for client, color in zip(unique_clients, style.get_colors(len(unique_clients)), strict=False)}

    rng = np.random.default_rng(42)
    jitter_scale = max(0.01, 0.02 * (mu_values.max() - mu_values.min() or 1.0))
    for mu, metric, client in zip(mu_values, metric_values, clients, strict=False):
        jitter = rng.normal(0, jitter_scale)
        ax.scatter(
            mu + jitter,
            metric,
            color=client_colors[client],
            alpha=style.alpha,
            s=style.markersize * 10,
            label=f"Client {client}",
        )

    grouped = {}
    for mu, metric in zip(mu_values, metric_values, strict=False):
        grouped.setdefault(mu, []).append(metric)

    sorted_mu = sorted(grouped.keys())
    means = [float(np.mean(grouped[mu])) for mu in sorted_mu]

    lower_bounds: list[float | None] = []
    upper_bounds: list[float | None] = []
    if ci_config and ci_config.enabled:
        from scipy import stats

        for mu in sorted_mu:
            values = grouped[mu]
            if len(values) < 2:
                lower_bounds.append(None)
                upper_bounds.append(None)
                continue
            mean = np.mean(values)
            se = stats.sem(values)
            margin = se * stats.t.ppf((1 + ci_config.confidence) / 2, len(values) - 1)
            lower_bounds.append(mean - margin)
            upper_bounds.append(mean + margin)
    else:
        lower_bounds = [None] * len(sorted_mu)
        upper_bounds = [None] * len(sorted_mu)

    ax.plot(
        sorted_mu,
        means,
        "o-",
        color="black",
        linewidth=style.linewidth,
        markersize=style.markersize,
        alpha=1.0,
        label="Global Mean",
    )

    for mu, low, high in zip(sorted_mu, lower_bounds, upper_bounds, strict=False):
        if low is None or high is None:
            continue
        ax.fill_between(
            [mu - jitter_scale * 0.5, mu + jitter_scale * 0.5],
            [low, low],
            [high, high],
            color="black",
            alpha=ci_config.alpha if ci_config else 0.1,
            label=f"{int(ci_config.confidence * 100)}% CI" if mu == sorted_mu[0] else "",
        )

    ax.set_xlabel("μ (FedProx)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Per-client μ Scatter with Global Mean")
    return True
