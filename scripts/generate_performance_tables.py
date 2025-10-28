#!/usr/bin/env python3
"""Generate PERFORMANCE_COMPARISON_TABLES.md from recorded experiment artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MACRO_F1_MIN, MACRO_F1_MAX = 0.0, 1.0


def load_attack_resilience() -> List[Dict[str, float]]:
    csv_path = (
        REPO_ROOT
        / "results"
        / "comparative_analysis"
        / "attack_resilience_stats.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: List[Dict[str, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            agg = row.get("aggregation")
            if not agg:
                continue
            rows.append(
                {
                    "aggregation": agg,
                    "adversary_fraction": float(row["adversary_fraction"]),
                    "macro_f1_mean": float(row["macro_f1_mean"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                    "degradation_pct": float(row["degradation_pct"]),
                    "samples": int(float(row["n"])),
                }
            )
    return rows


def summarise_attack_resilience(
    rows: Iterable[Dict[str, float]], adv_levels: Tuple[float, ...] = (0.0, 0.1, 0.3)
) -> List[Dict[str, float]]:
    metrics: Dict[str, Dict[float, Dict[str, float]]] = defaultdict(dict)
    for row in rows:
        metrics[row["aggregation"]][row["adversary_fraction"]] = row

    summary: List[Dict[str, float]] = []
    for aggregation, values in sorted(metrics.items()):
        entry: Dict[str, float] = {"aggregation": aggregation}
        for level in adv_levels:
            row = values.get(level)
            if not row:
                continue
            entry[f"f{int(level * 100)}_macro_f1"] = row["macro_f1_mean"]
            entry[f"f{int(level * 100)}_ci_lower"] = row["ci_lower"]
            entry[f"f{int(level * 100)}_ci_upper"] = row["ci_upper"]
            entry[f"f{int(level * 100)}_n"] = row["samples"]
            if level == 0.3:
                entry["f30_drop_pct"] = row["degradation_pct"]
        summary.append(entry)
    return summary


def load_personalization_summary() -> Dict[str, object]:
    json_path = (
        REPO_ROOT / "analysis" / "personalization" / "personalization_summary.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return json.loads(json_path.read_text())


def load_fedprox_summary() -> Dict[str, object]:
    json_path = (
        REPO_ROOT / "analysis" / "fedprox_nightly" / "fedprox_comparison_summary.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return json.loads(json_path.read_text())


def load_privacy_curve() -> List[Dict[str, object]]:
    csv_path = REPO_ROOT / "results" / "privacy_check" / "privacy_utility_curve.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows: List[Dict[str, object]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epsilon_raw = row.get("epsilon") or ""
            epsilon_raw = epsilon_raw.strip()
            if epsilon_raw:
                try:
                    epsilon: float | str = float(epsilon_raw)
                except ValueError:
                    epsilon = epsilon_raw
            else:
                epsilon = "baseline"

            rows.append(
                {
                    "epsilon": epsilon,
                    "macro_f1_mean": (
                        float(row["macro_f1_mean"])
                        if row.get("macro_f1_mean")
                        else None
                    ),
                    "ci_lower": (
                        float(row["ci_lower"]) if row.get("ci_lower") else None
                    ),
                    "ci_upper": (
                        float(row["ci_upper"]) if row.get("ci_upper") else None
                    ),
                    "noise_multiplier": (
                        float(row["dp_noise_multiplier"])
                        if row.get("dp_noise_multiplier")
                        else None
                    ),
                    "is_baseline": bool(int(row.get("is_baseline", "0") or 0)),
                    "samples": (
                        int(float(row["n"])) if row.get("n") not in (None, "") else None
                    ),
                }
            )
    return rows


def format_macro(value: float | None) -> str:
    if value is None:
        return "—"
    value = max(MACRO_F1_MIN, min(MACRO_F1_MAX, value))
    return f"{value:.3f}"


def compute_attack_table(summary: List[Dict[str, float]]) -> str:
    headers = [
        "Aggregation",
        "Macro-F1 (f=0%)",
        "Macro-F1 (f=10%)",
        "Macro-F1 (f=30%)",
        "Δ at 30% (drop)",
        "Runs",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for row in summary:
        runs = int(row.get("f0_n") or row.get("f10_n") or row.get("f30_n") or 0)
        lines.append(
            "| {aggregation} | {f0} | {f10} | {f30} | {drop:.1f}% | {runs} |".format(
                aggregation=row["aggregation"].title(),
                f0=format_macro(row.get("f0_macro_f1")),
                f10=format_macro(row.get("f10_macro_f1")),
                f30=format_macro(row.get("f30_macro_f1")),
                drop=row.get("f30_drop_pct", 0.0),
                runs=runs,
            )
        )
    return "\n".join(lines)


def compute_personalization_table(summary: Dict[str, object]) -> str:
    overall = summary["overall"]
    by_dataset = summary.get("by_dataset", {})
    headers = ["Scope", "Mean ΔF1", "Std Dev", "Clients"]
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    total_clients = sum(d["n_clients"] for d in by_dataset.values())
    lines.append(
        "| Overall | {mean:.3f} | {std:.3f} | {clients} |".format(
            mean=overall["mean_gain"],
            std=overall["std_gain"],
            clients=total_clients,
        )
    )
    for dataset, stats in sorted(by_dataset.items()):
        lines.append(
            "| Dataset: {dataset} | {mean:.3f} | — | {clients} |".format(
                dataset=dataset.upper(),
                mean=stats["mean_gain"],
                clients=stats["n_clients"],
            )
        )
    return "\n".join(lines)


def compute_fedprox_table(summary: Dict[str, object]) -> str:
    improvements = summary["raw_analysis_results"]["improvement_ratios"]
    headers = ["Scenario", "L2 Improvement (×)", "Time Overhead (×)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for scenario, stats in sorted(improvements.items()):
        lines.append(
            "| {scenario} | {l2:.2f} | {time:.2f} |".format(
                scenario=scenario.replace("_", " "),
                l2=stats["l2_improvement"],
                time=stats["time_overhead"],
            )
        )
    return "\n".join(lines)


def compute_privacy_table(rows: List[Dict[str, object]]) -> str:
    headers = ["ε (target)", "Macro-F1", "Noise σ", "Samples"]
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for row in rows:
        epsilon = "baseline" if row["is_baseline"] else row["epsilon"]
        lines.append(
            "| {epsilon} | {macro} | {sigma} | {samples} |".format(
                epsilon=epsilon,
                macro=format_macro(row["macro_f1_mean"]),
                sigma=(
                    f"{row['noise_multiplier']:.2f}"
                    if row["noise_multiplier"] is not None
                    else "—"
                ),
                samples=row["samples"] or "—",
            )
        )
    return "\n".join(lines)


def build_markdown() -> str:
    attack_rows = load_attack_resilience()
    attack_summary = summarise_attack_resilience(attack_rows)
    personalization_summary = load_personalization_summary()
    fedprox_summary = load_fedprox_summary()
    privacy_rows = load_privacy_curve()

    sections: List[str] = []
    sections.append("# Performance Comparison Tables")
    sections.append(
        "Consolidated metrics derived from the latest committed experiment artifacts."
    )

    sections.append("## Attack Resilience (Macro-F1)")
    sections.append(
        "Source: `results/comparative_analysis/attack_resilience_stats.csv`. "
        "Macro-F1 values are clipped to [0,1] to account for a known logging anomaly "
        "on the FedAvg baseline while preserving the relative degradation signals."
    )
    sections.append(compute_attack_table(attack_summary))

    sections.append("## Personalization Gains (Δ Macro-F1)")
    sections.append(compute_personalization_table(personalization_summary))

    sections.append("## FedAvg vs FedProx (Heterogeneity Stability)")
    sections.append(compute_fedprox_table(fedprox_summary))

    sections.append("## Privacy–Utility Trade-off (Differential Privacy)")
    sections.append(compute_privacy_table(privacy_rows))

    sections.append("---")
    sections.append(
        "_Last regenerated via `python scripts/generate_performance_tables.py`._"
    )

    return "\n\n".join(sections) + "\n"


def main() -> None:
    markdown = build_markdown()
    output_path = REPO_ROOT / "PERFORMANCE_COMPARISON_TABLES.md"
    output_path.write_text(markdown)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
