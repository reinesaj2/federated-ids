#!/usr/bin/env python3
"""Generate a publication-oriented heterogeneity claim report from census artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

DATASET_CONTEXT = {
    "cic": {
        "label": "CIC-IDS2017",
        "effective_class_count": 2.16,
        "majority_share": 0.803,
    },
    "edge-iiotset-full": {
        "label": "Edge-IIoTset",
        "effective_class_count": 3.35,
        "majority_share": 0.728,
    },
    "unsw": {
        "label": "UNSW-NB15",
        "effective_class_count": 1.32,
        "majority_share": 0.952,
    },
}

ALPHA_ORDER = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, math.inf]


def format_alpha(alpha: float) -> str:
    """Format alpha values for human-readable reporting."""
    if math.isinf(alpha):
        return "inf"
    return f"{alpha:g}"


def classify_heterogeneity_regime(alpha_to_metric: dict[float, float]) -> str:
    """Classify the dataset-level heterogeneity response regime."""
    extreme_alpha = min(alpha for alpha in alpha_to_metric if not math.isinf(alpha))
    iid_alpha = math.inf if math.inf in alpha_to_metric else max(alpha_to_metric)
    best_alpha = max(alpha_to_metric, key=alpha_to_metric.get)

    if best_alpha == extreme_alpha and alpha_to_metric[extreme_alpha] > alpha_to_metric[iid_alpha]:
        return "rare_class_concentration_favored"
    if math.isinf(best_alpha) or best_alpha >= 0.5:
        return "iid_favored"
    return "mixed"


def infer_mechanism(dataset: str, regime: str) -> str:
    """Infer a concise mechanism statement from dataset context and regime."""
    context = DATASET_CONTEXT[dataset]
    effective_class_count = context["effective_class_count"]
    majority_share = context["majority_share"]

    if regime == "rare_class_concentration_favored":
        return (
            f"Effective class count {effective_class_count:.2f} and majority share {majority_share:.1%} "
            "indicate an almost binary regime where severe non-IID partitioning concentrates rare attacks "
            "into a subset of clients, strengthening local minority-class signal."
        )
    if regime == "iid_favored":
        return (
            f"Effective class count {effective_class_count:.2f} and majority share {majority_share:.1%} "
            "leave enough multi-class support for broader client coverage to help; as partitions approach IID, "
            "clients collectively expose the server to more attack families per round."
        )
    return (
        f"Effective class count {effective_class_count:.2f} and majority share {majority_share:.1%} suggest "
        "heterogeneity effects are present but not dominated by a single regime across the alpha sweep."
    )


def load_heterogeneity_coverage(coverage_path: Path) -> pd.DataFrame:
    """Load the heterogeneity slice from the confirmatory coverage artifact."""
    coverage = pd.read_csv(coverage_path)
    heterogeneity = coverage[coverage["slice"] == "heterogeneity_fedavg"].copy()
    heterogeneity["alpha"] = heterogeneity["alpha"].map(
        lambda value: math.inf if str(value) == "inf" else float(value)
    )
    heterogeneity["metric_mean"] = pd.to_numeric(heterogeneity["metric_mean"], errors="coerce")
    heterogeneity["alpha_order"] = heterogeneity["alpha"].map(ALPHA_ORDER.index)
    heterogeneity = heterogeneity.sort_values(["dataset", "alpha_order"]).drop(columns=["alpha_order"])
    return heterogeneity


def build_regime_summary(heterogeneity: pd.DataFrame) -> pd.DataFrame:
    """Build a publication-ready regime summary for each dataset."""
    rows: list[dict[str, object]] = []
    for dataset, group in heterogeneity.groupby("dataset", sort=False):
        alpha_to_metric = {
            float(row.alpha): float(row.metric_mean)
            for row in group.itertuples(index=False)
            if not pd.isna(row.metric_mean)
        }
        regime = classify_heterogeneity_regime(alpha_to_metric)
        best_alpha, best_metric = max(alpha_to_metric.items(), key=lambda item: item[1])
        worst_alpha, worst_metric = min(alpha_to_metric.items(), key=lambda item: item[1])
        iid_metric = alpha_to_metric[math.inf]
        extreme_metric = alpha_to_metric[min(alpha for alpha in alpha_to_metric if not math.isinf(alpha))]
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_CONTEXT[dataset]["label"],
                "effective_class_count": DATASET_CONTEXT[dataset]["effective_class_count"],
                "majority_share": DATASET_CONTEXT[dataset]["majority_share"],
                "regime": regime,
                "best_alpha": format_alpha(best_alpha),
                "best_metric": best_metric,
                "worst_alpha": format_alpha(worst_alpha),
                "worst_metric": worst_metric,
                "delta_iid_minus_extreme": iid_metric - extreme_metric,
                "mechanism": infer_mechanism(dataset, regime),
            }
        )
    return pd.DataFrame(rows)


def tightened_claim(summary: pd.DataFrame) -> str:
    """Render the core tightened heterogeneity claim."""
    iid_favored = summary[summary["regime"] == "iid_favored"]["dataset_label"].tolist()
    rare_class_favored = summary[summary["regime"] == "rare_class_concentration_favored"]["dataset_label"].tolist()
    return (
        "FedAvg heterogeneity response is dataset-dependent: "
        f"{', '.join(iid_favored)} improve as partitions approach IID, while "
        f"{', '.join(rare_class_favored)} benefits from extreme heterogeneity because minority attacks "
        "become concentrated within fewer clients."
    )


def render_markdown(summary: pd.DataFrame) -> str:
    """Render the report as Markdown."""
    lines = [
        "# Heterogeneity Claim Report",
        "",
        "## Tightened Claim",
        "",
        tightened_claim(summary),
        "",
        "## Dataset Regimes",
        "",
        "| Dataset | Effective classes | Best alpha | Best macro-F1 | Worst alpha | Worst macro-F1 | IID-extreme delta | Regime |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset_label} | {row.effective_class_count:.2f} | {row.best_alpha} | {row.best_metric:.3f} | "
            f"{row.worst_alpha} | {row.worst_metric:.3f} | {row.delta_iid_minus_extreme:.3f} | {row.regime} |"
        )

    lines.extend(["", "## Mechanism Notes", ""])

    for row in summary.itertuples(index=False):
        lines.append(f"### {row.dataset_label}")
        lines.append(row.mechanism)
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate heterogeneity publication report.")
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=Path("reports") / "runs_census" / "coverage_confirmatory.csv",
        help="Path to coverage_confirmatory.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports") / "publication" / "heterogeneity_claim_report.md",
        help="Where to write the Markdown report.",
    )
    parser.add_argument(
        "--summary-csv-path",
        type=Path,
        default=Path("reports") / "publication" / "heterogeneity_claim_summary.csv",
        help="Where to write the CSV summary.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    heterogeneity = load_heterogeneity_coverage(args.coverage_path)
    summary = build_regime_summary(heterogeneity)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(render_markdown(summary))
    summary.to_csv(args.summary_csv_path, index=False)

    print(f"Wrote {args.output_path}")
    print(f"Wrote {args.summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
