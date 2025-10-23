#!/usr/bin/env python3
"""
Consolidate thesis reporting by orchestrating experiment discovery, validation,
and publication-ready artifact generation.

Workflow:
1. Scan runs/ for experiment directories (comp_* and d2_*)
2. Build experiment metadata manifest with version tracking
3. Generate plots + caption tables for each dimension
4. Emit structured outputs for LaTeX inclusion
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentMetadata:
    """Metadata for a single experiment run."""

    run_dir: str
    dimension: str
    config: dict[str, Any]
    metrics_file: str
    client_metrics_files: list[str]
    has_plots: bool
    final_macro_f1: float | None


def scan_experiment_runs(runs_root: Path) -> list[ExperimentMetadata]:
    """Discover and catalog all experiment runs in runs_root."""
    experiments: list[ExperimentMetadata] = []

    for pattern in ["comp_*", "d2_*"]:
        for run_dir in sorted(runs_root.glob(pattern)):
            if not run_dir.is_dir():
                continue

            config_file = run_dir / "config.json"
            metrics_file = run_dir / "metrics.csv"
            client_metrics = sorted(run_dir.glob("client_*_metrics.csv"))

            if not metrics_file.exists() or not client_metrics:
                continue

            config = {}
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

            dimension = _infer_dimension(config, run_dir.name)
            final_macro_f1 = _extract_final_macro_f1(metrics_file, client_metrics)

            has_plots = bool(list(run_dir.glob("*.png"))) or bool(list(run_dir.glob("*.pdf")))

            experiments.append(
                ExperimentMetadata(
                    run_dir=str(run_dir),
                    dimension=dimension,
                    config=config,
                    metrics_file=str(metrics_file),
                    client_metrics_files=[str(f) for f in client_metrics],
                    has_plots=has_plots,
                    final_macro_f1=final_macro_f1,
                )
            )

    return experiments


def _infer_dimension(config: dict[str, Any], run_dir_name: str) -> str:
    """Infer experiment dimension from config or directory name."""
    if config.get("dimension"):
        return config["dimension"]

    dimension_markers = {
        "adv": "attack",
        "dp": "privacy",
        "pers": "personalization",
        "mu": "heterogeneity_fedprox",
        "alpha": "heterogeneity",
    }

    for marker, dim in dimension_markers.items():
        if marker in run_dir_name.lower():
            return dim

    return "aggregation"


def _extract_final_macro_f1(
    metrics_file: Path, client_metrics_files: list[Path]
) -> float | None:
    """Extract final macro-F1 from client metrics."""
    if not client_metrics_files:
        return None

    # Try reading the first available client metrics file
    for client_file in client_metrics_files:
        try:
            df = pd.read_csv(client_file)
            if df.empty:
                continue

            for col in ["macro_f1_after", "macro_f1_argmax", "macro_f1"]:
                if col in df.columns:
                    # Get the last non-null value
                    last_val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
                    if last_val is not None:
                        return float(last_val)

        except (FileNotFoundError, pd.errors.EmptyDataError, IndexError, KeyError) as e:
            # Silently ignore common pandas errors for individual files
            print(f"  - Warning: Could not read or parse {client_file.name}: {e}")
            continue

    return None


def build_manifest(experiments: list[ExperimentMetadata], output_file: Path) -> None:
    """Write experiment manifest as JSON."""
    manifest = {
        "version": "1.0",
        "total_experiments": len(experiments),
        "by_dimension": {},
        "experiments": [asdict(e) for e in experiments],
    }

    for exp in experiments:
        if exp.dimension not in manifest["by_dimension"]:
            manifest["by_dimension"][exp.dimension] = 0
        manifest["by_dimension"][exp.dimension] += 1

    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest: {output_file}")
    print(f"  Total experiments: {len(experiments)}")
    for dim, count in manifest["by_dimension"].items():
        print(f"    {dim}: {count}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate thesis reporting artifacts")
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=ROOT / "runs",
        help="Directory containing experiment runs",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "results" / "official",
        help="Output directory for consolidated artifacts",
    )
    parser.add_argument(
        "--manifest_only",
        action="store_true",
        help="Only generate manifest; do not regenerate plots",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning experiments in {args.runs_dir}...")
    experiments = scan_experiment_runs(args.runs_dir)

    if not experiments:
        print("No experiments found!")
        return

    manifest_file = args.output_dir / "manifest.json"
    build_manifest(experiments, manifest_file)

    if args.manifest_only:
        print("Manifest-only mode; skipping plot generation")
        return

    print("\nIntegrating plot and caption table generation...")

    # Load the full dataframe for plotting functions
    # This logic is simplified from generate_thesis_plots.py's load_experiment_results
    all_data = []
    for exp in experiments:
        df = pd.read_csv(exp.metrics_file)
        for key, value in exp.config.items():
            df[key] = value
        if "aggregation" not in df.columns:
            df["aggregation"] = exp.config.get("aggregation", "fedavg")
        if "seed" not in df.columns:
            df["seed"] = exp.config.get("seed", 42)
        all_data.append(df)

    if not all_data:
        print("No data found for plotting.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Dynamically import and call generation functions
    try:
        from generate_thesis_plots import (
            plot_aggregation_comparison,
            plot_attack_resilience,
            plot_heterogeneity_comparison,
            plot_personalization_benefit,
            plot_privacy_utility,
        )
        from caption_tables import (
            generate_aggregation_caption_table,
            generate_attack_caption_table,
            generate_heterogeneity_caption_table,
            generate_personalization_caption_table,
            generate_privacy_caption_table,
        )

        plot_functions = {
            "aggregation": plot_aggregation_comparison,
            "attack": plot_attack_resilience,
            "heterogeneity": plot_heterogeneity_comparison,
            "personalization": plot_personalization_benefit,
            "privacy": plot_privacy_utility,
        }
        caption_functions = {
            "aggregation": generate_aggregation_caption_table,
            "attack": generate_attack_caption_table,
            "heterogeneity": generate_heterogeneity_caption_table,
            "personalization": generate_personalization_caption_table,
            "privacy": generate_privacy_caption_table,
        }

        for dimension, count in manifest["by_dimension"].items():
            print(f"\n--- Generating artifacts for dimension: {dimension} ({count} runs) ---")
            if dimension in plot_functions:
                plot_functions[dimension](combined_df, args.output_dir)
                print(f"  -> Generated plot for {dimension}")
            if dimension in caption_functions:
                caption_out = args.output_dir / f"{dimension}_caption.csv"
                caption_functions[dimension](combined_df, caption_out)

    except ImportError as e:
        print(f"\nERROR: Could not import plotting or captioning functions: {e}")
        print("Please ensure generate_thesis_plots.py and caption_tables.py are in the same directory.")
        sys.exit(1)

    print("\nConsolidation complete.")


if __name__ == "__main__":
    main()
