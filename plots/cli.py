#!/usr/bin/env python3
"""
Unified Plotting CLI

Single entrypoint for generating all thesis plots.

Usage:
    python -m plots chapter4 --data data/summary.csv --output results/plots
    python -m plots objective --objective 1 --data data/summary.csv
    python -m plots list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_chapter4(args: argparse.Namespace) -> int:
    """Generate Chapter 4 figures."""
    from plots.data.loader import load_chapter4_data
    from plots.suites.chapter4 import generate_chapter4_figures

    print(f"Loading data from: {args.data}")
    df = load_chapter4_data(args.data)
    print(f"Loaded {len(df)} experiment records")

    figures = args.figures.split(",") if args.figures else None
    formats = args.formats.split(",") if args.formats else ["png", "pdf"]

    results = generate_chapter4_figures(
        df=df,
        output_dir=Path(args.output),
        formats=formats,
        figures=figures,
    )

    print(f"\nGenerated {len(results)} figures to {args.output}")
    return 0


def cmd_objective(args: argparse.Namespace) -> int:
    """Generate objective-specific figures."""
    from plots.data.loader import load_chapter4_data

    print(f"Loading data from: {args.data}")
    df = load_chapter4_data(args.data)
    print(f"Loaded {len(df)} experiment records")

    formats = args.formats.split(",") if args.formats else ["png", "pdf"]
    output_dir = Path(args.output)

    if args.objective == 1:
        from plots.figures.robustness import generate_objective1_figures

        results = generate_objective1_figures(df, output_dir, formats)
    elif args.objective == 2:
        from plots.figures.heterogeneity import generate_objective2_figures

        results = generate_objective2_figures(df, output_dir, formats)
    elif args.objective == 3:
        from plots.figures.cross_dataset import generate_objective3_figures

        results = generate_objective3_figures(df, output_dir, formats)
    else:
        print(f"Objective {args.objective} not yet implemented (valid: 1, 2, 3)")
        return 1

    print(f"\nGenerated {len(results)} figures to {args.output}")
    return 0


def cmd_comprehensive(args: argparse.Namespace) -> int:
    """Generate comprehensive thesis figures (dataset characterization, etc.)."""
    from plots.figures.comprehensive import generate_comprehensive_figures

    formats = args.formats.split(",") if args.formats else ["png", "pdf"]
    output_dir = Path(args.output)

    results = generate_comprehensive_figures(output_dir, formats)

    print(f"\nGenerated {len(results)} comprehensive figures to {args.output}")
    return 0


def cmd_neurips(args: argparse.Namespace) -> int:
    """Generate all 28 NeurIPS-grade thesis plots."""
    import subprocess

    script_path = Path(__file__).parent.parent / "scripts" / "plot_neurips_thesis_plots.py"

    cmd = ["python", str(script_path)]
    if args.data:
        cmd.extend(["--data-csv", args.data])
    if args.output:
        cmd.extend(["--output-dir", args.output])

    print("Running NeurIPS plotting script...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def cmd_list(args: argparse.Namespace) -> int:
    """List available plot types."""
    print(
        """
Available Plot Commands:
========================

chapter4        Generate all Chapter 4 thesis figures (4.1-4.5)
                --data      Path to CSV or experiment directory (required)
                --output    Output directory (default: results/plots/chapter4)
                --figures   Comma-separated figure IDs: 4.1,4.2,4.3,4.4,4.5
                --formats   Output formats: png,pdf,svg (default: png,pdf)

objective       Generate objective-specific figures
                --objective Objective number: 1, 2, 3 (required)
                --data      Path to CSV or experiment directory (required)
                --output    Output directory (default: results/plots/objectives)
                --formats   Output formats (default: png,pdf)

comprehensive   Generate comprehensive figures (dataset characterization,
                CIC vs IIoT comparison, efficiency/overhead analysis)
                --output    Output directory (default: results/plots/comprehensive)
                --formats   Output formats (default: png,pdf)

neurips         Generate all 28 NeurIPS-grade thesis plots following
                publication standards (colorblind-safe palette, 95% CI,
                statistical annotations, vector PDF + high-DPI PNG)
                --data      Path to summary CSV (default: results/summary.csv)
                --output    Output directory (default: results/neurips_plots)

list            Show this help message

Examples:
---------
# Generate all Chapter 4 figures from CSV
python -m plots chapter4 --data results/summary.csv --output results/plots/chapter4

# Generate only figures 4.1 and 4.3
python -m plots chapter4 --data results/summary.csv --figures 4.1,4.3

# Generate Objective 1 (Robustness) figures
python -m plots objective --objective 1 --data results/summary.csv

# Generate Objective 3 (Cross-Dataset) figures
python -m plots objective --objective 3 --data results/summary.csv

# Generate comprehensive figures (no data required)
python -m plots comprehensive --output results/plots/comprehensive

# Generate from experiment directory
python -m plots chapter4 --data runs/experiments/ --output results/plots

# Generate all 28 NeurIPS-grade plots
python -m plots neurips --data results/summary.csv --output results/neurips_plots
"""
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="plots",
        description="Unified plotting CLI for federated learning thesis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    chapter4_parser = subparsers.add_parser("chapter4", help="Generate Chapter 4 figures")
    chapter4_parser.add_argument("--data", "-d", required=True, help="Path to data CSV or experiment directory")
    chapter4_parser.add_argument("--output", "-o", default="results/plots/chapter4", help="Output directory")
    chapter4_parser.add_argument("--figures", "-f", help="Comma-separated figure IDs (default: all)")
    chapter4_parser.add_argument("--formats", help="Output formats (default: png,pdf)")
    chapter4_parser.set_defaults(func=cmd_chapter4)

    objective_parser = subparsers.add_parser("objective", help="Generate objective-specific figures")
    objective_parser.add_argument("--objective", "-obj", type=int, required=True, help="Objective number (1-3)")
    objective_parser.add_argument("--data", "-d", required=True, help="Path to data CSV or experiment directory")
    objective_parser.add_argument("--output", "-o", default="results/plots/objectives", help="Output directory")
    objective_parser.add_argument("--formats", help="Output formats (default: png,pdf)")
    objective_parser.set_defaults(func=cmd_objective)

    comprehensive_parser = subparsers.add_parser("comprehensive", help="Generate comprehensive thesis figures")
    comprehensive_parser.add_argument("--output", "-o", default="results/plots/comprehensive", help="Output directory")
    comprehensive_parser.add_argument("--formats", help="Output formats (default: png,pdf)")
    comprehensive_parser.set_defaults(func=cmd_comprehensive)

    neurips_parser = subparsers.add_parser("neurips", help="Generate all 28 NeurIPS-grade plots")
    neurips_parser.add_argument("--data", "-d", help="Path to summary CSV (default: results/summary.csv)")
    neurips_parser.add_argument("--output", "-o", help="Output directory (default: results/neurips_plots)")
    neurips_parser.set_defaults(func=cmd_neurips)

    list_parser = subparsers.add_parser("list", help="List available plot types")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
