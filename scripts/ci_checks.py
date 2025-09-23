#!/usr/bin/env python3
"""
CI validation script for federated learning experiment artifacts.
Validates schemas and basic sanity of generated metrics files.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Set


class ArtifactValidationError(Exception):
    """Raised when artifact validation fails."""

    pass


def validate_csv_schema(csv_path: Path, expected_columns: Set[str]) -> None:
    """Validate that a CSV file exists and has expected columns."""
    if not csv_path.exists():
        raise ArtifactValidationError(f"Required CSV file missing: {csv_path}")

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            actual_columns = set(reader.fieldnames or [])

            if not expected_columns.issubset(actual_columns):
                missing = expected_columns - actual_columns
                raise ArtifactValidationError(
                    f"CSV {csv_path} missing required columns: {missing}. "
                    f"Found: {actual_columns}"
                )

            # Validate at least one data row exists
            try:
                next(reader)
            except StopIteration:
                raise ArtifactValidationError(f"CSV {csv_path} has no data rows")

    except Exception as e:
        if isinstance(e, ArtifactValidationError):
            raise
        raise ArtifactValidationError(f"Failed to read CSV {csv_path}: {e}")


def validate_plot_files(run_dir: Path) -> None:
    """Validate that required plot files exist."""
    required_plots = ["client_metrics_plot.png", "server_metrics_plot.png"]

    for plot_file in required_plots:
        plot_path = run_dir / plot_file
        if not plot_path.exists():
            raise ArtifactValidationError(
                f"Required plot file missing: {plot_path}"
            )


def validate_run_directory(run_dir: Path) -> None:
    """Validate a single FL run directory."""
    print(f"Validating run directory: {run_dir}")

    # Validate server metrics - require basic structure
    server_metrics_path = run_dir / "metrics.csv"
    required_server_columns = {"round"}  # Flexible schema: only require 'round'
    validate_csv_schema(server_metrics_path, required_server_columns)

    # Find and validate client metrics files - required
    client_metrics_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_metrics_files:
        raise ArtifactValidationError(f"No client metrics files found in {run_dir}")

    # Validate client files have at least 'round' column
    required_client_columns = {"round"}
    for client_file in client_metrics_files:
        validate_csv_schema(client_file, required_client_columns)

    # Validate plot files - required
    validate_plot_files(run_dir)

    print(f"✓ Run directory {run_dir.name} validation passed")


def find_run_directories(runs_dir: Path) -> List[Path]:
    """Find all FL run directories in the runs directory."""
    if not runs_dir.exists():
        raise ArtifactValidationError(f"Runs directory does not exist: {runs_dir}")

    run_dirs = [
        d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    if not run_dirs:
        raise ArtifactValidationError(f"No run directories found in {runs_dir}")

    return sorted(run_dirs)


def main() -> None:
    """Main CI validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate FL experiment artifacts for CI"
    )

    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory containing FL run subdirectories",
    )

    args = parser.parse_args()

    try:
        runs_dir = Path(args.runs_dir)
        run_directories = find_run_directories(runs_dir)

        print(f"Found {len(run_directories)} run directories to validate")

        for run_dir in run_directories:
            validate_run_directory(run_dir)

        print(f"✓ All {len(run_directories)} run directories passed validation")

    except ArtifactValidationError as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
