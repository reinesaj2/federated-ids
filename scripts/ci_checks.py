#!/usr/bin/env python3
"""
CI validation script for federated learning experiment artifacts.
Validates schemas and basic sanity of generated metrics files.
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ArtifactValidationError(Exception):
    """Raised when artifact validation fails."""

    pass


MIN_WEIGHTED_MACRO_F1 = 0.70
MIN_WEIGHTED_ACCURACY = 0.70
MAX_FINAL_L2_DISTANCE = 1.5


def _safe_float(value: str | None) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Load all rows from a CSV file as dictionaries."""
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


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
                raise ArtifactValidationError(f"CSV {csv_path} missing required columns: {missing}. " f"Found: {actual_columns}")

            # Validate at least one data row exists
            try:
                next(reader)
            except StopIteration:
                raise ArtifactValidationError(f"CSV {csv_path} has no data rows")

    except Exception as e:
        if isinstance(e, ArtifactValidationError):
            raise
        raise ArtifactValidationError(f"Failed to read CSV {csv_path}: {e}")


def check_convergence_quality(rows: List[Dict[str, str]]) -> None:
    """Validate convergence quality: accuracy should improve over rounds."""
    if not rows:
        raise ArtifactValidationError("No data rows to validate convergence")
    
    # Check for NaN or Inf in critical columns
    for row in rows:
        for col in ["weighted_macro_f1", "weighted_accuracy"]:
            val = row.get(col, "")
            if val and val.lower() in ("nan", "inf", "-inf"):
                raise ArtifactValidationError(f"Found {val} in {col}: {row}")
    
    # Check final accuracy meets minimum threshold
    final_f1_vals = [_safe_float(row.get("weighted_macro_f1")) for row in rows[-5:] if row.get("weighted_macro_f1")]
    final_f1_vals = [v for v in final_f1_vals if v is not None]
    
    if final_f1_vals and min(final_f1_vals) < MIN_WEIGHTED_MACRO_F1:
        avg_final = sum(final_f1_vals) / len(final_f1_vals)
        raise ArtifactValidationError(f"Final F1 {avg_final:.4f} below minimum {MIN_WEIGHTED_MACRO_F1}")


def check_no_nans_or_infs(rows: List[Dict[str, str]], critical_columns: List[str]) -> None:
    """Ensure no NaN or Inf values in critical metric columns."""
    for i, row in enumerate(rows):
        for col in critical_columns:
            val = row.get(col, "")
            if val and val.lower() in ("nan", "inf", "-inf", ""):
                raise ArtifactValidationError(f"Row {i} column {col} has invalid value: {val}")


def check_seed_consistency(rows: List[Dict[str, str]], expected_seeds: int = 5) -> None:
    """Validate that sufficient seeds are present in results."""
    try:
        seed_col = "seed" if "seed" in rows[0] else None
        if not seed_col:
            return
        
        seeds = set()
        for row in rows:
            if row.get(seed_col):
                try:
                    seeds.add(int(row[seed_col]))
                except (ValueError, KeyError):
                    pass
        
        if len(seeds) < expected_seeds:
            raise ArtifactValidationError(f"Only {len(seeds)} seeds found; expected at least {expected_seeds}")
    except (KeyError, IndexError):
        pass


def validate_plot_files(run_dir: Path) -> None:
    """Validate that required plot files exist."""
    required_plots = ["client_metrics_plot.png", "server_metrics_plot.png"]

    for plot_file in required_plots:
        plot_path = run_dir / plot_file
        if not plot_path.exists():
            raise ArtifactValidationError(f"Required plot file missing: {plot_path}")


def validate_fpr_tolerance(
    run_dir: Path,
    target_fpr: float = 0.10,
    tolerance: float = 0.02,
    strict: bool = True,
) -> None:
    """Validate that benign FPR at tau is within tolerance of target when using low_fpr mode.

    Args:
        run_dir: Directory containing client metrics
        target_fpr: Target FPR value (default 0.10)
        tolerance: Tolerance band around target (default 0.02)
        strict: If False, log warnings instead of raising errors (default True)
    """
    client_metrics_files = list(run_dir.glob("client_*_metrics.csv"))

    if not client_metrics_files:
        return  # Skip if no client files

    for client_file in client_metrics_files:
        try:
            with open(client_file, "r") as f:
                reader = csv.DictReader(f)
                headers = set(reader.fieldnames or [])

                # Check if this run has tau metrics (extended metrics mode)
                if "benign_fpr_bin_tau" not in headers:
                    return  # Skip validation if extended metrics not present

                # Read all rows and check last round FPR
                rows = list(reader)
                if not rows:
                    continue

                last_row = rows[-1]
                benign_fpr_str = last_row.get("benign_fpr_bin_tau", "")

                if benign_fpr_str and benign_fpr_str.strip():
                    try:
                        benign_fpr = float(benign_fpr_str)
                        fpr_diff = abs(benign_fpr - target_fpr)

                        if fpr_diff > tolerance:
                            message = (
                                f"FPR tolerance check failed for {client_file.name}: "
                                f"benign_fpr_bin_tau={benign_fpr:.3f}, target={target_fpr:.3f}, "
                                f"diff={fpr_diff:.3f} > tolerance={tolerance:.3f}"
                            )
                            if strict:
                                raise ArtifactValidationError(message)
                            else:
                                print(f"[WARNING] {message}")
                    except ValueError:
                        pass  # Skip if conversion fails
        except Exception as e:
            if isinstance(e, ArtifactValidationError):
                raise
            # Skip validation errors for files that can't be read
            pass


def validate_run_directory(run_dir: Path, fpr_strict: bool = True) -> None:
    """Validate a single FL run directory.

    Args:
        run_dir: Directory to validate
        fpr_strict: If False, FPR tolerance violations are warnings (default True)
    """
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

    # Compute convergence metrics from client summaries
    macro_weight = 0.0
    macro_sum = 0.0
    macro_samples = 0
    acc_weight = 0.0
    acc_sum = 0.0
    acc_samples = 0

    for client_file in client_metrics_files:
        rows = _load_csv_rows(client_file)
        if not rows:
            continue
        last_row = rows[-1]
        dataset_size = _safe_float(last_row.get("dataset_size")) or 0.0
        weight = dataset_size if dataset_size > 0 else 1.0

        macro_value = _safe_float(last_row.get("macro_f1_after"))
        if macro_value is not None:
            macro_sum += weight * macro_value
            macro_weight += weight
            macro_samples += 1

        acc_value = _safe_float(last_row.get("acc_after"))
        if acc_value is not None:
            acc_sum += weight * acc_value
            acc_weight += weight
            acc_samples += 1

    if macro_samples == 0 or macro_weight == 0:
        raise ArtifactValidationError(f"No macro_f1_after values found in {run_dir}")

    weighted_macro_f1 = macro_sum / macro_weight
    if not math.isfinite(weighted_macro_f1) or weighted_macro_f1 < MIN_WEIGHTED_MACRO_F1:
        raise ArtifactValidationError(
            f"Weighted macro_f1_after={weighted_macro_f1:.3f} below minimum {MIN_WEIGHTED_MACRO_F1:.2f}"
        )

    if acc_samples == 0 or acc_weight == 0:
        raise ArtifactValidationError(f"No acc_after values found in {run_dir}")

    weighted_accuracy = acc_sum / acc_weight
    if not math.isfinite(weighted_accuracy) or weighted_accuracy < MIN_WEIGHTED_ACCURACY:
        raise ArtifactValidationError(
            f"Weighted acc_after={weighted_accuracy:.3f} below minimum {MIN_WEIGHTED_ACCURACY:.2f}"
        )

    # Validate server convergence metrics (L2 distance)
    server_rows = _load_csv_rows(server_metrics_path)
    if not server_rows:
        raise ArtifactValidationError(f"No server metrics rows found in {server_metrics_path}")
    final_server_row = server_rows[-1]
    l2_value = _safe_float(final_server_row.get("l2_to_benign_mean"))
    if l2_value is None:
        raise ArtifactValidationError(
            f"Server metrics missing l2_to_benign_mean in {server_metrics_path}"
        )
    if not math.isfinite(l2_value) or l2_value > MAX_FINAL_L2_DISTANCE:
        raise ArtifactValidationError(
            f"Final l2_to_benign_mean={l2_value:.3f} exceeds maximum {MAX_FINAL_L2_DISTANCE:.1f}"
        )

    # Validate FPR tolerance if using low_fpr tau mode
    validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=fpr_strict)

    print(f"[PASS] Run directory {run_dir.name} validation passed")


def find_run_directories(runs_dir: Path) -> List[Path]:
    """Find all FL run directories in the runs directory."""
    if not runs_dir.exists():
        raise ArtifactValidationError(f"Runs directory does not exist: {runs_dir}")

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not run_dirs:
        raise ArtifactValidationError(f"No run directories found in {runs_dir}")

    return sorted(run_dirs)


RUN_NAME_PATTERN = re.compile(r"nightly_fedprox_alpha(?P<alpha>[0-9.]+)_mu(?P<mu>[0-9.]+)_seed(?P<seed>\d+)")


def _collect_seed_counts(run_directories: List[Path]) -> Dict[Tuple[str, str], Set[int]]:
    seed_map: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    for run_dir in run_directories:
        match = RUN_NAME_PATTERN.match(run_dir.name)
        if not match:
            continue
        key = (match.group("alpha"), match.group("mu"))
        seed_map[key].add(int(match.group("seed")))
    return seed_map


def validate_seed_coverage(run_directories: List[Path], minimum_seeds: int = 5) -> None:
    """Ensure each FedProx nightly configuration has at least ``minimum_seeds`` runs."""
    seed_map = _collect_seed_counts(run_directories)
    if not seed_map:
        return
    for (alpha, mu), seeds in seed_map.items():
        if len(seeds) < minimum_seeds:
            raise ArtifactValidationError(
                f"FedProx nightly runs for alpha={alpha} mu={mu} have only {len(seeds)} seeds; " f"require at least {minimum_seeds}."
            )


def main() -> None:
    """Main CI validation entry point."""
    import os

    parser = argparse.ArgumentParser(description="Validate FL experiment artifacts for CI")

    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory containing FL run subdirectories",
    )

    parser.add_argument(
        "--fpr_strict",
        action="store_true",
        help="Enforce strict FPR tolerance (raises error on violations). Default: warnings only.",
    )

    args = parser.parse_args()

    # Allow environment variable to override FPR strictness
    # FPR_STRICT=1 for strict validation (errors), FPR_STRICT=0 for warnings only
    fpr_strict_env = os.environ.get("FPR_STRICT", "0")
    fpr_strict = args.fpr_strict or (fpr_strict_env == "1")

    if not fpr_strict:
        print("[INFO] FPR tolerance check: warnings only (not blocking)")

    try:
        runs_dir = Path(args.runs_dir)
        run_directories = find_run_directories(runs_dir)
        validate_seed_coverage(run_directories, minimum_seeds=5)

        print(f"Found {len(run_directories)} run directories to validate")

        for run_dir in run_directories:
            validate_run_directory(run_dir, fpr_strict=fpr_strict)

        print(f"[PASS] All {len(run_directories)} run directories passed validation")

    except ArtifactValidationError as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error during validation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
