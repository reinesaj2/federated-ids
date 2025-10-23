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
        return float(str(value))
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
        raise ArtifactValidationError(f"Weighted macro_f1_after={weighted_macro_f1:.3f} below minimum {MIN_WEIGHTED_MACRO_F1:.2f}")

    if acc_samples == 0 or acc_weight == 0:
        raise ArtifactValidationError(f"No acc_after values found in {run_dir}")

    weighted_accuracy = acc_sum / acc_weight
    if not math.isfinite(weighted_accuracy) or weighted_accuracy < MIN_WEIGHTED_ACCURACY:
        raise ArtifactValidationError(f"Weighted acc_after={weighted_accuracy:.3f} below minimum {MIN_WEIGHTED_ACCURACY:.2f}")

    # Validate server convergence metrics (L2 distance)
    server_rows = _load_csv_rows(server_metrics_path)
    if not server_rows:
        raise ArtifactValidationError(f"No server metrics rows found in {server_metrics_path}")
    final_server_row = server_rows[-1]
    l2_value = _safe_float(final_server_row.get("l2_to_benign_mean"))
    if l2_value is None:
        raise ArtifactValidationError(f"Server metrics missing l2_to_benign_mean in {server_metrics_path}")
    if not math.isfinite(l2_value) or l2_value > MAX_FINAL_L2_DISTANCE:
        raise ArtifactValidationError(f"Final l2_to_benign_mean={l2_value:.3f} exceeds maximum {MAX_FINAL_L2_DISTANCE:.1f}")

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


def validate_privacy_experiments(runs_dir: Path) -> None:
    """Validate that privacy experiments have proper DP parameters and epsilon computation.

    Args:
        runs_dir: Directory containing experiment results

    Raises:
        ArtifactValidationError: If privacy experiments are invalid
    """
    all_dirs = find_run_directories(runs_dir)
    privacy_dirs = [d for d in all_dirs if "comparative-analysis-privacy" in d.name]

    if not privacy_dirs:
        raise ArtifactValidationError("No privacy experiment directories found")

    for exp_dir in privacy_dirs:
        client_metrics_files = list(exp_dir.glob("client_*_metrics.csv"))

        if not client_metrics_files:
            raise ArtifactValidationError(f"No client metrics found in {exp_dir}")

        for metrics_file in client_metrics_files:
            rows = _load_csv_rows(metrics_file)

            if not rows:
                raise ArtifactValidationError(f"Empty metrics file: {metrics_file}")

            # Check for required DP columns
            required_dp_columns = {"dp_enabled", "dp_noise_multiplier", "dp_epsilon", "dp_delta"}
            if not required_dp_columns.issubset(rows[0].keys()):
                missing = required_dp_columns - set(rows[0].keys())
                raise ArtifactValidationError(f"Missing DP parameters in {metrics_file}: {missing}")

            # Validate DP parameters
            dp_enabled_values = set(row.get("dp_enabled", "") for row in rows)
            if "True" not in dp_enabled_values and "true" not in dp_enabled_values:
                raise ArtifactValidationError(f"DP not enabled in {metrics_file}")

            # Validate epsilon values
            epsilon_values = []
            noise_multipliers = []

            for row in rows:
                epsilon_val = _safe_float(row.get("dp_epsilon"))
                noise_val = _safe_float(row.get("dp_noise_multiplier"))

                if epsilon_val is not None:
                    epsilon_values.append(epsilon_val)
                if noise_val is not None:
                    noise_multipliers.append(noise_val)

            if not epsilon_values:
                raise ArtifactValidationError(f"No valid epsilon values in {metrics_file}")

            # Check epsilon values are reasonable
            for epsilon in epsilon_values:
                if epsilon <= 0 or epsilon > 100:
                    raise ArtifactValidationError(f"Invalid epsilon value {epsilon} in {metrics_file}")

            # Check noise multiplier consistency
            if len(set(noise_multipliers)) > 1:
                raise ArtifactValidationError(f"Inconsistent noise multiplier in {metrics_file}")


def check_privacy_regressions(baseline_dir: Path, current_dir: Path, tolerance: float = 0.1) -> None:
    """Check for privacy regressions between baseline and current runs.

    Args:
        baseline_dir: Directory containing baseline experiment results
        current_dir: Directory containing current experiment results
        tolerance: Tolerance for epsilon value changes (default: 0.1)

    Raises:
        ArtifactValidationError: If privacy regression detected
    """
    baseline_all_dirs = find_run_directories(baseline_dir)
    current_all_dirs = find_run_directories(current_dir)
    baseline_privacy_dirs = [d for d in baseline_all_dirs if "comparative-analysis-privacy" in d.name]
    current_privacy_dirs = [d for d in current_all_dirs if "comparative-analysis-privacy" in d.name]

    if not baseline_privacy_dirs or not current_privacy_dirs:
        return  # No privacy experiments to compare

    # Compare epsilon values for similar experiments
    baseline_epsilons = _extract_epsilon_values(baseline_privacy_dirs)
    current_epsilons = _extract_epsilon_values(current_privacy_dirs)

    for noise_level in baseline_epsilons:
        if noise_level not in current_epsilons:
            continue

        baseline_eps = baseline_epsilons[noise_level]
        current_eps = current_epsilons[noise_level]

        if len(baseline_eps) != len(current_eps):
            continue

        # Check for significant changes in epsilon values
        for i, (baseline_eps_val, current_eps_val) in enumerate(zip(baseline_eps, current_eps)):
            if abs(baseline_eps_val - current_eps_val) > tolerance:
                raise ArtifactValidationError(
                    f"Privacy regression detected: epsilon changed from {baseline_eps_val} to {current_eps_val} "
                    f"for noise level {noise_level} (tolerance: {tolerance})"
                )


def validate_privacy_utility_curve_data(curve_csv: Path) -> None:
    """Validate privacy-utility curve CSV data structure and values.

    Args:
        curve_csv: Path to privacy-utility curve CSV file

    Raises:
        ArtifactValidationError: If curve data is invalid
    """
    if not curve_csv.exists():
        raise ArtifactValidationError(f"Privacy-utility curve CSV not found: {curve_csv}")

    rows = _load_csv_rows(curve_csv)

    if not rows:
        raise ArtifactValidationError(f"Empty privacy-utility curve CSV: {curve_csv}")

    # Check required columns
    required_columns = {"epsilon", "macro_f1_mean", "ci_lower", "ci_upper", "n", "dp_noise_multiplier", "is_baseline"}
    if not required_columns.issubset(rows[0].keys()):
        missing = required_columns - set(rows[0].keys())
        raise ArtifactValidationError(f"Missing required columns in {curve_csv}: {missing}")

    # Validate data values
    for i, row in enumerate(rows):
        epsilon = _safe_float(row.get("epsilon"))
        macro_f1 = _safe_float(row.get("macro_f1_mean"))
        ci_lower = _safe_float(row.get("ci_lower"))
        ci_upper = _safe_float(row.get("ci_upper"))
        n = _safe_float(row.get("n"))
        is_baseline = row.get("is_baseline", "0")

        # Skip baseline rows (epsilon can be None)
        if is_baseline in ("1", "True", "true"):
            continue

        # Validate epsilon values
        if epsilon is not None:
            if epsilon <= 0 or epsilon > 50:
                raise ArtifactValidationError(f"Epsilon values out of range: {epsilon} in row {i+1}")

        # Validate macro F1 values
        if macro_f1 is not None:
            if macro_f1 < 0 or macro_f1 > 1:
                raise ArtifactValidationError(f"Invalid macro F1 value: {macro_f1} in row {i+1}")

        # Validate confidence intervals
        if ci_lower is not None and ci_upper is not None and macro_f1 is not None:
            if ci_lower > ci_upper:
                raise ArtifactValidationError(f"Invalid confidence intervals: ci_lower={ci_lower} > ci_upper={ci_upper} in row {i+1}")

            if ci_lower > macro_f1 or ci_upper < macro_f1:
                raise ArtifactValidationError(
                    f"Confidence intervals don't contain mean: {ci_lower} <= {macro_f1} <= {ci_upper} in row {i+1}"
                )

        # Validate sample size
        if n is not None and n < 1:
            raise ArtifactValidationError(f"Invalid sample size: {n} in row {i+1}")


def _extract_epsilon_values(privacy_dirs: List[Path]) -> Dict[float, List[float]]:
    """Extract epsilon values grouped by noise multiplier from privacy experiment directories.
    
    Args:
        privacy_dirs: List of privacy experiment directories
        
    Returns:
        Dictionary mapping noise multiplier to list of epsilon values
    """
    epsilon_by_noise: Dict[float, List[float]] = {}

    for exp_dir in privacy_dirs:
        client_metrics_files = list(exp_dir.glob("client_*_metrics.csv"))

        for metrics_file in client_metrics_files:
            rows = _load_csv_rows(metrics_file)

            if not rows:
                continue

            # Get noise multiplier from first row
            noise_multiplier = _safe_float(rows[0].get("dp_noise_multiplier"))
            if noise_multiplier is None:
                continue

            # Extract epsilon values
            epsilon_values = []
            for row in rows:
                epsilon = _safe_float(row.get("dp_epsilon"))
                if epsilon is not None:
                    epsilon_values.append(epsilon)

            if epsilon_values:
                if noise_multiplier not in epsilon_by_noise:
                    epsilon_by_noise[noise_multiplier] = []
                epsilon_by_noise[noise_multiplier].extend(epsilon_values)

    return epsilon_by_noise


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

    parser.add_argument(
        "--validate_privacy",
        action="store_true",
        help="Validate privacy experiments for DP parameters and epsilon computation.",
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

        # Validate privacy experiments if requested
        if args.validate_privacy:
            print("Validating privacy experiments...")
            validate_privacy_experiments(runs_dir)
            print("[PASS] Privacy experiments validation passed")

    except ArtifactValidationError as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error during validation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
