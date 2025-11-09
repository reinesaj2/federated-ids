#!/usr/bin/env python3
"""
CI validation script for federated learning experiment artifacts.
Validates schemas and basic sanity of generated metrics files.
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ArtifactValidationError(Exception):
    """Raised when artifact validation fails."""

    pass


MIN_WEIGHTED_MACRO_F1 = float(os.environ.get("MIN_WEIGHTED_MACRO_F1", "0.70"))
MIN_WEIGHTED_ACCURACY = float(os.environ.get("MIN_WEIGHTED_ACCURACY", "0.70"))
MAX_FINAL_L2_DISTANCE = float(os.environ.get("MAX_FINAL_L2_DISTANCE", "1.5"))
L2_ALPHA_SCALE = float(os.environ.get("L2_ALPHA_SCALE", "3.0"))


def _safe_float(value: str | None) -> float | None:
    if value in ("", None):
        return None
    try:
        assert value is not None  # type narrowing for mypy
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_alpha_from_run_name(run_dir: Path | str) -> Optional[float]:
    """Extract Dirichlet alpha value from nightly FedProx run directory name."""
    name = Path(run_dir).name
    pattern = re.compile(r"nightly_fedprox_alpha(?P<alpha>[0-9.]+)_mu(?P<mu>[0-9.]+)_seed(?P<seed>\d+)")
    match = pattern.match(name)
    if not match:
        return None
    try:
        return float(match.group("alpha"))
    except (TypeError, ValueError):
        return None


def _compute_adaptive_l2_threshold(alpha: Optional[float]) -> float:
    """Scale the L2 threshold based on heterogeneity (Dirichlet alpha)."""
    if alpha is None:
        return MAX_FINAL_L2_DISTANCE

    # Clamp alpha to [0, 1]; lower alpha => stronger heterogeneity
    alpha = max(0.0, min(1.0, alpha))
    return MAX_FINAL_L2_DISTANCE + (1.0 - alpha) * L2_ALPHA_SCALE


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
    final_f1_vals_raw = [_safe_float(row.get("weighted_macro_f1")) for row in rows[-5:] if row.get("weighted_macro_f1")]
    final_f1_vals: list[float] = [v for v in final_f1_vals_raw if v is not None]

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


def validate_run_directory(run_dir: Path, fpr_strict: bool = True, require_plots: bool = True) -> None:
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

    # Validate plot files if required
    if require_plots:
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
    alpha_value = _extract_alpha_from_run_name(run_dir)
    adaptive_limit = _compute_adaptive_l2_threshold(alpha_value)
    if not math.isfinite(l2_value) or l2_value > adaptive_limit:
        alpha_msg = f" (alpha={alpha_value:.2f})" if alpha_value is not None else ""
        raise ArtifactValidationError(f"Final l2_to_benign_mean={l2_value:.3f} exceeds maximum {adaptive_limit:.1f}{alpha_msg}")

    # Validate FPR tolerance if using low_fpr tau mode
    validate_fpr_tolerance(run_dir, target_fpr=0.10, tolerance=0.02, strict=fpr_strict)

    print(f"[PASS] Run directory {run_dir.name} validation passed")


def validate_privacy_experiments(runs_root: Path) -> None:
    """Validate presence and schema of DP-related experiment artifacts.

    Scans for comparative-analysis privacy runs and ensures that when DP is
    enabled, required DP columns exist and contain valid values.

    Args:
        runs_root: Root directory containing runs (e.g., CI working dir or artifacts dir)

    Raises:
        ArtifactValidationError: If required columns are missing or invalid when DP is enabled.
    """
    # Locate directories like comparative-analysis-privacy-*
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("comparative-analysis-privacy")]

    if not candidates:
        return  # Nothing to validate

    required_cols = {"dp_enabled", "dp_epsilon", "dp_delta", "dp_sigma", "dp_clip_norm"}

    for run_dir in candidates:
        client_files = list(run_dir.glob("client_*_metrics.csv"))
        if not client_files:
            # No client files to check in this run
            continue

        for csv_path in client_files:
            rows = _load_csv_rows(csv_path)
            if not rows:
                continue
            headers = set(rows[0].keys())
            missing = required_cols - headers
            if missing:
                raise ArtifactValidationError(f"DP metrics missing required columns {missing} in {csv_path}")

            dp_enabled_rows = False
            epsilon_values: List[float] = []
            noise_values: List[float] = []

            for row in rows:
                dp_enabled_raw = row.get("dp_enabled")
                dp_enabled = str(dp_enabled_raw).lower() in {"1", "true", "yes"}

                if not dp_enabled:
                    continue

                dp_enabled_rows = True
                eps = _safe_float(row.get("dp_epsilon"))
                delt = _safe_float(row.get("dp_delta"))
                sigma = _safe_float(row.get("dp_sigma"))
                clip = _safe_float(row.get("dp_clip_norm"))
                noise = _safe_float(row.get("dp_noise_multiplier"))

                if any(v is None or not math.isfinite(v) for v in [eps, delt, sigma, clip]):
                    raise ArtifactValidationError(
                        f"Invalid DP values in {csv_path}: "
                        f"epsilon={row.get('dp_epsilon')}, delta={row.get('dp_delta')}, "
                        f"sigma={row.get('dp_sigma')}, clip={row.get('dp_clip_norm')}"
                    )

                epsilon_values.append(eps)  # type: ignore[arg-type]
                if noise is not None:
                    noise_values.append(noise)

            if dp_enabled_rows:
                if not epsilon_values:
                    raise ArtifactValidationError(f"No valid epsilon values in {csv_path}")

                unique_noise = {round(val, 6) for val in noise_values} if noise_values else set()
                if len(unique_noise) > 1:
                    raise ArtifactValidationError(f"Inconsistent noise multiplier in {csv_path}: {sorted(unique_noise)}")


def _extract_epsilon_values(privacy_dirs: List[Path]) -> Dict[float, List[float]]:
    """Extract epsilon values grouped by noise multiplier from privacy experiments."""
    epsilon_by_noise: Dict[float, List[float]] = {}

    for exp_dir in privacy_dirs:
        client_metrics_files = list(exp_dir.glob("client_*_metrics.csv"))

        for metrics_file in client_metrics_files:
            rows = _load_csv_rows(metrics_file)
            if not rows:
                continue

            noise_multiplier = _safe_float(rows[0].get("dp_noise_multiplier"))
            if noise_multiplier is None:
                continue

            epsilon_values = []
            for row in rows:
                epsilon = _safe_float(row.get("dp_epsilon"))
                if epsilon is not None:
                    epsilon_values.append(epsilon)

            if epsilon_values:
                epsilon_by_noise.setdefault(noise_multiplier, []).extend(epsilon_values)

    return epsilon_by_noise


def check_privacy_regressions(baseline_dir: Path, current_dir: Path, tolerance: float = 0.1) -> None:
    """Compare epsilon values between baseline and current privacy experiments."""
    baseline_runs = find_run_directories(baseline_dir)
    current_runs = find_run_directories(current_dir)

    baseline_privacy_dirs = [d for d in baseline_runs if "comparative-analysis-privacy" in d.name]
    current_privacy_dirs = [d for d in current_runs if "comparative-analysis-privacy" in d.name]

    if not baseline_privacy_dirs or not current_privacy_dirs:
        return

    baseline_eps = _extract_epsilon_values(baseline_privacy_dirs)
    current_eps = _extract_epsilon_values(current_privacy_dirs)

    for noise_level, baseline_values in baseline_eps.items():
        current_values = current_eps.get(noise_level)
        if not current_values or len(current_values) != len(baseline_values):
            continue

        for base_val, curr_val in zip(baseline_values, current_values):
            if abs(base_val - curr_val) > tolerance:
                raise ArtifactValidationError(
                    f"Privacy regression detected: epsilon changed from {base_val} to {curr_val} "
                    f"for noise level {noise_level} (tolerance {tolerance})"
                )


def validate_privacy_utility_curve_data(curve_csv: Path) -> None:
    """Validate privacy-utility curve CSV structure and values."""
    if not curve_csv.exists():
        raise ArtifactValidationError(f"Privacy-utility curve CSV not found: {curve_csv}")

    rows = _load_csv_rows(curve_csv)
    if not rows:
        raise ArtifactValidationError(f"Empty privacy-utility curve CSV: {curve_csv}")

    required_columns = {"epsilon", "macro_f1_mean", "ci_lower", "ci_upper", "n", "dp_noise_multiplier", "is_baseline"}
    if not required_columns.issubset(rows[0].keys()):
        missing = required_columns - set(rows[0].keys())
        raise ArtifactValidationError(f"Missing required columns in {curve_csv}: {missing}")

    for idx, row in enumerate(rows):
        epsilon = _safe_float(row.get("epsilon"))
        macro_f1 = _safe_float(row.get("macro_f1_mean"))
        ci_lower = _safe_float(row.get("ci_lower"))
        ci_upper = _safe_float(row.get("ci_upper"))
        n = _safe_float(row.get("n"))
        is_baseline = str(row.get("is_baseline", "0")).lower() in {"1", "true"}

        if not is_baseline:
            if epsilon is None or epsilon <= 0 or epsilon > 100:
                raise ArtifactValidationError(f"Invalid epsilon value '{row.get('epsilon')}' (row {idx})")

        if macro_f1 is None or not math.isfinite(macro_f1):
            raise ArtifactValidationError(f"Invalid macro_f1_mean '{row.get('macro_f1_mean')}' (row {idx})")

        if ci_lower is None or ci_upper is None or ci_lower > ci_upper:
            raise ArtifactValidationError(f"Invalid CI bounds in row {idx}: lower={ci_lower}, upper={ci_upper}")

        if n is None or n <= 0:
            raise ArtifactValidationError(f"Invalid sample size 'n' in row {idx}: {row.get('n')}")


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


def validate_no_regression(
    regression_report_path: Path,
    fail_on_regression: bool = True,
) -> None:
    """Validate that no performance regression occurred compared to baseline.

    Args:
        regression_report_path: Path to regression report JSON file
        fail_on_regression: If True, raise error on regression; otherwise warn

    Raises:
        ArtifactValidationError: If regression detected and fail_on_regression is True
    """
    if not regression_report_path.exists():
        return

    try:
        import json

        with open(regression_report_path, "r", encoding="utf-8") as f:
            regression_report = json.load(f)

        if not regression_report.get("any_regression_detected", False):
            print("[PASS] No performance regression detected vs 90-day baseline")
            return

        regression_results = regression_report.get("regression_results", [])
        regressed_metrics = [r for r in regression_results if r.get("regression_detected", False)]

        if not regressed_metrics:
            return

        threshold = regression_report.get("threshold_std", 2.0)
        messages = []

        for result in regressed_metrics:
            metric = result.get("metric", "unknown")
            z_score = result.get("z_score", 0.0)
            current = result.get("current", 0.0)
            baseline_mean = result.get("baseline_mean", 0.0)
            alpha = result.get("alpha", "N/A")
            mu = result.get("mu", "N/A")

            messages.append(
                f"  - {metric} (alpha={alpha}, mu={mu}): "
                f"z-score={z_score:.2f} > {threshold:.1f}, "
                f"current={current:.4f}, baseline_mean={baseline_mean:.4f}"
            )

        msg = "Performance regression detected:\n" + "\n".join(messages)

        if fail_on_regression:
            raise ArtifactValidationError(msg)
        else:
            print(f"[WARNING] {msg}")

    except json.JSONDecodeError as e:
        print(f"[WARNING] Failed to parse regression report: {e}")
    except ArtifactValidationError:
        raise
    except Exception as e:
        print(f"[WARNING] Regression validation error: {e}")


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
        "--min-seeds",
        type=int,
        default=5,
        help="Minimum number of seeds required per alpha/mu configuration (default: 5)",
    )

    parser.add_argument(
        "--regression_report",
        type=str,
        required=False,
        help="Path to regression report JSON for validation",
    )

    parser.add_argument(
        "--regression_strict",
        action="store_true",
        help="Fail CI if regression detected. Default: warnings only.",
    )

    parser.add_argument(
        "--validate_privacy",
        action="store_true",
        help="Validate privacy experiment artifacts (DP parameters, epsilon accounting).",
    )

    args = parser.parse_args()

    fpr_strict_env = os.environ.get("FPR_STRICT", "0")
    fpr_strict = args.fpr_strict or (fpr_strict_env == "1")

    regression_strict_env = os.environ.get("REGRESSION_STRICT", "0")
    regression_strict = args.regression_strict or (regression_strict_env == "1")

    if not fpr_strict:
        print("[INFO] FPR tolerance check: warnings only (not blocking)")

    if not regression_strict:
        print("[INFO] Regression check: warnings only (not blocking)")

    try:
        runs_dir = Path(args.runs_dir)
        run_directories = find_run_directories(runs_dir)
        validate_seed_coverage(run_directories, minimum_seeds=args.min_seeds)

        print(f"Found {len(run_directories)} run directories to validate")

        require_plots_env = os.environ.get("REQUIRE_PLOTS", "1")
        require_plots = require_plots_env == "1"

        for run_dir in run_directories:
            validate_run_directory(run_dir, fpr_strict=fpr_strict, require_plots=require_plots)

        print(f"[PASS] All {len(run_directories)} run directories passed validation")

        if args.validate_privacy:
            print("[INFO] Validating privacy experiments...")
            validate_privacy_experiments(runs_dir)
            print("[PASS] Privacy experiment validation complete")

        if args.regression_report:
            regression_report_path = Path(args.regression_report)
            validate_no_regression(regression_report_path, fail_on_regression=regression_strict)

    except ArtifactValidationError as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error during validation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
