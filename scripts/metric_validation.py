#!/usr/bin/env python3
"""
Metric Validation Framework for Issue #78.

Provides comprehensive validation of federated learning metrics to prevent
scientific validity issues in thesis plots.

This module implements the validation framework described in Issue #78:
- Automated metric validation before plotting
- Detection of impossible metric values
- Statistical sanity checks
- Quality gates for thesis-ready plots
"""

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PlotQualityChecklist:
    """Quality gates before plot generation."""

    # Data quality
    no_nan_in_key_metrics: bool = False
    all_seeds_present: bool = False
    no_duplicate_runs: bool = False

    # Metric sanity
    cosine_in_valid_range: bool = False  # [0.5, 1.0] for FL
    l2_positive_nonzero: bool = False
    f1_below_ceiling: bool = False  # < 0.999

    # Statistical power
    min_seeds: int = 3
    min_experiments_per_group: int = 3

    # Coverage
    multiple_parameter_levels: bool = False

    def validate(self) -> list[str]:
        """Return list of failing checks."""
        failures = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) and not value:
                failures.append(field.name)
        return failures


class MetricValidator:
    """Validates federated learning metrics for data quality issues."""

    def __init__(self, min_expected_cosine: float = 0.5, f1_ceiling_threshold: float = 0.999):
        """Initialize validator with thresholds."""
        self.min_expected_cosine = min_expected_cosine
        self.f1_ceiling_threshold = f1_ceiling_threshold

    def validate_plot_metrics(self, df: pd.DataFrame, dimension: str) -> list[str]:
        """Validate metrics before plotting.

        Args:
            df: DataFrame with experiment results
            dimension: Description of what dimension/experiment these metrics are from

        Returns:
            List of warning messages (empty if no issues detected)
        """
        warnings = []

        # Cosine similarity validation
        if "cos_to_benign_mean" in df.columns:
            cosine_warnings = self._validate_cosine_similarity(df["cos_to_benign_mean"], dimension)
            warnings.extend(cosine_warnings)

        # L2 distance validation
        if "l2_to_benign_mean" in df.columns:
            l2_warnings = self._validate_l2_distance(df["l2_to_benign_mean"], dimension)
            warnings.extend(l2_warnings)

        # F1 ceiling detection
        if "macro_f1" in df.columns:
            f1_warnings = self._validate_f1_ceiling(df["macro_f1"], dimension)
            warnings.extend(f1_warnings)

        # Sample size validation
        if "seed" in df.columns:
            seed_warnings = self._validate_sample_size(df["seed"], dimension)
            warnings.extend(seed_warnings)

        return warnings

    def _validate_cosine_similarity(self, cosine_series: pd.Series, dimension: str) -> list[str]:
        """Validate cosine similarity values."""
        warnings = []
        cosine = cosine_series.dropna()

        if len(cosine) == 0:
            return warnings

        # Check for values outside valid range
        epsilon = 1e-6  # Floating point tolerance
        invalid = cosine[(cosine < -1.0 - epsilon) | (cosine > 1.0 + epsilon)]
        if len(invalid) > 0:
            warnings.append(f"{dimension}: Cosine values outside valid range [-1, 1]: {invalid.tolist()}")

        # Check for suspiciously low values
        suspicious = cosine[cosine < self.min_expected_cosine]
        if len(suspicious) > 0:
            warnings.append(
                f"{dimension}: Suspiciously low cosine {cosine.min():.6f} for FL models "
                f"(expected >{self.min_expected_cosine} for same-architecture models)"
            )

        # Check for variance collapse (all values identical)
        if cosine.std() < 1e-6 and len(cosine) > 1:
            warnings.append(f"{dimension}: No variance in cosine (all values = {cosine.iloc[0]:.6f})")

        return warnings

    def _validate_l2_distance(self, l2_series: pd.Series, dimension: str) -> list[str]:
        """Validate L2 distance values."""
        warnings = []
        l2 = l2_series.dropna()

        if len(l2) == 0:
            return warnings

        # Check for negative values
        negative = l2[l2 < 0.0]
        if len(negative) > 0:
            warnings.append(f"{dimension}: Negative L2 distance values: {negative.tolist()}")

        # Check for exact zeros (suspicious)
        zeros = l2[l2 == 0.0]
        if len(zeros) > 0:
            if len(zeros) == 1:
                warnings.append(f"{dimension}: L2 distance is exactly 0.0 (aggregated model identical to reference)")
            else:
                warnings.append(f"{dimension}: {len(zeros)} L2 values = 0.0 exactly - possible bug")

        return warnings

    def _validate_f1_ceiling(self, f1_series: pd.Series, dimension: str) -> list[str]:
        """Validate F1 scores for ceiling effect."""
        warnings = []
        f1 = f1_series.dropna()

        if len(f1) == 0:
            return warnings

        # Check for ceiling effect
        ceiling_count = (f1 >= self.f1_ceiling_threshold).sum()
        ceiling_ratio = ceiling_count / len(f1)

        if ceiling_ratio > 0.8:
            warnings.append(
                f"{dimension}: Ceiling effect: {ceiling_ratio:.0%} of runs at F1≥{self.f1_ceiling_threshold} "
                f"({ceiling_count}/{len(f1)})"
            )

        return warnings

    def _validate_sample_size(self, seed_series: pd.Series, dimension: str) -> list[str]:
        """Validate sample size for statistical power."""
        warnings = []
        unique_seeds = seed_series.nunique()

        if unique_seeds < 3:
            warnings.append(f"{dimension}: Only {unique_seeds} seeds - CIs may be unreliable")

        return warnings

    def create_quality_checklist(self, df: pd.DataFrame) -> PlotQualityChecklist:
        """Create quality checklist for DataFrame."""
        checklist = PlotQualityChecklist()

        # Data quality checks
        key_metrics = ["cos_to_benign_mean", "l2_to_benign_mean", "macro_f1"]
        checklist.no_nan_in_key_metrics = not df[key_metrics].isnull().any().any()

        if "seed" in df.columns:
            checklist.all_seeds_present = df["seed"].nunique() >= 3

        # Check for duplicate runs (same config + seed)
        if "run_dir" in df.columns:
            checklist.no_duplicate_runs = not df.duplicated(subset=["run_dir"]).any()

        # Metric sanity checks
        if "cos_to_benign_mean" in df.columns:
            cosine = df["cos_to_benign_mean"].dropna()
            checklist.cosine_in_valid_range = len(cosine) > 0 and cosine.min() >= self.min_expected_cosine and cosine.max() <= 1.0

        if "l2_to_benign_mean" in df.columns:
            l2 = df["l2_to_benign_mean"].dropna()
            checklist.l2_positive_nonzero = len(l2) > 0 and l2.min() > 0.0 and (l2 == 0.0).sum() <= 1  # Allow at most one zero

        if "macro_f1" in df.columns:
            f1 = df["macro_f1"].dropna()
            ceiling_ratio = (f1 >= self.f1_ceiling_threshold).sum() / len(f1) if len(f1) > 0 else 0
            checklist.f1_below_ceiling = ceiling_ratio < 0.8

        # Coverage checks
        if "aggregation" in df.columns:
            checklist.multiple_parameter_levels = df["aggregation"].nunique() > 1

        return checklist


def validate_experiment_data(runs_dir: Path, output_dir: Path | None = None) -> dict[str, list[str]]:
    """Validate all experiment data in runs directory.

    Args:
        runs_dir: Directory containing experiment results
        output_dir: Optional directory to save validation report

    Returns:
        Dictionary mapping file paths to validation warnings
    """
    validator = MetricValidator()
    all_warnings = {}

    logger.info(f"Validating experiment data in {runs_dir}")

    # Find all metrics files
    metrics_files = list(runs_dir.rglob("metrics.csv"))
    logger.info(f"Found {len(metrics_files)} metrics files")

    for metrics_file in metrics_files:
        try:
            df = pd.read_csv(metrics_file)
            warnings = validator.validate_plot_metrics(df, str(metrics_file))

            if warnings:
                all_warnings[str(metrics_file)] = warnings
                logger.warning(f"Issues found in {metrics_file}: {len(warnings)} warnings")

        except Exception as e:
            logger.error(f"Error validating {metrics_file}: {e}")
            all_warnings[str(metrics_file)] = [f"Error reading file: {e}"]

    # Generate summary report
    total_files = len(metrics_files)
    files_with_issues = len(all_warnings)

    logger.info(f"Validation complete: {files_with_issues}/{total_files} files have issues")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "metric_validation_report.txt"

        with open(report_file, "w") as f:
            f.write("Metric Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files validated: {total_files}\n")
            f.write(f"Files with issues: {files_with_issues}\n\n")

            for file_path, warnings in all_warnings.items():
                f.write(f"File: {file_path}\n")
                f.write("-" * 30 + "\n")
                for warning in warnings:
                    f.write(f"  WARNING: {warning}\n")
                f.write("\n")

        logger.info(f"Validation report saved to {report_file}")

    return all_warnings


def main():
    """Command-line interface for metric validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate federated learning metrics")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"), help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=Path, help="Directory to save validation report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Run validation
    warnings = validate_experiment_data(args.runs_dir, args.output_dir)

    # Exit with error code if issues found
    if warnings:
        print(f"\n❌ Found issues in {len(warnings)} files")
        return 1
    else:
        print("\n✅ All metrics validated successfully")
        return 0


if __name__ == "__main__":
    exit(main())
