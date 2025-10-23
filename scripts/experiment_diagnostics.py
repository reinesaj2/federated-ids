#!/usr/bin/env python3
"""
Structured failure diagnosis for comparative analysis experiments.

Categorizes experiment failures into distinct root causes for improved
observability and faster debugging.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple


FailureReason = Literal["timeout", "constraint_violation", "error", "no_metrics", "success"]


@dataclass
class ExperimentDiagnostic:
    """Structured diagnostic record for a single experiment run."""

    experiment_name: str
    status: Literal["success", "failed"]
    failure_reason: FailureReason
    error_message: str = ""
    duration_seconds: float = 0.0
    timestamp: str = ""
    exit_code: int = 0
    metrics_rows: int = 0

    def __post_init__(self):
        """Auto-populate timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class DiagnosticsCollector:
    """Collects and summarizes diagnostics for a batch of experiments."""

    def __init__(self, output_dir: Path):
        """Initialize collector with output directory.

        Args:
            output_dir: Directory to save diagnostic logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics: list[ExperimentDiagnostic] = []

    def record(self, diagnostic: ExperimentDiagnostic) -> None:
        """Record a diagnostic event.

        Args:
            diagnostic: Diagnostic record to store
        """
        self.diagnostics.append(diagnostic)

    def summarize(self) -> dict:
        """Generate summary statistics of all diagnostics.

        Returns:
            Dictionary with counts by failure reason and success rate
        """
        failure_counts = {}
        success_count = 0

        for diag in self.diagnostics:
            if diag.status == "success":
                success_count += 1
            else:
                reason = diag.failure_reason
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

        total = len(self.diagnostics)
        return {
            "total_experiments": total,
            "success_count": success_count,
            "success_rate_pct": (100.0 * success_count / total if total > 0 else 0.0),
            "failure_counts": failure_counts,
            "total_failed": total - success_count,
        }

    def print_summary(self) -> None:
        """Print human-readable summary to stdout."""
        summary = self.summarize()

        print("\n" + "=" * 70)
        print("EXPERIMENT BATCH DIAGNOSTICS")
        print("=" * 70)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['success_count']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Success rate: {summary['success_rate_pct']:.1f}%")

        if summary["failure_counts"]:
            print("\nFailure breakdown:")
            for reason, count in sorted(
                summary["failure_counts"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {reason}: {count}")

        print("=" * 70 + "\n")

    def save_diagnostics(self, filename: str = "diagnostics.jsonl") -> Path:
        """Save all diagnostics to JSONL file.

        Args:
            filename: Output filename for diagnostic records

        Returns:
            Path to saved file
        """
        output_file = self.output_dir / filename
        with open(output_file, "w") as f:
            for diag in self.diagnostics:
                f.write(diag.to_json() + "\n")

        return output_file

    def categorize_failure(
        self,
        exit_code: int,
        timeout_expired: bool,
        metrics_exist: bool,
        error_msg: str = "",
    ) -> Tuple[FailureReason, str]:
        """Categorize a failure into root cause.

        Args:
            exit_code: Process exit code
            timeout_expired: Whether process timed out
            metrics_exist: Whether metrics CSV was generated
            error_msg: Error message from process

        Returns:
            (failure_reason, diagnostic_message) tuple
        """
        if timeout_expired:
            return "timeout", "Process exceeded timeout limit"

        if "constraint" in error_msg.lower() or "n >=" in error_msg:
            return "constraint_violation", error_msg

        if not metrics_exist:
            return (
                "no_metrics",
                f"No metrics.csv found (exit_code={exit_code})",
            )

        if exit_code != 0:
            return "error", f"Non-zero exit code: {exit_code}"

        return "success", "Experiment completed successfully"
