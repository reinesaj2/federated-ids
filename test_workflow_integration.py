#!/usr/bin/env python3
"""
Integration tests for workflow configurations and plot commit functionality.

Tests workflow configuration, job dependencies, and end-to-end integration
between GitHub Actions and plot repository storage.

Includes tests for:
- fedprox-nightly workflow
- robust-agg-weekly workflow
"""

import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest


class TestWorkflowConfiguration:
    """Test GitHub workflow configuration and syntax."""

    def test_fedprox_nightly_workflow_has_valid_yaml_syntax(self):
        """Test that workflow YAML syntax is valid."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")
        assert workflow_path.exists(), "Workflow file must exist"

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        # Verify basic workflow structure
        assert "name" in workflow_content
        # YAML parses 'on' as boolean True, so check for that
        assert True in workflow_content or "on" in workflow_content
        assert "jobs" in workflow_content
        assert workflow_content["name"] == "FedProx Nightly Comparison"

    def test_workflow_has_required_permissions(self):
        """Test that workflow has contents:write permission for plot commits."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        assert "permissions" in workflow_content
        permissions = workflow_content["permissions"]
        assert permissions["contents"] == "write"
        assert permissions["actions"] == "read"

    def test_commit_plots_job_dependencies_are_correct(self):
        """Test that commit_plots job depends on fedprox_summary."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        jobs = workflow_content["jobs"]
        assert "commit_plots" in jobs

        commit_plots_job = jobs["commit_plots"]
        assert "needs" in commit_plots_job
        assert commit_plots_job["needs"] == ["fedprox_summary"]
        assert commit_plots_job["if"] == "always() && needs.fedprox_summary.result == 'success'"

    def test_commit_plots_job_uses_correct_script(self):
        """Test that commit_plots job calls the correct Python script."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        commit_plots_job = workflow_content["jobs"]["commit_plots"]
        steps = commit_plots_job["steps"]

        # Find step that calls commit_plots.py
        script_step = None
        for step in steps:
            if "run" in step and "commit_plots.py" in step["run"]:
                script_step = step
                break

        assert script_step is not None, "commit_plots.py script step not found"

        # Verify script parameters
        run_command = script_step["run"]
        assert "--source_dir fedprox-summary" in run_command
        assert "--experiment_type fedprox-nightly" in run_command
        assert "--plots_dir plots" in run_command

    def test_artifact_retention_extends_to_ninety_days(self):
        """Nightly artifacts should be retained for 90 days to support trend analysis."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path, "r") as f:
            workflow_content = yaml.safe_load(f)

        comparison_upload = next(
            step for step in workflow_content["jobs"]["fedprox_comparison"]["steps"] if step.get("uses") == "actions/upload-artifact@v4"
        )
        assert comparison_upload["with"]["retention-days"] == 90

        summary_upload = next(
            step for step in workflow_content["jobs"]["fedprox_summary"]["steps"] if step.get("uses") == "actions/upload-artifact@v4"
        )
        assert summary_upload["with"]["retention-days"] >= 90

    def test_manual_artifact_download_guarded_by_event(self):
        """Manual artifact download should only trigger on workflow_dispatch to avoid failures."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path, "r") as f:
            workflow_content = yaml.safe_load(f)

        manual_download_step = next(
            step for step in workflow_content["jobs"]["commit_plots"]["steps"] if step.get("name") == "Download manual comparison results"
        )
        assert manual_download_step.get("if") == "github.event_name == 'workflow_dispatch'"
        assert manual_download_step.get("continue-on-error") is True


class TestWorkflowScriptIntegration:
    """Test integration between workflow and commit_plots script."""

    def test_script_handles_workflow_directory_structure(self):
        """Test that script works with workflow artifact directory structure."""
        from scripts.commit_plots import copy_plots_to_repository

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Simulate workflow artifact structure
            fedprox_summary_dir = temp_path / "fedprox-summary"
            fedprox_summary_dir.mkdir()

            # Create plots matching workflow output
            workflow_plots = [
                "fedprox_comparison_summary.json",
                "fedprox_performance_plots.png",
                "fedprox_thesis_tables.tex"
            ]

            for plot_file in workflow_plots:
                (fedprox_summary_dir / plot_file).write_bytes(b"workflow_data")

            plots_dir = temp_path / "plots"

            # Execute script as workflow would
            copied_files = copy_plots_to_repository(
                str(fedprox_summary_dir),
                str(plots_dir),
                "fedprox-nightly"
            )

            # Verify PNG files copied (JSON/TEX should be skipped)
            assert len(copied_files) == 1
            assert "fedprox_performance_plots.png" in copied_files[0]

    def test_script_handles_manual_comparison_artifacts(self):
        """Test script works with manual comparison artifact structure."""
        from scripts.commit_plots import copy_plots_to_repository

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Simulate manual comparison structure (runs/**)
            manual_dir = temp_path / "manual-comparison"
            runs_dir = manual_dir / "runs"

            experiment_dirs = [
                "manual_fedprox_alpha0.05_mu0.0",
                "manual_fedprox_alpha0.05_mu0.1",
                "manual_fedprox_alpha0.1_mu0.0"
            ]

            for exp_dir in experiment_dirs:
                exp_path = runs_dir / exp_dir
                exp_path.mkdir(parents=True)
                (exp_path / "comparison.png").write_bytes(b"comparison_data")
                (exp_path / "server_metrics_plot.png").write_bytes(b"metrics_data")

            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(
                str(manual_dir),
                str(plots_dir),
                "fedprox-manual"
            )

            # Should copy all PNG files from all experiment directories
            assert len(copied_files) == 6  # 2 plots × 3 experiments
            assert all("fedprox-manual" in path for path in copied_files)

    def test_gitignore_includes_plots_directory(self):
        """Test that .gitignore is configured to include plots directory."""
        gitignore_path = Path(".gitignore")
        assert gitignore_path.exists()

        gitignore_content = gitignore_path.read_text()

        # Verify plots directory is included
        assert "!/plots/" in gitignore_content
        # Verify runs directory is still excluded
        assert "/runs/" in gitignore_content


class TestFailureScenarios:
    """Test error handling and failure scenarios."""

    def test_script_handles_missing_source_directory(self):
        """Test script behavior when source directory doesn't exist."""
        from scripts.commit_plots import copy_plots_to_repository

        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_source = str(Path(temp_dir) / "does_not_exist")
            plots_dir = str(Path(temp_dir) / "plots")

            # Should handle gracefully and return empty list
            copied_files = copy_plots_to_repository(
                non_existent_source, plots_dir, "test-experiment"
            )

            assert copied_files == []

    def test_script_handles_permission_errors(self):
        """Test script behavior when encountering permission errors."""
        from scripts.commit_plots import copy_plots_to_repository

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_dir = temp_path / "source"
            source_dir.mkdir()
            (source_dir / "test.png").write_bytes(b"data")

            # Create read-only plots directory
            plots_dir = temp_path / "readonly_plots"
            plots_dir.mkdir(mode=0o444)

            try:
                # Should handle permission error gracefully
                copied_files = copy_plots_to_repository(
                    str(source_dir), str(plots_dir), "permission-test"
                )
                # May succeed or fail depending on system, but shouldn't crash
            except PermissionError:
                # Expected on some systems
                pass
            finally:
                # Restore permissions for cleanup
                plots_dir.chmod(0o755)

    @patch('scripts.commit_plots.run_git_command')
    def test_commit_plots_handles_git_failures(self, mock_git):
        """Test commit_plots handles git command failures gracefully."""
        from scripts.commit_plots import commit_plots

        # Simulate git add success but commit failure
        mock_git.side_effect = [True, False]  # add succeeds, commit fails

        with patch('subprocess.run') as mock_subprocess:
            # Simulate changes exist
            mock_subprocess.return_value.returncode = 1

            result = commit_plots("plots", "test", ["file1.png"])

            assert result is False

    def test_cleanup_handles_permission_denied_on_removal(self):
        """Test cleanup handles permission errors when removing directories."""
        from scripts.commit_plots import cleanup_old_plots

        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            # Create old directory with read-only permissions
            old_date = "2024-01-01"  # Very old date
            old_dir = plots_dir / old_date
            old_dir.mkdir()
            old_dir.chmod(0o444)  # Read-only

            try:
                # Should handle permission error gracefully
                removed_dirs = cleanup_old_plots(str(plots_dir), retention_days=30)
                # May or may not remove depending on system
            except PermissionError:
                # Expected behavior - should not crash
                pass
            finally:
                # Restore permissions for cleanup (only if directory still exists)
                if old_dir.exists():
                    old_dir.chmod(0o755)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cleanup_with_zero_retention_days(self):
        """Test cleanup behavior with zero retention period."""
        from scripts.commit_plots import cleanup_old_plots

        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            # Create today's directory
            today = "2025-09-28"
            today_dir = plots_dir / today
            today_dir.mkdir()

            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days=0)

            # Today's directory should be removed with zero retention
            assert today in removed_dirs
            assert not today_dir.exists()

    def test_copy_plots_with_very_long_filename(self):
        """Test handling of very long filenames."""
        from scripts.commit_plots import copy_plots_to_repository

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_dir = temp_path / "source"
            source_dir.mkdir()

            # Create file with long name (but within filesystem limits)
            long_name = "a" * 200 + ".png"
            (source_dir / long_name).write_bytes(b"data")

            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), "long-name-test"
            )

            assert len(copied_files) == 1
            assert long_name in copied_files[0]

    def test_script_main_function_with_invalid_arguments(self):
        """Test main function argument validation."""
        from scripts.commit_plots import main

        # Test with missing required arguments
        with patch('sys.argv', ['commit_plots.py']):
            with pytest.raises(SystemExit):
                main()

    def test_workflow_handles_missing_artifacts(self):
        """Test that workflow continues gracefully when artifacts are missing."""
        # This would be tested in actual CI, but we can verify the continue-on-error
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        commit_plots_job = workflow_content["jobs"]["commit_plots"]
        steps = commit_plots_job["steps"]

        # Find download manual comparison step
        manual_step = None
        for step in steps:
            if "Download manual comparison results" in step.get("name", ""):
                manual_step = step
                break

        assert manual_step is not None
        assert manual_step.get("continue-on-error") is True


class TestRobustAggWeeklyWorkflow:
    """Test robust aggregation weekly workflow configuration."""

    def test_robust_agg_workflow_has_valid_yaml_syntax(self):
        """Test that robust-agg-weekly workflow YAML syntax is valid."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")
        assert workflow_path.exists(), "Robust-agg-weekly workflow file must exist"

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        # Verify basic workflow structure
        assert "name" in workflow_content
        assert True in workflow_content or "on" in workflow_content
        assert "jobs" in workflow_content
        assert workflow_content["name"] == "Robust Aggregation Weekly"

    def test_workflow_has_saturday_schedule(self):
        """Test that workflow runs on Saturday (weekly)."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        # Get schedule trigger
        schedule = workflow_content.get(True, {}).get("schedule") or workflow_content.get("on", {}).get("schedule")
        assert schedule is not None, "Workflow must have schedule trigger"

        # Verify cron expression runs on Saturday (day 6)
        cron_expr = schedule[0]["cron"]
        # Format: "0 4 * * 6" = 4 AM UTC every Saturday
        assert "* * 6" in cron_expr, "Should run on Saturday (day 6)"

    def test_workflow_matrix_has_correct_dimensions(self):
        """Test that workflow matrix covers all algorithm and adversary combinations."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        jobs = workflow_content["jobs"]
        assert "robust_agg_experiments" in jobs

        matrix = jobs["robust_agg_experiments"]["strategy"]["matrix"]

        # Verify algorithms
        assert set(matrix["aggregation"]) == {"fedavg", "krum", "bulyan", "median"}

        # Verify adversary fractions
        assert set(matrix["adv_fraction"]) == {0.0, 0.2, 0.4}

    def test_workflow_calls_adversarial_validation(self):
        """Test that workflow calls ci_checks.py with adversarial validation."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        experiment_job = workflow_content["jobs"]["robust_agg_experiments"]
        steps = experiment_job["steps"]

        # Find validation step
        validation_step = None
        for step in steps:
            if "run" in step and "ci_checks.py" in step.get("run", ""):
                validation_step = step
                break

        assert validation_step is not None, "ci_checks.py validation step not found"

        # Verify adversarial validation is enabled
        run_command = validation_step["run"]
        assert "--adversarial_validation" in run_command

    def test_workflow_generates_summary_statistics(self):
        """Test that workflow calls summarize_robust_agg.py."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        experiment_job = workflow_content["jobs"]["robust_agg_experiments"]
        steps = experiment_job["steps"]

        # Find summary generation step
        summary_step = None
        for step in steps:
            if "run" in step and "summarize_robust_agg.py" in step.get("run", ""):
                summary_step = step
                break

        assert summary_step is not None, "summarize_robust_agg.py step not found"

    def test_workflow_generates_comparison_plots(self):
        """Test that consolidated analysis job generates comparison plots."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        consolidated_job = workflow_content["jobs"]["consolidated_analysis"]
        steps = consolidated_job["steps"]

        # Find plot generation step
        plot_step = None
        for step in steps:
            if "run" in step and "plot_robust_agg_comparison.py" in step.get("run", ""):
                plot_step = step
                break

        assert plot_step is not None, "plot_robust_agg_comparison.py step not found"

    def test_artifacts_have_90_day_retention(self):
        """Test that artifacts are retained for 90 days."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        # Check experiment artifacts
        experiment_upload = next(
            step for step in workflow_content["jobs"]["robust_agg_experiments"]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        )
        assert experiment_upload["with"]["retention-days"] == 90

        # Check consolidated artifacts
        consolidated_upload = next(
            step for step in workflow_content["jobs"]["consolidated_analysis"]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        )
        assert consolidated_upload["with"]["retention-days"] == 90

    def test_analysis_results_committed_to_repo(self):
        """Test that analysis results are committed to analysis/robust_agg_weekly/."""
        workflow_path = Path(".github/workflows/robust-agg-weekly.yml")

        with open(workflow_path, 'r') as f:
            workflow_content = yaml.safe_load(f)

        commit_job = workflow_content["jobs"]["commit_analysis_results"]
        steps = commit_job["steps"]

        # Find commit step
        commit_step = None
        for step in steps:
            if "Commit analysis results" in step.get("name", ""):
                commit_step = step
                break

        assert commit_step is not None
        assert "analysis/robust_agg_weekly/" in commit_step["run"]


class TestRobustAggScripts:
    """Test robust aggregation scripts work correctly."""

    def test_summarize_robust_agg_script_exists(self):
        """Test that summarize_robust_agg.py script exists and is executable."""
        script_path = Path("scripts/summarize_robust_agg.py")
        assert script_path.exists()
        assert script_path.stat().st_mode & 0o111  # Check executable bit

    def test_plot_robust_agg_comparison_script_exists(self):
        """Test that plot_robust_agg_comparison.py script exists."""
        script_path = Path("scripts/plot_robust_agg_comparison.py")
        assert script_path.exists()

    def test_ci_checks_has_adversarial_validation_argument(self):
        """Test that ci_checks.py accepts --adversarial_validation argument."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/ci_checks.py", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "--adversarial_validation" in result.stdout

    def test_analysis_directory_structure_exists(self):
        """Test that analysis/robust_agg_weekly/ directory exists with README."""
        analysis_dir = Path("analysis/robust_agg_weekly")
        assert analysis_dir.exists()
        assert analysis_dir.is_dir()

        readme_path = analysis_dir / "README.md"
        assert readme_path.exists()
