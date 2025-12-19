#!/usr/bin/env python3
"""
Integration tests for fedprox-nightly workflow and plot commit functionality.

Tests workflow configuration, job dependencies, and end-to-end integration
between GitHub Actions and plot repository storage.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


class TestWorkflowConfiguration:
    """Test GitHub workflow configuration and syntax."""

    def test_fedprox_nightly_workflow_has_valid_yaml_syntax(self):
        """Test that workflow YAML syntax is valid."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")
        assert workflow_path.exists(), "Workflow file must exist"

        with open(workflow_path) as f:
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

        with open(workflow_path) as f:
            workflow_content = yaml.safe_load(f)

        assert "permissions" in workflow_content
        permissions = workflow_content["permissions"]
        assert permissions["contents"] == "write"
        assert permissions["actions"] == "read"

    def test_commit_plots_job_dependencies_are_correct(self):
        """Test that commit_plots job depends on fedprox_summary."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path) as f:
            workflow_content = yaml.safe_load(f)

        jobs = workflow_content["jobs"]
        assert "commit_plots" in jobs

        commit_plots_job = jobs["commit_plots"]
        assert "needs" in commit_plots_job
        assert commit_plots_job["needs"] == ["fedprox_summary"]
        assert (
            commit_plots_job["if"]
            == "always() && needs.fedprox_summary.result == 'success' && needs.fedprox_summary.outputs.has_summary == 'true'"
        )

    def test_commit_plots_job_uses_correct_script(self):
        """Test that commit_plots job calls the correct Python script."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path) as f:
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

        with open(workflow_path) as f:
            workflow_content = yaml.safe_load(f)

        comparison_upload = next(
            step for step in workflow_content["jobs"]["fedprox_comparison"]["steps"] if "actions/upload-artifact" in step.get("uses", "")
        )
        assert comparison_upload["with"]["retention-days"] == 90

        summary_upload = next(
            step for step in workflow_content["jobs"]["fedprox_summary"]["steps"] if "actions/upload-artifact" in step.get("uses", "")
        )
        assert summary_upload["with"]["retention-days"] >= 90

    def test_manual_artifact_download_guarded_by_event(self):
        """Manual artifact download should only trigger on workflow_dispatch to avoid failures."""
        workflow_path = Path(".github/workflows/fedprox-nightly.yml")

        with open(workflow_path) as f:
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
            workflow_plots = ["fedprox_comparison_summary.json", "fedprox_performance_plots.png", "fedprox_thesis_tables.tex"]

            for plot_file in workflow_plots:
                (fedprox_summary_dir / plot_file).write_bytes(b"workflow_data")

            plots_dir = temp_path / "plots"

            # Execute script as workflow would
            copied_files = copy_plots_to_repository(str(fedprox_summary_dir), str(plots_dir), "fedprox-nightly")

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

            experiment_dirs = ["manual_fedprox_alpha0.05_mu0.0", "manual_fedprox_alpha0.05_mu0.1", "manual_fedprox_alpha0.1_mu0.0"]

            for exp_dir in experiment_dirs:
                exp_path = runs_dir / exp_dir
                exp_path.mkdir(parents=True)
                (exp_path / "comparison.png").write_bytes(b"comparison_data")
                (exp_path / "server_metrics_plot.png").write_bytes(b"metrics_data")

            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(str(manual_dir), str(plots_dir), "fedprox-manual")

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
        # Verify problematic runs directories are excluded
        assert "/runs_buggy_*/" in gitignore_content or "/runs_incomplete_*/" in gitignore_content


class TestFailureScenarios:
    """Test error handling and failure scenarios."""

    def test_script_handles_missing_source_directory(self):
        """Test script behavior when source directory doesn't exist."""
        from scripts.commit_plots import copy_plots_to_repository

        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_source = str(Path(temp_dir) / "does_not_exist")
            plots_dir = str(Path(temp_dir) / "plots")

            # Should handle gracefully and return empty list
            copied_files = copy_plots_to_repository(non_existent_source, plots_dir, "test-experiment")

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
                copy_plots_to_repository(str(source_dir), str(plots_dir), "permission-test")
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
                cleanup_old_plots(str(plots_dir), retention_days=30)
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

            copied_files = copy_plots_to_repository(str(source_dir), str(plots_dir), "long-name-test")

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

        with open(workflow_path) as f:
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
