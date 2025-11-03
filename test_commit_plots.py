#!/usr/bin/env python3
"""
Unit tests for commit_plots.py script

Tests plot repository commit functionality with file system operations,
git integration, and error handling scenarios.
"""

import subprocess
import tempfile
import unittest.mock
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.commit_plots import (
    cleanup_old_plots,
    commit_plots,
    copy_plots_to_repository,
    run_git_command,
    setup_git_config,
)


class TestCopyPlotsToRepository:
    """Test copy_plots_to_repository function."""

    def test_copies_png_files_to_date_experiment_structure(self):
        """Test that PNG files are copied to correct directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source directory with test plots
            source_dir = temp_path / "source"
            source_dir.mkdir()

            test_plot_content = b"fake_png_data"
            test_files = [
                "experiment_plot.png",
                "summary_chart.png",
                "metrics_visualization.jpg"
            ]

            for filename in test_files:
                (source_dir / filename).write_bytes(test_plot_content)

            plots_dir = temp_path / "plots"
            experiment_type = "fedprox-test"
            test_date = "2025-01-15"

            # Execute copy operation
            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), experiment_type, test_date
            )

            # Verify directory structure created
            expected_dir = plots_dir / test_date / experiment_type
            assert expected_dir.exists()

            # Verify all files copied
            assert len(copied_files) == len(test_files)

            for filename in test_files:
                target_file = expected_dir / filename
                assert target_file.exists()
                assert target_file.read_bytes() == test_plot_content

                # Verify relative path in returned list
                relative_path = f"{test_date}/{experiment_type}/{filename}"
                assert relative_path in copied_files

    def test_handles_filename_conflicts_with_numeric_suffix(self):
        """Test that duplicate filenames get numeric suffixes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source with duplicate filenames in subdirectories
            source_dir = temp_path / "source"
            (source_dir / "run1").mkdir(parents=True)
            (source_dir / "run2").mkdir(parents=True)

            duplicate_filename = "plot.png"
            content1 = b"plot_data_1"
            content2 = b"plot_data_2"

            (source_dir / "run1" / duplicate_filename).write_bytes(content1)
            (source_dir / "run2" / duplicate_filename).write_bytes(content2)

            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), "test-experiment"
            )

            # Verify both files copied with conflict resolution
            assert len(copied_files) == 2

            target_dir = plots_dir / datetime.now().strftime("%Y-%m-%d") / "test-experiment"
            original_file = target_dir / "plot.png"
            conflict_file = target_dir / "plot_1.png"

            assert original_file.exists()
            assert conflict_file.exists()

            # Files should have different content
            assert original_file.read_bytes() != conflict_file.read_bytes()

    def test_skips_matplotlib_cache_and_temp_files(self):
        """Test that matplotlib cache and temp files are excluded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_dir = temp_path / "source"
            source_dir.mkdir()

            # Create mix of valid plots and excluded files
            valid_plot = source_dir / "valid_plot.png"
            matplotlib_cache = source_dir / ".matplotlib" / "cache.png"
            temp_file = source_dir / "temp_visualization.png"

            matplotlib_cache.parent.mkdir()

            for file_path in [valid_plot, matplotlib_cache, temp_file]:
                file_path.write_bytes(b"test_data")

            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), "filter-test"
            )

            # Only valid plot should be copied
            assert len(copied_files) == 1
            assert "valid_plot.png" in copied_files[0]

    def test_creates_target_directory_if_not_exists(self):
        """Test that target directory structure is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_dir = temp_path / "source"
            source_dir.mkdir()
            (source_dir / "test.png").write_bytes(b"data")

            # Non-existent plots directory
            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), "create-test"
            )

            assert len(copied_files) == 1
            assert plots_dir.exists()

            today = datetime.now().strftime("%Y-%m-%d")
            target_dir = plots_dir / today / "create-test"
            assert target_dir.exists()


class TestCleanupOldPlots:
    """Test cleanup_old_plots function."""

    def test_removes_directories_older_than_retention_period(self):
        """Test that directories older than retention period are removed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            retention_days = 30
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Create old and recent directories
            old_date = (cutoff_date - timedelta(days=5)).strftime("%Y-%m-%d")
            recent_date = (cutoff_date + timedelta(days=5)).strftime("%Y-%m-%d")

            old_dir = plots_dir / old_date / "experiment"
            recent_dir = plots_dir / recent_date / "experiment"

            old_dir.mkdir(parents=True)
            recent_dir.mkdir(parents=True)

            # Add files to verify removal
            (old_dir / "old_plot.png").write_bytes(b"old_data")
            (recent_dir / "recent_plot.png").write_bytes(b"recent_data")

            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days)

            # Verify old directory removed, recent kept
            assert old_date in removed_dirs
            assert not (plots_dir / old_date).exists()
            assert (plots_dir / recent_date).exists()

    def test_skips_non_date_format_directories(self):
        """Test that directories not matching date format are preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            # Create directories with various names
            non_date_dirs = ["index.html", "assets", "invalid-date", "2025-13-45"]

            for dir_name in non_date_dirs:
                (plots_dir / dir_name).mkdir()

            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days=0)

            # No directories should be removed
            assert len(removed_dirs) == 0

            for dir_name in non_date_dirs:
                assert (plots_dir / dir_name).exists()

    def test_handles_empty_plots_directory(self):
        """Test cleanup behavior with empty plots directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days=30)

            assert len(removed_dirs) == 0
            assert plots_dir.exists()


class TestRunGitCommand:
    """Test run_git_command function."""

    @patch('subprocess.run')
    def test_returns_true_on_successful_git_command(self, mock_run):
        """Test that successful git commands return True."""
        mock_run.return_value = MagicMock(returncode=0, stdout="success", stderr="")

        result = run_git_command(["status"], cwd="/test/dir")

        assert result is True
        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/test/dir",
            capture_output=True,
            text=True,
            check=True
        )

    @patch('subprocess.run')
    def test_returns_false_on_git_command_failure(self, mock_run):
        """Test that failed git commands return False and print error."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "git status", output="", stderr="fatal: not a git repository"
        )

        with patch('builtins.print') as mock_print:
            result = run_git_command(["status"])

            assert result is False
            assert mock_print.call_count >= 1

    @patch('subprocess.run')
    def test_handles_git_command_with_no_cwd(self, mock_run):
        """Test git command execution without working directory."""
        mock_run.return_value = MagicMock(returncode=0)

        result = run_git_command(["--version"])

        assert result is True
        mock_run.assert_called_once_with(
            ["git", "--version"],
            cwd=None,
            capture_output=True,
            text=True,
            check=True
        )


class TestSetupGitConfig:
    """Test setup_git_config function."""

    @patch('scripts.commit_plots.run_git_command')
    def test_configures_git_user_settings(self, mock_git):
        """Test that git user configuration is set correctly."""
        mock_git.return_value = True

        result = setup_git_config()

        assert result is True

        expected_calls = [
            unittest.mock.call(["config", "user.name", "GitHub Actions"]),
            unittest.mock.call(["config", "user.email", "actions@github.com"]),
            unittest.mock.call(["config", "core.autocrlf", "false"])
        ]

        mock_git.assert_has_calls(expected_calls)

    @patch('scripts.commit_plots.run_git_command')
    def test_returns_false_if_git_config_fails(self, mock_git):
        """Test that setup fails if any git config command fails."""
        # First command succeeds, second fails
        mock_git.side_effect = [True, False, True]

        result = setup_git_config()

        assert result is False


class TestCommitPlots:
    """Test commit_plots function."""

    @patch('scripts.commit_plots.run_git_command')
    @patch('subprocess.run')
    def test_commits_plots_with_proper_message(self, mock_subprocess, mock_git):
        """Test that plots are committed with descriptive message."""
        # Mock git diff to show changes exist
        mock_subprocess.return_value = MagicMock(returncode=1)  # Changes exist
        mock_git.return_value = True

        plots_dir = "plots"
        experiment_type = "fedprox-test"
        copied_files = [
            "2025-01-15/fedprox-test/plot1.png",
            "2025-01-15/fedprox-test/plot2.png"
        ]

        result = commit_plots(plots_dir, experiment_type, copied_files)

        assert result is True

        # Verify git add and commit called
        mock_git.assert_any_call(["add", plots_dir])

        # Verify commit message contains experiment type and file count
        commit_call = [call for call in mock_git.call_args_list if "commit" in str(call)]
        assert len(commit_call) == 1
        commit_message = commit_call[0][0][0][2]  # Extract commit message
        assert "fedprox-test" in commit_message
        assert "2 plot files" in commit_message

    @patch('scripts.commit_plots.run_git_command')
    @patch('subprocess.run')
    def test_skips_commit_when_no_changes(self, mock_subprocess, mock_git):
        """Test that commit is skipped when no changes exist."""
        # Mock git diff to show no changes
        mock_subprocess.return_value = MagicMock(returncode=0)  # No changes

        result = commit_plots("plots", "test", ["file1.png"])

        assert result is True

        # Verify git add called but not commit
        mock_git.assert_called_once_with(["add", "plots"])

    def test_returns_true_when_no_files_to_commit(self):
        """Test that function succeeds with empty file list."""
        result = commit_plots("plots", "test", [])

        assert result is True


class TestIntegration:
    """Integration tests for commit_plots workflow."""

    def test_end_to_end_plot_copying_and_cleanup(self):
        """Test complete workflow from source plots to repository structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Setup source directory with plots
            source_dir = temp_path / "artifacts"
            source_dir.mkdir()
            (source_dir / "performance.png").write_bytes(b"perf_data")
            (source_dir / "comparison.png").write_bytes(b"comp_data")

            plots_dir = temp_path / "plots"

            # Copy current experiment
            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), "integration-test"
            )

            # Create old directory for cleanup test
            old_date = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
            old_experiment_dir = plots_dir / old_date / "old-experiment"
            old_experiment_dir.mkdir(parents=True)
            (old_experiment_dir / "old_plot.png").write_bytes(b"old_data")

            # Perform cleanup
            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days=30)

            # Verify results
            assert len(copied_files) == 2
            assert len(removed_dirs) == 1
            assert old_date in removed_dirs

            # Verify current experiment preserved
            today = datetime.now().strftime("%Y-%m-%d")
            current_dir = plots_dir / today / "integration-test"
            assert current_dir.exists()
            assert (current_dir / "performance.png").exists()
            assert (current_dir / "comparison.png").exists()

            # Verify old experiment removed
            assert not (plots_dir / old_date).exists()


class TestFailureScenarios:
    """Test error handling and edge cases for commit_plots functions."""

    def test_copy_plots_handles_corrupted_source_files(self):
        """Test that copy operation handles corrupted or unreadable files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_dir = temp_path / "source"
            source_dir.mkdir()

            # Create valid and invalid files
            (source_dir / "valid.png").write_bytes(b"valid_data")

            # Create file with restricted permissions (unreadable)
            restricted_file = source_dir / "restricted.png"
            restricted_file.write_bytes(b"restricted_data")
            restricted_file.chmod(0o000)  # No permissions

            plots_dir = temp_path / "plots"

            try:
                copied_files = copy_plots_to_repository(
                    str(source_dir), str(plots_dir), "corruption-test"
                )

                # Should copy valid file, may skip restricted file
                assert len(copied_files) >= 1
                assert any("valid.png" in f for f in copied_files)
            finally:
                # Restore permissions for cleanup
                try:
                    restricted_file.chmod(0o644)
                except OSError:
                    pass

    def test_cleanup_handles_malformed_date_directories_gracefully(self):
        """Test cleanup with directories that have malformed date names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            # Create directories with malformed dates that truly cannot be parsed
            malformed_dirs = [
                "2025-13-01",  # Invalid month
                "2025-12-32",  # Invalid day
                "not-a-date",
                "invalid-format",
                "2025-1",      # Too short
                ""
            ]

            for dir_name in malformed_dirs:
                if dir_name:  # Skip empty string
                    (plots_dir / dir_name).mkdir()

            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days=0)

            # Should not crash and should not remove malformed directories
            assert len(removed_dirs) == 0
            for dir_name in malformed_dirs:
                if dir_name:
                    assert (plots_dir / dir_name).exists()

    @patch('scripts.commit_plots.shutil.copy2')
    def test_copy_plots_handles_copy_failures(self, mock_copy):
        """Test handling of file copy failures."""
        mock_copy.side_effect = OSError("Disk full")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_dir = temp_path / "source"
            source_dir.mkdir()
            (source_dir / "test.png").write_bytes(b"data")

            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), "copy-failure-test"
            )

            # Should handle failure gracefully and return empty list
            assert len(copied_files) == 0

    def test_run_git_command_with_unicode_characters(self):
        """Test git command handling with unicode characters in output."""
        from scripts.commit_plots import run_git_command

        with patch('subprocess.run') as mock_run:
            # Simulate git output with unicode characters
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="SUCCESS: Unicode commit message",
                stderr=""
            )

            result = run_git_command(["log", "--oneline", "-1"])

            assert result is True

    def test_commit_plots_with_very_large_file_list(self):
        """Test commit message generation with very large file list."""
        from scripts.commit_plots import commit_plots

        # Create large list of files
        large_file_list = [f"2025-09-28/test/plot_{i}.png" for i in range(100)]

        with patch('scripts.commit_plots.run_git_command') as mock_git:
            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=1)  # Changes exist
                mock_git.return_value = True

                result = commit_plots("plots", "large-test", large_file_list)

                assert result is True

                # Verify commit message is truncated appropriately
                commit_call = [call for call in mock_git.call_args_list if "commit" in str(call)]
                assert len(commit_call) == 1
                commit_message = commit_call[0][0][0][2]

                # Message should mention file count and truncation
                assert "100 plot files" in commit_message
                assert "... and" in commit_message  # Indicates truncation


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_cleanup_exactly_at_retention_boundary(self):
        """Test cleanup behavior at exact retention boundary."""
        from datetime import datetime, timedelta

        from scripts.commit_plots import cleanup_old_plots

        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            retention_days = 30
            cutoff_datetime = datetime.now() - timedelta(days=retention_days)

            # Create directory exactly at the boundary
            boundary_date = cutoff_datetime.strftime("%Y-%m-%d")
            boundary_dir = plots_dir / boundary_date
            boundary_dir.mkdir()

            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days)

            # Directory exactly at boundary should be removed
            assert boundary_date in removed_dirs
            assert not boundary_dir.exists()

    def test_copy_plots_with_empty_source_directory(self):
        """Test copy behavior with empty source directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            source_dir = temp_path / "empty_source"
            source_dir.mkdir()  # Empty directory

            plots_dir = temp_path / "plots"

            copied_files = copy_plots_to_repository(
                str(source_dir), str(plots_dir), "empty-test"
            )

            assert len(copied_files) == 0

            # Target directory should still be created
            today = datetime.now().strftime("%Y-%m-%d")
            target_dir = plots_dir / today / "empty-test"
            assert target_dir.exists()

    def test_cleanup_with_large_retention_days(self):
        """Test cleanup with large retention period."""
        from datetime import datetime, timedelta

        from scripts.commit_plots import cleanup_old_plots

        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = Path(temp_dir) / "plots"
            plots_dir.mkdir()

            # Create directory from yesterday
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            recent_dir = plots_dir / yesterday
            recent_dir.mkdir()

            # Use large retention period (5 years) - should preserve recent directory
            removed_dirs = cleanup_old_plots(str(plots_dir), retention_days=1825)

            # Should not remove recent directory
            assert len(removed_dirs) == 0
            assert recent_dir.exists()
