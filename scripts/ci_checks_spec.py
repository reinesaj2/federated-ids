from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import ci_checks


def test_collect_seed_counts_groups_by_alpha_mu(tmp_path: Path) -> None:
    dirs = []
    for seed in range(3):
        run_dir = tmp_path / f"nightly_fedprox_alpha0.1_mu0.01_seed{seed}"
        run_dir.mkdir()
        dirs.append(run_dir)
    extra_dir = tmp_path / "other_run"
    extra_dir.mkdir()

    seed_map = ci_checks._collect_seed_counts(dirs + [extra_dir])
    assert seed_map[("0.1", "0.01")] == {0, 1, 2}


def test_validate_seed_coverage_raises_when_fewer_than_five(tmp_path: Path) -> None:
    runs_dir = tmp_path
    for seed in range(3):
        run_dir = runs_dir / f"nightly_fedprox_alpha0.2_mu0.05_seed{seed}"
        run_dir.mkdir()
        (run_dir / "metrics.csv").write_text("round\n1\n", encoding="utf-8")
        client_csv = run_dir / "client_0_metrics.csv"
        client_csv.write_text("round\n1\n", encoding="utf-8")
        (run_dir / "client_metrics_plot.png").write_bytes(b"0")
        (run_dir / "server_metrics_plot.png").write_bytes(b"0")

    with pytest.raises(ci_checks.ArtifactValidationError):
        ci_checks.validate_seed_coverage(list(runs_dir.iterdir()), minimum_seeds=5)


def test_safe_float_handles_real_like_inputs() -> None:
    assert ci_checks._safe_float(1.5) == pytest.approx(1.5)
    assert ci_checks._safe_float(np.float64(2.0)) == pytest.approx(2.0)
    assert ci_checks._safe_float(" 3.25 ") == pytest.approx(3.25)
    assert ci_checks._safe_float("") is None
    assert ci_checks._safe_float(None) is None
    assert ci_checks._safe_float("not-a-number") is None


def test_validate_plot_files_reports_missing_when_not_strict(tmp_path: Path) -> None:
    run_dir = tmp_path / "plots" / "fedprox-manual"
    run_dir.mkdir(parents=True)
    (run_dir / "client_metrics_plot.png").write_bytes(b"data")

    status = ci_checks.validate_plot_files(run_dir, strict=False)

    assert status["present"] == ["client_metrics_plot.png"]
    assert status["missing"] == ["server_metrics_plot.png"]


def test_validate_plot_files_keeps_strict_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "plots" / "fedprox-manual"
    run_dir.mkdir(parents=True)

    with pytest.raises(ci_checks.ArtifactValidationError):
        ci_checks.validate_plot_files(run_dir)


def test_collect_plot_inventory_targets_depth_two(tmp_path: Path) -> None:
    date_dir = tmp_path / "2025-10-12"
    target_dir = date_dir / "fedprox-manual"
    target_dir.mkdir(parents=True)
    extra_dir = tmp_path / "2025-10-13"
    extra_dir.mkdir()

    targets = ci_checks.collect_plot_inventory_targets(tmp_path, depth=2)

    assert targets == [target_dir]


def test_summarize_plot_inventory_counts_missing(tmp_path: Path) -> None:
    ok_dir = tmp_path / "2025-10-12" / "fedprox-manual"
    ok_dir.mkdir(parents=True)
    (ok_dir / "client_metrics_plot.png").write_bytes(b"ok")
    (ok_dir / "server_metrics_plot.png").write_bytes(b"ok")

    missing_dir = tmp_path / "2025-10-13" / "fedprox-manual"
    missing_dir.mkdir(parents=True)
    (missing_dir / "client_metrics_plot.png").write_bytes(b"ok")

    summary = ci_checks.summarize_plot_inventory([ok_dir, missing_dir])

    assert summary["total_runs"] == 2
    assert summary["complete_runs"] == 1
    assert summary["incomplete_runs"] == 1
    assert summary["missing_runs"] == [
        {"run": str(missing_dir), "missing": ["server_metrics_plot.png"], "present": ["client_metrics_plot.png"]}
    ]
