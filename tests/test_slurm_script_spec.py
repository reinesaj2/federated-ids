from __future__ import annotations

from pathlib import Path


def test_hybrid_slurm_script_uses_three_clients() -> None:
    script_path = Path("scripts/slurm/hybrid_cross_source_array.sbatch")
    script_text = script_path.read_text()
    expected_clients = 3
    expected_label = "1 per source"

    assert f"NUM_CLIENTS={expected_clients}" in script_text
    assert expected_label in script_text
