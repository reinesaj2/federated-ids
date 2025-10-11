import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_thesis_plots import plot_privacy_utility


def _write_config(path: Path, seed: int, dp_enabled: bool, dp_noise: float) -> None:
    config = {
        "aggregation": "fedavg",
        "alpha": 0.5,
        "adversary_fraction": 0.0,
        "dp_enabled": dp_enabled,
        "dp_noise_multiplier": dp_noise,
        "personalization_epochs": 0,
        "num_clients": 2,
        "num_rounds": 3,
        "seed": seed,
    }
    path.write_text(json.dumps(config))


@pytest.mark.parametrize("dp_noise", [0.7])
def test_privacy_utility_curve_outputs(tmp_path, dp_noise):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    dp_run_seed42 = runs_dir / "comp_fedavg_alpha0.5_dp1_seed42"
    dp_run_seed43 = runs_dir / "comp_fedavg_alpha0.5_dp1_seed43"
    baseline_run = runs_dir / "comp_fedavg_alpha0.5_dp0_seed42"

    for run_dir in [dp_run_seed42, dp_run_seed43, baseline_run]:
        run_dir.mkdir()

    _write_config(dp_run_seed42 / "config.json", seed=42, dp_enabled=True, dp_noise=dp_noise)
    _write_config(dp_run_seed43 / "config.json", seed=43, dp_enabled=True, dp_noise=dp_noise)
    _write_config(baseline_run / "config.json", seed=42, dp_enabled=False, dp_noise=0.0)

    metrics_df = pd.DataFrame(
        [
            {"round": 1, "dp_enabled": True, "dp_noise_multiplier": dp_noise, "seed": 42},
            {"round": 3, "dp_enabled": True, "dp_noise_multiplier": dp_noise, "seed": 42},
        ]
    )
    metrics_df.to_csv(dp_run_seed42 / "metrics.csv", index=False)
    metrics_df.assign(seed=43).to_csv(dp_run_seed43 / "metrics.csv", index=False)
    metrics_df.assign(dp_enabled=False, dp_noise_multiplier=0.0).to_csv(
        baseline_run / "metrics.csv", index=False
    )

    for run_dir, epsilon, f1_values in [
        (dp_run_seed42, 1.5, [0.82, 0.80]),
        (dp_run_seed43, 1.5, [0.78, 0.76]),
        (baseline_run, float("nan"), [0.90, 0.88]),
    ]:
        for idx, f1 in enumerate(f1_values):
            client_df = pd.DataFrame(
                [
                    {"round": 1, "macro_f1_after": f1 - 0.05, "dp_epsilon": epsilon / 3},
                    {"round": 3, "macro_f1_after": f1, "dp_epsilon": epsilon},
                ]
            )
            (run_dir / f"client_{idx}_metrics.csv").write_text(client_df.to_csv(index=False))

    df = pd.DataFrame(
        [
            {
                "dp_enabled": True,
                "dp_noise_multiplier": dp_noise,
                "seed": 42,
                "round": 3,
                "run_dir": str(dp_run_seed42),
            },
            {
                "dp_enabled": True,
                "dp_noise_multiplier": dp_noise,
                "seed": 43,
                "round": 3,
                "run_dir": str(dp_run_seed43),
            },
            {
                "dp_enabled": False,
                "dp_noise_multiplier": 0.0,
                "seed": 42,
                "round": 3,
                "run_dir": str(baseline_run),
            },
        ]
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    plot_privacy_utility(df, output_dir, runs_dir=runs_dir)

    curve_path = output_dir / "privacy_utility_curve.png"
    summary_path = output_dir / "privacy_utility_curve.csv"

    assert curve_path.exists()
    assert summary_path.exists()

    summary_df = pd.read_csv(summary_path)

    assert set(summary_df["is_baseline"].unique()) == {0, 1}

    dp_row = summary_df.loc[summary_df["is_baseline"] == 0].iloc[0]
    baseline_row = summary_df.loc[summary_df["is_baseline"] == 1].iloc[0]

    assert pytest.approx(dp_row["epsilon"], rel=1e-3) == 1.5
    assert dp_row["n"] == 2
    assert pytest.approx(dp_row["macro_f1_mean"], rel=1e-3) == 0.79

    assert pd.isna(baseline_row["epsilon"])
    assert baseline_row["n"] == 1
    assert pytest.approx(baseline_row["macro_f1_mean"], rel=1e-3) == 0.89
