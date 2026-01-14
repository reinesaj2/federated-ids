"""
Deprecation Utilities

Provides warnings for legacy plotting scripts that have been superseded
by the unified plots package.
"""

import warnings
from pathlib import Path


def warn_deprecated_script(script_name: str, replacement_cmd: str) -> None:
    """
    Emit deprecation warning for legacy plotting scripts.

    Args:
        script_name: Name of the deprecated script
        replacement_cmd: The new command to use instead
    """
    warnings.warn(
        f"\n{'=' * 60}\n"
        f"DEPRECATION WARNING: {script_name} is deprecated.\n"
        f"\n"
        f"Use the unified plotting CLI instead:\n"
        f"    {replacement_cmd}\n"
        f"\n"
        f"This script will be removed in a future version.\n"
        f"{'=' * 60}\n",
        DeprecationWarning,
        stacklevel=3,
    )


SCRIPT_REPLACEMENTS: dict[str, str] = {
    "plot_chapter4_comprehensive.py": "python -m plots chapter4 --data <CSV>",
    "generate_chapter4_plots.py": "python -m plots chapter4 --data <CSV>",
    "plot_obj1_attack_resilience.py": "python -m plots objective --objective 1 --data <CSV>",
    "plot_obj2_comprehensive.py": "python -m plots objective --objective 2 --data <CSV>",
    "plot_obj2_comprehensive_v2.py": "python -m plots objective --objective 2 --data <CSV>",
    "plot_obj2_client_drift.py": "python -m plots objective --objective 2 --data <CSV>",
    "plot_objective1_robust_aggregation.py": "python -m plots objective --objective 1 --data <CSV>",
    "plot_objective2_fedprox_heterogeneity.py": "python -m plots objective --objective 2 --data <CSV>",
    "plot_objective3_multi_dataset.py": "python -m plots objective --objective 3 --data <CSV>",
    "plot_obj3_personalization.py": "python -m plots objective --objective 3 --data <CSV>",
    "generate_thesis_plots.py": "python -m plots chapter4 --data <CSV>",
    "generate_comprehensive_thesis_plots.py": "python -m plots comprehensive",
}


def check_and_warn() -> None:
    """
    Check if current script is deprecated and emit warning.

    Call this at the top of deprecated scripts:
        from plots.deprecation import check_and_warn
        check_and_warn()
    """
    import sys

    if not sys.argv:
        return

    script_path = Path(sys.argv[0])
    script_name = script_path.name

    if script_name in SCRIPT_REPLACEMENTS:
        warn_deprecated_script(script_name, SCRIPT_REPLACEMENTS[script_name])


def run_chapter4_wrapper(data_path: str | Path, output_dir: str | Path) -> int:
    """
    Wrapper function to run Chapter 4 plots via the new CLI.

    Args:
        data_path: Path to CSV or experiment directory
        output_dir: Output directory for plots

    Returns:
        Exit code (0 for success)
    """
    from plots.cli import main as cli_main

    check_and_warn()
    return cli_main(["chapter4", "--data", str(data_path), "--output", str(output_dir)])


def run_objective_wrapper(objective: int, data_path: str | Path, output_dir: str | Path) -> int:
    """
    Wrapper function to run objective-specific plots via the new CLI.

    Args:
        objective: Objective number (1, 2, or 3)
        data_path: Path to CSV or experiment directory
        output_dir: Output directory for plots

    Returns:
        Exit code (0 for success)
    """
    from plots.cli import main as cli_main

    check_and_warn()
    return cli_main(["objective", "--objective", str(objective), "--data", str(data_path), "--output", str(output_dir)])


def run_comprehensive_wrapper(output_dir: str | Path) -> int:
    """
    Wrapper function to run comprehensive plots via the new CLI.

    Args:
        output_dir: Output directory for plots

    Returns:
        Exit code (0 for success)
    """
    from plots.cli import main as cli_main

    check_and_warn()
    return cli_main(["comprehensive", "--output", str(output_dir)])
