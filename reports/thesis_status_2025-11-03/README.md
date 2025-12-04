# Thesis Status Report — 2025-11-03

This package bundles the key artifacts demonstrating completion of the five thesis objectives from `deliverable1/FL.txt`. Each subdirectory lists the evidence you can share directly with your advisor.

## Contents

- `objective1_robust_aggregation/`
  - `aggregation_comparison.pdf` plus `aggregation_comparison_thesis.{png,pdf}`: Macro-F1 comparison of FedAvg, Krum, Median, and Bulyan across thesis seeds (final thesis-styled layout pulled from `plots/thesis/2025-10-24`).
  - `attack_resilience.pdf`, `attack_resilience_thesis.{png,pdf}`, and `attack_resilience_stats.csv`: Robustness curves and supporting statistics as adversarial participation increases (publication cut and underlying data).
  - `metrics_bulyan_alpha0.5_seed42.csv`: Round-level metrics from a representative Bulyan run (seed 42) confirming the production implementation.

- `objective2_heterogeneity/`
  - `heterogeneity_comparison.png`: Final FedAvg vs FedProx macro-F1 results across α heterogeneity settings.
  - `server_metrics_plot.png`: Nightly monitoring output for μ=0.1, α=0.5, showing convergence behaviour under FedProx.
  - `fedprox_heterogeneity_experiment_report.md`: Methodology and findings for the 3×3×3 FedProx study.

- `objective3_personalization/`
  - `personalization_benefit.png`: ΔF1 distribution highlighting personalization gains per client.
  - `table_personalization_cic.tex` and `table_personalization_unsw.tex`: LaTeX tables for CIC-IDS2017 and UNSW-NB15 personalization results.
  - `personalization_summary.json`: Aggregated statistics backing the plotted improvements.

- `objective4_privacy_secure_agg/`
  - `privacy_utility_curve.png`, `privacy_utility_curve_thesis.png`, and `privacy_utility_curve.csv`: Epsilon vs macro-F1 trade-off from the DP sweep, including the polished thesis-ready rendering.
  - `privacy_utility_thesis.png`: Alternate framing (macro-F1 vs noise multiplier) for quick-glance discussions.
  - `PRIVACY_UTILITY_CURVE.md`: Accountant configuration and interpretation notes.
  - `IMPLEMENTATION_SUMMARY_ISSUE_21.md`: Secure aggregation (deterministic masking) implementation summary and test coverage.

- `objective5_empirical_validation/`
  - `table_aggregation_cic.tex`, `table_aggregation_unsw.tex`, and `thesis_tables.tex`: Aggregated thesis-ready tables covering both IDS datasets.
  - `aggregation_comparison_thesis.{png,pdf}` and `attack_resilience_thesis.{png,pdf}`: Publication-grade plots summarising comparative performance and attack resilience across the full IDS datasets.
  - `attack_experiments_bulyan_final.log`: Execution log showing completion of robustness experiments across seeds and adversary rates.
  - `cic_objectives.md`: Checklist demonstrating the CIC-IDS2017 campaign covering all objectives.

## Next Steps

- Zip or share the entire `reports/thesis_status_2025-11-03/` directory when sending to your advisor.
- Pair the artifacts with the latest narrative write-up (if needed) to provide context around methodology.
