# FedProx Heterogeneity Matrix Documentation

## Overview

This directory contains documentation for the FedProx heterogeneity matrix experiments implemented in Issue #86. The experiments evaluate FedProx algorithm effectiveness across varying levels of data heterogeneity in federated learning.

## Files

- `fedprox_heterogeneity_experiment_report.md`: Complete experimental methodology and setup documentation
- `README.md`: This file

## Important Notes

**DO NOT manually generate CSV files or statistical data.** All experimental data should be generated using the automated pipeline:

```bash
# Run the FedProx heterogeneity experiments
python scripts/comparative_analysis.py --dimension heterogeneity_fedprox

# Generate analysis plots and statistics
python scripts/generate_thesis_plots.py --dimension heterogeneity_fedprox
```

## CI Integration

The CI workflow has been updated to include the `heterogeneity_fedprox` dimension. When merged to main, the nightly comparative analysis will automatically run the FedProx experiments.

## Experimental Matrix

- **Heterogeneity levels**: 3 (alpha = 0.1, 0.5, 1.0)
- **FedProx strengths**: 3 (mu = 0.01, 0.1, 1.0)
- **Random seeds**: 3 (42, 43, 44)
- **Total experiments**: 27

## Data Integrity

All experimental data is stored in the `runs/` directory with the following structure:
- `runs/comp_fedavg_alpha{alpha}_adv0_dp0_pers0_mu{mu}_seed{seed}/`
  - `config.json`: Experiment parameters
  - `metrics.csv`: Server-side aggregation metrics
  - `client_*_metrics.csv`: Per-client training metrics
  - `server.log`: Server execution logs
  - `client_*.log`: Per-client execution logs
