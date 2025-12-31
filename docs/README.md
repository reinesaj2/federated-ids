# Federated IDS Documentation

This directory contains comprehensive documentation for the Federated Intrusion Detection System research project.

## Table of Contents

### Cross-Dataset Comparison (CIC-IDS2017 vs Edge-IIoTset)

- [CIC_VS_IIOT_COMPREHENSIVE_COMPARISON.md](CIC_VS_IIOT_COMPREHENSIVE_COMPARISON.md) - Complete comparison of both datasets including performance metrics, class distributions, and key findings
- [ATTACK_CLASS_FEATURE_ANALYSIS.md](ATTACK_CLASS_FEATURE_ANALYSIS.md) - Detailed feature importance analysis for top 5 attack classes in each dataset
- [DATASET_STATISTICAL_ANALYSIS.md](DATASET_STATISTICAL_ANALYSIS.md) - Statistical analysis including Shannon entropy, imbalance ratios, and ML implications
- [FEDERATED_LEARNING_PERFORMANCE_ANALYSIS.md](FEDERATED_LEARNING_PERFORMANCE_ANALYSIS.md) - FL performance analysis across aggregation methods, adversary fractions, and heterogeneity levels

### Dataset-Specific Documentation

- [FULL_IIOT_COMPREHENSIVE_ANALYSIS.md](FULL_IIOT_COMPREHENSIVE_ANALYSIS.md) - Deep analysis of Edge-IIoTset full dataset experiments
- [IIOT_RESULTS_SUMMARY.md](IIOT_RESULTS_SUMMARY.md) - Summary of IIOT experimental results
- [edge_iiotset_integration.md](edge_iiotset_integration.md) - Edge-IIoTset dataset integration guide
- [edge_iiotset_full_strategy.md](edge_iiotset_full_strategy.md) - Strategy for full IIOT dataset experiments

### Federated Learning Algorithms

- [FEDPROX_NOVELTY_ANALYSIS.md](FEDPROX_NOVELTY_ANALYSIS.md) - FedProx algorithm analysis and novelty assessment
- [FEDPROX_OPTIMIZER_RESEARCH.md](FEDPROX_OPTIMIZER_RESEARCH.md) - FedProx optimizer implementation research
- [FEDPROX_VALIDATION_RESULTS.md](FEDPROX_VALIDATION_RESULTS.md) - Validation results for FedProx implementation
- [FEDPROX_VS_FEDAVG_FINDINGS.md](FEDPROX_VS_FEDAVG_FINDINGS.md) - Comparative findings between FedProx and FedAvg

### Experimental Design

- [experiment_checklist.md](experiment_checklist.md) - Comprehensive experiment checklist
- [manual_thesis_experiments.md](manual_thesis_experiments.md) - Manual experiment execution guide
- [EXPERIMENT_CONSTRAINTS.md](EXPERIMENT_CONSTRAINTS.md) - Experimental design constraints
- [ALPHA_FEASIBILITY_ANALYSIS.md](ALPHA_FEASIBILITY_ANALYSIS.md) - Alpha parameter feasibility analysis
- [BULYAN_CONSTRAINT_RESOLUTION.md](BULYAN_CONSTRAINT_RESOLUTION.md) - Bulyan aggregation constraints

### Cluster and Infrastructure

- [CLUSTER_ARCHITECTURE.md](CLUSTER_ARCHITECTURE.md) - Cluster architecture documentation
- [CLUSTER_RUNS_ANALYSIS.md](CLUSTER_RUNS_ANALYSIS.md) - Analysis of cluster experiment runs
- [JMU_CS470_CLUSTER_RUNBOOK.md](JMU_CS470_CLUSTER_RUNBOOK.md) - JMU CS470 cluster runbook
- [ISSUE_128_CLUSTER_DEPLOYMENT.md](ISSUE_128_CLUSTER_DEPLOYMENT.md) - Cluster deployment for Issue #128

### Privacy and Security

- [PRIVACY_UTILITY_CURVE.md](PRIVACY_UTILITY_CURVE.md) - Privacy-utility tradeoff analysis
- [DP_MERGE_SUMMARY.md](DP_MERGE_SUMMARY.md) - Differential privacy merge summary
- [threat_model.md](threat_model.md) - Threat model documentation

### Thesis and Publication

- [THESIS_PLOTS_EXPLAINED.md](THESIS_PLOTS_EXPLAINED.md) - Explanation of thesis plots
- [cross_dataset_publishability_analysis.md](cross_dataset_publishability_analysis.md) - Publishability analysis
- [NEURIPS_FINAL_RUNBOOK.md](NEURIPS_FINAL_RUNBOOK.md) - NeurIPS submission runbook
- [NEURIPS_IIOT_FULL_ANALYSIS.md](NEURIPS_IIOT_FULL_ANALYSIS.md) - IIOT analysis for NeurIPS

### Technical References

- [TEMPORAL_VALIDATION_PROTOCOL.md](TEMPORAL_VALIDATION_PROTOCOL.md) - Temporal validation methodology
- [TEMPORAL_VALIDATION_RUNBOOK.md](TEMPORAL_VALIDATION_RUNBOOK.md) - Temporal validation runbook
- [gradient_clipping_theory.md](gradient_clipping_theory.md) - Gradient clipping theory
- [compute_constraints_and_solutions.md](compute_constraints_and_solutions.md) - Compute constraints and solutions

## Quick Reference

### Key Findings Summary

| Metric | CIC-IDS2017 | Edge-IIoTset |
|--------|-------------|--------------|
| Mean Macro-F1 | 0.177 | 0.432 |
| Effective Classes | 2.16 / 15 | 13.07 / 15 |
| Imbalance Ratio | 206,645:1 | 1,621:1 |
| Best Aggregation | Bulyan | Bulyan |

### Running Experiments

```bash
# Run comparative analysis
python scripts/comparative_analysis.py --dimension heterogeneity

# Generate thesis plots
python scripts/generate_thesis_plots.py --dimension all

# Run cross-dataset comparison
python scripts/plot_cic_vs_iiot_comparison.py
```

### Data Locations

- Experimental runs: `runs/`
- Generated plots: `plots/`
- Raw datasets: `data/`

## Document Standards

All documentation follows these conventions:
- No emojis (professional formatting)
- Markdown tables for structured data
- Code blocks for commands and file paths
- Clear section headers with table of contents

## Last Updated

2024-12-31 - Added comprehensive CIC vs IIOT comparison documentation
