# Robust Aggregation Weekly Analysis

This directory contains automated weekly experiments comparing robust aggregation algorithms under Byzantine attacks.

## Experiment Configuration

- **Schedule**: Every Saturday at 4 AM UTC
- **Algorithms**: FedAvg, Krum, Bulyan, Median
- **Adversary Rates**: 0.0 (no attack), 0.2 (20% adversaries), 0.4 (40% adversaries)
- **Clients**: 10 per experiment
- **Rounds**: 10 training rounds
- **Seeds**: 3 seeds for statistical significance
- **Dataset**: Synthetic data with Dirichlet non-IID partitioning (alpha=0.1)

## Directory Structure

```
analysis/robust_agg_weekly/
├── README.md                    # This file
├── YYYY-MM-DD/                  # Results for each weekly run
│   ├── f1_vs_adversary.png     # F1 score comparison
│   ├── l2_vs_adversary.png     # L2 distance comparison
│   ├── algorithm_comparison_heatmap.png
│   ├── robustness_degradation.png
│   ├── algorithm_bar_comparison.png
│   └── README.md               # Run-specific summary
└── consolidated/                # Latest consolidated analysis
    └── (same plots as above)
```

## Metrics

### Performance Metrics

- **Macro F1 Score**: Primary metric for classification performance
  - Threshold: ≥0.70 with no adversaries
  - Robust algorithms should maintain >0.60 with 20% adversaries
  - Robust algorithms should maintain >0.50 with 40% adversaries

- **Accuracy**: Overall classification accuracy
  - Complementary metric to F1

### Robustness Metrics

- **L2 Distance to Benign Mean**: Measures model divergence
  - Lower is better
  - FedAvg: Should be <1.5 with no adversaries
  - Robust aggregators: Should be <3.0 with 20% adversaries, <5.0 with 40% adversaries

- **Cosine Similarity**: Agreement between aggregated and benign models
  - Higher is better (range: -1 to 1)
  - Values >0.9 indicate strong agreement

## Algorithm Behavior

### FedAvg (Baseline)
- **Expected**: High performance with no adversaries
- **Under Attack**: Significant degradation expected
- Performance drops ~20-40% with Byzantine clients

### Krum
- **Expected**: Moderate robustness
- **Under Attack**: Better than FedAvg but may struggle with >30% adversaries
- Selects single client update closest to majority

### Bulyan
- **Expected**: High robustness
- **Under Attack**: Should maintain performance up to 40% adversaries
- Multi-Krum with additional aggregation step

### Median
- **Expected**: Good robustness
- **Under Attack**: Robust to Byzantine attacks through coordinate-wise median
- Simple but effective defense

## Interpreting Results

### Good Results
- Robust algorithms (Krum, Bulyan, Median) maintain F1 >0.60 with adversaries
- FedAvg shows clear degradation under attack (validates adversarial setup)
- L2 distance increases with adversary rate (expected behavior)

### Concerning Results
- Robust algorithms perform worse than FedAvg with no adversaries (overly conservative)
- No performance difference between 0% and 40% adversaries (Byzantine attack ineffective)
- All algorithms collapse to random performance (F1 <0.5)

## Validation

Experiments are validated using `scripts/ci_checks.py` with adversarial validation:

```bash
python scripts/ci_checks.py \
  --adversarial_validation \
  --aggregation <algorithm> \
  --adv_fraction <fraction>
```

Validation checks:
1. Minimum F1 thresholds based on adversary rate
2. Maximum L2 distance based on algorithm and adversary rate
3. Presence of both benign and adversarial clients
4. CSV schema compliance
5. Plot file generation

## Manual Execution

To run experiments manually:

```bash
# Trigger workflow via GitHub UI or API
gh workflow run robust-agg-weekly.yml

# Or run locally with specific configuration
python scripts/comparative_analysis.py \
  --dimension aggregation \
  --adv_fraction 0.2 \
  --aggregation bulyan
```

## Citation

If using these results, cite:

```bibtex
@misc{federated-ids-robust-agg,
  title={Robust Aggregation Weekly Experiments},
  author={Federated IDS Research},
  year={2025},
  url={https://github.com/reinesaj2/federated-ids}
}
```

## Maintenance

- Results are automatically committed to this directory by GitHub Actions
- Artifacts are retained for 90 days in GitHub Actions
- Old experiment directories can be archived after 6 months
