# Byzantine Attack Type Analysis - Investigation Summary

## Files Generated

This investigation produced the following analysis files:

1. **attack_type_analysis_report.txt**
   - Detailed statistical report with all metrics
   - Organized by aggregator and adversary percentage
   - Includes attack effectiveness rankings

2. **attack_type_analysis_data.json**
   - Machine-readable summary statistics
   - Mean, std, min, max for all metrics
   - Grouped by (aggregator, adv_pct, attack_mode)

3. **attack_type_detailed_data.csv**
   - Per-experiment raw data
   - 300 experiments with individual metrics
   - Suitable for further analysis and visualization

4. **byzantine_attack_analysis_summary.md**
   - Comprehensive narrative summary
   - Attack mechanisms explained
   - Key findings and recommendations
   - Defense effectiveness analysis

5. **attack_comparison_table.md**
   - Side-by-side comparison tables
   - Performance degradation metrics
   - Stealthiness comparisons
   - Practical implications

## Quick Findings

### Attack Types Tested

1. **label_flip**: Rotates labels by +1 mod n_classes
2. **sign_flip_topk**: Flips sign of top 10% largest gradients
3. **targeted_label**: Forces all labels to class 0

### Key Results (30% adversaries, FedAvg)

| Attack         | F1 Score | L2 Distance | Damage Rank                       |
| -------------- | -------- | ----------- | --------------------------------- |
| sign_flip_topk | 0.0565   | 873.40      | 1 (Most damaging)                 |
| label_flip     | 0.3862   | 5.61        | 2                                 |
| targeted_label | 0.4153   | 0.66        | 3 (Least damaging, most stealthy) |

### Defense Effectiveness (vs FedAvg at 30% adversaries)

**Median Aggregator:**

- vs sign_flip_topk: +625% improvement
- vs label_flip: +9.5% improvement
- vs targeted_label: +3.8% improvement

**Conclusion**: Robust aggregators excel against magnitude-based attacks (sign_flip_topk) but struggle against semantic attacks (label_flip, targeted_label).

## Experiment Configuration

- Dataset: Edge-IIoTset-full (15 classes)
- Data heterogeneity: alpha = 0.5 (moderate)
- Adversary percentages: 10%, 20%, 30%
- Aggregators: FedAvg, Median, Krum, Bulyan
- Runs per config: 10 (seeds 42-51)
- Total experiments analyzed: 300

## Source Code

Analysis performed by:

- `attack_type_analysis_v2.py` - Main analysis script
- Source data: `/Users/abrahamreines/Documents/Thesis/cluster-experiments/cluster-runs/`

## Next Steps

1. Investigate why Bulyan data is missing for 20%/30% adversaries
2. Extract benign client FPR data (currently showing 0.0000)
3. Analyze grad_ascent attack (only 4 experiments found)
4. Create visualizations comparing attack effectiveness
5. Develop attack-specific defenses for label_flip and targeted_label

## Usage

To regenerate analysis:

```bash
cd /Users/abrahamreines/Documents/Thesis
python3 attack_type_analysis_v2.py
```

This will update all generated files with latest data from cluster experiments.
