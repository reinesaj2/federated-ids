# IIoT Comprehensive Plotting Framework

## Overview

The IIoT Comprehensive Plotting Framework generates publication-quality multi-panel visualizations for thesis objectives in federated learning for Industrial IoT intrusion detection systems. Each objective produces a 6-panel figure (3x3 GridSpec layout) with complementary views of experimental results.

**Key Features:**
- 4 thesis objectives with 6 visualization panels each
- Consistent styling, color schemes, and confidence intervals
- Automatic data loading from experiment run directories
- Statistical rigor: 95% CI from n=5 seeds per configuration
- Publication-ready output: 300 DPI PNG files

## Architecture

### Data Pipeline

```
runs/dsedge-iiotset-*/
├── metrics.csv              # Server-side: L2 distance, dispersion, timing
└── client_*_metrics.csv     # Client-side: F1 scores, personalization
                ↓
scripts/load_iiot_data.py    # Merge server + client metrics by round
                ↓
        Unified DataFrame     # Config + per-round server + averaged client metrics
                ↓
scripts/generate_comprehensive_thesis_plots.py  # 4 objective plotting functions
                ↓
thesis_plots_iiot/
├── obj1_robustness_comprehensive.png
├── obj2_heterogeneity_comprehensive.png
├── obj3_personalization_comprehensive.png
└── obj4_system_overhead_comprehensive.png
```

### Data Schema

The unified DataFrame from `load_iiot_data.py` contains:

**Configuration Columns:**
- `aggregation`: fedavg | krum | bulyan | median
- `alpha`: Dirichlet parameter (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf)
- `adv_pct`: Adversary percentage (0, 10, 30)
- `pers_epochs`: Personalization epochs (0, 3, 5)
- `seed`: Random seed (42, 43, 44, 45, 46)
- `run_id`: Full experiment directory name

**Per-Round Metrics:**
- `round`: Communication round number
- `l2_to_benign_mean`: L2 distance from benign consensus (robustness)
- `l2_dispersion_mean`: L2 dispersion among clients (drift)
- `t_aggregate_ms`: Aggregation time in milliseconds (overhead)
- `macro_f1_global`: Averaged client F1 on global model (performance)
- `macro_f1_personalized`: Averaged client F1 after personalization (if pers_epochs > 0)
- `personalization_gain`: Computed as `macro_f1_personalized - macro_f1_global`

## Objective 1: Robust Aggregation Strategies for Byzantine-Resilient IIoT IDS

**Research Question:** How do robust aggregation methods (Krum, Bulyan, Median) compare to FedAvg under Byzantine gradient ascent attacks?

**6 Panels:**

1. **Robustness: L2 Distance vs Attack Level** (top-left)
   - X: Adversary percentage (0%, 10%, 30%)
   - Y: L2 distance to benign mean (log scale)
   - Lines: 4 aggregators with 95% CI bands
   - Key Insight: Measures how far poisoned consensus drifts from benign

2. **Utility: F1 Degradation Under Attack** (top-center)
   - X: Adversary percentage
   - Y: Final F1 score (round 20)
   - Lines: 4 aggregators with 95% CI bands
   - Key Insight: FedAvg catastrophic failure at 30% (F1=0.002), robust methods stable

3. **Convergence Under Attack** (top-right)
   - X: Communication round
   - Y: Macro F1 score
   - Lines: 4 aggregators (alpha=0.5, adv_pct=30%)
   - Key Insight: FedAvg flat at 0.0, Bulyan/Krum/Median converge to 0.55-0.67

4. **Attack Resilience Matrix** (middle-left, heatmap)
   - Rows: Aggregators
   - Columns: Attack levels (0%, 10%, 30%)
   - Values: Final F1 scores (color-coded)
   - Key Insight: Visual summary of robustness-utility tradeoff

5. **Robustness-Utility Tradeoff** (middle-center, scatter)
   - X: L2 distance to benign mean (robustness, lower=better)
   - Y: Final F1 score (utility, higher=better)
   - Points: Aggregator x attack level x seed combinations
   - Key Insight: Pareto frontier analysis

6. **Performance Comparison Across Attack Intensities** (bottom, bar chart)
   - X: Aggregators
   - Y: Final F1 score
   - Grouped bars: 0%, 10%, 30% attack with 95% CI error bars
   - Key Insight: Direct statistical comparison

**Data Filters:**
- Alpha: 0.5 (moderate heterogeneity)
- Attack levels: 0, 10, 30
- Seeds: All (n=5)

**Key Functions:**
- `plot_objective1_robustness(df, output_dir)`
- `plot_robustness_l2_vs_attack(df, ax)`
- `plot_utility_under_attack(df, ax)`
- `plot_convergence_under_attack(df, ax)`
- `plot_attack_resilience_matrix(df, ax)`
- `plot_robustness_utility_tradeoff(df, ax)`
- `plot_aggregator_performance_bars(df, ax)`

## Objective 2: Handling Data Heterogeneity in IIoT Federated IDS

**Research Question:** How do aggregation methods perform across the IID to non-IID spectrum (Dirichlet alpha)?

**6 Panels:**

1. **Performance vs Data Heterogeneity (Benign)** (top-left, wide)
   - X: Dirichlet alpha (log scale: 0.02 → inf)
   - Y: Final F1 score
   - Lines: 4 aggregators with 95% CI bands
   - Key Insight: Performance degrades as alpha decreases (0.72 IID → 0.62 non-IID)

2. **Convergence Speed** (top-right)
   - X: Dirichlet alpha (log scale)
   - Y: Rounds to 90% final F1
   - Lines: 4 aggregators
   - Key Insight: Non-intuitive result - fastest convergence at alpha=0.1

3. **Client Model Drift** (middle-left)
   - X: Communication round
   - Y: L2 dispersion among clients
   - Lines: Alpha values (0.02, 0.1, 0.5, 1.0)
   - Key Insight: Drift increases with heterogeneity, converges over time

4. **Heterogeneity Impact Matrix** (middle-center, heatmap)
   - Rows: Aggregators
   - Columns: Alpha values
   - Values: Final F1 scores (color-coded)
   - Key Insight: Consistent 1-2% Bulyan advantage across all alpha

5. **IID vs Non-IID Performance** (middle-right, bar chart)
   - X: Aggregators
   - Y: Final F1 score
   - Grouped bars: IID (alpha=inf) vs Non-IID (alpha=0.1)
   - Key Insight: Quantifies heterogeneity penalty per aggregator

6. **FedAvg Convergence Under Different Heterogeneity Levels** (bottom, full width)
   - X: Communication round
   - Y: Macro F1 score
   - Lines: Alpha values (0.02, 0.1, 0.5, 1.0, inf) with 95% CI bands
   - Key Insight: Detailed convergence trajectories for baseline method

**Data Filters:**
- Attack level: 0 (benign)
- All alpha values
- Seeds: All (n=5)

**Key Functions:**
- `plot_objective2_heterogeneity(df, output_dir)`
- `plot_f1_vs_alpha(df, ax)`
- `plot_convergence_speed(df, ax)`
- `plot_drift_by_alpha(df, ax)`
- `plot_alpha_heatmap(df, ax)`
- `plot_iid_vs_noniid(df, ax)`
- `plot_alpha_convergence_trajectories(df, ax)`

## Objective 3: Personalization for Client-Specific IIoT Attack Detection

**Research Question:** Does local fine-tuning (personalization) improve client-specific F1 scores? What are the risks under attack?

**6 Panels:**

1. **Personalization Benefit vs Attack Intensity** (top-left)
   - X: Adversary percentage
   - Y: Mean personalization gain (Delta F1)
   - Line: FedAvg only (others lack personalization data)
   - Horizontal dashed line: Break-even (gain=0)
   - Key Insight: Positive gains even under attack (~0.19 Delta F1)

2. **Gain vs Heterogeneity** (top-right)
   - X: Dirichlet alpha (log scale)
   - Y: Mean personalization gain
   - Points: FedAvg per alpha
   - Key Insight: Gain stable across heterogeneity levels

3. **Gain by Training Duration** (middle-left)
   - X: Personalization epochs (3, 5)
   - Y: Mean personalization gain
   - Line: FedAvg showing linear growth
   - Key Insight: More epochs = more gain (0.175 → 0.200 Delta F1)

4. **Personalization Risk Profile** (middle-center, stacked bar)
   - X: Adversary percentage
   - Y: Client distribution (%)
   - Stacked bars: Positive gain (green), Neutral (gray), Negative/Risk (red)
   - Key Insight: 100% positive gains in benign conditions

5. **Gain Distribution (Benign)** (middle-right, boxplot)
   - X: Aggregators (only FedAvg has data)
   - Y: Personalization gain distribution
   - Box: Quartiles with outliers
   - Key Insight: Median gain ~0.18, range 0.1-0.7

6. **Personalization Impact: Before vs After** (bottom, scatter)
   - X: Global model F1 (before personalization)
   - Y: Personalized model F1 (after local fine-tuning)
   - Diagonal dashed line: No improvement
   - Points: Aggregator x seed combinations (colored by method)
   - Key Insight: Most points above diagonal = consistent improvement

**Data Filters:**
- Personalization epochs > 0
- All attack levels
- Seeds: All (n=5)

**Key Functions:**
- `plot_objective3_personalization(df, output_dir)`
- `plot_gain_vs_adversary(df, ax)`
- `plot_gain_vs_alpha(df, ax)`
- `plot_gain_vs_epochs(df, ax)`
- `plot_risk_profile(df, ax)`
- `plot_gain_distribution(df, ax)`
- `plot_before_after_comparison(df, ax)`

## Objective 4: Computational Overhead vs Security in IIoT Federated IDS

**Research Question:** What is the cost-benefit tradeoff of robust aggregation? Does Byzantine resilience justify the computational overhead?

**6 Panels:**

1. **Aggregation Overhead** (top-left, boxplot)
   - X: Aggregators
   - Y: Aggregation time (ms, log scale)
   - Box: Distribution across rounds and seeds
   - Key Insight: Bulyan 45x slower than FedAvg (20ms vs 0.5ms)

2. **Overhead vs Attack Level** (top-center)
   - X: Adversary percentage
   - Y: Mean aggregation time (ms, log scale)
   - Lines: 4 aggregators
   - Key Insight: Overhead relatively constant across attack levels

3. **Overhead vs Heterogeneity** (top-right)
   - X: Dirichlet alpha (log scale)
   - Y: Mean aggregation time (ms, log scale)
   - Lines: 4 aggregators
   - Key Insight: FedAvg scales better with heterogeneity than robust methods

4. **Cost-Benefit Tradeoff: Computational Cost vs Security** (middle, scatter)
   - X: Aggregation time (ms, log scale)
   - Y: Final F1 score (utility)
   - Points: Aggregator x attack x seed combinations (colored by method)
   - Key Insight: Clear clustering - FedAvg fast but vulnerable, robust methods slow but secure

5. **Overhead Multiplier** (bottom-left, bar chart)
   - X: Aggregators
   - Y: Relative overhead (FedAvg baseline = 1.0x)
   - Bars: Krum 26.5x, Bulyan 45.2x, Median 26.8x
   - Key Insight: Quantifies computational penalty for security

6. **Total Computational Cost Over Training** (bottom-right)
   - X: Communication round
   - Y: Cumulative aggregation time (ms)
   - Lines: 4 aggregators (alpha=0.5, benign)
   - Key Insight: Bulyan accumulates 450ms over 20 rounds vs FedAvg 30ms

**Data Filters:**
- Alpha: 0.5 (moderate heterogeneity) for most panels
- Attack level: 0 (benign) for most panels
- Seeds: All (n=5)

**Key Functions:**
- `plot_objective4_system_overhead(df, output_dir)`
- `plot_aggregation_time(df, ax)`
- `plot_time_vs_attack(df, ax)`
- `plot_time_vs_alpha(df, ax)`
- `plot_cost_benefit(df, ax)`
- `plot_overhead_comparison(df, ax)`
- `plot_cumulative_time(df, ax)`

## Usage

### Basic Invocation

```bash
python scripts/generate_comprehensive_thesis_plots.py \
    --runs_dir runs \
    --output_dir thesis_plots_iiot
```

### Parameters

- `--runs_dir`: Directory containing experiment runs (default: `runs`)
  - Expected structure: `runs/dsedge-iiotset-*/{metrics.csv, client_*_metrics.csv}`
- `--output_dir`: Output directory for plots (default: `thesis_plots_iiot`)
  - Will be created if it doesn't exist

### Output Files

- `thesis_plots_iiot/obj1_robustness_comprehensive.png` (18x12 inches, 300 DPI)
- `thesis_plots_iiot/obj2_heterogeneity_comprehensive.png` (18x12 inches, 300 DPI)
- `thesis_plots_iiot/obj3_personalization_comprehensive.png` (18x12 inches, 300 DPI)
- `thesis_plots_iiot/obj4_system_overhead_comprehensive.png` (18x12 inches, 300 DPI)

### Expected Console Output

```
================================================================================
COMPREHENSIVE THESIS PLOTTING FRAMEWORK
================================================================================

Loading all experiments from runs...
SUCCESS: Loaded 2625 records from 5 seeds
  - Aggregators: ['bulyan', 'fedavg', 'krum', 'median']
  - Alpha values: [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf]
  - Attack levels: [0, 10, 30]

================================================================================
GENERATING OBJECTIVE 1: ROBUSTNESS PLOTS
================================================================================
Saved: thesis_plots_iiot/obj1_robustness_comprehensive.png

================================================================================
GENERATING OBJECTIVE 2: HETEROGENEITY PLOTS
================================================================================
Saved: thesis_plots_iiot/obj2_heterogeneity_comprehensive.png

================================================================================
GENERATING OBJECTIVE 3: PERSONALIZATION PLOTS
================================================================================
Saved: thesis_plots_iiot/obj3_personalization_comprehensive.png

================================================================================
GENERATING OBJECTIVE 4: SYSTEM OVERHEAD PLOTS
================================================================================
Saved: thesis_plots_iiot/obj4_system_overhead_comprehensive.png

================================================================================
COMPLETE!
================================================================================
All 4 comprehensive plots saved to: thesis_plots_iiot/
```

## Customization Guide

### Color Schemes

The framework uses a consistent color palette defined in `COLORS` dictionary:

```python
COLORS = {
    "fedavg": "#1f77b4",   # Blue
    "krum": "#ff7f0e",     # Orange
    "bulyan": "#2ca02c",   # Green
    "median": "#d62728",   # Red
}
```

**To customize:** Edit the `COLORS` dictionary at the top of `generate_comprehensive_thesis_plots.py`.

### Confidence Intervals

All line plots show 95% confidence intervals computed from n=5 seeds using:

```python
def compute_confidence_interval(values: np.ndarray) -> tuple:
    """Compute 95% CI using t-distribution."""
    mean = np.mean(values)
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(values) - 1)
    return (mean, mean - ci, mean + ci)
```

**To customize:** Adjust the confidence level (0.95) in the `compute_confidence_interval` function.

### Panel Layout

Each objective uses a 3x3 GridSpec with custom panel spans:

```python
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# Example panel assignments:
ax1 = fig.add_subplot(gs[0, :2])   # Top-left, spans 2 columns
ax2 = fig.add_subplot(gs[0, 2])    # Top-right
ax6 = fig.add_subplot(gs[2, :])    # Bottom, full width
```

**To customize:** Adjust GridSpec spans and figure size in each `plot_objectiveN_*` function.

### Metrics and Filters

To modify which metrics are plotted or filter criteria:

1. **Robustness metrics:** Edit filters in `plot_objective1_robustness()`
   - Current: `alpha == 0.5`, `adv_pct in [0, 10, 30]`
2. **Heterogeneity metrics:** Edit filters in `plot_objective2_heterogeneity()`
   - Current: `adv_pct == 0`, all alpha values
3. **Personalization metrics:** Edit filters in `plot_objective3_personalization()`
   - Current: `pers_epochs > 0`, all conditions
4. **Overhead metrics:** Edit filters in `plot_objective4_system_overhead()`
   - Current: `alpha == 0.5`, `adv_pct == 0` for most panels

## Troubleshooting

### Issue: "ERROR: No data loaded!"

**Cause:** No experiment runs found matching expected naming pattern.

**Solution:** Verify run directories follow pattern `runs/dsedge-iiotset-*` and contain:
- `metrics.csv` (server metrics)
- `client_*_metrics.csv` (client metrics)

### Issue: ValueError - "x and y must have same first dimension"

**Cause:** Variable-length experiment runs (different numbers of rounds per seed).

**Solution:** The `plot_cumulative_time()` function automatically trims to minimum length. If error persists, check for corrupted CSV files.

### Issue: Empty panels or missing aggregators

**Cause:** Insufficient experimental coverage for that objective.

**Solution:**
- Objective 3 requires `pers_epochs > 0` experiments
- Check data availability: `df['aggregation'].value_counts()`
- Verify specific alpha/attack combinations exist

### Issue: Matplotlib deprecation warnings

**Cause:** Older matplotlib versions use deprecated parameter names.

**Solution:** Update matplotlib: `pip install --upgrade matplotlib>=3.9`
- Framework uses `tick_labels` parameter (not deprecated `labels`)

### Issue: Memory errors with large datasets

**Cause:** Loading thousands of experiment runs simultaneously.

**Solution:**
- Process runs in batches using `load_iiot_data()` with custom filtering
- Reduce DPI: Change `dpi=300` to `dpi=150` in `plt.savefig()` calls
- Use selective date ranges in run directory filtering

## Extension Guide: Adding Objective 5

To add a new thesis objective following the same architecture:

### Step 1: Define plotting function

```python
def plot_objective5_your_objective(df: pd.DataFrame, output_dir: Path):
    """Objective 5: Your research question here."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    fig.suptitle(
        "Objective 5: Your Objective Title",
        fontsize=18,
        fontweight="bold",
    )

    # Filter data for your objective
    obj5_df = df[YOUR_FILTER_CONDITIONS].copy()

    # Create 6 panels
    ax1 = fig.add_subplot(gs[0, :2])
    plot_your_panel1(obj5_df, ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    plot_your_panel2(obj5_df, ax2)

    # ... panels 3-6 ...

    plt.savefig(output_dir / "obj5_your_objective_comprehensive.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'obj5_your_objective_comprehensive.png'}")
```

### Step 2: Implement panel functions

```python
def plot_your_panel1(df: pd.DataFrame, ax: plt.Axes):
    """Panel 1 description."""
    # Compute statistics with CI
    for agg in ["fedavg", "krum", "bulyan", "median"]:
        agg_data = df[df["aggregation"] == agg]

        stats = agg_data.groupby("YOUR_X_VARIABLE")["YOUR_Y_METRIC"].apply(
            lambda x: compute_confidence_interval(x.dropna())
        ).apply(pd.Series)
        stats.columns = ["mean", "ci_low", "ci_up"]

        # Plot line with CI band
        ax.plot(stats.index, stats["mean"],
               label=agg.capitalize(),
               color=COLORS.get(agg, "gray"),
               linewidth=2)
        ax.fill_between(stats.index, stats["ci_low"], stats["ci_up"],
                       alpha=0.15)

    ax.set_xlabel("Your X Label", fontsize=11)
    ax.set_ylabel("Your Y Label", fontsize=11)
    ax.set_title("Panel 1 Title", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
```

### Step 3: Add to main()

```python
def main():
    # ... existing code ...

    print("\n" + "=" * 80)
    print("GENERATING OBJECTIVE 5: YOUR OBJECTIVE PLOTS")
    print("=" * 80)
    plot_objective5_your_objective(df, args.output_dir)
```

### Step 4: Document in this file

Add a new section following the template of Objectives 1-4.

## Dependencies

```python
# Core
import argparse
from pathlib import Path

# Data processing
import numpy as np
import pandas as pd

# Statistics
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Custom
from load_iiot_data import load_iiot_data
```

**Required packages:**
```bash
pip install numpy pandas scipy matplotlib
```

## Testing

The framework currently lacks automated tests. Recommended test coverage:

### Unit Tests (to be implemented)

```python
def test_compute_confidence_interval():
    """Test CI computation with known values."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, ci_low, ci_up = compute_confidence_interval(values)
    assert mean == 3.0
    assert ci_low < mean < ci_up

def test_load_iiot_data_empty_dir():
    """Test data loader with empty directory."""
    df = load_iiot_data(Path("/tmp/nonexistent"))
    assert df.empty

def test_plot_objective1_generates_file(tmp_path):
    """Test that Objective 1 creates output file."""
    # Create minimal test data
    df = create_minimal_test_dataframe()
    plot_objective1_robustness(df, tmp_path)
    assert (tmp_path / "obj1_robustness_comprehensive.png").exists()
```

### Integration Tests (to be implemented)

```python
def test_end_to_end_pipeline(test_runs_dir, tmp_path):
    """Test full pipeline: load data -> generate all 4 plots."""
    df = load_iiot_data(test_runs_dir)
    assert not df.empty

    plot_objective1_robustness(df, tmp_path)
    plot_objective2_heterogeneity(df, tmp_path)
    plot_objective3_personalization(df, tmp_path)
    plot_objective4_system_overhead(df, tmp_path)

    assert all((tmp_path / f"obj{i}_*.png").exists() for i in range(1, 5))
```

## Performance Considerations

- **Data loading:** O(n_runs * n_rounds) - typically ~2600 records in <1 second
- **Plot generation:** O(n_panels * n_aggregators * n_seeds) - ~30 seconds for all 4 objectives
- **Memory usage:** ~50 MB for typical dataset (2600 rows x 15 columns)

**Optimization opportunities:**
- Cache loaded data between plot generations
- Parallelize panel rendering using `multiprocessing`
- Use `matplotlib.use('Agg')` backend for headless servers

## References

- **Data loading:** `scripts/load_iiot_data.py`
- **Statistical utilities:** `scripts/statistical_utils.py` (shared CI computation)
- **Plot configuration:** `scripts/plot_config.py` (shared color schemes)
- **Thesis document:** `deliverables/thesis.pdf` (research context)
- **Experiment tracking:** `docs/experiment_checklist.md` (run status)

## Version History

- **v1.0 (2025-11-24):** Initial comprehensive framework with 4 objectives
  - 24 panel plotting functions
  - Robust data loader handling server + client metrics
  - Statistical rigor with 95% CI from n=5 seeds
  - Publication-quality 300 DPI output
