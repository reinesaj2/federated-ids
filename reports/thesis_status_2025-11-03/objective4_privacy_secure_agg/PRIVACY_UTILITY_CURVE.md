# Privacy-Utility Curve Documentation

## Overview

The privacy-utility curve visualization demonstrates the fundamental tradeoff between **differential privacy (DP) protection** and **model utility** in federated learning for intrusion detection.

This addresses **Research Objective 4** from FL.txt: "Maintain utility while introducing privacy-preserving mechanisms."

---

## Metrics

### Privacy Budget: ε (Epsilon)

The privacy budget `ε` quantifies the privacy guarantee provided by differential privacy:

- **Lower ε → Stronger privacy** (more noise added to gradients)
- **Higher ε → Weaker privacy** (less noise, closer to non-private baseline)
- **ε → ∞ → No privacy** (baseline FedAvg without DP)

**Computation**: Uses **Rényi Differential Privacy (RDP)** accounting via Opacus `RDPAccountant`:

```python
from privacy_accounting import compute_epsilon

epsilon = compute_epsilon(
    noise_multiplier=1.0,  # σ (Gaussian noise stddev)
    delta=1e-5,            # Target δ for (ε, δ)-DP
    num_steps=20,          # FL rounds
    sample_rate=1.0        # Sampling rate per round
)
```

**References**:

- Mironov (2017): Rényi Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy
- McMahan et al. (2017): Learning Differentially Private Language Models

### Utility: Macro-F1

Model performance measured by **macro-averaged F1 score** across all attack classes:

- Computed from client-level metrics (averaged across all clients)
- Aggregated across multiple seeds (5 seeds per configuration)
- Reported with **95% confidence intervals**

---

## Plot Structure

### Primary Plot: Macro-F1 vs ε

**X-axis**: Epsilon (privacy budget), formal DP accountant
**Y-axis**: Macro-F1 score (0.0 to 1.0)
**Error bars**: 95% confidence intervals across seeds
**Baseline**: Horizontal dashed line showing non-DP performance (ε → ∞)

**Interpretation**:

- Points **above baseline**: Unlikely (DP typically degrades performance)
- Points **near baseline**: Good privacy-utility tradeoff
- Points **far below baseline**: High privacy cost

### Supplementary Plots (Appendix)

1. **L2 Distance vs σ (noise multiplier)**
   - Shows model drift relative to benign baseline
   - Higher noise → higher drift

2. **Cosine Similarity Distribution**
   - Violin plots comparing DP-enabled vs DP-disabled
   - Shows alignment with benign model

---

## Generated Outputs

### 1. Plot Files

**Location**: `results/comparative_analysis/<dataset>/privacy_utility_curve.{png,pdf}`

**Formats**:

- PNG: 300 DPI, publication-ready
- PDF: Vector graphics for LaTeX inclusion

### 2. CSV Summary

**Location**: `results/comparative_analysis/<dataset>/privacy_utility_curve.csv`

**Columns**:

- `epsilon`: Privacy budget (computed via RDP accountant)
- `macro_f1_mean`: Mean macro-F1 across seeds
- `ci_lower`: Lower bound of 95% CI
- `ci_upper`: Upper bound of 95% CI
- `n`: Number of seeds aggregated
- `dp_noise_multiplier`: Gaussian noise σ (metadata)
- `is_baseline`: 1 if no DP (baseline), 0 otherwise

**Usage**: For LaTeX caption tables and quantitative analysis.

### 3. Logged Metadata in summary.json

Each DP experiment logs the following fields:

- `dp_epsilon`: Formal privacy budget (RDP-computed)
- `dp_delta`: Target δ for (ε, δ)-DP (default: 1e-5)
- `dp_sigma`: Noise multiplier (Gaussian stddev)
- `dp_clip_norm`: Gradient clipping threshold
- `dp_sample_rate`: Sampling rate per round
- `num_steps`: Total FL rounds (used for ε computation)

---

## Experimental Configuration

### DP Parameters (Issue #44 Expanded Grids)

**Noise Multipliers** (σ):

- 0.0 (baseline, no DP)
- 0.5 (weak privacy)
- 1.0 (moderate privacy)
- 1.5 (strong privacy)

**Fixed Parameters**:

- δ = 1e-5 (standard for DP literature)
- Clipping norm = 1.0 (gradient clipping threshold)
- Sample rate = 1.0 (full-batch aggregation per round)
- Steps = 20 (FL rounds)

**Seeds**: 5 replications (42, 43, 44, 45, 46)

### Expected ε Ranges

For 20 FL rounds with δ=1e-5:

| σ (Noise) | Approximate ε | Privacy Level   |
| --------- | ------------- | --------------- |
| 0.0       | ∞             | None (baseline) |
| 0.5       | ~20-30        | Weak            |
| 1.0       | ~5-10         | Moderate        |
| 1.5       | ~2-4          | Strong          |

---

## Usage

### Generate Privacy-Utility Curve

```bash
# For all privacy experiments (UNSW + CIC)
python scripts/generate_thesis_plots.py --dimension privacy --runs_dir runs --output_dir results/comparative_analysis

# For specific dataset
python scripts/generate_thesis_plots.py --dimension privacy --runs_dir runs --output_dir results/comparative_analysis/cic
```

### Validate DP Accounting

```bash
# Run privacy accounting integration tests
pytest test_privacy_accounting_integration.py -v

# Check epsilon computation consistency
pytest test_privacy_accounting_integration.py::test_privacy_utility_consistency_check -v
```

### Inspect Generated Data

```bash
# View CSV summary
cat results/comparative_analysis/unsw/privacy_utility_curve.csv

# Example output:
# epsilon,macro_f1_mean,ci_lower,ci_upper,n,dp_noise_multiplier,is_baseline
# 2.5,0.82,0.80,0.84,5,1.5,0
# 7.8,0.87,0.85,0.89,5,1.0,0
# 25.3,0.90,0.88,0.92,5,0.5,0
# nan,0.91,0.89,0.93,5,0.0,1
```

---

## Acceptance Criteria (Issue #59)

- [x] Plot macro-F1 vs formal ε (computed via RDP accountant)
- [x] Include 95% CIs across ≥5 seeds per (ε, config)
- [x] Log ε/δ, clip, σ, sampling rate, and steps in summary.json
- [x] Caption lists accountant type, δ, clipping norm, sampling rate, steps, seeds (n), dataset
- [x] Emit CSV of ε→F1 mappings for caption tables
- [x] Integration with existing `generate_thesis_plots.py --dimension privacy`
- [x] Unit tests for epsilon computation and edge cases
- [x] Integration tests for plot generation and data validation

---

## Interpretation Guidelines

### For Thesis Results Section

**Good Result** (achievable tradeoff):

- At ε = 5 (moderate privacy), macro-F1 drops by ≤ 10% relative to baseline
- Example: Baseline F1 = 0.90, DP (ε=5) F1 = 0.82 → 8.9% degradation

**Acceptable Result** (slight degradation):

- At ε = 10 (weak privacy), macro-F1 drops by ≤ 5%
- Example: Baseline F1 = 0.90, DP (ε=10) F1 = 0.86 → 4.4% degradation

**Poor Result** (high privacy cost):

- At ε = 2 (strong privacy), macro-F1 drops by > 20%
- Example: Baseline F1 = 0.90, DP (ε=2) F1 = 0.70 → 22% degradation

### For Thesis Defense

**Question**: "Why is differential privacy important for federated IDS?"

**Answer**:

- IDS data contains sensitive network traffic patterns
- DP prevents reconstruction attacks on individual client datasets
- Formal privacy guarantees (ε, δ) provide mathematical provability
- Trade-off analysis shows feasibility for real-world deployment

**Question**: "What privacy level do you recommend for production?"

**Answer**:

- For CIC-IDS2017 with macro-F1 baseline ≈ 0.90:
  - **ε = 5 (σ=1.0)**: Moderate privacy, ≤10% F1 degradation, **recommended**
  - **ε = 10 (σ=0.5)**: Weak privacy, ≤5% F1 degradation, acceptable for less sensitive data
  - **ε = 2 (σ=1.5)**: Strong privacy, may require model architecture improvements

---

## Technical Notes

### Why RDP Accountant?

**Advantages over basic DP composition**:

- **Tighter bounds**: More accurate privacy accounting for multiple rounds
- **Standard in FL**: Used by Google's TensorFlow Privacy, Meta's Opacus
- **Peer-reviewed**: Mironov (2017) theoretical foundation

**Alternative**: Zhu et al. (2019) analytical accountant for FedAvg

- Not implemented (requires custom accounting per aggregation method)
- RDP is aggregation-agnostic and works with FedAvg, Krum, Bulyan, Median

### Known Limitations

1. **ε → ∞ for σ = 0**: Baseline experiments have no privacy, ε is undefined (plotted as separate reference line)
2. **Sample rate = 1.0 assumption**: Full-batch aggregation (standard for FL with 6 clients)
3. **Single clipping norm**: Fixed at 1.0 (not swept in Issue #44 grid)
4. **No per-aggregation-method DP analysis**: Current experiments only use DP with FedAvg (Krum/Bulyan/Median + DP is future work)

### Future Enhancements (Post-Thesis)

- Per-client ε tracking (heterogeneous privacy budgets)
- Adaptive noise scheduling (higher noise early, lower noise late)
- DP-SGD integration for local client training (currently only server-side noise)
- Per-layer noise allocation (embedding vs classifier layers)

---

## References

1. **Mironov, I. (2017)**. "Rényi Differential Privacy." IEEE CSF 2017.
2. **Abadi, M., et al. (2016)**. "Deep Learning with Differential Privacy." ACM CCS 2016.
3. **McMahan, H. B., et al. (2017)**. "Learning Differentially Private Language Models Without Losing Accuracy." arXiv:1710.06963.
4. **Geyer, R. C., Klein, T., & Nabi, M. (2017)**. "Differentially Private Federated Learning: A Client Level Perspective." NIPS 2017 Workshop.

---

## Related Issues

- **Issue #10**: DP accounting scaffold + SecAgg toggle clarity (M2: Privacy & Security)
- **Issue #23**: Consolidate experiments into thesis reporting (M3: Visualization)
- **Issue #44**: Run comprehensive CIC-IDS2017 experiments with expanded DP grids (M1: Experiments)
- **Issue #59**: THIS ISSUE - Privacy-utility curve with formal ε (M3: Visualization)

---

_Documentation generated as part of Issue #59 acceptance criteria._
_Last updated: 2025-10-21_
