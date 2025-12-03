# Edge-IIoTset Dataset Integration

## Overview

This document describes the integration of the Edge-IIoTset dataset into the federated learning for intrusion detection thesis project. Edge-IIoTset is a modern (2022) comprehensive dataset specifically designed for IoT/IIoT intrusion detection in both centralized and federated learning scenarios.

## Dataset Information

### Citation

```
Ferrag, M.A., Friha, O., Hamouda, D., Maglaras, L., Janicke, H. (2022)
"Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT
and IIoT Applications for Centralized and Federated Learning"
IEEE Access, 2022
DOI: 10.36227/techrxiv.18857336.v1
```

### Dataset Characteristics

- **Total Samples**: 2,219,201
- **Attack Types**: 14 distinct attack categories
- **Classification Modes**:
  - Binary: Attack_label (0=Normal, 1=Attack)
  - Multi-class: Attack_type (15 classes including BENIGN)
- **Features**: 61 network flow features
- **Format**: CSV
- **License**: Free for academic research (CC BY-NC-SA 4.0)

### Attack Types

1. BENIGN (Normal)
2. DDoS_UDP
3. DDoS_ICMP
4. DDoS_TCP
5. DDoS_HTTP
6. SQL_injection
7. Password
8. Vulnerability_scanner
9. Uploading
10. Backdoor
11. Port_Scanning
12. XSS
13. Ransomware
14. MITM
15. Fingerprinting

### Distribution

- Normal traffic: ~73% (1,615,643 samples)
- Attack traffic: ~27% (603,558 samples)
- Largest attack classes: DDoS variants
- Smallest attack classes: MITM, Fingerprinting

## Integration Architecture

### Three-Tier Testing Strategy

The integration uses a three-tier approach to balance testing speed with experimental comprehensiveness:

#### Tier 1: Quick CI Validation
- **Sample Size**: 50,000 (2.3% of dataset)
- **Clients**: 3
- **Rounds**: 5
- **Purpose**: Fast PR validation
- **Duration**: ~10 minutes per experiment
- **Schedule**: Every pull request

#### Tier 2: Comprehensive Nightly
- **Sample Size**: 500,000 (22.5% of dataset)
- **Clients**: 6
- **Rounds**: 20
- **Purpose**: Robust aggregation and DP validation
- **Duration**: ~45 minutes per experiment
- **Schedule**: Nightly

#### Tier 3: Full-Scale Thesis
- **Sample Size**: 2,000,000 (90.1% of dataset)
- **Clients**: 10
- **Rounds**: 50
- **Purpose**: Publication-quality results
- **Duration**: ~2 hours per experiment
- **Schedule**: Manual trigger or weekly

## Implementation Details

### Data Loading

The `load_edge_iiotset()` function in `data_preprocessing.py` provides a consistent interface for loading Edge-IIoTset data:

```python
from data_preprocessing import load_edge_iiotset

# Multi-class classification (15 classes)
df, label_col, proto_col = load_edge_iiotset(
    "data/edge-iiotset/edge_iiotset_full.csv",
    use_multiclass=True
)

# Binary classification
df, label_col, proto_col = load_edge_iiotset(
    "data/edge-iiotset/edge_iiotset_full.csv",
    use_multiclass=False
)

# Quick testing with sample limit
df, label_col, proto_col = load_edge_iiotset(
    "data/edge-iiotset/edge_iiotset_full.csv",
    use_multiclass=True,
    max_samples=10000
)
```

### Sample Generation

Generate stratified samples using the provided script:

```bash
# Generate all tiers
python scripts/prepare_edge_iiotset_samples.py --tier all

# Generate specific tier
python scripts/prepare_edge_iiotset_samples.py --tier quick

# Custom parameters
python scripts/prepare_edge_iiotset_samples.py \
    --tier nightly \
    --source datasets/edge-iiotset/Edge-IIoTset\ dataset/Selected\ dataset\ for\ ML\ and\ DL/DNN-EdgeIIoT-dataset.csv \
    --output-dir data/edge-iiotset \
    --seed 42
```

### Running Experiments

Edge-IIoTset is integrated into the comparative analysis framework:

```bash
# Quick experiment (local testing)
python scripts/comparative_analysis.py \
    --dataset edge-iiotset-quick \
    --preset comp_fedavg_alpha1.0_seed42 \
    --clients 3 \
    --rounds 5

# Full-scale experiment
python scripts/comparative_analysis.py \
    --dataset edge-iiotset-full \
    --preset comp_krum_alpha0.5_seed42 \
    --clients 10 \
    --rounds 50
```

### Supported Datasets

The comparative analysis framework now supports:

- `unsw`: UNSW-NB15 (legacy)
- `cic`: CIC-IDS2017 (legacy)
- `edge-iiotset-quick`: 50k sample
- `edge-iiotset-nightly`: 500k sample
- `edge-iiotset-full`: 2M sample (90% of full dataset)

## CI Integration

### Three-Tier Automated Workflow Architecture

Edge-IIoTset experiments run automatically via GitHub Actions with three tiers:

#### Tier 1: Quick PR Validation

**File:** `.github/workflows/edge-iiotset-quick.yml`

**Trigger:** Every pull request to `main`

**Configuration:**
- Dataset: edge-iiotset-quick (50k samples)
- Clients: 3
- Rounds: 5
- Timeout: 60 minutes
- Presets: 3 critical experiments (FedAvg, Krum, FedProx)

**Purpose:** Fast validation before merging changes

**Artifacts:** Metrics CSVs and plots (30-day retention)

#### Tier 2: Nightly Comprehensive

**File:** `.github/workflows/edge-iiotset-nightly.yml`

**Trigger:** Daily at 2 AM UTC + pushes to `main`

**Configuration:**
- Dataset: edge-iiotset-nightly (500k samples)
- Clients: 6
- Rounds: 20
- Timeout: 8 hours per dimension
- Dimensions: All 6 (aggregation, heterogeneity, attack, privacy, personalization, heterogeneity_fedprox)
- Parallelization: max-parallel: 2

**Process:**
1. Run experiments for all dimensions in parallel
2. Generate run-level plots for each experiment
3. Generate thesis-quality plots per dimension
4. Consolidate all plots
5. Commit plots to `plots/thesis/YYYY-MM-DD/edge-iiotset-nightly/`

**Artifacts:** Metrics, plots, logs (90-day retention)

#### Tier 3: Weekly Full-Scale

**File:** `.github/workflows/edge-iiotset-full-scale.yml`

**Trigger:** Weekly on Sundays at 1 AM UTC

**Configuration:**
- Dataset: edge-iiotset-full (2M samples = 90% of full dataset)
- Clients: 10
- Rounds: 50
- Timeout: 8 hours per group (24 hours total)
- Dimension Groups:
  - Group 1: aggregation, heterogeneity
  - Group 2: attack, privacy
  - Group 3: personalization, heterogeneity_fedprox
- Parallelization: max-parallel: 3 (one group per parallel job)

**Process:**
1. Run 3 dimension groups in parallel (8 hours each)
2. Each group runs its dimensions sequentially
3. Generate comprehensive thesis plots for all experiments
4. Consolidate publication-quality results
5. Commit plots to `plots/thesis/YYYY-MM-DD/edge-iiotset-full/`

**Artifacts:** All metrics, plots, logs (365-day retention for thesis publication)

### Automatic Plot Commitment

All nightly and weekly workflows automatically:
1. Generate thesis-quality plots (PNG + PDF)
2. Consolidate results across all dimensions
3. Commit plots to `plots/thesis/` with date-based organization
4. Push to `main` branch for easy browser access

### Manual Workflow Triggers

All workflows support `workflow_dispatch` for manual execution:

```bash
# Quick validation
gh workflow run edge-iiotset-quick.yml

# Nightly with custom parameters
gh workflow run edge-iiotset-nightly.yml \
  -f dimensions=aggregation,attack \
  -f num_clients=8 \
  -f num_rounds=30

# Full-scale specific group
gh workflow run edge-iiotset-full-scale.yml \
  -f dimension_group=group1 \
  -f num_clients=12 \
  -f num_rounds=60
```

## Validation Thresholds

Edge-IIoTset has adjusted validation thresholds due to the increased difficulty of 15-class classification:

### Binary Classification
- Minimum F1 Score: 0.75
- Minimum Accuracy: 0.80

### Multi-class Classification
- Minimum F1 Score: 0.60
- Minimum Accuracy: 0.65

These thresholds are configured in `scripts/ci_checks.py`.

## Thesis Integration

### Research Objectives Mapping

1. **Robust Aggregation** (Krum, Bulyan, Median)
   - Validated on 27x larger dataset (2.2M vs 82K samples)
   - 14 attack types stress-test Byzantine robustness

2. **Data Heterogeneity** (FedProx, Non-IID)
   - IoT-specific non-IID scenarios
   - Per-sensor data partitioning

3. **Personalization**
   - Per-sensor model fine-tuning
   - Heterogeneous device adaptation

4. **Privacy** (DP, Secure Aggregation)
   - Large-scale DP validation
   - Multi-class privacy analysis

5. **Empirical Validation**
   - Modern dataset (2022 vs 2017/2015)
   - Publication-quality results

### Expected Contributions

- **Scalability Validation**: Demonstrate robust aggregation maintains performance on datasets 27x larger than prior work
- **Modern Threat Landscape**: Validate against 2022 IoT/IIoT attacks
- **Attack Diversity**: 14 attack types vs 2-5 in legacy datasets
- **FL-Native Design**: Dataset explicitly designed for federated scenarios
- **Statistical Significance**: 144 experiments with 3 seeds each

## Troubleshooting

### Dataset Not Found

If you see "Source dataset not found" errors:

1. Verify Edge-IIoTset is extracted:
   ```bash
   ls datasets/edge-iiotset/Edge-IIoTset\ dataset/Selected\ dataset\ for\ ML\ and\ DL/
   ```

2. Check file size:
   ```bash
   ls -lh datasets/edge-iiotset/Edge-IIoTset\ dataset/Selected\ dataset\ for\ ML\ and\ DL/DNN-EdgeIIoT-dataset.csv
   ```
   Expected: ~1.1GB

3. Re-extract if needed:
   ```bash
   cd datasets/edge-iiotset
   unzip -q ../archive.zip
   ```

### Memory Issues

For systems with limited RAM:

1. Use smaller tiers:
   ```bash
   python scripts/comparative_analysis.py --dataset edge-iiotset-quick
   ```

2. Reduce batch size in experiments

3. Generate custom sample size:
   ```python
   df, label_col, _ = load_edge_iiotset(csv_path, max_samples=100000)
   ```

### Convergence Issues

If models don't converge:

1. Check class distribution in sample:
   ```bash
   python scripts/analyze_data_splits.py --dataset edge-iiotset-quick
   ```

2. Increase number of rounds (50+ for full dataset)

3. Adjust learning rate for larger dataset

## References

1. Ferrag et al., "Edge-IIoTset", IEEE Access, 2022
2. Dataset on Kaggle: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
3. Dataset on IEEE DataPort: https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications
4. Official Website: Contact mohamed.amine.ferrag@gmail.com

## Version History

- **2025-01-12**: Initial integration for Issue #130
  - Added `load_edge_iiotset()` function
  - Created three-tier sample generation
  - Updated comparative analysis framework
  - Integrated CI workflows
