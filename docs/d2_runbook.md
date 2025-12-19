# D2 Runbook: Federated IDS Experiment Workflow

## Purpose

This runbook documents the end-to-end workflow for running federated learning experiments on intrusion detection datasets, from data preprocessing through result publication. It serves as a reference for reproducibility and advisor-facing documentation.

---

## Workflow Overview

```
┌─────────────────┐
│ 1. Preprocess   │  Setup datasets (CIC-IDS2017, UNSW-NB15)
│    Data         │  → scripts/setup_real_datasets.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Run          │  Execute federated training experiments
│    Experiments  │  → scripts/comparative_analysis.py
└────────┬────────┘  → server.py + client.py
         │
         ▼
┌─────────────────┐
│ 3. Generate     │  Create thesis-ready visualizations
│    Plots        │  → scripts/generate_thesis_plots.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Summarize    │  Aggregate metrics and statistics
│    Results      │  → scripts/summarize_metrics.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Commit       │  Version control and artifact retention
│    Artifacts    │  → plots/, analysis/, runs/
└─────────────────┘
```

---

## Step 1: Data Preprocessing

### Command
```bash
python scripts/setup_real_datasets.py
```

### Purpose
- Downloads and prepares CIC-IDS2017 and UNSW-NB15 datasets
- Creates sample datasets for CI/testing
- Validates data integrity

### Output
- `data/cic/cic_ids2017_multiclass.csv`
- `data/unsw/unsw_nb15.csv`
- Sample files for rapid testing

### Validation
```bash
ls -lh data/cic/cic_ids2017_multiclass.csv
ls -lh data/unsw/unsw_nb15.csv
```

---

## Step 2: Run Experiments

### Centralized Training (Baseline)
```bash
python server.py --rounds 20 --aggregation fedavg
python client.py --dataset unsw --num_clients 1 --client_id 0
```

### Federated Training (Standard)
```bash
# Terminal 1: Start server
python server.py --rounds 20 --aggregation fedavg --server_address 127.0.0.1:8099

# Terminal 2-7: Start 6 clients
for i in {0..5}; do
  python client.py --server_address 127.0.0.1:8099 \
    --dataset unsw --num_clients 6 --client_id $i \
    --partition_strategy dirichlet --alpha 0.1 \
    --seed 42 &
done
```

### Federated Training (Protocol-based Drift)
```bash
# Predefine protocol-to-client mapping (cic example provided)
python server.py --rounds 20 --aggregation fedavg --server_address 127.0.0.1:8098

for i in {0..4}; do
  python client.py --server_address 127.0.0.1:8098 \
    --dataset cic --data_path data/cic/cic_ids2017_multiclass.csv \
    --num_clients 5 --client_id $i \
    --partition_strategy protocol --protocol_col Protocol \
    --protocol_mapping_path configs/protocol_groups/cic_protocol_clients.json \
    --seed 7 &
done
```
`protocol_mapping_path` pins important protocols (HTTP, DNS, SSH, etc.) to specific clients, while any unlisted protocols fall back to balanced round-robin assignments without altering feature schemas.

### Per-dataset encoder toggle

```bash
# Launch CIC client with dataset-specific encoder
python client.py --server_address 127.0.0.1:8099 \
  --dataset cic --data_path data/cic/cic_ids2017_multiclass.csv \
  --num_clients 5 --client_id 0 \
  --partition_strategy dirichlet --alpha 0.1 \
  --model_arch per_dataset_encoder --encoder_latent_dim 256
```

`--model_arch auto` (default) automatically selects the encoder for CIC/UNSW clients while synthetic and ad-hoc datasets continue to use the lightweight `SimpleNet`.

### Comparative Analysis (Automated)
```bash
# Run all 5 thesis objectives on UNSW-NB15
python scripts/comparative_analysis.py \
  --dimension heterogeneity \
  --dataset unsw \
  --output_dir results/comparative_analysis/unsw

# Run on CIC-IDS2017
python scripts/comparative_analysis.py \
  --dimension heterogeneity \
  --dataset cic \
  --output_dir results/comparative_analysis/cic
```

### Available Dimensions
- `aggregation` - FedAvg vs robust methods (Krum, Bulyan, Median)
- `attack` - Byzantine attack resilience (0%, 10%, 30% adversarial clients)
- `heterogeneity` - Data heterogeneity sweep (alpha values)
- `heterogeneity_fedprox` - FedProx mu parameter sweep
- `privacy` - Differential privacy noise sweep
- `personalization` - Personalization epochs sweep
- `mixed` - Cross-dataset federation (CIC + UNSW)

### Output
- `runs/comp_*/` - Individual experiment directories
- Each run contains artifacts (see Artifact Map below)

---

## Step 3: Generate Plots

### Command
```bash
python scripts/generate_thesis_plots.py \
  --dimension heterogeneity \
  --runs_dir runs \
  --output_dir results/thesis_plots/heterogeneity
```

### Options
- `--dimension` - Experiment dimension to plot
- `--runs_dir` - Directory containing experiment runs
- `--output_dir` - Output directory for plots
- `--dataset` - Filter by dataset (unsw, cic)

### Output
- PNG plots with confidence intervals
- PDF plots for thesis inclusion
- CSV summaries with statistical data

### Validation
```bash
find results/thesis_plots/ -name "*.png" -o -name "*.pdf"
```

---

## Step 4: Summarize Results

### Command
```bash
python scripts/summarize_metrics.py \
  --run_dir runs/comp_fedavg_alpha0.1_adv0_dp0_pers0_seed42 \
  --output runs/comp_fedavg_alpha0.1_adv0_dp0_pers0_seed42/summary.json
```

### Purpose
- Aggregates client-level metrics
- Computes fairness indicators
- Generates summary statistics

### Output
- `summary.json` with aggregated metrics
- Includes: macro_f1_argmax, worst/best client performance, CV, FPR fraction
- Multi-class runs also emit a `confusion_matrix` block storing global + per-client counts/percentages for thesis visuals.

### Confusion Matrices
```bash
python scripts/plot_metrics.py \
  --run_dir runs/comp_fedavg_alpha0.1_adv0_dp0_pers0_seed42 \
  --output_dir runs/comp_fedavg_alpha0.1_adv0_dp0_pers0_seed42 \
  --save_confusion_matrix --confusion_matrix_scope both
```
- Saves normalized global heatmap to `runs/.../confusion_matrix.png`
- Stores per-client count/percentage heatmaps under `runs/.../confusion_matrices/`

### Validation
```bash
jq . runs/comp_*/summary.json
```

---

## Step 5: Commit Artifacts

### Plots
```bash
git add plots/thesis/
git commit -m "feat(plots): add thesis visualizations for heterogeneity experiments"
```

### Analysis Results
```bash
git add analysis/
git commit -m "feat(analysis): add statistical summaries for Issue #44"
```

### Experiment Runs (Selective)
```bash
# Commit summaries only, not raw logs
git add runs/*/summary.json
git add runs/*/config.json
git commit -m "feat(experiments): add experiment configurations and summaries"
```

---

## Artifact Map

### Experiment Run Directory Structure

```
runs/comp_{aggregation}_alpha{α}_adv{%}_dp{σ}_pers{epochs}_mu{μ}_seed{n}/
├── config.json              # Experiment configuration
├── metrics.csv              # Server-level metrics per round
├── server.log               # Server execution log
├── client_0_metrics.csv     # Client 0 per-round metrics
├── client_0.log             # Client 0 execution log
├── client_1_metrics.csv     # Client 1 per-round metrics
├── client_1.log             # Client 1 execution log
├── ...                      # Additional clients
├── client_N_metrics.csv     # Client N per-round metrics
├── client_N.log             # Client N execution log
└── summary.json             # Aggregated statistics (generated post-run)
```

### File Specifications

#### config.json
```json
{
  "aggregation": "fedavg",
  "alpha": 0.1,
  "adversarial_clients": 0,
  "dp_noise_multiplier": 0.0,
  "personalization_epochs": 0,
  "fedprox_mu": 0.01,
  "seed": 42,
  "num_clients": 6,
  "num_rounds": 20,
  "dataset": "unsw"
}
```

#### metrics.csv
Columns:
- `round` - Training round number
- `accuracy` - Global model accuracy
- `loss` - Global model loss
- `macro_f1` - Macro F1-score across all classes
- `benign_fpr` - False positive rate on benign traffic
- `timestamp` - Execution timestamp

#### client_N_metrics.csv
Columns:
- `round` - Training round number
- `client_id` - Client identifier
- `macro_f1_argmax` - F1-score at argmax threshold
- `benign_fpr_argmax` - FPR at argmax threshold
- `f1_bin_tau` - F1-score at binary threshold tau
- `benign_fpr_bin_tau` - FPR at binary threshold tau
- `tau_bin` - Binary classification threshold
- `loss_before` - Loss before local training
- `loss_after` - Loss after local training
- `grad_norm_l2` - L2 norm of gradients
- `weight_norm_l2` - L2 norm of model weights
- `weight_update_norm_l2` - L2 norm of weight updates
- `personalization_gain` - F1 improvement from personalization (if enabled)
- `dp_epsilon` - Privacy budget (if DP enabled)
- `dp_delta` - Privacy parameter (if DP enabled)
- `dp_sigma` - Noise multiplier (if DP enabled)
- `dp_clip_norm` - Gradient clipping norm (if DP enabled)

#### summary.json
```json
{
  "macro_f1_argmax": {
    "mean": 0.856,
    "min": 0.821,
    "max": 0.869,
    "cv": 0.028
  },
  "benign_fpr_argmax": {
    "mean": 0.045,
    "min": 0.032,
    "max": 0.067,
    "cv": 0.312
  },
  "worst_client_macro_f1_argmax": 0.821,
  "best_client_macro_f1_argmax": 0.869,
  "cv_macro_f1_argmax": 0.028,
  "fraction_clients_fpr_le_0_10": 0.833
}
```

---

## CI/CD Workflow Dispatch

### Manual Trigger via GitHub Actions

```bash
# Trigger comparative analysis workflow
gh workflow run comparative-analysis-nightly.yml \
  --ref exp/issue-44-comprehensive-experiments \
  -f dimensions=heterogeneity \
  -f num_clients=6 \
  -f num_rounds=20 \
  -f run_cic=true
```

### Parameters
- `dimensions` - Comma-separated list or "all"
- `num_clients` - Number of federated clients (default: 6)
- `num_rounds` - Training rounds (default: 20)
- `run_cic` - Run CIC-IDS2017 experiments (default: true)

### Workflow Files
- `.github/workflows/comparative-analysis-nightly.yml` - Main experiment workflow
- `.github/workflows/fedprox-nightly.yml` - FedProx comparison workflow
- `.github/workflows/ci.yml` - Standard CI checks

---

## Common Workflows

### Full Thesis Experiment Suite
```bash
# Run all 5 objectives on both datasets
for dimension in aggregation attack heterogeneity privacy personalization; do
  python scripts/comparative_analysis.py \
    --dimension $dimension \
    --dataset unsw \
    --output_dir results/comparative_analysis/unsw
  
  python scripts/comparative_analysis.py \
    --dimension $dimension \
    --dataset cic \
    --output_dir results/comparative_analysis/cic
done

# Generate all plots
for dimension in aggregation attack heterogeneity privacy personalization; do
  python scripts/generate_thesis_plots.py \
    --dimension $dimension \
    --runs_dir runs \
    --output_dir results/thesis_plots/$dimension
done
```

### Quick Smoke Test
```bash
# Validate setup with minimal experiment
python scripts/comparative_analysis.py \
  --dimension heterogeneity \
  --dataset unsw \
  --num_clients 3 \
  --num_rounds 5 \
  --dry_run  # Preview commands without execution
```

### Reproduce Specific Experiment
```bash
# Extract config from existing run
CONFIG=$(cat runs/comp_fedavg_alpha0.1_adv0_dp0_pers0_seed42/config.json)

# Re-run with same parameters
python server.py --rounds 20 --aggregation fedavg --seed 42
python client.py --dataset unsw --num_clients 6 --client_id 0 \
  --partition_strategy dirichlet --alpha 0.1 --seed 42
```

---

## Troubleshooting

### Issue: Port Already in Use
```bash
# Check for existing processes
lsof -i :8099

# Kill existing server
pkill -f "python server.py"
```

### Issue: Out of Memory
```bash
# Reduce batch size or client count
python client.py --batch_size 32 --num_clients 3
```

### Issue: Experiments Timeout in CI
```bash
# Check workflow timeout settings
grep "timeout-minutes" .github/workflows/comparative-analysis-nightly.yml

# Current: 900 minutes (15 hours)
```

### Issue: Missing Artifacts
```bash
# Validate experiment completed
ls -la runs/comp_*/summary.json

# Re-generate summaries
for dir in runs/comp_*/; do
  python scripts/summarize_metrics.py --run_dir "$dir" --output "$dir/summary.json"
done
```

---

## Performance Benchmarks

### Local Execution
- Single experiment (6 clients, 20 rounds): 3-5 minutes
- Full dimension sweep (5 seeds): 15-25 minutes
- Complete thesis suite (5 dimensions, 2 datasets): 2-3 hours

### CI Execution
- Single dimension (UNSW): 30-45 minutes
- Single dimension (CIC): 45-60 minutes
- Full matrix (6 dimensions, 2 datasets): 8-12 hours

---

## References

### Related Documentation
- [CIC Objectives Matrix](cic_objectives.md) - Experiment dimension details
- [CI Optimization Guide](ci-optimization.md) - Timeout and resource configuration
- [Threat Model](threat_model.md) - Security assumptions and defenses

### Key Scripts
- `scripts/comparative_analysis.py` - Main experiment runner
- `scripts/generate_thesis_plots.py` - Visualization generator
- `scripts/summarize_metrics.py` - Metrics aggregation
- `scripts/setup_real_datasets.py` - Data preparation

### External Resources
- Flower Framework: https://flower.dev/docs/
- CIC-IDS2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15 Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

**Last Updated:** 2025-10-22  
**Maintained By:** Abraham J. Reines  
**Version:** 1.0
