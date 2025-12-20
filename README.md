# Federated IDS Demo (Flower + PyTorch)

Federated learning demo using [Flower](https://flower.dev) for orchestration and PyTorch for local training.
Supports synthetic and real IDS datasets (CIC-IDS2017, UNSW-NB15) with preprocessing (scaling and one‑hot encoding)
and non‑IID partitioning (IID, Dirichlet, protocol). Includes robust aggregation implementations (Median, Krum, simplified Bulyan) and FedProx algorithm comparison.

## Table of Contents

1. Prerequisites (what you need installed)
2. One‑command verification (recommended first run)
3. Manual Quickstart (server + two clients)
4. Expected output (so you know it worked)
5. Reproducibility & logging (seeds, logs, plots)
6. Algorithm comparison (FedAvg vs FedProx)
7. Real datasets (UNSW‑NB15, CIC‑IDS2017)
8. Experimental Results & Performance Analysis
9. Visualization Gallery & Plot References
10. Quick Links to Analysis & Results
11. Troubleshooting (common errors and fixes)
12. Project structure
13. Privacy & robustness disclosure (D2 scope)
14. D2 Runbook (experiment workflow & artifact map)

## 1) Prerequisites

- macOS or Linux (Windows works via WSL2).
- Python 3.13 recommended (CPU‑only is fine). Check with:
  ```bash
  python3 --version
  ```
- Enough disk for datasets (optional demos use a 10% UNSW sample).

Clone or open the project folder. In what follows, replace <ABS_PATH> with your absolute path:
`/Users/you/Documents/Thesis/federated-ids`.

---

## 2) One‑command verification (recommended)

This runs the server and two synthetic clients twice on two ports, and checks reproducibility. It creates `.verify_logs/`.

```bash
cd <ABS_PATH>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

export PORT_MAIN=8099 PORT_ALT=8100 ROUNDS=2 TIMEOUT_SECS=30 SEED=42
bash scripts/verify_readme.sh
```

You should see “All checks passed”. If this completes, you’re ready to demo.

---

## 3) Manual Quickstart (server + two clients)

Run everything from the project root. Use three terminals (one for server, two for clients).

### 3.1 Create and activate a virtual environment (if not already done)

```bash
cd <ABS_PATH>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3.2 Start the Flower server (FedAvg, 2 rounds)

```bash
export SEED=42
python server.py --rounds 2 --aggregation fedavg --server_address 127.0.0.1:8099
```

Notes:

- Deprecation warnings about `start_server`/`start_client` are expected on flwr==1.21.0.
- If port 8099 is busy, choose another (e.g., 8100) and use it for both server and clients.

### 3.3 Start two synthetic clients (in two new terminals)

Terminal B:

```bash
cd <ABS_PATH>
source .venv/bin/activate
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2
```

Terminal C:

```bash
cd <ABS_PATH>
source .venv/bin/activate
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2
```

The server will run 2 rounds and then print a summary and exit.

---

## 4) Expected output

On the server terminal, a successful 2‑round run ends with output similar to:

```
INFO :      [SUMMARY]
INFO :      Run finished 2 round(s) in ~6-8s
INFO :          History (loss, distributed):
INFO :                  round 1: 0.047... (varies by seed)
INFO :                  round 2: 0.041... (varies by seed)
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 0.98), (2, 0.975)]} (varies by seed)
```

On each client terminal, you'll see lines such as:

```
[Client X] Logging metrics to: ./logs/client_X_metrics.csv
[Data] Train samples=1600, class_counts={0: 800, 1: 800}; Test samples=400, class_counts={0: 200, 1: 200}
[Client X] Model validation passed: out_features=2, num_classes_global=2
[Client X] Label histogram: {"0": 1016, "1": 984} (varies by partitioning)
```

**Note**: Exact values will vary based on random seed and data partitioning, but the structure should be identical.

---

## 5) Reproducibility, logs, and plots

- Reproducibility: set `SEED` on the server and `--seed` on clients. Example:
  ```bash
  export SEED=42
  python server.py --rounds 2
  python client.py --seed 42 ...
  ```
- Logs: CSV files are written to `./logs/` (e.g., `metrics.csv`, `client_0_metrics.csv`).
- Plots: generate figures from any run directory that contains CSVs:

  ```bash
  # Create output directory
  mkdir -p ./runs/smoke_metrics

  # Server + client plots → saves PNGs to output directory
  python scripts/plot_metrics.py --run_dir ./logs --output_dir ./runs/smoke_metrics

  # JSON summary of client metrics
  python scripts/summarize_metrics.py --run_dir ./logs --output ./runs/smoke_metrics/summary.json
  ```

**Important**: If plotting fails with "Expected X fields, saw Y" error, clean logs between different demo runs:

```bash
rm -rf logs/; mkdir logs
```

---

## 6) Algorithm comparison (FedAvg vs FedProx)

Test the FedProx algorithm with proximal regularization to improve convergence on non-IID data:

### Single comparison

```bash
# Clean logs and run FedAvg baseline
rm -rf logs/; mkdir logs
export SEED=42
python server.py --rounds 3 --aggregation fedavg --server_address 127.0.0.1:8099 &
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2 --fedprox_mu 0.0 &
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2 --fedprox_mu 0.0 &
wait

# Run FedProx with regularization
python server.py --rounds 3 --aggregation fedavg --server_address 127.0.0.1:8098 &
python client.py --server_address 127.0.0.1:8098 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2 --fedprox_mu 0.01 &
python client.py --server_address 127.0.0.1:8098 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2 --fedprox_mu 0.01 &
wait
```

### Matrix comparison script

```bash
# Test multiple α (non-IID levels) and μ (regularization strengths)
export ALPHA_VALUES="0.1,0.5" MU_VALUES="0.0,0.01,0.1" ROUNDS=5 LOGDIR="./fedprox_comparison"
bash scripts/compare_fedprox_fedavg.sh

# Generate analysis plots and thesis tables
python scripts/analyze_fedprox_comparison.py --artifacts_dir ./fedprox_comparison --output_dir ./fedprox_analysis
```

**Parameters**:

- `--fedprox_mu 0.0`: Standard FedAvg (no regularization)
- `--fedprox_mu 0.01`: Light FedProx regularization
- `--fedprox_mu 0.1`: Strong FedProx regularization

---

## 6.5) Personalization: Client-level model adaptation

After federated training completes, each client can optionally fine-tune the global model on its local data to improve local performance. This is useful in heterogeneous (non-IID) environments where each client has unique traffic patterns.

### Enable personalization

```bash
# Run FL training with 2 local epochs, then 3 personalization epochs
python server.py --rounds 5 --aggregation fedavg --server_address 127.0.0.1:8099 &
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2 --local_epochs 2 --personalization_epochs 3 &
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2 --local_epochs 2 --personalization_epochs 3 &
wait
```

**Key points:**

- Personalization happens **after** each FL round, locally on the client
- The **global model weights** are sent back to the server (personalized weights stay local)
- Each client logs both global and personalized performance metrics
- Useful for non-IID data where clients have different data distributions

### Metrics logged

When `--personalization_epochs > 0` and `D2_EXTENDED_METRICS=1`, client CSVs include:

- `macro_f1_global`: F1 score of global model before personalization
- `macro_f1_personalized`: F1 score after local fine-tuning
- `benign_fpr_global`: False positive rate of global model
- `benign_fpr_personalized`: False positive rate after personalization
- `personalization_gain`: Improvement from personalization (`personalized - global`)

### Example: Compare with and without personalization

```bash
rm -rf logs/; mkdir logs

# Baseline: No personalization
export SEED=42 D2_EXTENDED_METRICS=1
python server.py --rounds 3 --aggregation fedavg --server_address 127.0.0.1:8099 &
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2 --partition_strategy dirichlet --dirichlet_alpha 0.1 --personalization_epochs 0 &
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2 --partition_strategy dirichlet --dirichlet_alpha 0.1 --personalization_epochs 0 &
wait

# With personalization
python server.py --rounds 3 --aggregation fedavg --server_address 127.0.0.1:8098 &
python client.py --server_address 127.0.0.1:8098 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2 --partition_strategy dirichlet --dirichlet_alpha 0.1 --personalization_epochs 3 &
python client.py --server_address 127.0.0.1:8098 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2 --partition_strategy dirichlet --dirichlet_alpha 0.1 --personalization_epochs 3 &
wait

# Compare metrics
cat logs/client_0_metrics.csv | grep -v "^client_id" | cut -d',' -f29,30,33
# Columns: macro_f1_global, macro_f1_personalized, personalization_gain
```

### When personalization helps

Personalization shows **positive gains** when:

1. **Highly heterogeneous clients** (use `--dirichlet_alpha 0.05` or lower)
2. **Protocol-based partitioning** where each client sees specific attack types
3. **Sufficient personalization epochs** (5-10 epochs recommended)
4. **Appropriate learning rate** (0.01-0.02 works well)
5. **Global model not fully converged** (room for local adaptation)

**When to expect zero gains (this is correct behavior!):**

- IID data (`alpha=1.0` or uniform partitioning)
- Stratified train/test splits (maintains same class distribution)
- Global model already achieves >95% F1

**Latest real-data experiments (2025-10-07):**

- `UNSW, α=0.1, 5 epochs, lr=0.01` → mean gain **+7.0%** (client 2: +17%)
- `UNSW, α=0.05, 10 epochs, lr=0.01` → skewed shard gain **+4.5%**, other shards already saturated
- `UNSW, α=1.0, 5 epochs, lr=0.01` → mean gain **+0.25%** (IID ≈ zero)
- `CIC sample` (single-class shards) → global and personalized F1 both **1.0** (no headroom)

Full tables and log paths are documented in `docs/personalization_investigation.md` (`logs_debug/`).

**Troubleshooting:**

```bash
# Enable debug logging to diagnose zero-gain issues
export DEBUG_PERSONALIZATION=1
python client.py --personalization_epochs 5 ...

# Expected output:
# [Client 0] Personalization R1: Starting with 5 epochs, global F1=0.7234, ...
# [Client 0] After epoch 1: weight_norm=5.5123, delta=0.002341
# [Client 0] Personalization results: global_F1=0.7234, personalized_F1=0.7456, gain=0.022200
#
# If gain < 0.001, you'll see:
# [Client 0] WARNING: Near-zero gain detected!
# Possible causes: (1) train/test same distribution, (2) insufficient epochs, (3) LR too low
```

**Diagnostic tools:**

```bash
# Analyze train/test data distributions
python scripts/analyze_data_splits.py --dataset unsw --data_path data/unsw/unsw_nb15_sample.csv --alpha 0.1

# Run comprehensive diagnostic experiments
python scripts/debug_personalization.py --dataset unsw --num_clients 3
```

See [docs/personalization_investigation.md](docs/personalization_investigation.md) for detailed investigation findings.

---

## 6.6) Multi-class attack detection

The framework supports multi-class attack detection (e.g., 8+ attack types) in addition to binary classification (BENIGN vs attack). Multi-class support enables per-attack-type performance analysis.

### Synthetic multi-class experiments

Use the `--num_classes` parameter to test multi-class scenarios:

```bash
# 8-class synthetic experiment (simulates DoS, DDoS, PortScan, etc.)
python server.py --rounds 5 --aggregation fedavg --server_address 127.0.0.1:8080 &

python client.py \
  --server_address 127.0.0.1:8080 \
  --dataset synthetic \
  --samples 2000 \
  --features 20 \
  --num_classes 8 \
  --client_id 0 \
  --num_clients 2 \
  --partition_strategy dirichlet \
  --alpha 0.1 &

python client.py \
  --server_address 127.0.0.1:8080 \
  --dataset synthetic \
  --samples 2000 \
  --features 20 \
  --num_classes 8 \
  --client_id 1 \
  --num_clients 2 \
  --partition_strategy dirichlet \
  --alpha 0.1 &

wait
```

### Per-class metrics

When using extended metrics (`D2_EXTENDED_METRICS=1`), the following per-class metrics are logged:

- **`f1_per_class_after`**: F1-score for each class (JSON format: `{"0": 0.92, "1": 0.88, ...}`)
- **`precision_per_class`**: Precision for each class
- **`recall_per_class`**: Recall for each class

Example:

```bash
export D2_EXTENDED_METRICS=1
# Run experiment as above, then inspect metrics
cat logs/client_0_metrics.csv | grep -v "^client_id" | cut -d',' -f13,14,15
# Columns: f1_per_class_after, precision_per_class, recall_per_class
```

### Real multi-class datasets

For CIC-IDS2017 and UNSW-NB15, `num_classes` is automatically detected from the dataset labels. No manual configuration needed.

```bash
# CIC-IDS2017 multi-class (8 attack types + BENIGN)
python client.py \
  --dataset cic \
  --data_path data/cic/cic_ids2017_multiclass.csv \
  --num_clients 3 \
  --client_id 0 \
  --partition_strategy dirichlet \
  --alpha 0.1

# num_classes automatically set to 9 (8 attacks + BENIGN)
```

### Per-dataset encoder architecture

To better handle CIC and UNSW feature spaces we added a per-dataset encoder that maps raw features into a shared latent space before the global classifier. Enable it explicitly with `--model_arch per_dataset_encoder` or simply rely on the default `auto` mode (which activates the encoder for CIC/UNSW and keeps `SimpleNet` for synthetic clients).

```bash
python client.py \
  --dataset cic \
  --data_path data/cic/cic_ids2017_multiclass.csv \
  --num_clients 5 --client_id 0 \
  --partition_strategy dirichlet --alpha 0.1 \
  --model_arch per_dataset_encoder \
  --encoder_latent_dim 256
```

Dataset-specific defaults:

- **CIC-IDS2017**: encoder hidden layers 768→384→192, latent 256, shared head 192→96→logits, dropout 0.25.
- **UNSW-NB15**: encoder hidden layers 512→256, latent 192, shared head 128→64→logits, dropout 0.2.

Use `--encoder_latent_dim` to override the latent size for ablations; passing `0` (default) keeps the dataset heuristic.

---

## 7) Real datasets (UNSW‑NB15, CIC‑IDS2017, Edge‑IIoTset)

Important rule: all clients connected to the same server must use the same dataset and preprocessing settings.
Do not mix synthetic with UNSW/CIC (or different feature configs) on the same server run, or you will get a
“state_dict size mismatch” error.

### 7.1 Lightweight samples shipped in-repo

Nightly CI consumes real UNSW-NB15 and CIC-IDS2017 slices that live under `datasets/real/*.csv.gz`.
To materialize them locally, run:

```bash
python scripts/setup_real_datasets.py
```

This inflates the archives into `data/unsw/unsw_nb15_sample.csv` and `data/cic/cic_ids2017_sample.csv`.
Feel free to regenerate larger or different samples with the helper scripts below—just remember to update the
archives if CI should pick them up.

### 7.2 UNSW‑NB15 (Dirichlet non‑IID; 3+ clients)

Prepare a fresh sample if you need a different size:

```bash
cd <ABS_PATH>
mkdir -p data/unsw
python scripts/prepare_unsw_sample.py \
  --input data/unsw/UNSW_NB15_training-set.csv \
  --output data/unsw/UNSW_NB15_training-set.sample.csv \
  --frac 0.10 --seed 42
```

### 7.3 Edge‑IIoTset (IoT/IIoT; 14 attack types, 15 classes)

Edge‑IIoTset (2022) is integrated with three tiers and dedicated preprocessing fixes to keep memory in check.

Tiers (materialized by `scripts/setup_real_datasets.py` if present, or by the sample prep script below):
- `edge-iiotset-quick` (~50k rows, CI smoke)
- `edge-iiotset-nightly` (~500k rows, scheduled runs)
- `edge-iiotset-full` (~1.7M rows, full thesis runs)

Prepare or regenerate tiers from the raw Kaggle/IEEE drop (place CSVs under `datasets/edge-iiotset/Edge-IIoTset dataset/`):

```bash
python scripts/prepare_edge_iiotset_samples.py --tier all \
  --source "datasets/edge-iiotset/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv" \
  --output-dir data/edge-iiotset --seed 42
```

Run comparative experiments on the quick tier (works on laptop):

```bash
python scripts/comparative_analysis.py --dataset edge-iiotset-quick --num_clients 5 --rounds 5 \
  --dimension aggregation --aggregation_methods fedavg,median,krum,bulyan
python scripts/plot_metrics.py --run_dir runs --output_dir runs/plots_edge_quick
```

Notes:
- Preprocessing drops high-cardinality identifier columns to avoid OOM (see `docs/EDGE_IIOTSET_PREPROCESSING_FIX.md`).
- For full-scale runs, use chunked loader via `scripts/setup_real_datasets.py` or the prep script above; ensure enough RAM/disk.

---

## 8) Experimental Results & Performance Analysis

### Key Findings Summary

Synthesis of the committed artifacts (see [`PERFORMANCE_COMPARISON_TABLES.md`](PERFORMANCE_COMPARISON_TABLES.md)):

- **Robust aggregation under attack:** With 30 % Byzantine clients, macro‑F1 remains at 0.854 for Bulyan (−14 %) and 0.894 for Median (−10 %), while FedAvg drops to 0.675 (−75 %). Source: `results/comparative_analysis/attack_resilience_stats.csv`.
- **Heterogeneity control:** FedProx with μ = 0.1 at Dirichlet α = 0.05 yields a 2.67× lower L2 drift than FedAvg, at ~3 % additional runtime. Smaller μ = 0.01 still improves stability 1.35× with ~1 % overhead. Source: `analysis/fedprox_nightly/fedprox_comparison_summary.json`.
- **Personalization gains:** Client fine‑tuning delivers an average +0.035 macro‑F1 across 30 clients, with +0.060 on CIC‑IDS2017 slices and +0.018 on UNSW‑NB15. Source: `analysis/personalization/personalization_summary.json`.
- **Privacy–utility trade-off:** Client‑side DP noise σ = 0.7 (ε≈1.5) preserves macro‑F1≈0.79 compared to the 0.89 baseline, an ~11 % drop while adding differential privacy guarantees. Source: `results/privacy_check/privacy_utility_curve.csv`.

### Performance Comparison Tables

| Aggregation | Macro‑F1 (benign) | Macro‑F1 (30 % adv.) | Δ at 30 % | Source                                                     |
| ----------- | ----------------- | -------------------- | --------- | ---------------------------------------------------------- |
| FedAvg      | 1.000             | 0.675                | −74.5 %   | `results/comparative_analysis/attack_resilience_stats.csv` |
| Krum        | 0.969             | 0.787                | −18.8 %   | `results/comparative_analysis/attack_resilience_stats.csv` |
| Bulyan      | 0.994             | 0.854                | −14.1 %   | `results/comparative_analysis/attack_resilience_stats.csv` |
| Median      | 0.999             | 0.894                | −10.5 %   | `results/comparative_analysis/attack_resilience_stats.csv` |

Additional tables (FedProx stability, personalization breakdown, privacy curve) live in [`PERFORMANCE_COMPARISON_TABLES.md`](PERFORMANCE_COMPARISON_TABLES.md).

### Experimental Dimensions

1. **Aggregation Comparison:** Robust vs non-robust strategies on CIC-IDS2017 with Byzantine clients (fractions 0 %, 10 %, 30 %).
2. **Heterogeneity Impact:** Dirichlet splits α ∈ {0.05, 0.1, 0.5} comparing FedAvg and FedProx (μ ∈ {0, 0.01, 0.1}).
3. **Attack Resilience:** Bounded-gradient adversaries vs honest updates across 5 seeds (reference: `attack_experiments_*.log` and stats CSV).
4. **Privacy–Utility:** Gaussian DP noise sweep (σ ∈ {0, 0.7}) with derived ε values.
5. **Personalization Benefits:** Post-round fine‑tuning epochs {0, 3, 5, 10} on CIC-IDS2017 and UNSW-NB15 partitions.

---

## 9) Visualization Gallery & Plot References

### Thesis Plots Directory

All publication-ready plots are organized in [`plots/thesis/`](plots/thesis/):

- **Latest Plots**: [`plots/thesis/2025-10-21/`](plots/thesis/2025-10-21/)
- **Heterogeneity Impact**: [`heterogeneity_comparison.png`](plots/thesis/2025-10-21/thesis-comparative/heterogeneity_comparison.png)
- **FedProx Analysis**: [`fedprox_heterogeneity_analysis.png`](plots/thesis/2025-10-21/thesis-comparative/fedprox_heterogeneity_analysis.png)

**Complete Plot Collection**:

- **Aggregation Comparison**: [`plots/thesis/2025-10-05/thesis-comparative/aggregation_comparison.png`](plots/thesis/2025-10-05/thesis-comparative/aggregation_comparison.png)
- **Attack Resilience**: [`plots/thesis/2025-10-05/thesis-comparative/attack_resilience.png`](plots/thesis/2025-10-05/thesis-comparative/attack_resilience.png)
- **Personalization Benefits**: [`plots/thesis/2025-10-20/thesis-comparative/personalization_benefit.png`](plots/thesis/2025-10-20/thesis-comparative/personalization_benefit.png)
- **Privacy-Utility Tradeoff**: [`plots/thesis/2025-10-20/thesis-comparative/privacy_utility.png`](plots/thesis/2025-10-20/thesis-comparative/privacy_utility.png)

### FedProx Analysis Plots

- **Performance Comparison**: [`analysis/fedprox_nightly/fedprox_performance_plots.png`](analysis/fedprox_nightly/fedprox_performance_plots.png)
- **Summary Data**: [`analysis/fedprox_nightly/fedprox_comparison_summary.json`](analysis/fedprox_nightly/fedprox_comparison_summary.json)
- **LaTeX Tables**: [`analysis/fedprox_nightly/fedprox_thesis_tables.tex`](analysis/fedprox_nightly/fedprox_thesis_tables.tex)

### Personalization Analysis

- **Gains Visualization**: [`analysis/personalization/personalization_gains_analysis.png`](analysis/personalization/personalization_gains_analysis.png)
- **Summary Statistics**: [`analysis/personalization/personalization_summary.json`](analysis/personalization/personalization_summary.json)
- **LaTeX Tables**: [`analysis/personalization/personalization_gains_table.tex`](analysis/personalization/personalization_gains_table.tex)

### Interactive Plot Browser

Browse all plots chronologically: [`plots/index.html`](plots/index.html)

---

## 10) Quick Links to Analysis & Results

### Experimental Data

- **Aggregated Outputs**: [`results/`](results/) – Plot-ready CSV/PNG bundles
- **Comparative Run Logs**: `comparative-analysis-*/runs/` – per-seed metrics (FedAvg, FedProx, personalization)
- **Analysis Scripts**: [`scripts/`](scripts/) – Plot generation and analysis tools
- **Documentation**: [`docs/`](docs/) – Detailed experimental methodology

### Key Analysis Files

- **Performance Tables**: [`PERFORMANCE_COMPARISON_TABLES.md`](PERFORMANCE_COMPARISON_TABLES.md)
- **FedProx Analysis**: [`analysis/fedprox_nightly/`](analysis/fedprox_nightly/)
- **Personalization Study**: [`analysis/personalization/`](analysis/personalization/)
- **Notebook Walkthrough**: [`analysis/notebooks/performance_overview.ipynb`](analysis/notebooks/performance_overview.ipynb)
- **Threat Model**: [`docs/threat_model.md`](docs/threat_model.md)

### Reproducibility

- **Experiment Bundles**: Aggregated CSV/JSON outputs under `results/` and `analysis/` (see comparative-analysis folders)
- **Seed Control**: Reported runs use documented seeds (42, 43, 44, 101–505 depending on study)
- **Data Sources**: UNSW-NB15 (82k samples) and CIC-IDS2017 datasets

---

## 11) Troubleshooting

- **Deprecation warnings (Flower)**: you might see messages about `start_server`/`start_client` being deprecated.
  This demo targets flwr==1.21.0 and is known to work despite the warnings.

- **Address already in use**: change the port (e.g., to 8100) and pass the same port to the clients.

  ```bash
  # Find what is using the port 8099 (macOS/Linux)
  lsof -i :8099
  ```

- **CSV plotting errors** ("Expected X fields, saw Y"): Clean logs directory between different demo runs.

  ```bash
  rm -rf logs/; mkdir logs
  ```

  This happens when CSV files accumulate data from runs with different column structures.

- State dict size mismatch: all clients in a given run must use the same dataset and preprocessing
  (do not mix synthetic with UNSW/CIC in the same server run).

- File not found for dataset: verify your `--data_path` exists. If your file is `.gz`, decompress or
  pass the correct path.

- Plots not showing: this script saves `.png` files; no GUI needed. If you run headless and see backend errors,
  try: `export MPLBACKEND=Agg` before running plotting scripts.

- Permissions for scripts: if `verify_readme.sh` is not executable, run `chmod +x scripts/verify_readme.sh`.

- CPU vs GPU: no GPU required. Torch CPU build is sufficient for the demo.

---

## 12) Project structure

- `server.py` – Flower server with FedAvg and robust aggregation options (`median`, `krum`, simplified `bulyan`).
  - For `fedavg`, aggregation is sample‑size weighted; robust methods are intentionally unweighted.
- `client.py` – PyTorch `NumPyClient` with a small MLP; supports synthetic, UNSW‑NB15, and CIC‑IDS2017 datasets
  with IID/Dirichlet/protocol partitions. Includes FedProx proximal regularization via `--fedprox_mu`.
- `data_preprocessing.py` – CSV loaders, preprocessing (StandardScaler + OneHotEncoder), partitioning
  (iid/dirichlet/protocol), and DataLoader builders.
- `robust_aggregation.py` – Aggregation method enum and robust implementations (Median, Krum, simplified Bulyan).
- `scripts/verify_readme.sh` – Non‑interactive verification for automated demo sanity checks.
- `scripts/plot_metrics.py` – Generate server/client metric plots from CSV logs.
- `scripts/summarize_metrics.py` – Emit a compact JSON summary of client metrics.
- `scripts/compare_fedprox_fedavg.sh` – Matrix comparison script for FedAvg vs FedProx across different parameters.
- `scripts/analyze_fedprox_comparison.py` – Analysis tool for generating thesis-ready plots and LaTeX tables.
- `requirements.txt` – Python dependencies.

---

## 13) Privacy & robustness disclosure (D2 scope)

For a comprehensive threat model including adversary assumptions, attack scenarios, and defense mechanisms, see [docs/threat_model.md](docs/threat_model.md).

**Current implementation status:**

- Differential Privacy (scaffold): client‑side clipping with Gaussian noise applied to the model update
  before sending. This is not DP‑SGD and does not include privacy accounting.
- Secure Aggregation (stub): toggle provided and status logged, but updates are not cryptographically masked.
  Integration of secure summation/masking is planned for a later milestone.

---

## 14) D2 Runbook (experiment workflow & artifact map)

For reproducibility and advisor-facing documentation, this section provides an overview of the end-to-end workflow for running federated learning experiments.

### Quick Workflow

```
Preprocess → Run Experiments → Generate Plots → Summarize → Commit Artifacts
```

### Step 1: Preprocess Data

```bash
python scripts/setup_real_datasets.py
```

### Step 2: Run Experiments

```bash
# Automated comparative analysis (recommended)
python scripts/comparative_analysis.py \
  --dimension heterogeneity \
  --dataset unsw \
  --output_dir results/comparative_analysis/unsw
```

Available dimensions: `aggregation`, `attack`, `heterogeneity`, `heterogeneity_fedprox`, `privacy`, `personalization`

### Step 3: Generate Plots

```bash
python scripts/generate_thesis_plots.py \
  --dimension heterogeneity \
  --runs_dir runs \
  --output_dir results/thesis_plots/heterogeneity
```

### Step 4: Summarize Results

```bash
python scripts/summarize_metrics.py \
  --run_dir runs/comp_fedavg_alpha0.1_adv0_dp0_pers0_seed42 \
  --output runs/comp_fedavg_alpha0.1_adv0_dp0_pers0_seed42/summary.json
```

### Artifact Structure

Each experiment run creates a directory `runs/comp_{config}/` containing:

- `config.json` - Experiment configuration
- `metrics.csv` - Server-level metrics per round
- `client_N_metrics.csv` - Per-client metrics per round
- `summary.json` - Aggregated statistics (generated post-run)
- `*.log` - Execution logs

### Detailed Documentation

For complete workflow details, artifact specifications, CI/CD instructions, and troubleshooting, see [docs/d2_runbook.md](docs/d2_runbook.md).
