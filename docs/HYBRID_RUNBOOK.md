# Hybrid Cross-Source Runbook (NeurIPS-Scale)

**Created:** 2025-12-18
**Purpose:** End-to-end guide for large-scale hybrid dataset experiments on the JMU cluster using `data/hybrid/hybrid_ids_dataset_full.csv.gz`.

---

## Overview

This runbook targets NeurIPS-scale robustness and statistical power. It emphasizes:

- Cross-source heterogeneity using the hybrid dataset (CIC + UNSW + Edge-IIoTset).
- Large alpha and mu grids with many seeds for tight confidence intervals.
- Reproducible staging, monitoring, and analysis.

| Phase                           | Jobs | Seeds | Purpose                               |
| ------------------------------- | ---- | ----- | ------------------------------------- |
| Audit                           | 1    | n/a   | Dataset integrity and metadata checks |
| Smoke                           | 1    | 1     | Verify pipeline end-to-end            |
| NeurIPS Sweep                   | 1400 | 20    | Full alpha x mu grid with robust CIs  |
| Robustness Extension (Optional) | 320+ | 20    | Attack resilience on hybrid data      |

---

## Pre-flight Checklist

### 1. Verify local dataset and audit

```bash
ls -lh data/hybrid/hybrid_ids_dataset_full.csv.gz
python scripts/verify_hybrid_dataset.py \
  --input data/hybrid/hybrid_ids_dataset_full.csv.gz
```

Expected: `passes_minimum_checks=true` and audit JSON at `data/hybrid/hybrid_ids_dataset_full.csv.gz.audit.json` (or `hybrid_ids_dataset_full.audit.json` if you prefer a fixed name).

### 2. Confirm hybrid dataset support is enabled

Hybrid runs assume the client can load the hybrid dataset and partition it in a source-aware way. If your branch does not yet support:

- `--dataset hybrid` in `client.py`, and
- source-aware partitioning using the `source_dataset` column (`--partition_strategy source`),

enable those first before launching the sweep.

### 3. Stage dataset to cluster

```bash
rsync -av data/hybrid/hybrid_ids_dataset_full.csv.gz cluster:/scratch/$USER/federated-ids/data/hybrid/
rsync -av data/hybrid/hybrid_ids_dataset_full.audit.json cluster:/scratch/$USER/federated-ids/data/hybrid/
```

Validate on cluster:

```bash
ssh cluster 'ls -lh /scratch/$USER/federated-ids/data/hybrid/hybrid_ids_dataset_full.csv.gz'
```

### 4. Verify environment

```bash
ssh cluster 'source /scratch/$USER/venvs/fedids-py311/bin/activate && python -c "import torch, flwr; print(torch.__version__, flwr.__version__)"'
```

---

## Phase 0: Smoke Test (Single Config)

Use the existing Slurm array script and run a single index:

```bash
ssh cluster 'cd /scratch/$USER/federated-ids && sbatch --array=0-0 scripts/slurm/hybrid_cross_source_array.sbatch'
```

Monitor:

```bash
ssh cluster 'squeue -u $USER'
```

---

## Phase 1: NeurIPS-Scale Hybrid Sweep (Full Grid)

### Target grid (robust + copious)

- **Clients:** 9 (3 per source: cic, unsw, iiot)
- **Rounds:** 20
- **Alphas:** 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf (7 values)
- **FedProx mu:** 0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5 (10 values)
- **Seeds:** 42-61 (20 values)

**Total jobs:** 7 x 10 x 20 = **1400**

### Update the Slurm array script

Before scaling, ensure the hybrid config generator is not hard-coded. The hybrid grid should respect `--alpha-values`, `--fedprox-mu-values`, and `--seeds` passed in. Remove any fixed alpha/mu lists and seed slicing so the full grid is honored.

Edit `scripts/slurm/hybrid_cross_source_array.sbatch` on the cluster to match the grid above:

```
NUM_CLIENTS=9
NUM_ROUNDS=20
ALPHA_VALUES="0.02,0.05,0.1,0.2,0.5,1.0,inf"
FEDPROX_MU_VALUES="0.0,0.002,0.005,0.01,0.02,0.05,0.08,0.1,0.2,0.5"
SEEDS="42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61"
SPLIT_TOTAL=1400
```

### Submit the sweep

```bash
ssh cluster 'cd /scratch/$USER/federated-ids && sbatch --array=0-1399%17 scripts/slurm/hybrid_cross_source_array.sbatch'
```

**Expected runtime:** ~14-28 hours at 17 concurrent jobs (depends on node load and dataset size).

### Progress checks

```bash
ssh cluster 'squeue -u $USER | wc -l'
ssh cluster 'cd /scratch/$USER/federated-ids && find runs -maxdepth 2 -name metrics.csv | wc -l'
```

---

## Phase 2: Robustness Extension (Optional, NeurIPS-Grade)

Attack-resilience on the hybrid dataset (FedAvg vs robust aggregators):

- **Aggregations:** fedavg, krum, bulyan, median
- **Adversary fraction:** 0%, 10%, 20%, 30%
- **Seeds:** 42-61 (20)
- **Clients:** 15 (5 per source; needed for Bulyan at higher adversary rates)

Estimated jobs: 4 x 4 x 20 = 320

Create a dedicated Slurm array (copy `scripts/slurm/hybrid_cross_source_array.sbatch`) and switch the command to `--dimension attack`. Then submit with an array sized to 320 jobs:

```bash
ssh cluster 'cd /scratch/$USER/federated-ids && sbatch --array=0-319%17 scripts/slurm/hybrid_attack_array.sbatch'
```

In `scripts/slurm/hybrid_attack_array.sbatch`, set:

```
NUM_CLIENTS=15
NUM_ROUNDS=20
SEEDS="42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61"
SPLIT_TOTAL=320
```

---

## Analysis & Reporting

### Sync runs locally (optional)

```bash
cd federated-ids
bash scripts/sync_cluster_runs.sh --once
```

### Generate plots

```bash
python scripts/generate_thesis_plots.py \
  --dimension heterogeneity_fedprox \
  --dataset hybrid \
  --runs_dir runs \
  --output_dir results/hybrid_neurips
```

### Summarize a representative run

```bash
python scripts/summarize_metrics.py \
  --run_dir runs/<RUN_DIR> \
  --output runs/<RUN_DIR>/summary.json
```

---

## Troubleshooting

### Jobs fail with ModuleNotFoundError

```bash
ssh cluster 'source /scratch/$USER/venvs/fedids-py311/bin/activate && python -c "import flwr; print(flwr.__version__)"'
```

### Out-of-memory / shared library errors

Reduce clients per node or lower concurrency:

```bash
sbatch --array=0-1399%8 scripts/slurm/hybrid_cross_source_array.sbatch
```

---

## Notes

- The hybrid dataset path is fixed by default in `scripts/comparative_analysis.py` as `data/hybrid/hybrid_ids_dataset_full.csv.gz`.
- Keep `FEDIDS_USE_OPACUS=1` enabled in Slurm scripts (even when DP is off).
- Use `/scratch/$USER` for all large artifacts to avoid NFS slowdowns.
