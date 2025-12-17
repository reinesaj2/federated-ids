# Cluster Runbook: Full Edge-IIoTset Heterogeneity (FedAvg vs FedProx)

**Goal:** run the **full Edge-IIoTset dataset** on the CS470 Slurm cluster for the **same heterogeneity + FedProx parameter space already exercised in nightly presets**, but at **full tier scale**, queued efficiently so the cluster stays busy in the background.

## Scope (what we will run)

### Dataset (full)
- **Dataset:** `edge-iiotset-full`
- **Cluster path (authoritative):** `/scratch/$USER/datasets/edge-iiotset/edge_iiotset_full.csv`
- **Run location:** `/scratch/$USER/federated-ids` (so `runs/` stays on scratch, not NFS home)

### Full-tier runtime settings (fixed)
- **Clients:** `10`
- **Rounds:** `20`
- **Local epochs:** `1` (hardcoded by `scripts/comparative_analysis.py`)

### Parameter space (match presets)
- **Heterogeneity alphas:** `[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf]`
- **FedProx mus:** `[0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2]` (exclude `0.0`; that’s the FedAvg baseline)
- **Seeds (default):** `42,43,44,45,46`

### Experiment counts (with defaults)
- **FedAvg baseline sweep:** `7 alphas × 5 seeds = 35 jobs`
- **FedProx sweep:** `7 alphas × 8 mus × 5 seeds = 280 jobs`
- **Total:** `315 jobs`

## Cluster constraints we design around

- **Partition:** `cs` (includes `compute21-29` and `gpu01-08`)
- **Total nodes available:** `17` (16 CPU cores each)
- **Max array size:** `1001` (315 is safe)
- **Memory reporting is broken (`MEMORY=1`)**: do **not** specify `--mem` in Slurm scripts.

## Preflight (one-time)

### 1) Ensure repo + venv are on scratch
```bash
ssh cluster
cd /scratch/$USER/federated-ids
```

### 2) Ensure the dataset is accessible via the repo default path
`scripts/comparative_analysis.py` defaults to `data/edge-iiotset/edge_iiotset_full.csv` for `edge-iiotset-full`.

On the cluster, create a symlink so we do not need `--data_path` overrides:
```bash
cd /scratch/$USER/federated-ids
mkdir -p data/edge-iiotset
ln -sf /scratch/$USER/datasets/edge-iiotset/edge_iiotset_full.csv data/edge-iiotset/edge_iiotset_full.csv
ls -lah data/edge-iiotset/edge_iiotset_full.csv
```

### 3) Sanity smoke (already-proven path)
```bash
cd /scratch/$USER/federated-ids
sbatch scripts/slurm/iiot_full_heterogeneity_smoke_10c_20r.sbatch
```

> Note: This smoke uses `OHE_SPARSE=1` so Edge-IIoTset preprocessing stays sparse and avoids dense OOMs.

## Queueing strategy (keep cluster busy, avoid node contention)

### Principle: 1 experiment = 1 Slurm job = 1 node
Each experiment spawns a server + 20 clients (21 Python processes). The simplest stable approach is to request a full node:
- `#SBATCH --nodes=1`
- `#SBATCH --cpus-per-task=16`
- `#SBATCH --exclusive`

### Use Slurm arrays with a concurrency cap
We submit arrays and cap concurrency:
- **Sequential (this plan):** `--array=0-279%1` runs **280 jobs total**, **1 at a time**.
- **Parallel (only after validation):** raise the cap (e.g. `%17`) to use more nodes.

### Use `--split-total/--split-index` so each array task runs exactly one config
`scripts/comparative_analysis.py` supports deterministic splitting:
- `--split-total N` + `--split-index i` selects exactly the *i*‑th config when `N` equals the total config count.

This avoids maintaining a separate manifest file and keeps the mapping reproducible.

## Recommended Slurm scripts (two arrays)

Create two batch scripts (names are suggestions) that differ only in `--dimension` and `--split-total`.

### Shared SBATCH skeleton (recommended)
```bash
#!/bin/bash
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/results/iiot_full/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/iiot_full/%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source /scratch/$USER/venvs/fedids-py311/bin/activate
cd /scratch/$USER/federated-ids

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" /scratch/$USER/results/iiot_full
```

### A) FedAvg baseline heterogeneity array (35 jobs)
Core command (inside `sbatch` script):
```bash
python scripts/comparative_analysis.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --num_clients 10 \
  --num_rounds 20 \
  --alpha-values 0.02,0.05,0.1,0.2,0.5,1.0,inf \
  --seeds 42,43,44,45,46 \
  --server_timeout 21600 \
  --client_timeout 21600 \
  --split-total 35 \
  --split-index "${SLURM_ARRAY_TASK_ID}"
```

Submit:
```bash
sbatch --array=0-34%1 scripts/slurm/iiot_full_heterogeneity_fedavg_array.sbatch
```

### B) FedProx heterogeneity sweep array (280 jobs)
Core command (inside `sbatch` script):
```bash
python scripts/comparative_analysis.py \
  --dimension heterogeneity_fedprox \
  --dataset edge-iiotset-full \
  --num_clients 10 \
  --num_rounds 20 \
  --alpha-values 0.02,0.05,0.1,0.2,0.5,1.0,inf \
  --fedprox-mu-values 0.002,0.005,0.01,0.02,0.05,0.08,0.1,0.2 \
  --seeds 42,43,44,45,46 \
  --server_timeout 21600 \
  --client_timeout 21600 \
  --split-total 280 \
  --split-index "${SLURM_ARRAY_TASK_ID}"
```

Submit:
```bash
sbatch --array=0-279%1 scripts/slurm/iiot_full_heterogeneity_fedprox_array.sbatch
```

## One-command submission (recommended)

From `/scratch/$USER/federated-ids`:
```bash
bash scripts/slurm/submit_iiot_full_heterogeneity.sh
```

Controls:
- `MAX_CONCURRENT=1 bash scripts/slurm/submit_iiot_full_heterogeneity.sh` (sequential; safest for memory)
- `MAX_CONCURRENT=17 bash scripts/slurm/submit_iiot_full_heterogeneity.sh` (parallel; use only after canary success)

## Monitoring + recovery

### Watch queue
```bash
squeue -u "$USER"
```

### Inspect a specific job
```bash
scontrol show job <JOBID> | head -n 80
```

### Find failures (no metrics.csv)
```bash
cd /scratch/$USER/federated-ids
find runs -maxdepth 2 -name metrics.csv -print | wc -l
find runs -maxdepth 2 -name config.json -print | wc -l
```

### Re-run only missing configs
Use the deterministic mapping: re-submit the same array, but restrict to the missing indices (Slurm supports comma-separated task lists like `--array=3,19,44`).

## Post-run validation + plots

From `/scratch/$USER/federated-ids`:
```bash
python scripts/validate_experiment_matrix.py --runs_dir runs
python scripts/generate_thesis_plots.py --dimension heterogeneity_fedprox --runs_dir runs
```

## Open choices (confirm before running at scale)
- **Seeds:** keep `{42..46}` (315 jobs) or expand to `{42..51}` (630 jobs)?
- **Node usage:** allow `gpu[01-08]` as CPU nodes, or exclude them?
- **Packing:** keep `--exclusive` (1 job/node) or allow multiple jobs/node (requires port/thread tuning)?
