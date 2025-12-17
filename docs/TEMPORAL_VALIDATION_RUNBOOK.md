# Temporal Validation Protocol: Cluster Runbook

**Created:** 2025-12-17
**Purpose:** Step-by-step guide to running the temporal validation experiments on the JMU cluster.

---

## Overview

This runbook implements the experimental protocol defined in `TEMPORAL_VALIDATION_PROTOCOL.md`:

| Phase | Jobs | Seeds | Purpose |
|-------|------|-------|---------|
| Tuning | 231 | 42, 43, 44 | Select optimal mu* per alpha |
| Evaluation | 70 | 45-49 | Report final results with CIs |
| **Total** | **301** | - | ~2.1 hours on 17 nodes |

---

## Pre-flight Checklist

### 1. Verify cluster code is up-to-date

```bash
ssh stu 'cd ~/federated-ids && git log --oneline -1'
```

Expected: `a6b03b1c feat(validation): implement temporal validation...`

### 2. Verify temporal validation tests pass

```bash
ssh stu 'cd ~/federated-ids && source venv/bin/activate && python -m pytest tests/test_temporal_validation.py -v'
```

Expected: `20 passed`

### 3. Verify dataset symlink

```bash
ssh stu 'ls -la ~/federated-ids/data/edge-iiotset/edge_iiotset_full.csv'
```

Should point to `/scratch/$USER/datasets/edge-iiotset/edge_iiotset_full.csv`

### 4. Create results directory

```bash
ssh stu 'mkdir -p /scratch/$USER/results/temporal_validation'
```

---

## Phase 1: Tuning (231 jobs)

### Submit tuning jobs

```bash
ssh stu 'cd ~/federated-ids && bash scripts/slurm/submit_temporal_validation.sh'
```

This submits:
- **FedProx tuning:** 210 jobs (7 alphas x 10 mu x 3 seeds)
- **FedAvg baseline:** 21 jobs (7 alphas x 3 seeds)

### Monitor progress

```bash
# On cluster
ssh stu 'squeue -u $USER'
ssh stu 'squeue -u $USER | wc -l'

# Watch every 30 seconds
ssh stu 'watch -n 30 "squeue -u \$USER | tail -20"'
```

### Sync runs locally (separate terminal)

```bash
cd federated-ids
bash scripts/sync_cluster_runs.sh
```

Or one-time sync:

```bash
bash scripts/sync_cluster_runs.sh --once
```

### Expected runtime

| Concurrent Jobs | Est. Time |
|-----------------|-----------|
| 1 (sequential) | ~27 hours |
| 17 (full cluster) | ~1.6 hours |

---

## Phase 2: Mu* Selection

After tuning completes, analyze results to select optimal mu per alpha.

### Run selection script

```bash
ssh stu 'cd ~/federated-ids && source venv/bin/activate && python scripts/select_mu_star.py --runs_dir runs'
```

### Expected output

```
============================================================
Mu* Selection Results (Tuning Seeds: 42, 43, 44)
============================================================
Alpha      mu*        Mean Val F1      Seeds
------------------------------------------------------------
0.02       0.050      0.8234           3
0.05       0.020      0.8456           3
0.10       0.010      0.8678           3
...
inf        0.002      0.9012           3
============================================================
```

### Review selections

- Check if mu* varies systematically with alpha
- Verify all alphas have complete tuning data (3 seeds each)
- Save `mu_star_selection.json` for reproducibility

---

## Phase 3: Evaluation (70 jobs)

### Create evaluation script with selected mu*

After reviewing mu* selections, create the evaluation script:

```bash
# On cluster
cd ~/federated-ids

# Edit the evaluation script to use selected mu* values
# (or use the dynamic version that reads mu_star_selection.json)
```

### Submit evaluation jobs

```bash
ssh stu 'cd ~/federated-ids && sbatch --array=0-69%17 scripts/slurm/temporal_validation_eval.sbatch'
```

### Expected runtime

| Concurrent Jobs | Est. Time |
|-----------------|-----------|
| 17 (full cluster) | ~30 minutes |

---

## Analysis & Reporting

### Generate results table

```bash
ssh stu 'cd ~/federated-ids && source venv/bin/activate && python scripts/generate_temporal_results.py --runs_dir runs'
```

### Expected output format

```
| Alpha | FedAvg macro_f1 (95% CI) | FedProx macro_f1 (95% CI) | mu* | Delta | p-value |
|-------|--------------------------|---------------------------|-----|-------|---------|
| 0.02  | 0.812 (0.798, 0.826)    | 0.834 (0.821, 0.847)     | 0.05 | +0.022 | 0.023  |
...
```

---

## Troubleshooting

### Job fails with "ModuleNotFoundError"

```bash
# Check venv activation in sbatch script
ssh stu 'source ~/federated-ids/venv/bin/activate && python -c "import flwr; print(flwr.__version__)"'
```

### Jobs stuck in PENDING

```bash
# Check cluster load
ssh stu 'sinfo -N'

# Check job reason
ssh stu 'squeue -u $USER -o "%.10i %.9P %.8j %.8u %.8T %.10M %.6D %R"'
```

### Missing metrics.csv

```bash
# Check for incomplete runs
ssh stu 'cd ~/federated-ids && find runs -maxdepth 2 -name config.json | wc -l'
ssh stu 'cd ~/federated-ids && find runs -maxdepth 2 -name metrics.csv | wc -l'
```

### Out of memory

Reduce concurrent jobs:
```bash
MAX_CONCURRENT=8 bash scripts/slurm/submit_temporal_validation.sh
```

---

## Quick Reference

### Key paths on cluster

| Purpose | Path |
|---------|------|
| Repository | `~/federated-ids` |
| Virtual env | `~/federated-ids/venv` |
| Dataset | `data/edge-iiotset/edge_iiotset_full.csv` |
| Results | `/scratch/$USER/results/temporal_validation/` |
| Runs | `~/federated-ids/runs/` |

### Slurm scripts

| Script | Jobs | Purpose |
|--------|------|---------|
| `temporal_validation_tuning.sbatch` | 210 | FedProx tuning |
| `temporal_validation_baseline.sbatch` | 21 | FedAvg baseline |
| `submit_temporal_validation.sh` | 231 | Submit all tuning |

### Local sync

```bash
# Continuous sync (every 60s)
bash scripts/sync_cluster_runs.sh

# One-time sync
bash scripts/sync_cluster_runs.sh --once

# Custom interval
INTERVAL=30 bash scripts/sync_cluster_runs.sh
```

---

## Experiment Matrix Summary

### Tuning Phase (231 jobs)

| Dimension | Values | Count |
|-----------|--------|-------|
| Alpha | 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf | 7 |
| Mu (FedProx) | 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0 | 10 |
| Seeds | 42, 43, 44 | 3 |
| **FedProx configs** | 7 x 10 x 3 | **210** |
| **FedAvg configs** | 7 x 1 x 3 | **21** |

### Evaluation Phase (70 jobs)

| Dimension | Values | Count |
|-----------|--------|-------|
| Alpha | 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf | 7 |
| Mu | mu*[alpha] (selected from tuning) | 1 per alpha |
| Seeds | 45, 46, 47, 48, 49 | 5 |
| **FedProx configs** | 7 x 1 x 5 | **35** |
| **FedAvg configs** | 7 x 1 x 5 | **35** |

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-17 | Initial runbook created |
