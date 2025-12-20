# Local Laptop Single-Configuration Runner

**Purpose:** Run individual Flower configurations locally for rapid iteration, debugging, and validation before committing to multi-hour AWS runs.

**Date:** November 17, 2025
**Branch:** `exp/iiot-experiments`

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Running a Single Configuration](#running-a-single-configuration)
5. [Dataset Tiers](#dataset-tiers)
6. [Monitoring and Validation](#monitoring-and-validation)
7. [Tracking Template](#tracking-template)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This runbook enables running **single Flower configurations** on a laptop with conservative resource settings. Unlike AWS or CI environments that run 10-20 workers in parallel, this approach uses **1 worker** to prevent memory exhaustion and thermal throttling.

**Use Cases:**

- Quick validation of experiment logic changes
- Debugging preprocessing or aggregation issues
- Testing new strategies before full AWS deployment
- Offline development without cloud dependencies

**Key Constraints:**

- Memory: Recommend 16GB RAM minimum for `full` tier; 8GB for `nightly` tier
- CPU: Expect thermal throttling on long runs; keep laptop plugged in and well-ventilated
- Runtime: Full tier configs can take 2-6 hours per config on laptop CPUs

---

## Prerequisites

### System Requirements

| Resource  | Minimum                      | Recommended |
| --------- | ---------------------------- | ----------- |
| RAM       | 8 GB                         | 16 GB       |
| Free Disk | 5 GB                         | 10 GB       |
| CPU Cores | 2                            | 4+          |
| OS        | macOS 10.15+ / Ubuntu 20.04+ | Latest      |

### Software Dependencies

- Python 3.13 (recommended)
- `virtualenv` or `venv`
- AWS CLI (if syncing datasets from S3)

---

## Environment Setup

### 1. Navigate to Repository

```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments
```

### 2. Create and Activate Virtual Environment

If not already created:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

If already exists, just activate:

```bash
source .venv/bin/activate
```

### 3. Verify Dataset

Check that the dataset tier you want to use exists:

```bash
ls -lh data/edge-iiotset/
```

Expected output:

```
edge_iiotset_quick.csv    (~26 MB)
edge_iiotset_nightly.csv  (~262 MB)
edge_iiotset_full.csv     (~934 MB)
```

If missing, download from S3:

```bash
# Quick tier
aws s3 cp s3://thesis-data-iiot-20251114/datasets/edge_iiotset_quick.csv data/edge-iiotset/

# Nightly tier
aws s3 cp s3://thesis-data-iiot-20251114/datasets/edge_iiotset_nightly.csv data/edge-iiotset/

# Full tier
aws s3 cp s3://thesis-data-iiot-20251114/datasets/edge_iiotset_full.csv data/edge-iiotset/
```

---

## Running a Single Configuration

### Command Template

```bash
python scripts/run_experiments_optimized.py \
  --dimension <DIMENSION> \
  --dataset <DATASET_TIER> \
  --dataset-type full \
  --workers 1 \
  --strategy <STRATEGY> \
  --seed <SEED> \
  --client-timeout-sec 7200 \
  > /tmp/laptop_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Parameter Reference

| Parameter              | Options                                                                | Description                                      |
| ---------------------- | ---------------------------------------------------------------------- | ------------------------------------------------ |
| `--dimension`          | `aggregation`, `heterogeneity`, `attack`, `privacy`, `personalization` | Thesis objective dimension                       |
| `--dataset`            | `edge-iiotset-quick`, `edge-iiotset-nightly`, `edge-iiotset-full`      | Dataset tier                                     |
| `--dataset-type`       | `full`                                                                 | Dataset type (always `full` for edge-iiotset)    |
| `--workers`            | `1`                                                                    | Number of parallel workers (ALWAYS 1 for laptop) |
| `--strategy`           | `fedavg`, `krum`, `bulyan`, `median`                                   | Aggregation strategy                             |
| `--seed`               | `42`, `43`, `44`, `45`, `46`                                           | Random seed for reproducibility                  |
| `--client-timeout-sec` | `7200`                                                                 | Max seconds per client (2 hours recommended)     |

### Example: Aggregation Dimension, Krum Strategy, Seed 42

```bash
python scripts/run_experiments_optimized.py \
  --dimension aggregation \
  --dataset edge-iiotset-nightly \
  --dataset-type full \
  --workers 1 \
  --strategy krum \
  --seed 42 \
  --client-timeout-sec 7200 \
  > /tmp/laptop_agg_krum_seed42.log 2>&1 &
```

**IMPORTANT:** Run in background using `&` to detach from terminal. This prevents the process from terminating if your shell session closes.

### Retrieve Background Job PID

After launching, note the process ID:

```bash
echo $!
```

Or find it later:

```bash
pgrep -af run_experiments_optimized
```

---

## Dataset Tiers

### Quick Tier (Smoke Testing)

- **File:** `edge_iiotset_quick.csv`
- **Size:** 26 MB
- **Rows:** ~50,000
- **Clients:** 3
- **Rounds:** 5
- **Expected Runtime:** 10-15 minutes

**Use for:** Quick logic validation, syntax checks, ensuring no crashes

### Nightly Tier (Development)

- **File:** `edge_iiotset_nightly.csv`
- **Size:** 262 MB
- **Rows:** ~500,000
- **Clients:** 6
- **Rounds:** 20
- **Expected Runtime:** 45-90 minutes

**Use for:** Iterative development, debugging aggregation logic, testing new features

### Full Tier (Publication Quality)

- **File:** `edge_iiotset_full.csv`
- **Size:** 934 MB
- **Rows:** ~1.7M (after 90% train split)
- **Clients:** 10
- **Rounds:** 50
- **Expected Runtime:** 2-6 hours (CPU dependent)

**Use for:** Final validation before AWS runs, local baseline collection

**WARNING:** Full tier may cause thermal throttling on laptops. Ensure good cooling and power supply.

---

## Monitoring and Validation

### Check Running Process

```bash
ps aux | grep run_experiments_optimized
```

### Monitor Log Output

```bash
tail -f /tmp/laptop_agg_krum_seed42.log
```

Press `Ctrl+C` to stop tailing (process continues running).

### Check System Resources

**CPU and Memory:**

```bash
top
```

Look for `python` process consuming CPU/memory.

**Disk Space:**

```bash
df -h .
```

Ensure sufficient space for `runs/` output.

### Verify Completion

After the process finishes (check with `ps`), inspect results:

```bash
# Find the generated run directory
ls -lt runs/ | head -5

# Check metrics file (should have header + num_rounds rows)
wc -l runs/<preset_name>/metrics.csv
head -3 runs/<preset_name>/metrics.csv

# Check server log for completion
tail -20 runs/<preset_name>/server.log
```

**Success Criteria:**

- `metrics.csv` has `num_rounds + 1` rows (header + 1 per round)
- Server log shows "aggregate_fit" messages for all rounds
- No Python tracebacks in log file

---

## Tracking Template

Create a tracking spreadsheet or markdown table to log each config:

### CSV Template

Save as `docs/laptop_run_tracker.csv`:

```csv
dimension,strategy,seed,dataset_tier,started_at,finished_at,runtime_min,metrics_rows,status,notes
aggregation,krum,42,nightly,2025-11-17 14:30,2025-11-17 15:45,75,21,SUCCESS,Baseline run
aggregation,fedavg,42,nightly,2025-11-17 16:00,2025-11-17 17:10,70,21,SUCCESS,
aggregation,bulyan,42,nightly,2025-11-17 17:15,,,,,RUNNING,
```

### Markdown Template

```markdown
## Laptop Run Log

| Dimension   | Strategy | Seed | Tier    | Started | Finished | Runtime | Status  | Notes    |
| ----------- | -------- | ---- | ------- | ------- | -------- | ------- | ------- | -------- |
| aggregation | krum     | 42   | nightly | 14:30   | 15:45    | 75 min  | SUCCESS | Baseline |
| aggregation | fedavg   | 42   | nightly | 16:00   | 17:10    | 70 min  | SUCCESS |          |
| aggregation | bulyan   | 42   | nightly | 17:15   | -        | -       | RUNNING |          |
```

---

## Troubleshooting

### Issue: Process Killed with Exit Code 137 (Memory)

**Cause:** Laptop ran out of RAM.

**Solutions:**

1. Use smaller tier: Switch from `full` to `nightly` or `quick`
2. Close other applications to free memory
3. Add swap space (Linux) or increase swap (macOS)
4. Reduce dataset size further with custom sampling

### Issue: Process Killed with Exit Code 143 (SIGTERM)

**Cause:** System sent termination signal, likely due to resource pressure.

**Solutions:**

1. Ensure laptop is plugged in (prevents sleep/hibernation)
2. Disable sleep mode during run
3. Check system logs for OOM killer activity

### Issue: Very Slow Performance / Thermal Throttling

**Cause:** CPU overheating, clock speed reduced.

**Solutions:**

1. Elevate laptop for better airflow
2. Use laptop cooling pad
3. Run during cooler times (evening)
4. Accept slower runtime (still faster than waiting for AWS slot)

### Issue: Port Already in Use

**Error:** `Address already in use: 8080`

**Cause:** Previous experiment still running or crashed without cleanup.

**Solution:**

```bash
# Find process using port
lsof -i :8080

# Kill it
kill -9 <PID>

# Or kill all Python processes
pkill -9 python
```

### Issue: Dataset Not Found

**Error:** `FileNotFoundError: data/edge-iiotset/edge_iiotset_full.csv`

**Solution:**

```bash
# Verify symlink
ls -la data/edge-iiotset

# Re-download if missing
aws s3 cp s3://thesis-data-iiot-20251114/datasets/edge_iiotset_full.csv data/edge-iiotset/
```

### Issue: ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'flwr'`

**Cause:** Virtual environment not activated or dependencies not installed.

**Solution:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## S3 Sync (Optional)

After successful local run, back up results to S3:

```bash
aws s3 sync runs/ s3://thesis-data-iiot-20251114/artifacts/local-runs --only-show-errors
aws s3 sync results/ s3://thesis-data-iiot-20251114/artifacts/local-results --only-show-errors
```

This ensures work is preserved and can be compared against AWS baseline runs.

---

## Best Practices

1. **Start with quick tier** before committing to nightly/full
2. **Run one config at a time** to avoid resource contention
3. **Track everything** in the template to avoid duplicate work
4. **Verify metrics immediately** after completion before starting next config
5. **Name logs descriptively** using dimension, strategy, seed in filename
6. **Use background execution** to prevent accidental termination
7. **Monitor first run closely** to establish baseline resource usage

---

## Related Documentation

- [Edge-IIoTset Integration](./edge_iiotset_integration.md) - Dataset preprocessing details
- [Compute Constraints](./compute_constraints_and_solutions.md) - Memory analysis and AWS architecture
- [Experimental Design](./cic_objectives.md) - Thesis objectives mapping

---

**Last Updated:** November 17, 2025
**Maintainer:** Thesis Development Team
