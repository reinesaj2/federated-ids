# Issue #128: Mixed-Silo 3-Dataset Federation - Cluster Deployment Guide

## Overview

This guide provides complete instructions for deploying **Issue #128** experiments on the JMU CS470 cluster. The experiment validates cross-organizational federated learning across **12 clients** (4 CIC-IDS2017 + 4 UNSW-NB15 + 4 Edge-IIoTset) with robust aggregation and FedProx heterogeneity handling.

**Research Contribution**: Demonstrates that robust aggregation + FedProx can unify models across organizational silos using incompatible IDS datasets.

---

## Experimental Design

### Configuration Matrix

- **Clients**: 12 (4 per dataset: CIC, UNSW, Edge-IIoTset)
- **Aggregators**: FedAvg, Krum, Bulyan, Median (4 methods)
- **FedProx mu**: 0.0 (baseline), 0.01, 0.1 (3 values)
- **Adversary fractions**: 0%, 10%, 20% (3 values, capped for Bulyan constraint)
- **Seeds**: 42-51 (10 seeds for statistical rigor)
- **Rounds**: 15 federated rounds per experiment

**Total configs**: 4 × 3 × 3 × 10 = **360 experiments**

### Resource Estimates

- **Per-experiment runtime**: ~2-3 minutes (15 rounds × ~10s/round)
- **Total runtime per split**: 60 configs × 2.5 min = **~2.5-3 hours** (conservative: 15-20 hours with overhead)
- **Memory**: 8 GB per array task (12 clients × ~500 MB each)
- **CPU**: 4 cores per task (parallelizable client execution)
- **Storage**: ~500 MB metrics + logs per split

---

## Prerequisites

### 1. Cluster Access

**Off-campus**: Use SSH jump host via `stu.cs.jmu.edu`

```bash
# Configure SSH jump host in ~/.ssh/config
Host jmu-cluster
    HostName login02.cluster.cs.jmu.edu
    User <your-eid>
    ProxyJump stu.cs.jmu.edu
```

**On-campus**: Direct SSH

```bash
ssh <your-eid>@login02.cluster.cs.jmu.edu
```

### 2. Dataset Preparation

Upload datasets to cluster scratch space:

```bash
# On local machine
EID="<your-eid>"
SCRATCH_DIR="/scratch/${EID}/federated-ids-128"

# Create directories on cluster
ssh jmu-cluster "mkdir -p ${SCRATCH_DIR}/datasets/{cic,unsw,edge-iiotset}"

# Upload CIC-IDS2017
scp data/cic/cic_ids2017_multiclass.csv \
    jmu-cluster:${SCRATCH_DIR}/datasets/cic/

# Upload UNSW-NB15
scp data/unsw/UNSW_NB15_training-set.csv \
    jmu-cluster:${SCRATCH_DIR}/datasets/unsw/

# Upload Edge-IIoTset
scp data/edge-iiotset/edge_iiotset_quick.csv \
    jmu-cluster:${SCRATCH_DIR}/datasets/edge-iiotset/
```

**Dataset sizes**:

- CIC: ~3.5 MB (9,971 samples)
- UNSW: ~20 MB (82,332 samples)
- Edge-IIoTset quick: ~26 MB (50,000 samples)

**Note**: Use `edge_iiotset_quick.csv` for initial validation, then scale to `edge_iiotset_nightly.csv` (500K) or `edge_iiotset_full.csv` (1.7M) for final runs.

---

## Deployment Steps

### Step 1: SSH to Cluster

```bash
ssh jmu-cluster
# Or: ssh <eid>@login02.cluster.cs.jmu.edu
```

### Step 2: Verify Environment

```bash
# Check scratch quota
df -h /scratch/$USER

# Load Python module
module load python/3.11  # Or available version

# Verify Python
python3 --version
```

### Step 3: Submit Array Job

The sbatch script uses **array jobs** to parallelize across 6 tasks (60 configs each):

```bash
cd /scratch/$USER/federated-ids-128/repo

# Submit all 6 array tasks
sbatch scripts/cluster/sbatch_mixed_silo_3dataset.sh
```

**Output**:

```
Submitted batch job 174500
```

### Step 4: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Tail logs (replace JOB_ID with actual ID)
tail -f /scratch/$USER/federated-ids-128/logs/slurm-174500_0.out

# Check all array tasks
watch -n 5 'squeue -u $USER'
```

**Job states**:

- `PD`: Pending (waiting for resources)
- `R`: Running
- `CG`: Completing
- `CD`: Completed

### Step 5: Cancel Jobs (if needed)

```bash
# Cancel specific job
scancel 174500

# Cancel specific array task
scancel 174500_3

# Cancel all your jobs
scancel -u $USER
```

---

## Array Job Configuration

The sbatch script splits 360 configs across **6 array tasks**:

| Task ID | Split Index | Configs | Estimated Runtime |
| ------- | ----------- | ------- | ----------------- |
| 0       | 0/6         | 60      | 15-20 hours       |
| 1       | 1/6         | 60      | 15-20 hours       |
| 2       | 2/6         | 60      | 15-20 hours       |
| 3       | 3/6         | 60      | 15-20 hours       |
| 4       | 4/6         | 60      | 15-20 hours       |
| 5       | 5/6         | 60      | 15-20 hours       |

**Parallelization**: All 6 tasks run simultaneously (subject to cluster availability).

---

## Output Structure

```
/scratch/<eid>/federated-ids-128/
├── repo/                     # Git repository (auto-cloned)
├── datasets/                 # Input datasets
│   ├── cic/
│   ├── unsw/
│   └── edge-iiotset/
├── results/                  # Experiment outputs
│   ├── runs/                 # Per-experiment metrics
│   │   ├── dsmixed_silo_3dataset_*_seed42/
│   │   │   ├── config.json
│   │   │   ├── metrics.csv
│   │   │   ├── server.log
│   │   │   └── client_*.log
│   │   └── ...
│   └── experiment_manifest_mixed_silo_3dataset_split*.json
├── logs/                     # Slurm logs
│   ├── slurm-174500_0.out
│   ├── slurm-174500_0.err
│   └── ...
└── venv/                     # Python virtual environment
```

---

## Post-Execution

### Step 1: Verify Completion

```bash
# On cluster
cd /scratch/$USER/federated-ids-128/results

# Check manifests
cat experiment_manifest_mixed_silo_3dataset_split1of6.json | jq '.total_experiments'
cat experiment_manifest_mixed_silo_3dataset_split1of6.json | jq '[.results[] | select(.metrics_exist == true)] | length'

# Count metrics files
find runs -name "metrics.csv" | wc -l
# Expected: 360 (if all successful)
```

### Step 2: Download Results

```bash
# On local machine
EID="<your-eid>"
SCRATCH_DIR="/scratch/${EID}/federated-ids-128"

# Download results directory
scp -r jmu-cluster:${SCRATCH_DIR}/results ./cluster_results_issue128

# Download logs (for debugging)
scp -r jmu-cluster:${SCRATCH_DIR}/logs ./cluster_logs_issue128
```

### Step 3: Analyze Results

```bash
# On local machine
cd /Users/abrahamreines/Documents/Thesis/federated-ids

# Generate consolidated metrics
python scripts/analyze_cluster_results.py \
    --results_dir ./cluster_results_issue128 \
    --output_dir ./analysis_issue128

# Generate thesis plots
python scripts/generate_thesis_plots.py \
    --results_dir ./cluster_results_issue128 \
    --output_dir ./plots_issue128
```

---

## Troubleshooting

### Issue: Job Pending for Long Time

**Cause**: Cluster resource contention

**Solution**:

```bash
# Check queue status
squeue

# Check partition availability
sinfo -p normal

# Consider running fewer array tasks
sbatch --array=0-2 scripts/cluster/sbatch_mixed_silo_3dataset.sh  # 3 tasks instead of 6
```

### Issue: Out of Memory (OOM)

**Cause**: 12 clients × dataset size exceeds 8 GB

**Solution**:

```bash
# Edit sbatch script to increase memory
#SBATCH --mem=16G  # Double memory allocation

# Or use smaller dataset variant (edge_iiotset_quick instead of full)
```

### Issue: Dataset Not Found

**Cause**: Missing dataset files in scratch

**Solution**:

```bash
# Verify files exist
ssh jmu-cluster "ls -lh /scratch/$USER/federated-ids-128/datasets/cic/"

# Re-upload if missing (see Prerequisites)
```

### Issue: Import Errors

**Cause**: Missing Python dependencies

**Solution**:

```bash
# On cluster
cd /scratch/$USER/federated-ids-128/repo
source /scratch/$USER/federated-ids-128/venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Disk Quota Exceeded

**Cause**: Too many logs or metrics files

**Solution**:

```bash
# Check quota
df -h /scratch/$USER

# Clean old results
rm -rf /scratch/$USER/federated-ids-128/results/runs/old_*

# Compress logs
gzip /scratch/$USER/federated-ids-128/logs/*.out
```

---

## Resource Monitoring

### CPU Usage

```bash
# While job is running
ssh jmu-cluster
sstat -j 174500 --format=JobID,AveCPU,MaxRSS,NTasks
```

### Memory Usage

```bash
# Real-time monitoring
ssh jmu-cluster
top -u $USER
```

### Disk Usage

```bash
# Check scratch usage
du -sh /scratch/$USER/federated-ids-128/*
```

---

## Validation Checklist

- [ ] SSH access to JMU cluster working
- [ ] Datasets uploaded to `/scratch/$USER/federated-ids-128/datasets/`
- [ ] Sbatch script is executable (`chmod +x scripts/cluster/sbatch_mixed_silo_3dataset.sh`)
- [ ] Git branch `feat/issue-128-mixed-silo-3dataset` exists on GitHub
- [ ] Array job submitted successfully (`sbatch` returns job ID)
- [ ] At least one array task running (`squeue` shows `R` state)
- [ ] First experiment completes without errors (`tail slurm-*.out` shows "SUCCESS")
- [ ] Metrics files generated (`ls /scratch/$USER/federated-ids-128/results/runs/*/metrics.csv`)
- [ ] All 360 experiments complete (check manifests: `jq '.total_experiments'`)
- [ ] Results downloaded to local machine
- [ ] Thesis plots generated successfully

---

## Expected Results

### Metrics Files

Each experiment generates:

- `metrics.csv`: Server-side metrics (accuracy, loss, aggregation stats)
- `client_*_metrics.csv`: Per-client metrics (local loss, gradients, DP stats)
- `config.json`: Experiment configuration
- `server.log`, `client_*.log`: Execution logs

### Key Metrics

- **Global accuracy**: Aggregated model performance on test sets
- **Per-dataset accuracy**: CIC, UNSW, Edge-IIoTset breakdown
- **Byzantine resilience**: Accuracy degradation under adversaries (0%, 10%, 20%)
- **FedProx benefit**: mu=0.01/0.1 vs mu=0.0 (baseline)
- **Cross-dataset generalization**: Model trained on mixed data vs single-dataset baseline

---

## References

- **Issue #128**: https://github.com/reinesaj2/federated-ids/issues/128
- **JMU CS470 Cluster Runbook**: `docs/JMU_CS470_CLUSTER_RUNBOOK.md`
- **Bulyan Byzantine Constraint**: El Mhamdi et al. 2018 (n >= 4f + 3)
- **Slurm Documentation**: https://slurm.schedmd.com/sbatch.html

---

## Contact

For cluster access issues, contact JMU CS470 support: cs470@jmu.edu

For experiment/code issues, see Issue #128 on GitHub.
