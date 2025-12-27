# JMU CS 470 Cluster: Architecture, Setup, and Operations Guide

**Last Updated:** 2025-12-16
**Author:** Abraham Reines
**Purpose:** Comprehensive documentation for running federated learning experiments on the JMU CS 470 Slurm cluster.

---

## Table of Contents

1. [Cluster Overview](#1-cluster-overview)
2. [Hardware Architecture](#2-hardware-architecture)
3. [Network Architecture](#3-network-architecture)
4. [Software Environment](#4-software-environment)
5. [Storage Architecture](#5-storage-architecture)
6. [Slurm Job Scheduler](#6-slurm-job-scheduler)
7. [Environment Setup (Reproduction Guide)](#7-environment-setup-reproduction-guide)
8. [Running Experiments](#8-running-experiments)
9. [Known Issues and Workarounds](#9-known-issues-and-workarounds)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Troubleshooting](#11-troubleshooting)
12. [Quick Reference](#12-quick-reference)

---

## 1. Cluster Overview

### 1.1 Purpose

The JMU CS 470 cluster is a teaching and research HPC cluster designed for:

- Parallel and distributed computing coursework
- Machine learning and deep learning experiments
- Research projects requiring GPU acceleration

### 1.2 Access Model

```
[Off-Campus] --> [stu.cs.jmu.edu (Jump Host)] --> [login02.cluster.cs.jmu.edu]
                                                           |
                                                           v
                                              [compute21-29, gpu01-08]
```

- **Login Node:** `login02.cluster.cs.jmu.edu` - For job submission, file management, light tasks
- **Jump Host:** `stu.cs.jmu.edu` - Required for off-campus SSH access
- **Compute Nodes:** 17 nodes for actual computation (accessed via Slurm)

### 1.3 Key Constraints

| Resource           | Limit                         | Notes                            |
| ------------------ | ----------------------------- | -------------------------------- |
| Home directory     | ~20 GB                        | NFS-mounted, backed up           |
| Scratch space      | Large (unquoted)              | Local to cluster, may be purged  |
| Job time limit     | Unlimited (partition default) | 2 hours recommended for testing  |
| Login node compute | Prohibited                    | Use Slurm for all intensive work |

---

## 2. Hardware Architecture

### 2.1 Node Specifications

#### Login Node (login02)

| Component | Specification                        |
| --------- | ------------------------------------ |
| CPU       | AMD EPYC 7252 8-Core (2 sockets)     |
| Cores     | 16 physical, 32 threads (SMT)        |
| RAM       | 62 GiB                               |
| Purpose   | Job submission, file management only |

#### Compute Nodes (compute21-29)

| Component       | Specification                    |
| --------------- | -------------------------------- |
| Count           | 9 nodes                          |
| CPU             | AMD EPYC 7252 8-Core (2 sockets) |
| Cores per node  | 16 physical, 32 threads          |
| RAM per node    | 62 GiB                           |
| Total CPU cores | 144 physical (288 threads)       |
| Total RAM       | ~558 GiB                         |

#### GPU Nodes (gpu01-08)

| Component          | Specification                    |
| ------------------ | -------------------------------- |
| Count              | 8 nodes                          |
| CPU                | AMD EPYC 7252 8-Core (2 sockets) |
| Cores per node     | 16 physical, 32 threads          |
| RAM per node       | 62 GiB                           |
| GPU                | NVIDIA A2 (1 per node)           |
| GPU Memory         | 15 GB VRAM per GPU               |
| Compute Capability | 8.6 (Ampere architecture)        |
| Total GPUs         | 8                                |
| Total GPU VRAM     | 120 GB                           |

### 2.2 Cluster Totals

```
+--------------------------------------------------+
|           JMU CS 470 CLUSTER SUMMARY             |
+--------------------------------------------------+
| Total Nodes:        17                           |
| Total CPU Cores:    272 physical (544 threads)   |
| Total RAM:          ~1 TB                        |
| Total GPUs:         8 x NVIDIA A2                |
| Total GPU VRAM:     120 GB                       |
+--------------------------------------------------+
```

### 2.3 NUMA Architecture

Each node has a 2-socket NUMA topology:

```
Socket 0: Cores 0-7, 16-23 (with SMT)
Socket 1: Cores 8-15, 24-31 (with SMT)
```

For memory-intensive workloads, consider NUMA-aware scheduling.

---

## 3. Network Architecture

### 3.1 External Access

```
Internet
    |
    v
[stu.cs.jmu.edu] (Jump Host / Bastion)
    |
    | SSH ProxyJump
    v
[login02.cluster.cs.jmu.edu] (Login Node)
    |
    | Slurm Job Submission
    v
[compute21-29, gpu01-08] (Compute Nodes)
```

### 3.2 SSH Configuration

Add to `~/.ssh/config` for seamless access:

```
EID for this project: reinesaj

Host stu
    HostName stu.cs.jmu.edu
    User <YOUR_EID>
    TCPKeepAlive yes
    ServerAliveInterval 15

Host cluster
    HostName login02.cluster.cs.jmu.edu
    User <YOUR_EID>
    ProxyJump stu
    TCPKeepAlive yes
    ServerAliveInterval 15
```

Usage: `ssh cluster`

### 3.3 SSH Key Setup

```bash
# Generate key (if not exists)
ssh-keygen -t ed25519

# Copy to jump host
ssh-copy-id <EID>@stu.cs.jmu.edu

# Copy to cluster (via jump host)
ssh-copy-id -o ProxyJump=<EID>@stu.cs.jmu.edu <EID>@login02.cluster.cs.jmu.edu
```

### 3.4 Internal Network

- Compute nodes communicate via internal cluster network
- NFS mounts available on all nodes
- Scratch storage accessible from all nodes

---

## 4. Software Environment

### 4.1 Operating System

- **OS:** RHEL 8 (Red Hat Enterprise Linux)
- **Kernel:** Linux (see `uname -r` for version)
- **Scheduler:** Slurm 20.11

### 4.2 Python Versions Available

| Version       | Path                  | Notes                    |
| ------------- | --------------------- | ------------------------ |
| Python 3.6.8  | `/usr/bin/python3`    | System default (too old) |
| Python 3.9    | `/usr/bin/python3.9`  | Available                |
| Python 3.11.9 | `/usr/bin/python3.11` | **Recommended**          |

### 4.3 Module System

```bash
# List available modules
module avail

# Load MPI (example)
module load mpi/mpich-4.2.0-x86_64
```

Available modules:

- `mpi/mpich-4.2.0-x86_64`
- `mpi/openmpi-x86_64`
- `pmi/pmix-x86_64`

### 4.4 GPU Software

- CUDA drivers installed on GPU nodes
- PyTorch CUDA support works with `torch==2.3.1`
- Use `--gres=gpu:1` in Slurm to request GPU

---

## 5. Storage Architecture

### 5.1 Storage Hierarchy

```
/nfs/home/<EID>/           # Home directory (NFS, backed up)
    |-- .bashrc
    |-- .ssh/
    |-- (small files only)

/scratch/<EID>/            # Scratch space (large, may be purged)
    |-- federated-ids/     # Repository clone
    |-- datasets/          # Large datasets
    |   |-- edge-iiotset/
    |       |-- edge_iiotset_full.csv (891 MB)
    |-- venvs/             # Python virtual environments
    |   |-- fedids-py311/
    |-- results/           # Experiment outputs
        |-- smoke_test/
        |-- comparative_analysis/
```

### 5.2 Storage Quotas

| Location          | Quota            | Purpose                     |
| ----------------- | ---------------- | --------------------------- |
| `/nfs/home/<EID>` | ~20 GB           | Config files, small scripts |
| `/scratch/<EID>`  | Large (unquoted) | Datasets, venvs, results    |

Check quota:

```bash
quota -s
```

### 5.3 Data Transfer

```bash
# SCP (single file)
scp local_file.csv cluster:/scratch/$USER/datasets/

# Rsync (directory, resumable)
rsync -avz --progress local_dir/ cluster:/scratch/$USER/remote_dir/
```

---

## 6. Slurm Job Scheduler

### 6.1 Partition Configuration

```
PartitionName: cs (default)
Nodes: compute[21-29], gpu[01-08]
State: UP
DefaultTime: 01:00:00
MaxTime: UNLIMITED
OverSubscribe: EXCLUSIVE
TotalCPUs: 272
TotalNodes: 17
```

### 6.2 Key Slurm Commands

| Command                  | Purpose          |
| ------------------------ | ---------------- |
| `sbatch script.sbatch`   | Submit batch job |
| `squeue -u $USER`        | View your jobs   |
| `scancel <jobid>`        | Cancel a job     |
| `sinfo -N`               | View node status |
| `scontrol show job <id>` | Job details      |
| `sacct -j <id>`          | Job accounting   |

### 6.3 Resource Request Syntax

```bash
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=16     # CPUs per task
#SBATCH --time=02:00:00        # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:1           # GPU request (gpu nodes only)
#SBATCH --output=/path/%j.out  # Stdout file
#SBATCH --error=/path/%j.err   # Stderr file
```

### 6.4 CRITICAL: Slurm Memory Misconfiguration

**Issue:** Slurm reports `RealMemory=1` for all nodes (should be ~62000 MB).

**Impact:** Cannot use `--mem=48G` or similar memory requests.

**Workaround:** Omit the `--mem` flag entirely. Nodes have 62 GB RAM which is available by default.

```bash
# BAD - Will fail
#SBATCH --mem=48G

# GOOD - Works
# (no --mem line)
```

---

## 7. Environment Setup (Reproduction Guide)

### 7.1 Prerequisites

- JMU EID with cluster access
- SSH key pair generated locally
- Git installed locally

### 7.2 Step-by-Step Setup

#### Step 1: Configure SSH Access

```bash
# Add to ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'
Host stu
    HostName stu.cs.jmu.edu
    User YOUR_EID
    TCPKeepAlive yes
    ServerAliveInterval 15

Host cluster
    HostName login02.cluster.cs.jmu.edu
    User YOUR_EID
    ProxyJump stu
    TCPKeepAlive yes
    ServerAliveInterval 15
EOF

# Copy SSH keys (will prompt for password twice)
ssh-copy-id YOUR_EID@stu.cs.jmu.edu
ssh-copy-id -o ProxyJump=YOUR_EID@stu.cs.jmu.edu YOUR_EID@login02.cluster.cs.jmu.edu

# Test connection
ssh cluster 'hostname'
```

#### Step 2: Create Directory Structure

```bash
ssh cluster 'mkdir -p /scratch/$USER/{federated-ids,datasets/edge-iiotset,venvs,results}'
```

#### Step 3: Clone Repository

```bash
ssh cluster 'cd /scratch/$USER && git clone https://github.com/reinesaj2/federated-ids.git'
```

#### Step 4: Create Python Virtual Environment

```bash
ssh cluster 'python3.11 -m venv /scratch/$USER/venvs/fedids-py311'
```

#### Step 5: Install Dependencies

```bash
ssh cluster 'source /scratch/$USER/venvs/fedids-py311/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    cd /scratch/$USER/federated-ids && \
    pip install -r requirements.txt'
```

#### Step 6: Transfer Dataset

```bash
# From local machine (891 MB, takes a few minutes)
scp /path/to/edge_iiotset_full.csv cluster:/scratch/$USER/datasets/edge-iiotset/
```

#### Step 7: Verify Setup

```bash
ssh cluster 'source /scratch/$USER/venvs/fedids-py311/bin/activate && \
    python -c "import torch; import flwr; print(f\"PyTorch: {torch.__version__}, Flower: {flwr.__version__}\")"'
```

---

## 8. Running Experiments

### 8.1 Smoke Test (Single Node)

```bash
# Submit smoke test job
ssh cluster 'cd /scratch/$USER/federated-ids && sbatch scripts/slurm/fedprox_smoke_test.sbatch'

# Monitor job
ssh cluster 'squeue -u $USER'

# View output
ssh cluster 'cat /scratch/$USER/results/smoke_test/slurm-<JOBID>.out'
```

### 8.2 Smoke Test Configuration

The `scripts/slurm/fedprox_smoke_test.sbatch` runs:

- **Clients:** 10 (reduced from 20 for memory)
- **Rounds:** 15
- **Aggregation:** FedAvg with FedProx (mu=0.01)
- **Heterogeneity:** alpha=0.5 (moderate non-IID)
- **Dataset:** Edge-IIoTset full (891 MB, 1.7M samples)
- **Time:** ~7 minutes on single compute node

### 8.3 Full Comparative Analysis

For thesis-quality experiments, use the comparative analysis framework:

```bash
# Heterogeneity dimension (alpha sweep)
sbatch --array=0-19 scripts/slurm/comparative_analysis.sbatch --dimension heterogeneity

# FedProx dimension (alpha x mu sweep)
sbatch --array=0-19 scripts/slurm/comparative_analysis.sbatch --dimension heterogeneity_fedprox

# Attack resilience dimension
sbatch --array=0-19 scripts/slurm/comparative_analysis.sbatch --dimension attack
```

### 8.4 GPU Experiments

To use GPU nodes:

```bash
#SBATCH --gres=gpu:1
#SBATCH --partition=cs  # Default partition includes GPU nodes

# In Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 8.5 Centralized Baseline (Encoder)

Run the centralized encoder baseline on the cluster with the dedicated Slurm script:

```bash
cd /scratch/$USER/federated-ids
sbatch scripts/slurm/centralized_baseline_encoder.sbatch
```

The script defaults to Edge-IIoTset full and writes outputs to:

```
/scratch/$USER/results/centralized_baseline/
```

---

## 9. Known Issues and Workarounds

### 9.1 Slurm Memory Configuration

**Issue:** `sinfo` reports `MEMORY=1` for all nodes.
**Cause:** Slurm `RealMemory` not configured by admin.
**Workaround:** Do not specify `--mem` in job scripts.

### 9.2 FEDIDS_USE_OPACUS Environment Variable

**Issue:** Clients fail with `RuntimeError: Opacus not available`.
**Cause:** The code requires `FEDIDS_USE_OPACUS=1` even when DP is disabled.
**Workaround:** Always set in job scripts:

```bash
export FEDIDS_USE_OPACUS=1
```

**Important:** Set this AFTER activating the virtual environment.

### 9.3 Memory Pressure with Many Clients

**Issue:** 20 clients on single node causes `MemoryError` and `failed to map segment`.
**Cause:** Each client loads 891MB dataset + PyTorch + scikit-learn.
**Workaround:**

- Reduce to 10 clients per node
- Or distribute across multiple nodes with Slurm arrays

### 9.4 Login Node Abuse

**Issue:** Running compute on login node maxes CPU and may violate policy.
**Cause:** All 20 clients running on login02 instead of compute nodes.
**Solution:** Always use Slurm to submit jobs to compute nodes.

### 9.5 SSH Connection Drops

**Issue:** Long-running SSH sessions disconnect.
**Workaround:** Add to `~/.ssh/config`:

```
TCPKeepAlive yes
ServerAliveInterval 15
```

---

## 10. Performance Characteristics

### 10.1 Smoke Test Benchmarks

| Configuration         | Nodes          | Time    | Rate       |
| --------------------- | -------------- | ------- | ---------- |
| 10 clients, 15 rounds | 1 (compute21)  | 7 min   | ~28s/round |
| 20 clients, 30 rounds | 1 (login node) | ~15 min | ~30s/round |

### 10.2 Per-Round Timing Breakdown

From smoke test logs:

- Round duration: ~26-28 seconds
- Aggregation time: ~0.23 ms
- Most time spent in client training

### 10.3 Scaling Estimates

| Experiment                          | Configs | Est. Time (1 node) | Est. Time (17 nodes) |
| ----------------------------------- | ------- | ------------------ | -------------------- |
| Heterogeneity (7 alpha x 10 seeds)  | 70      | ~8 hours           | ~30 min              |
| FedProx (7 alpha x 9 mu x 10 seeds) | 630     | ~3 days            | ~4 hours             |
| Attack (4 agg x 4 adv x 10 seeds)   | 160     | ~18 hours          | ~1 hour              |

---

## 11. Troubleshooting

### 11.1 Job Fails Immediately

Check error log:

```bash
cat /scratch/$USER/results/smoke_test/slurm-<JOBID>.err
```

Common causes:

- Unbound variable (use `${VAR:-default}` syntax)
- Missing environment variable
- Memory request too high

### 11.2 Clients Crash with ImportError

```
ImportError: .../sklearn/utils/murmurhash.cpython-311-x86_64-linux-gnu.so: failed to map segment
```

**Cause:** Memory exhaustion during shared library loading.
**Fix:** Reduce number of clients or use multiple nodes.

### 11.3 Training Stuck on Round 1

**Cause:** Not enough clients connected (server waiting).
**Check:** Look at client logs for connection errors.
**Fix:** Ensure `FEDIDS_USE_OPACUS=1` is set.

### 11.4 Cannot SSH to Cluster

**Cause:** Off-campus without jump host.
**Fix:** Use ProxyJump configuration (see Section 3.2).

---

## 12. Quick Reference

### 12.1 Essential Commands

```bash
# Connect to cluster
ssh cluster

# Submit job
cd /scratch/$USER/federated-ids
sbatch scripts/slurm/fedprox_smoke_test.sbatch

# Check queue
squeue -u $USER

# View job output
tail -f /scratch/$USER/results/smoke_test/slurm-<JOBID>.out

# Cancel job
scancel <JOBID>

# Check node status
sinfo -N
```

### 12.2 Key Paths

| Purpose       | Path                                                         |
| ------------- | ------------------------------------------------------------ |
| Repository    | `/scratch/$USER/federated-ids`                               |
| Virtual env   | `/scratch/$USER/venvs/fedids-py311`                          |
| Dataset       | `/scratch/$USER/datasets/edge-iiotset/edge_iiotset_full.csv` |
| Results       | `/scratch/$USER/results/`                                    |
| Slurm scripts | `/scratch/$USER/federated-ids/scripts/slurm/`                |

### 12.3 Environment Activation

```bash
source /scratch/$USER/venvs/fedids-py311/bin/activate
export FEDIDS_USE_OPACUS=1
export SEED=42
```

### 12.4 Job Template

```bash
#!/bin/bash
#SBATCH --job-name=fedids-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/results/slurm-%j.out
#SBATCH --error=/scratch/%u/results/slurm-%j.err

source /scratch/$USER/venvs/fedids-py311/bin/activate
export FEDIDS_USE_OPACUS=1
export SEED=42

cd /scratch/$USER/federated-ids
# Your experiment commands here
```

---

## Appendix A: Cluster Topology Diagram

```
                    +-------------------+
                    |    INTERNET       |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  stu.cs.jmu.edu   |
                    |   (Jump Host)     |
                    +--------+----------+
                             |
                    +--------v----------+
                    | login02.cluster   |
                    |   (Login Node)    |
                    |  32 threads, 62GB |
                    +--------+----------+
                             |
           +-----------------+-----------------+
           |                                   |
+----------v-----------+           +-----------v----------+
|   COMPUTE NODES      |           |     GPU NODES        |
|  compute[21-29]      |           |    gpu[01-08]        |
|  9 nodes             |           |    8 nodes           |
|  16 cores each       |           |    16 cores each     |
|  62 GB RAM each      |           |    62 GB RAM each    |
|                      |           |    NVIDIA A2 (15GB)  |
+----------------------+           +----------------------+
```

---

## Appendix B: Slurm Batch Script Template

See `scripts/slurm/fedprox_smoke_test.sbatch` for a complete working example.

---

## Appendix C: Change Log

| Date       | Change                                              |
| ---------- | --------------------------------------------------- |
| 2025-12-16 | Initial documentation created                       |
| 2025-12-16 | Added smoke test results and performance data       |
| 2025-12-16 | Documented Slurm memory misconfiguration workaround |
| 2025-12-16 | Added FEDIDS_USE_OPACUS requirement                 |
