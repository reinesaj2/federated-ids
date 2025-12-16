# JMU CS 470 Cluster Runbook (Slurm)

**Source documentation:** https://w3.cs.jmu.edu/lam2mo/cs470_2025_01/cluster.html  
**Last reviewed:** 2025-12-16  
**Project goal:** run full-scale Edge-IIoTset experiments to satisfy `docs/NEURIPS_READINESS_ANALYSIS.md`.

## Quick Facts (from source docs)

- **Login node:** `login02.cluster.cs.jmu.edu`
- **Scheduler:** Slurm (RHEL8, Slurm 20.11)
- **Nodes (newer cluster):** 17 compute nodes total, including GPU nodes (NVIDIA A2)
- **Shared software:** modules (`module avail`) and utilities in `/shared/cs470/bin`

## Access

### On-campus (SSH)

```bash
ssh <eid>@login02.cluster.cs.jmu.edu
```

### Off-campus (SSH jump host)

```bash
ssh -J <eid>@stu.cs.jmu.edu <eid>@login02.cluster.cs.jmu.edu
```

Optional `~/.ssh/config`:

```text
Host stu
    HostName stu.cs.jmu.edu
    User <eid>

Host cluster
    HostName login02.cluster.cs.jmu.edu
    User <eid>
    ProxyJump stu
```

Optional keepalive (useful if the connection drops):

```text
TCPKeepAlive yes
ServerAliveInterval 15
```

### SSH keys (recommended)

Generate keys (once per machine):

```bash
ssh-keygen -t ed25519
```

Copy key to cluster (Linux/macOS):

```bash
ssh-copy-id <eid>@login02.cluster.cs.jmu.edu
```

Windows option (from source docs):

```bash
type .ssh\id_rsa.pub | ssh <eid>@login02.cluster.cs.jmu.edu "cat >> .ssh/authorized_keys"
```

## Storage & File Transfer

### Home vs scratch

- **Home:** `/nfs/home/<eid>` (source docs mention a ~250MB quota). Check usage via:
  ```bash
  quota -s
  ```
- **Scratch:** `/scratch/<eid>` (more space, but **may be purged**). Use it for datasets, virtualenvs, and run artifacts.

### Transfer files

- **CLI:** `scp` (or `rsync` if available)
- **GUI:** FileZilla / WinSCP (SFTP to `login02.cluster.cs.jmu.edu`, folder `/scratch/<eid>` or `/nfs/home/<eid>`)

## Slurm: Running Jobs (do not run heavy compute on the login node)

### Interactive jobs (`srun`)

Run a program:

```bash
srun [Slurm options] /path/to/program [program options]
```

Examples from source docs:

```bash
srun -n 4  hostname
srun -n 32 hostname
srun -N 4  hostname
srun -N 4 -n 32 hostname
```

Interactive shell on a compute node (useful for debugging):

```bash
srun --pty /usr/bin/bash -i
```

### Batch jobs (`sbatch`)

Submit:

```bash
sbatch job.sh
```

Monitor:

```bash
squeue
```

Cancel:

```bash
scancel <jobid>
```

### GPU jobs

Per source docs, request a GPU by adding `--gres=gpu` to the job launch command (or `#SBATCH --gres=gpu` in a batch script).

## Modules / Software

List modules:

```bash
module avail
```

MPI example (source docs):

```bash
module load mpi/mpich-4.2.0-x86_64
```

If your batch environment doesn’t have `module`, source it (zsh example from source docs):

```bash
source /usr/share/Modules/init/zsh
```

## Running `federated-ids` Experiments on the Cluster (Edge-IIoTset full)

### Recommended scratch layout

Because home quota is small, prefer:

```text
/scratch/<eid>/
  federated-ids/                # repo clone
  datasets/edge-iiotset/         # Edge-IIoTset CSV(s)
  venvs/fedids/                  # python env (optional)
  results/                       # runs + plots + tables
```

### Dataset path expectations

`scripts/comparative_analysis.py` supports `--dataset edge-iiotset-full` and defaults to:

```text
data/edge-iiotset/edge_iiotset_full.csv
```

On the cluster, either:

- Place/symlink the full CSV at that path inside the repo, or
- Use `--data_path /scratch/<eid>/datasets/edge-iiotset/edge_iiotset_full.csv`.

### Parameter space (α and μ)

To be defensible for publication, use the repo’s default sweep definitions in `scripts/comparative_analysis.py` (`ComparisonMatrix`):

- **Heterogeneity (Dirichlet α):** `0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf`
- **FedProx μ:** `0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2`
- **Seeds:** `42..51` (10 seeds)

This gives broad coverage of heterogeneity strength and FedProx regularization strength.

### FedAvg vs FedProx (robust comparison)

Run both dimensions on the **full** dataset:

- **FedAvg baseline across α:** `--dimension heterogeneity`
- **FedProx across (α, μ):** `--dimension heterogeneity_fedprox`

`heterogeneity_fedprox` uses `aggregation=fedprox` and sweeps μ; it includes μ=0.0, but you should still run the FedAvg baseline dimension for a clean comparison.

### Robust aggregation (with adversaries)

Use:

```bash
python scripts/comparative_analysis.py --dimension attack --dataset edge-iiotset-full ...
```

This runs robust aggregators (FedAvg, Krum, Bulyan, Median) across adversary fractions. Note Bulyan’s Byzantine constraint `n >= 4f + 3` is enforced; the attack dimension uses a larger client count to satisfy it.

### Scaling out with Slurm arrays (`--split-index/--split-total`)

`scripts/comparative_analysis.py` can split the generated config list across multiple jobs:

- `--split-total N` = number of parallel jobs
- `--split-index I` = the 0-based slice index for this job

This maps cleanly to a Slurm array:

```bash
#SBATCH --array=0-19
python scripts/comparative_analysis.py \
  --dimension heterogeneity_fedprox \
  --dataset edge-iiotset-full \
  --split-total 20 \
  --split-index "$SLURM_ARRAY_TASK_ID"
```

### Minimal `sbatch` template (single split)

Adjust `--time/--mem/--cpus-per-task` based on your cluster limits and observed runtime.

```bash
#!/bin/bash
#SBATCH --job-name=fedids-obj2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/%x-%j.out

set -euo pipefail

source /usr/share/Modules/init/zsh || true

cd /scratch/$USER/federated-ids

# Keep matplotlib caches off the home directory (if plots are generated)
export MPLCONFIGDIR="/scratch/$USER/tmp/mplconfig"
mkdir -p "$MPLCONFIGDIR"

python scripts/comparative_analysis.py \
  --dimension heterogeneity_fedprox \
  --dataset edge-iiotset-full \
  --data_path "/scratch/$USER/datasets/edge-iiotset/edge_iiotset_full.csv" \
  --output_dir "/scratch/$USER/results/comparative_analysis" \
  --num_clients 10 \
  --num_rounds 50 \
  --split-total 20 \
  --split-index 0
```

## Post-run: defensible reporting (NeurIPS readiness)

To withstand reviewer criticism, keep every run reproducible and statistically grounded:

- Record: git commit hash, dataset file hash, full CLI args, and Slurm job IDs.
- Use enough seeds: the default matrix uses 10 (`42..51`).
- Use the existing analysis tooling for significance/effect sizes where applicable (see `scripts/analyze_statistical_rigor.py` and `scripts/statistical_analysis.py`).
