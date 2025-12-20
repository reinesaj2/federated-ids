# NeurIPS Final Experiment Runbook

**Created:** 2025-12-19
**Purpose:** Definitive guide to complete ALL experiments required for NeurIPS submission
**Status:** AUTHORITATIVE - supersedes all previous runbooks for gap experiments

---

## Executive Summary

### Current State (1,917 experiments)

| Dimension           | Coverage                              | Status           |
| ------------------- | ------------------------------------- | ---------------- |
| Aggregators         | FedAvg, FedProx, Krum, Bulyan, Median | COMPLETE         |
| Non-IID (alpha)     | 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf   | COMPLETE         |
| Adversary fractions | 0%, 10%, 20%, 30%                     | COMPLETE (0-30%) |
| Seeds               | 5-20 per config                       | COMPLETE         |
| Attack types        | grad_ascent only                      | INCOMPLETE       |
| Combined ablation   | 0 experiments                         | CRITICAL GAP     |
| FedProx + Byzantine | 0 experiments                         | CRITICAL GAP     |

### Required New Experiments: 420 jobs (P0) + 60 optional (P2)

| Gap Category                             | New Jobs | Priority          |
| ---------------------------------------- | -------- | ----------------- |
| Combined ablation (Robust Agg + FedProx) | 240      | P0 - Critical     |
| FedProx under Byzantine                  | 180      | P0 - Critical     |
| Additional attack types                  | 60       | P2 - Nice to have |
| **Total (P0)**                           | **420**  |                   |
| **Total (P0+P2)**                        | **480**  |                   |

---
## Scope Note: Cross-Silo, Hybrid, and CIC Parity

This runbook covers **Edge-IIoTset full-dataset NeurIPS gap closure only**. It does **not** include mixed-silo cross-dataset experiments (Issue #128) or the hybrid dataset sweep.

### Additional Required Work (Not Included in 420)

| Objective Area                              | Source Doc                             | Required Jobs | Optional Jobs | Notes |
| ------------------------------------------- | -------------------------------------- | ------------: | ------------: | ----- |
| Mixed-silo 3-dataset federation (Issue #128) | `docs/ISSUE_128_CLUSTER_DEPLOYMENT.md` | 360           | 0             | 4 aggregators × 3 μ × 3 adv × 10 seeds |
| Hybrid dataset NeurIPS sweep                | `docs/HYBRID_RUNBOOK.md`               | 1400          | 320           | Optional = hybrid robustness extension |

### CIC Full-Scale Parity Requirement

**Requirement:** By the end, CIC must have **the same number of full-scale runs as Edge-IIoTset**.

- Current Edge-IIoTset full-scale baseline: **1,740 valid complete runs** (see `docs/NEURIPS_IIOT_FULL_ANALYSIS.md`).
- CIC parity target: **1,740 full-scale runs**.
- Remaining CIC runs needed = **1,740 − current_CIC_full_runs** (compute after inventory of CIC runs on `main`).

---

## Gap 1: Combined Ablation (Robust Agg + FedProx)

**NeurIPS Requirement:** Demonstrate whether combining robust aggregation with FedProx provides synergistic benefits under simultaneous Byzantine + non-IID stress.

### Experiment Matrix

| Aggregator | Mu Values | Alpha Values       | Adv Fractions    | Seeds | Jobs |
| ---------- | --------- | ------------------ | ---------------- | ----- | ---- |
| Krum       | 0.01, 0.1 | 0.1, 0.5, 1.0, inf | 0, 10%, 20%, 30% | 5     | 160  |
| Bulyan     | 0.01, 0.1 | 0.1, 0.5, 1.0, inf | 0, 10%           | 5     | 80   |

**Note:** Bulyan limited to 10% adv due to n >= 4f+3 constraint with 10 clients.

**Total Combined Jobs:** 240

### Slurm Script: `neurips_combined_ablation.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=neurips-combined
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/results/neurips_combined/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/neurips_combined/%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source /usr/share/Modules/init/bash 2>/dev/null || true
source "/scratch/${USER}/venvs/fedids-py311/bin/activate"

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/neurips_combined"
cd "/scratch/${USER}/federated-ids"

NUM_CLIENTS=10
NUM_ROUNDS=20
SEEDS="42,43,44,45,46"

SPLIT_TOTAL=240
OFFSET="${OFFSET:-0}"
SPLIT_INDEX="$(( ${SLURM_ARRAY_TASK_ID:-0} + OFFSET ))"

python -u scripts/comparative_analysis.py \
  --dimension combined_robustness \
  --dataset edge-iiotset-full \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --seeds "${SEEDS}" \
  --server_timeout 21600 \
  --client_timeout 21600 \
  --split-total "${SPLIT_TOTAL}" \
  --split-index "${SPLIT_INDEX}"
```

---

## Gap 2: FedProx Under Byzantine Attack

**NeurIPS Requirement:** Verify Claim B - "FedProx alone fails under >20% Byzantine clients"

### Experiment Matrix

| Aggregator | Mu Values       | Alpha Values       | Adv Fractions | Seeds | Jobs |
| ---------- | --------------- | ------------------ | ------------- | ----- | ---- |
| FedProx    | 0.01, 0.05, 0.1 | 0.1, 0.5, 1.0, inf | 10%, 20%, 30% | 5     | 180  |

**Note:** This tests FedProx (standard averaging + proximal term) under Byzantine attack WITHOUT robust aggregation.

**Total FedProx+Byzantine Jobs:** 180

### Slurm Script: `neurips_fedprox_attack.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=neurips-fedprox-attack
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/results/neurips_fedprox_attack/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/neurips_fedprox_attack/%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source /usr/share/Modules/init/bash 2>/dev/null || true
source "/scratch/${USER}/venvs/fedids-py311/bin/activate"

export FEDIDS_USE_OPACUS=1
export MPLCONFIGDIR="/scratch/${USER}/tmp/mplconfig"
export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$MPLCONFIGDIR" "/scratch/${USER}/results/neurips_fedprox_attack"
cd "/scratch/${USER}/federated-ids"

NUM_CLIENTS=10
NUM_ROUNDS=20
SEEDS="42,43,44,45,46"

SPLIT_TOTAL=180
OFFSET="${OFFSET:-0}"
SPLIT_INDEX="$(( ${SLURM_ARRAY_TASK_ID:-0} + OFFSET ))"

python -u scripts/comparative_analysis.py \
  --dimension attack_fedprox \
  --dataset edge-iiotset-full \
  --num_clients "${NUM_CLIENTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --seeds "${SEEDS}" \
  --server_timeout 21600 \
  --client_timeout 21600 \
  --split-total "${SPLIT_TOTAL}" \
  --split-index "${SPLIT_INDEX}"
```

### Key Hypothesis

If FedProx fails under Byzantine attack (as expected), this motivates the Combined ablation experiments. If FedProx is surprisingly robust, that's also a publishable finding.

---

## Gap 3: Additional Attack Types (Optional P2)

**NeurIPS Requirement:** "At least 3 attack types" - currently only have grad_ascent

### Proposed Attack Types

1. **label_flipping** - Flip labels on adversarial clients
2. **gaussian_noise** - Add Gaussian noise to model updates

### Experiment Matrix (if time permits)

| Attack Type                         | Aggregators  | Alpha | Adv Fractions | Seeds | Jobs |
| ----------------------------------- | ------------ | ----- | ------------- | ----- | ---- |
| label_flipping                      | Krum, Median | 1.0   | 10%, 30%      | 5     | 20   |
| gaussian_noise                      | Krum, Median | 1.0   | 10%, 30%      | 5     | 20   |
| Combined attacks on FedAvg baseline | FedAvg       | 1.0   | 10%, 30%      | 5     | 20   |

**Total Attack Type Jobs:** 60 (stretch goal)

---

## Complete Job Summary

### Priority 0 (Must Have)

| Category            | Jobs    | Est. Time (17 nodes) |
| ------------------- | ------- | -------------------- |
| Combined Ablation   | 240     | ~4 hours             |
| FedProx + Byzantine | 180     | ~3 hours             |
| **Subtotal P0**     | **420** | **~7 hours**         |

### Priority 2 (Nice to Have)

| Category        | Jobs   | Est. Time (17 nodes) |
| --------------- | ------ | -------------------- |
| Attack Types    | 60     | ~1 hour              |
| **Subtotal P2** | **60** | **~1 hour**          |

### Grand Total

| Priority | Jobs | Cumulative |
| -------- | ---- | ---------- |
| P0       | 420  | 420        |
| P0+P2    | 480  | 480        |

**Estimated cluster time for all experiments:** ~8 hours on 17 nodes

---

## Pre-Submission Checklist

### Before Running

- [ ] Merge `cluster-experiments` to `main` (or run from cluster-experiments)
- [ ] Verify `combined_robustness` and `attack_fedprox` dimensions are available in `scripts/comparative_analysis.py`
- [ ] Verify attack_mode parameter works for `label_flip` (already implemented in client.py)
- [ ] Create results directory: `/scratch/$USER/results/neurips_final/`
- [ ] Sync latest code to cluster

### Implementation Notes

**Combined Robustness Dimension:** Implemented as `combined_robustness` in `comparative_analysis.py`.

**FedProx Under Attack Dimension:** Implemented as `attack_fedprox` in `comparative_analysis.py`.

Available dimensions: `aggregation`, `heterogeneity`, `heterogeneity_fedprox`, `attack`, `combined_robustness`, `attack_fedprox`, `privacy`, `personalization`, `hybrid`

**Reference (combined_robustness)**:

```python
def _generate_combined_robustness_configs(self) -> List[ExperimentConfig]:
    """Generate configs for robust aggregation + FedProx under Byzantine attack."""
    configs = []
    aggregations = ["krum", "bulyan"]
    fedprox_mus = [0.01, 0.1]
    alpha_values = [0.1, 0.5, 1.0, float("inf")]
    adversary_fractions = [0.0, 0.1, 0.2, 0.3]

    for agg in aggregations:
        for mu in fedprox_mus:
            for alpha in alpha_values:
                for adv in adversary_fractions:
                    for seed in self.seeds:
                        configs.append(ExperimentConfig(
                            aggregation=agg,
                            fedprox_mu=mu,
                            alpha=alpha,
                            adversary_fraction=adv,
                            seed=seed,
                            # ... other params
                        ))
    return configs
```

**Attack modes already supported:** `label_flip`, `grad_ascent` (see client.py:1137)

### After Running

- [ ] Verify all 420 P0 jobs completed with metrics.csv
- [ ] Run statistical analysis on combined ablation
- [ ] Generate publication-quality plots
- [ ] Update `docs/NEURIPS_IIOT_FULL_ANALYSIS.md` with new findings

---

## Quick Start Commands

### 1. Sync code to cluster

```bash
rsync -avz --exclude 'runs/' --exclude '.git/' --exclude '__pycache__/' \
  ~/Documents/Thesis/federated-ids/ stu:/scratch/$USER/federated-ids/
```

### 2. Submit P0 experiments (Combined + FedProx+Byzantine)

```bash
ssh stu 'cd /scratch/$USER/federated-ids && bash scripts/slurm/submit_neurips_p0.sh'
```

### 3. Monitor progress

```bash
ssh stu 'squeue -u $USER | wc -l'
```

### 4. Sync results back

```bash
rsync -avz stu:/scratch/$USER/federated-ids/runs/ \
  ~/Documents/Thesis/federated-ids/runs/
```

---

## Branch Strategy Recommendation

### Current State

- `main`: Stable, but missing cluster experiment features
- `cluster-experiments`: Has all cluster scripts and fixes
- `feat/issue-134-unified-schema`: Schema alignment (CLOSED)
- `feat/issue-135-mixed-vocab`: Mixed label handling (OPEN)

### Recommended Action

1. **Merge `cluster-experiments` into `main`** - brings cluster scripts, sparse DataLoader, temporal validation
2. **Run NeurIPS experiments from `main`** - ensures reproducibility
3. **Issue 135 can wait** - not needed for single-dataset (Edge-IIoTset) experiments

```bash
git checkout main
git merge cluster-experiments --no-ff -m "feat: merge cluster experiment infrastructure"
git push origin main
```

---

## Appendix: Mathematical Constraints

### Bulyan Byzantine Bound

```
n >= 4f + 3

With n=10 clients:
  f_max = (10 - 3) / 4 = 1.75 -> f_max = 1 (10% adversary)

Bulyan CANNOT be used at:
  - 20% adversary (f=2, need n>=11)
  - 30% adversary (f=3, need n>=15)
  - 40% adversary (f=4, need n>=19)
```

### Krum Byzantine Bound

```
n >= 2f + 3

With n=10 clients:
  f_max = (10 - 3) / 2 = 3.5 -> f_max = 3 (30% adversary)

Krum CANNOT be used at:
  - 40% adversary (f=4, need n>=11)
```

### Safe Configurations Summary

| Aggregator | Max Safe Adv% (n=10) |
| ---------- | -------------------- |
| FedAvg     | 100% (no bound)      |
| FedProx    | 100% (no bound)      |
| Median     | ~50% (practical)     |
| Krum       | 30% (f<=3)           |
| Bulyan     | 10% (f<=1)           |

---

## References

- `docs/NEURIPS_IIOT_FULL_ANALYSIS.md` - Current gap analysis
- `docs/EXPERIMENT_CONSTRAINTS.md` - Bulyan constraints
- `docs/CLUSTER_RUNS_ANALYSIS.md` - Existing experiment summary
- `docs/FEDPROX_NOVELTY_ANALYSIS.md` - FedProx findings
- `docs/TEMPORAL_VALIDATION_RUNBOOK.md` - FedProx mu selection protocol
