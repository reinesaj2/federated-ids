# NeurIPS Final Experiment Runbook

**Created:** 2025-12-19
**Purpose:** Definitive guide to complete ALL experiments required for NeurIPS submission
**Status:** AUTHORITATIVE - supersedes all previous runbooks for gap experiments

---

## Executive Summary

### Current State (1,917 experiments)

| Dimension | Coverage | Status |
|-----------|----------|--------|
| Aggregators | FedAvg, FedProx, Krum, Bulyan, Median | COMPLETE |
| Non-IID (alpha) | 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf | COMPLETE |
| Adversary fractions | 0%, 10%, 20%, 30% | PARTIAL (missing 40%) |
| Seeds | 5-20 per config | COMPLETE |
| Attack types | grad_ascent only | INCOMPLETE |
| Combined ablation | 0 experiments | CRITICAL GAP |
| FedProx + Byzantine | 0 experiments | CRITICAL GAP |

### Required New Experiments: 525 jobs

| Gap Category | New Jobs | Priority |
|--------------|----------|----------|
| Combined ablation (Robust Agg + FedProx) | 240 | P0 - Critical |
| FedProx under Byzantine | 105 | P0 - Critical |
| 40% adversary fraction | 80 | P1 - Important |
| Additional attack types | 100 | P2 - Nice to have |
| **Total** | **525** | |

---

## Gap 1: Combined Ablation (Robust Agg + FedProx)

**NeurIPS Requirement:** Demonstrate whether combining robust aggregation with FedProx provides synergistic benefits under simultaneous Byzantine + non-IID stress.

### Experiment Matrix

| Aggregator | Mu Values | Alpha Values | Adv Fractions | Seeds | Jobs |
|------------|-----------|--------------|---------------|-------|------|
| Krum | 0.01, 0.1 | 0.1, 0.5, 1.0, inf | 0, 10%, 20%, 30% | 5 | 160 |
| Bulyan | 0.01, 0.1 | 0.1, 0.5, 1.0, inf | 0, 10% | 5 | 80 |

**Note:** Bulyan limited to 10% adv due to n >= 4f+3 constraint with 10 clients.

**Total Combined Jobs:** 240

### Slurm Script: `neurips_combined_ablation.sbatch`

```bash
#!/bin/bash
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/results/neurips_combined/%x-%A_%a.out
#SBATCH --error=/scratch/%u/results/neurips_combined/%x-%A_%a.err
#SBATCH --exclusive

set -euo pipefail

source /scratch/$USER/venvs/fedids-py311/bin/activate
cd /scratch/$USER/federated-ids

export OHE_SPARSE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /scratch/$USER/results/neurips_combined

# Parameters will be set by array index
# See submit script for mapping
python scripts/comparative_analysis.py \
  --dimension combined_robustness \
  --dataset edge-iiotset-full \
  --num_clients 10 \
  --num_rounds 20 \
  --aggregation "${AGG}" \
  --fedprox-mu "${MU}" \
  --alpha-values "${ALPHA}" \
  --adversary-fraction "${ADV}" \
  --seeds "${SEED}" \
  --server_timeout 21600 \
  --client_timeout 21600
```

---

## Gap 2: FedProx Under Byzantine Attack

**NeurIPS Requirement:** Verify Claim B - "FedProx alone fails under >20% Byzantine clients"

### Experiment Matrix

| Aggregator | Mu Values | Alpha Values | Adv Fractions | Seeds | Jobs |
|------------|-----------|--------------|---------------|-------|------|
| FedProx | 0.01, 0.05, 0.1 | 0.1, 0.5, 1.0, inf | 10%, 20%, 30% | 5 | 105 |

**Note:** This tests FedProx (standard averaging + proximal term) under Byzantine attack WITHOUT robust aggregation.

**Total FedProx+Byzantine Jobs:** 105

### Key Hypothesis

If FedProx fails under Byzantine attack (as expected), this motivates the Combined ablation experiments. If FedProx is surprisingly robust, that's also a publishable finding.

---

## Gap 3: 40% Adversary Fraction

**NeurIPS Requirement:** Test extreme adversarial conditions (40% malicious clients)

### Experiment Matrix

| Aggregator | Alpha Values | Adv Fraction | Seeds | Jobs |
|------------|--------------|--------------|-------|------|
| FedAvg | 0.5, 1.0, inf | 40% | 5 | 15 |
| Krum | 0.5, 1.0, inf | 40% | 5 | 15 |
| Median | 0.5, 1.0, inf | 40% | 5 | 15 |
| FedProx | 0.5, 1.0, inf | 40% | 5 | 15 |

**Note:** Bulyan excluded (violates n >= 4f+3 at 40%)

**Total 40% Adversary Jobs:** 60

### Additional Combined at 40%

| Aggregator | Mu | Alpha Values | Adv Fraction | Seeds | Jobs |
|------------|-----|--------------|--------------|-------|------|
| Krum | 0.1 | 0.5, 1.0 | 40% | 5 | 10 |
| Median | 0.1 | 0.5, 1.0 | 40% | 5 | 10 |

**Total 40% Jobs:** 80

---

## Gap 4: Additional Attack Types (Optional P2)

**NeurIPS Requirement:** "At least 3 attack types" - currently only have grad_ascent

### Proposed Attack Types

1. **label_flipping** - Flip labels on adversarial clients
2. **gaussian_noise** - Add Gaussian noise to model updates

### Experiment Matrix (if time permits)

| Attack Type | Aggregators | Alpha | Adv Fractions | Seeds | Jobs |
|-------------|-------------|-------|---------------|-------|------|
| label_flipping | Krum, Median | 1.0 | 10%, 30% | 5 | 20 |
| gaussian_noise | Krum, Median | 1.0 | 10%, 30% | 5 | 20 |
| Combined attacks on FedAvg baseline | FedAvg | 1.0 | 10%, 30% | 5 | 20 |

**Total Attack Type Jobs:** 60 (stretch goal)

---

## Complete Job Summary

### Priority 0 (Must Have)

| Category | Jobs | Est. Time (17 nodes) |
|----------|------|----------------------|
| Combined Ablation | 240 | ~4 hours |
| FedProx + Byzantine | 105 | ~2 hours |
| **Subtotal P0** | **345** | **~6 hours** |

### Priority 1 (Should Have)

| Category | Jobs | Est. Time (17 nodes) |
|----------|------|----------------------|
| 40% Adversary | 80 | ~1.5 hours |
| **Subtotal P1** | **80** | **~1.5 hours** |

### Priority 2 (Nice to Have)

| Category | Jobs | Est. Time (17 nodes) |
|----------|------|----------------------|
| Attack Types | 60 | ~1 hour |
| **Subtotal P2** | **60** | **~1 hour** |

### Grand Total

| Priority | Jobs | Cumulative |
|----------|------|------------|
| P0 | 345 | 345 |
| P0+P1 | 425 | 425 |
| P0+P1+P2 | 525 | 525 |

**Estimated cluster time for all experiments:** ~8.5 hours on 17 nodes

---

## Pre-Submission Checklist

### Before Running

- [ ] Merge `cluster-experiments` to `main` (or run from cluster-experiments)
- [ ] **ADD** `combined_robustness` dimension to `scripts/comparative_analysis.py` (see Implementation Notes below)
- [ ] Verify attack_mode parameter works for `label_flip` (already implemented in client.py)
- [ ] Create results directory: `/scratch/$USER/results/neurips_final/`
- [ ] Sync latest code to cluster

### Implementation Notes

**Combined Robustness Dimension:** Currently NOT implemented in `comparative_analysis.py`. 

Available dimensions: `aggregation`, `heterogeneity`, `heterogeneity_fedprox`, `attack`, `privacy`, `personalization`, `hybrid`

**To add combined_robustness**, add to `scripts/comparative_analysis.py`:

```python
def _generate_combined_robustness_configs(self) -> List[ExperimentConfig]:
    """Generate configs for robust aggregation + FedProx under Byzantine attack."""
    configs = []
    aggregations = ["krum", "bulyan"]
    mus = [0.01, 0.1]
    alphas = [0.1, 0.5, 1.0, float("inf")]
    adv_fractions = [0.0, 0.1, 0.2, 0.3]
    
    for agg in aggregations:
        for mu in mus:
            for alpha in alphas:
                for adv in adv_fractions:
                    # Skip Bulyan at high adversary (violates n >= 4f+3)
                    if agg == "bulyan" and adv > 0.1:
                        continue
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

- [ ] Verify all 525 jobs completed with metrics.csv
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
|------------|----------------------|
| FedAvg | 100% (no bound) |
| FedProx | 100% (no bound) |
| Median | ~50% (practical) |
| Krum | 30% (f<=3) |
| Bulyan | 10% (f<=1) |

---

## References

- `docs/NEURIPS_IIOT_FULL_ANALYSIS.md` - Current gap analysis
- `docs/EXPERIMENT_CONSTRAINTS.md` - Bulyan constraints
- `docs/CLUSTER_RUNS_ANALYSIS.md` - Existing experiment summary
- `docs/FEDPROX_NOVELTY_ANALYSIS.md` - FedProx findings
- `docs/TEMPORAL_VALIDATION_RUNBOOK.md` - FedProx mu selection protocol
