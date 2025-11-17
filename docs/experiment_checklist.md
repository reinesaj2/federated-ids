# Thesis Experiments Checklist - Full Edge-IIoTSet Dataset

**Purpose:** Track completion of all 20 experiments for thesis validation
**Dataset:** edge-iiotset-full (1,701,692 samples)
**Workers:** 1 (sequential execution)
**Total Experiments:** 20
**Status:** 0/20 completed

---

## Progress Summary

- [ ] **Objective 1: Robust Aggregation** (4 experiments) - 0/4 complete
- [ ] **Objective 2: Data Heterogeneity** (5 experiments) - 0/5 complete
- [ ] **Objective 3: Attack Resilience** (4 experiments) - 0/4 complete
- [ ] **Objective 4: Privacy/Security** (4 experiments) - 0/4 complete
- [ ] **Objective 5: Personalization** (3 experiments) - 0/3 complete

---

## OBJECTIVE 1: ROBUST AGGREGATION METHODS (4/20)

Compare FedAvg, Krum, Bulyan, and Median aggregation strategies.

### [ ] Experiment 1: FedAvg (Baseline)
- **Status:** IN PROGRESS
- **Started:** 2025-11-17 12:19 PM
- **PID:** 43544
- **Expected Completion:** 2-4 PM
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension aggregation --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedavg --seed 42 --client-timeout-sec 7200 \
    > logs/exp01_aggregation_fedavg.log 2>&1 &
  ```

---

### [ ] Experiment 2: Krum
- **Status:** PENDING
- **Config:** aggregation=krum, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_krum_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension aggregation --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy krum --seed 42 --client-timeout-sec 7200 \
    > logs/exp02_aggregation_krum.log 2>&1 &
  ```

---

### [ ] Experiment 3: Bulyan
- **Status:** PENDING
- **Config:** aggregation=bulyan, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_bulyan_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension aggregation --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy bulyan --seed 42 --client-timeout-sec 7200 \
    > logs/exp03_aggregation_bulyan.log 2>&1 &
  ```

---

### [ ] Experiment 4: Median
- **Status:** PENDING
- **Config:** aggregation=median, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_median_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension aggregation --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy median --seed 42 --client-timeout-sec 7200 \
    > logs/exp04_aggregation_median.log 2>&1 &
  ```

---

## OBJECTIVE 2: DATA HETEROGENEITY (5/20)

Evaluate performance across different data distributions (Non-IID via Dirichlet alpha).

### [ ] Experiment 5: Alpha = 0.02 (Highly Non-IID)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=0.02, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha0.02_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --alpha 0.02 --seed 42 --client-timeout-sec 7200 \
    > logs/exp05_heterogeneity_alpha0.02.log 2>&1 &
  ```

---

### [ ] Experiment 6: Alpha = 0.1
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=0.1, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha0.1_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --alpha 0.1 --seed 42 --client-timeout-sec 7200 \
    > logs/exp06_heterogeneity_alpha0.1.log 2>&1 &
  ```

---

### [ ] Experiment 7: Alpha = 0.5
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=0.5, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha0.5_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --alpha 0.5 --seed 42 --client-timeout-sec 7200 \
    > logs/exp07_heterogeneity_alpha0.5.log 2>&1 &
  ```

---

### [ ] Experiment 8: Alpha = 1.0
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/`
- **Note:** Same as Experiment 1 - NO NEED TO RE-RUN
- **Command:** N/A (skip, already completed in Exp 1)

---

### [ ] Experiment 9: Alpha = inf (IID / Uniform)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=inf, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alphainf_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --alpha inf --seed 42 --client-timeout-sec 7200 \
    > logs/exp09_heterogeneity_alphainf.log 2>&1 &
  ```

---

## OBJECTIVE 3: ATTACK RESILIENCE (4/20)

Test Byzantine attack resistance with different aggregation strategies.

### [ ] Experiment 10: FedAvg + 10% Adversaries
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.1, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0.1_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension attack --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedavg --adversary-fraction 0.1 --seed 42 \
    --client-timeout-sec 7200 > logs/exp10_attack_fedavg_adv0.1.log 2>&1 &
  ```

---

### [ ] Experiment 11: Krum + 10% Adversaries
- **Status:** PENDING
- **Config:** aggregation=krum, alpha=1.0, adversary=0.1, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_krum_alpha1.0_adv0.1_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension attack --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy krum --adversary-fraction 0.1 --seed 42 \
    --client-timeout-sec 7200 > logs/exp11_attack_krum_adv0.1.log 2>&1 &
  ```

---

### [ ] Experiment 12: Median + 10% Adversaries
- **Status:** PENDING
- **Config:** aggregation=median, alpha=1.0, adversary=0.1, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_median_alpha1.0_adv0.1_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension attack --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy median --adversary-fraction 0.1 --seed 42 \
    --client-timeout-sec 7200 > logs/exp12_attack_median_adv0.1.log 2>&1 &
  ```

---

### [ ] Experiment 13: FedAvg + 30% Adversaries
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.3, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0.3_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension attack --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedavg --adversary-fraction 0.3 --seed 42 \
    --client-timeout-sec 7200 > logs/exp13_attack_fedavg_adv0.3.log 2>&1 &
  ```

---

## OBJECTIVE 4: PRIVACY/SECURITY (4/20)

Evaluate Differential Privacy impact on model utility.

### [ ] Experiment 14: DP Noise = 0.3 (Low Privacy)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.3, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0.3_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-noise 0.3 --seed 42 --client-timeout-sec 7200 \
    > logs/exp14_privacy_dp0.3.log 2>&1 &
  ```

---

### [ ] Experiment 15: DP Noise = 0.5 (Medium Privacy)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.5, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0.5_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-noise 0.5 --seed 42 --client-timeout-sec 7200 \
    > logs/exp15_privacy_dp0.5.log 2>&1 &
  ```

---

### [ ] Experiment 16: DP Noise = 1.0 (High Privacy)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=1.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp1.0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-noise 1.0 --seed 42 --client-timeout-sec 7200 \
    > logs/exp16_privacy_dp1.0.log 2>&1 &
  ```

---

### [ ] Experiment 17: DP Noise = 2.0 (Very High Privacy)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=2.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp2.0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-noise 2.0 --seed 42 --client-timeout-sec 7200 \
    > logs/exp17_privacy_dp2.0.log 2>&1 &
  ```

---

## OBJECTIVE 5: PERSONALIZATION (3/20)

Test local model fine-tuning after federated training.

### [ ] Experiment 18: No Personalization (Baseline)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/`
- **Note:** Same as Experiment 1 - NO NEED TO RE-RUN
- **Command:** N/A (skip, already completed in Exp 1)

---

### [ ] Experiment 19: Personalization = 3 Epochs
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=3, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers3_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension personalization --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --personalization-epochs 3 --seed 42 --client-timeout-sec 7200 \
    > logs/exp19_personalization_pers3.log 2>&1 &
  ```

---

### [ ] Experiment 20: Personalization = 5 Epochs
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=5, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers5_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension personalization --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --personalization-epochs 5 --seed 42 --client-timeout-sec 7200 \
    > logs/exp20_personalization_pers5.log 2>&1 &
  ```

---

## QUICK VERIFICATION COMMANDS

### Count completed experiments:
```bash
find ~/Documents/Thesis/worktrees/iiot-experiments/runs -name "metrics.csv" -type f | wc -l
```

### List all completed experiment directories:
```bash
ls -d ~/Documents/Thesis/worktrees/iiot-experiments/runs/dsedge-iiotset-full_*/
```

### Check specific experiment completion:
```bash
# Replace <config_name> with experiment config
cat ~/Documents/Thesis/worktrees/iiot-experiments/runs/<config_name>/metrics.csv | wc -l
# Should return > 1 if completed (header + data rows)
```

### Find currently running experiment:
```bash
ps aux | grep run_experiments_optimized | grep -v grep
```

---

## NOTES

- **Experiment 1 = Experiment 8 = Experiment 18:** FedAvg with alpha=1.0, no attacks, no DP, no personalization. Only needs to run ONCE.
- **Actual unique experiments:** 18 out of 20 listed
- **Estimated runtime per experiment:** 2-3 hours
- **Total sequential runtime:** ~36-54 hours (1.5-2.25 days continuous)
- **Started:** 2025-11-17
- **Target completion:** 2025-11-19 or 2025-11-20

---

**Last Updated:** 2025-11-17 12:21 PM
**Current Experiment:** Experiment 1 (PID 43544)
**Progress:** 0/18 unique experiments complete
