# Thesis Experiments Checklist - Full Edge-IIoTSet Dataset

**Purpose:** Track completion of all experiments for thesis validation
**Dataset:** edge-iiotset-full (1,701,692 samples)
**Workers:** 1 (sequential execution)
**Total Experiments:** 28 (18 original + 10 added for completeness)
**Status:** 9/28 completed (FedAvg baseline, Krum, Bulyan, Median, alpha=0.02/0.1/0.5/inf and IID baseline)

---

## Progress Summary

- [ ] **Objective 1: Robust Aggregation** (4 experiments) - 4/4 complete
- [ ] **Objective 2: Data Heterogeneity + FedProx** (9 experiments) - 5/9 complete
- [ ] **Objective 3: Attack Resilience** (6 experiments) - 0/6 complete
- [ ] **Objective 4: Privacy/Security** (6 experiments) - 0/6 complete
- [ ] **Objective 5: Personalization** (3 experiments) - 0/3 complete

---

## OBJECTIVE 1: ROBUST AGGREGATION METHODS (4/20)

Compare FedAvg, Krum, Bulyan, and Median aggregation strategies.

### [x] Experiment 1: FedAvg (Baseline)
- **Status:** COMPLETED (seed46 run; 15 rounds recorded, final round 5/6 clients)
- **Started:** 2025-11-18 12:34 PM (seed46)
- **PID:** n/a (no active run)
- **Expected Completion:** 2-4 PM
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed46/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension aggregation --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedavg --seed 42 --client-timeout-sec 7200 \
    > logs/exp01_aggregation_fedavg.log 2>&1 &
  ```

---

### [x] Experiment 2: Krum
- **Status:** COMPLETED (seed42, 15/15 rounds, stored artifacts retained)
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

### [x] Experiment 3: Bulyan
- **Status:** COMPLETED (seed42, 15/15 rounds)
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

### [x] Experiment 4: Median
- **Status:** COMPLETED (seed42, 15/15 rounds)
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

### [x] Experiment 5: Alpha = 0.02 (Highly Non-IID)
- **Status:** COMPLETED (seed42, 15/15 rounds; high heterogeneity FedAvg)
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

- **Completed heterogeneity configs:** dsedge-iiotset-full_comp_fedavg_alpha0.02_adv0_dp0_pers0_mu0.0_seed42 (15 rounds), dsedge-iiotset-full_comp_fedavg_alpha0.1_adv0_dp0_pers0_mu0.0_seed42 (15 rounds), dsedge-iiotset-full_comp_fedavg_alpha0.5_adv0_dp0_pers0_mu0.0_seed42 (15 rounds), dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42 (baseline reuse), dsedge-iiotset-full_comp_fedavg_alphainf_adv0_dp0_pers0_mu0.0_seed42 (15 rounds)

### [x] Experiment 6: Alpha = 0.1
- **Status:** COMPLETED (seed42, 15/15 rounds captured)
- **Config:** aggregation=fedavg, alpha=0.1, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha0.1_adv0_dp0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --preset dsedge-iiotset-full_comp_fedavg_alpha0.1_adv0_dp0_pers0_mu0.0_seed42 \
    --client-timeout-sec 7200 \
    > logs/exp06_heterogeneity_alpha0.1.log 2>&1 &
  ```

---

### [x] Experiment 7: Alpha = 0.5
- **Status:** COMPLETED (seed42, 15/15 rounds)
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

### [x] Experiment 8: Alpha = 1.0
- **Status:** COMPLETED (same as Experiment 1; artifacts reused from baseline)
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/`
- **Note:** Same as Experiment 1 - NO NEED TO RE-RUN
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --alpha 1.0 --seed 42 --client-timeout-sec 7200 \
    > logs/exp08_heterogeneity_alpha1.0.log 2>&1 &
  ```

---

### [x] Experiment 9: Alpha = inf (IID / Uniform)
- **Status:** COMPLETED (seed42 IID reference run, 15/15 rounds)
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

### [ ] Experiment 10: FedProx with mu=0.01 (Low Regularization)
- **Status:** PENDING
- **Config:** aggregation=fedprox, alpha=0.02, adversary=0.0, dp=0.0, pers=0, mu=0.01, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedprox_alpha0.02_adv0_dp0_pers0_mu0.01_seed42/`
- **Note:** Test FedProx on highly non-IID data (alpha=0.02) to show improvement over FedAvg
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity_fedprox --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedprox --alpha 0.02 --fedprox-mu 0.01 --seed 42 \
    --client-timeout-sec 7200 > logs/exp10_fedprox_mu0.01_alpha0.02.log 2>&1 &
  ```

---

### [ ] Experiment 11: FedProx with mu=0.1 (Medium Regularization)
- **Status:** PENDING
- **Config:** aggregation=fedprox, alpha=0.02, adversary=0.0, dp=0.0, pers=0, mu=0.1, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedprox_alpha0.02_adv0_dp0_pers0_mu0.1_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity_fedprox --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedprox --alpha 0.02 --fedprox-mu 0.1 --seed 42 \
    --client-timeout-sec 7200 > logs/exp11_fedprox_mu0.1_alpha0.02.log 2>&1 &
  ```

---

### [ ] Experiment 12: FedProx with mu=1.0 (High Regularization)
- **Status:** PENDING
- **Config:** aggregation=fedprox, alpha=0.02, adversary=0.0, dp=0.0, pers=0, mu=1.0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedprox_alpha0.02_adv0_dp0_pers0_mu1.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity_fedprox --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedprox --alpha 0.02 --fedprox-mu 1.0 --seed 42 \
    --client-timeout-sec 7200 > logs/exp12_fedprox_mu1.0_alpha0.02.log 2>&1 &
  ```

---

### [ ] Experiment 13: FedProx on IID data (Baseline Comparison)
- **Status:** PENDING
- **Config:** aggregation=fedprox, alpha=1.0, adversary=0.0, dp=0.0, pers=0, mu=0.1, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedprox_alpha1.0_adv0_dp0_pers0_mu0.1_seed42/`
- **Note:** Compare FedProx vs FedAvg on IID data to show it doesn't hurt performance
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension heterogeneity_fedprox --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --strategy fedprox --alpha 1.0 --fedprox-mu 0.1 --seed 42 \
    --client-timeout-sec 7200 > logs/exp13_fedprox_mu0.1_alpha1.0.log 2>&1 &
  ```

---

## OBJECTIVE 3: ATTACK RESILIENCE (6/28)

Test Byzantine attack resistance with different aggregation strategies.

### [ ] Experiment 14: FedAvg + 10% Adversaries
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

## OBJECTIVE 4: PRIVACY/SECURITY (6/28)

Evaluate Differential Privacy and Secure Aggregation impact on model utility.

### [ ] Experiment 20: Secure Aggregation Only (No DP)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=0, secure_agg=true, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_secagg1_mu0.0_seed42/`
- **Note:** Test overhead of secure aggregation without DP noise
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --secure-aggregation --seed 42 --client-timeout-sec 7200 \
    > logs/exp20_privacy_secagg_only.log 2>&1 &
  ```

---

### [ ] Experiment 21: DP Noise = 0.5 (Medium Privacy)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp_noise=0.5, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dpnoise0.5_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-noise 0.5 --seed 42 --client-timeout-sec 7200 \
    > logs/exp21_privacy_dp0.5.log 2>&1 &
  ```

---

### [ ] Experiment 22: DP Noise = 1.0 (High Privacy)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp_noise=1.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dpnoise1.0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-noise 1.0 --seed 42 --client-timeout-sec 7200 \
    > logs/exp22_privacy_dp1.0.log 2>&1 &
  ```

---

### [ ] Experiment 23: DP Noise = 2.0 (Very High Privacy)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp_noise=2.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dpnoise2.0_pers0_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-noise 2.0 --seed 42 --client-timeout-sec 7200 \
    > logs/exp23_privacy_dp2.0.log 2>&1 &
  ```

---

### [ ] Experiment 24: Secure Aggregation + DP Combined
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp_noise=0.5, pers=0, secure_agg=true, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dpnoise0.5_pers0_secagg1_mu0.0_seed42/`
- **Note:** Test both privacy mechanisms together (thesis requirement)
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --secure-aggregation --dp-noise 0.5 --seed 42 \
    --client-timeout-sec 7200 > logs/exp24_privacy_secagg_dp0.5.log 2>&1 &
  ```

---

### [ ] Experiment 25: DP with Clipping (Privacy Accounting)
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp_epsilon=3.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dpeps3.0_pers0_mu0.0_seed42/`
- **Note:** Use epsilon-based DP for proper privacy accounting
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension privacy --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --dp-epsilon 3.0 --seed 42 --client-timeout-sec 7200 \
    > logs/exp25_privacy_dpeps3.0.log 2>&1 &
  ```

---

## OBJECTIVE 5: PERSONALIZATION (3/28)

Test local model fine-tuning after federated training.

### [ ] Experiment 26: No Personalization (Baseline)
- **Status:** COMPLETED (same as Experiment 1)
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=0, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed46/`
- **Note:** Same as Experiment 1 - NO NEED TO RE-RUN
- **Command:** N/A (skip, already completed in Exp 1)

---

### [ ] Experiment 27: Personalization = 3 Epochs
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=3, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers3_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension personalization --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --personalization-epochs 3 --seed 42 --client-timeout-sec 7200 \
    > logs/exp27_personalization_pers3.log 2>&1 &
  ```

---

### [ ] Experiment 28: Personalization = 5 Epochs
- **Status:** PENDING
- **Config:** aggregation=fedavg, alpha=1.0, adversary=0.0, dp=0.0, pers=5, seed=42
- **Results Path:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers5_mu0.0_seed42/`
- **Command:**
  ```bash
  cd ~/Documents/Thesis/worktrees/iiot-experiments && \
  nohup .venv/bin/python scripts/run_experiments_optimized.py \
    --dimension personalization --dataset edge-iiotset-full --dataset-type full \
    --workers 1 --personalization-epochs 5 --seed 42 --client-timeout-sec 7200 \
    > logs/exp28_personalization_pers5.log 2>&1 &
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

- **Shared Baseline Experiments:**
  - Experiment 1 = Experiment 8 = Experiment 26: FedAvg baseline (already complete)
- **Actual unique experiments:** 26 out of 28 listed
- **Estimated runtime per experiment:** 2-3 hours
- **Total sequential runtime:** ~52-78 hours (2.2-3.25 days continuous)
- **Critical for thesis defense:** FedProx (Exp 10-13) and SecAgg (Exp 20, 24)

---

## INTELLIGENT EXPERIMENT ORDERING (Recommended Sequence)

1. **PRIORITY 1 - Complete Heterogeneity Sweep** (Exp 6-9): Finish alpha variations
2. **PRIORITY 2 - FedProx Validation** (Exp 10-13): Critical thesis objective, test on non-IID
3. **PRIORITY 3 - Attack Resilience** (Exp 14-19): Show robust aggregation benefits
4. **PRIORITY 4 - Privacy Mechanisms** (Exp 20-25): DP + SecAgg for thesis Objective 4
5. **PRIORITY 5 - Personalization** (Exp 27-28): Final comparison dimension

---

## KEY THESIS CONTRIBUTIONS

With all 28 experiments complete, you will have:

[OK] **Novel**: First comprehensive robust FL evaluation on Edge-IIoT dataset
[OK] **Complete**: All 5 thesis objectives fully addressed with experimental evidence
[OK] **Rigorous**: Attack resilience (0%, 10%, 20%, 30% adversaries) with smooth degradation curves
[OK] **Privacy**: Both Secure Aggregation AND Differential Privacy tested (separately + combined)
[OK] **Non-IID**: FedProx vs FedAvg comparison across heterogeneity spectrum (α=0.02 to 1.0)
[OK] **Personalization**: Local fine-tuning gains quantified (0, 3, 5 epochs)

---

**Last Updated:** 2025-11-19 2:10 PM (Intelligent additions by ML defense scientist review)
**Current Status:** 5/28 complete (18%)
**Next Recommended:** Complete Exp 6-9 (heterogeneity), then prioritize FedProx (Exp 10-13)
**Estimated Completion:** 2025-11-22 (if run continuously)
