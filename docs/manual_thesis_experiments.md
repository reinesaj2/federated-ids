# Manual Thesis Experiments - Full Edge-IIoTSet Dataset

**System Configuration:**
- CPU: 10 cores
- RAM: 32 GB
- Workers per experiment: 6 (conservative to avoid crashes)
- Dataset: edge-iiotset-full (1,701,692 samples, 934MB)
- Execution: ONE experiment at a time, manual launch

**Total Experiments: 20**
**Estimated Time per Experiment: 2-3 hours**
**Total Estimated Time: 40-60 hours**

---

## Thesis Objectives Coverage

This minimal set of 20 experiments satisfies all 5 thesis objectives:

1. **Robust Aggregation Methods** - Experiments 1-4 (compare FedAvg, Krum, Bulyan, Median)
2. **Data Heterogeneity** - Experiments 5-9 (vary alpha from 0.02 to inf)
3. **Personalization** - Experiments 18-20 (vary personalization epochs)
4. **Privacy/Security** - Experiments 14-17 (vary DP noise)
5. **Empirical Validation** - All experiments use Edge-IIoTSet dataset

Attack resilience: Experiments 10-13 (Byzantine attacks with different aggregation strategies)

---

## DIMENSION 1: AGGREGATION (4 experiments)

### Baseline Parameters
- alpha: 1.0 (homogeneous data distribution)
- adversary_fraction: 0.0 (no Byzantine attackers)
- dp_noise: 0.0 (no differential privacy)
- personalization_epochs: 0 (no personalization)
- seed: 42

### Experiment 1: FedAvg (Baseline)
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension aggregation \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy fedavg \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp01_aggregation_fedavg.log 2>&1 &
echo "Experiment 1 PID: $!"
```

**Monitoring:**
```bash
tail -f logs/exp01_aggregation_fedavg.log
ps aux | grep run_experiments_optimized
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 2: Krum
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension aggregation \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy krum \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp02_aggregation_krum.log 2>&1 &
echo "Experiment 2 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_krum_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 3: Bulyan
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension aggregation \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy bulyan \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp03_aggregation_bulyan.log 2>&1 &
echo "Experiment 3 PID: $!"
```

**Note:** Bulyan requires n >= 4f + 3 clients. With 6 workers, max adversaries f = 0 (no room for Byzantine nodes in this configuration).

**Expected Output:** `runs/dsedge-iiotset-full_comp_bulyan_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 4: Median
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension aggregation \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy median \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp04_aggregation_median.log 2>&1 &
echo "Experiment 4 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_median_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

## DIMENSION 2: HETEROGENEITY (5 experiments)

### Baseline Parameters
- aggregation: fedavg
- adversary_fraction: 0.0
- dp_noise: 0.0
- personalization_epochs: 0
- seed: 42

**Vary: alpha (data heterogeneity level)**

---

### Experiment 5: Alpha = 0.02 (Highly Non-IID)
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --alpha 0.02 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp05_heterogeneity_alpha0.02.log 2>&1 &
echo "Experiment 5 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha0.02_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 6: Alpha = 0.1
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --alpha 0.1 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp06_heterogeneity_alpha0.1.log 2>&1 &
echo "Experiment 6 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha0.1_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 7: Alpha = 0.5
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --alpha 0.5 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp07_heterogeneity_alpha0.5.log 2>&1 &
echo "Experiment 7 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha0.5_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 8: Alpha = 1.0
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --alpha 1.0 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp08_heterogeneity_alpha1.0.log 2>&1 &
echo "Experiment 8 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 9: Alpha = inf (IID / Uniform Distribution)
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension heterogeneity \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --alpha inf \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp09_heterogeneity_alphainf.log 2>&1 &
echo "Experiment 9 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alphainf_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

## DIMENSION 3: ATTACK RESILIENCE (4 experiments)

### Baseline Parameters
- alpha: 1.0
- dp_noise: 0.0
- personalization_epochs: 0
- seed: 42

**Vary: adversary_fraction (Byzantine attacker percentage) and aggregation strategy**

**NOTE:** With 6 workers, adversary_fraction=0.1 means 0.6 adversaries (rounds to 1), adversary_fraction=0.3 means 1.8 adversaries (rounds to 2).

---

### Experiment 10: FedAvg + 10% Adversaries
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension attack \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy fedavg \
  --adversary-fraction 0.1 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp10_attack_fedavg_adv0.1.log 2>&1 &
echo "Experiment 10 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0.1_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 11: Krum + 10% Adversaries
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension attack \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy krum \
  --adversary-fraction 0.1 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp11_attack_krum_adv0.1.log 2>&1 &
echo "Experiment 11 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_krum_alpha1.0_adv0.1_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 12: Median + 10% Adversaries
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension attack \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy median \
  --adversary-fraction 0.1 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp12_attack_median_adv0.1.log 2>&1 &
echo "Experiment 12 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_median_alpha1.0_adv0.1_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 13: FedAvg + 30% Adversaries
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension attack \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --strategy fedavg \
  --adversary-fraction 0.3 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp13_attack_fedavg_adv0.3.log 2>&1 &
echo "Experiment 13 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0.3_dp0_pers0_mu0.0_seed42/metrics.csv`

---

## DIMENSION 4: PRIVACY (4 experiments)

### Baseline Parameters
- aggregation: fedavg
- alpha: 1.0
- adversary_fraction: 0.0
- personalization_epochs: 0
- seed: 42

**Vary: dp_noise (Differential Privacy noise multiplier)**

---

### Experiment 14: DP Noise = 0.3 (Low Privacy)
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension privacy \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --dp-noise 0.3 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp14_privacy_dp0.3.log 2>&1 &
echo "Experiment 14 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0.3_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 15: DP Noise = 0.5 (Medium Privacy)
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension privacy \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --dp-noise 0.5 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp15_privacy_dp0.5.log 2>&1 &
echo "Experiment 15 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0.5_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 16: DP Noise = 1.0 (High Privacy)
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension privacy \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --dp-noise 1.0 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp16_privacy_dp1.0.log 2>&1 &
echo "Experiment 16 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp1.0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 17: DP Noise = 2.0 (Very High Privacy)
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension privacy \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --dp-noise 2.0 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp17_privacy_dp2.0.log 2>&1 &
echo "Experiment 17 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp2.0_pers0_mu0.0_seed42/metrics.csv`

---

## DIMENSION 5: PERSONALIZATION (3 experiments)

### Baseline Parameters
- aggregation: fedavg
- alpha: 1.0
- adversary_fraction: 0.0
- dp_noise: 0.0
- seed: 42

**Vary: personalization_epochs (local fine-tuning after federated training)**

---

### Experiment 18: No Personalization (Baseline - Already covered in Exp 1)
**Note:** This is the same as Experiment 1 (FedAvg baseline). No need to re-run.

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/metrics.csv`

---

### Experiment 19: Personalization = 3 Epochs
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension personalization \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --personalization-epochs 3 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp19_personalization_pers3.log 2>&1 &
echo "Experiment 19 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers3_mu0.0_seed42/metrics.csv`

---

### Experiment 20: Personalization = 5 Epochs
```bash
cd ~/Documents/Thesis/worktrees/iiot-experiments && \
nohup .venv/bin/python scripts/run_experiments_optimized.py \
  --dimension personalization \
  --dataset edge-iiotset-full \
  --dataset-type full \
  --workers 6 \
  --personalization-epochs 5 \
  --seed 42 \
  --client-timeout-sec 7200 \
  > logs/exp20_personalization_pers5.log 2>&1 &
echo "Experiment 20 PID: $!"
```

**Expected Output:** `runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers5_mu0.0_seed42/metrics.csv`

---

## Monitoring Commands

### Check Running Processes
```bash
ps aux | grep run_experiments_optimized
```

### Check System Resources
```bash
# CPU and Memory
top -l 1 | grep -E "CPU|PhysMem"

# Disk space
df -h ~/Documents/Thesis/worktrees/iiot-experiments/runs
```

### Count Completed Experiments
```bash
find ~/Documents/Thesis/worktrees/iiot-experiments/runs -name "metrics.csv" -type f | wc -l
```

### View Latest Log
```bash
tail -f logs/exp<NUMBER>_*.log
```

### Kill Stuck Process
```bash
pkill -f run_experiments_optimized
```

---

## Progress Tracking

Create/update tracking CSV after each experiment:

```bash
echo "exp_num,dimension,strategy,alpha,adversary_fraction,dp_noise,personalization_epochs,started_at,finished_at,status,metrics_file" > ~/Documents/Thesis/worktrees/iiot-experiments/docs/thesis_experiment_tracker.csv
```

After each experiment completes, append a row:
```bash
echo "1,aggregation,fedavg,1.0,0.0,0.0,0,2025-11-17 15:30,2025-11-17 17:45,SUCCESS,runs/dsedge-iiotset-full_comp_fedavg_alpha1.0_adv0_dp0_pers0_mu0.0_seed42/metrics.csv" >> docs/thesis_experiment_tracker.csv
```

---

## Verification Checklist

After completing all 20 experiments, verify:

- [ ] 20 metrics.csv files exist in runs/ directory
- [ ] All 5 thesis objectives have supporting data
- [ ] No experiments failed (check logs for errors)
- [ ] Disk space remaining > 10GB
- [ ] All results backed up to external storage

---

**NEXT STEP: Run Experiment 1**

When ready to begin, copy and paste the Experiment 1 command above.
