#!/bin/bash
set -euo pipefail

MAX_CONCURRENT="${MAX_CONCURRENT:-17}"

echo "=== Temporal Validation Protocol Experiment Submission ==="
echo "Max concurrent jobs: $MAX_CONCURRENT"
echo "Timestamp: $(date)"
echo ""

echo "Phase 1: Tuning (seeds 42-44)"
echo "================================"
echo ""

echo "1a) FedProx tuning sweep (210 jobs = 7 alphas x 10 mu x 3 seeds)"
FEDPROX_JOB=$(sbatch --parsable --array=0-209%${MAX_CONCURRENT} scripts/slurm/temporal_validation_tuning.sbatch)
echo "    Submitted: Job array $FEDPROX_JOB (tasks 0-209)"

echo ""
echo "1b) FedAvg baseline sweep (21 jobs = 7 alphas x 3 seeds)"
FEDAVG_JOB=$(sbatch --parsable --array=0-20%${MAX_CONCURRENT} scripts/slurm/temporal_validation_baseline.sbatch)
echo "    Submitted: Job array $FEDAVG_JOB (tasks 0-20)"

echo ""
echo "=== Submission Complete ==="
echo ""
echo "Total jobs submitted: 231"
echo "  - FedProx tuning: $FEDPROX_JOB (210 tasks)"
echo "  - FedAvg baseline: $FEDAVG_JOB (21 tasks)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER | wc -l'"
echo ""
echo "After tuning completes:"
echo "  1. Run: python scripts/select_mu_star.py --runs_dir runs"
echo "  2. Review mu* selections per alpha"
echo "  3. Run evaluation phase with seeds 45-49"
echo ""
echo "Estimated time with $MAX_CONCURRENT concurrent jobs:"
echo "  ~7 min/job x 231 jobs / $MAX_CONCURRENT nodes = ~$(( 7 * 231 / MAX_CONCURRENT )) minutes"
