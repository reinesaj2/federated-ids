#!/bin/bash

set -euo pipefail

MAX_CONCURRENT="${MAX_CONCURRENT:-17}"

cd "/scratch/${USER}/federated-ids"
mkdir -p "/scratch/${USER}/results/neurips_combined" "/scratch/${USER}/results/neurips_fedprox_attack"

mkdir -p data/edge-iiotset
ln -sf "/scratch/${USER}/datasets/edge-iiotset/edge_iiotset_full.csv" "data/edge-iiotset/edge_iiotset_full.csv"

echo "Submitting NeurIPS P0 experiments with MAX_CONCURRENT=${MAX_CONCURRENT}"

COMBINED_JOB_ID="$(sbatch --parsable --array=0-239%${MAX_CONCURRENT} scripts/slurm/neurips_combined_ablation.sbatch)"
echo "combined_array_job_id=${COMBINED_JOB_ID}"

FEDPROX_JOB_ID="$(sbatch --parsable --dependency=afterok:${COMBINED_JOB_ID} --array=0-179%${MAX_CONCURRENT} scripts/slurm/neurips_fedprox_attack.sbatch)"
echo "fedprox_attack_array_job_id=${FEDPROX_JOB_ID}"

echo "Done. Monitor with: squeue -u ${USER}"
