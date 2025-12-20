#!/bin/bash

set -euo pipefail

MAX_CONCURRENT="${MAX_CONCURRENT:-17}"

cd "/scratch/${USER}/federated-ids"
mkdir -p "/scratch/${USER}/results/iiot_full"

mkdir -p data/edge-iiotset
ln -sf "/scratch/${USER}/datasets/edge-iiotset/edge_iiotset_full.csv" "data/edge-iiotset/edge_iiotset_full.csv"

echo "Submitting IIoT full heterogeneity grid with MAX_CONCURRENT=${MAX_CONCURRENT}"

SMOKE_JOB_ID="$(sbatch --parsable scripts/slurm/iiot_full_heterogeneity_smoke_10c_20r.sbatch)"
echo "smoke_job_id=${SMOKE_JOB_ID}"

FEDAVG_JOB_ID="$(sbatch --parsable --dependency=afterok:${SMOKE_JOB_ID} --array=0-34%${MAX_CONCURRENT} scripts/slurm/iiot_full_heterogeneity_fedavg_array.sbatch)"
echo "fedavg_array_job_id=${FEDAVG_JOB_ID}"

FEDPROX_JOB_ID="$(sbatch --parsable --dependency=afterok:${FEDAVG_JOB_ID} --array=0-279%${MAX_CONCURRENT} scripts/slurm/iiot_full_heterogeneity_fedprox_array.sbatch)"
echo "fedprox_array_job_id=${FEDPROX_JOB_ID}"

echo "Done. Monitor with: squeue -u ${USER}"
