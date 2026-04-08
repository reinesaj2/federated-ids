#!/bin/bash

set -euo pipefail

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
SPLIT_TOTAL="${SPLIT_TOTAL:-20}"
ARRAY_END="$((SPLIT_TOTAL - 1))"
DATASETS="${DATASETS:-cic unsw edge-iiotset-full}"

cd "/scratch/${USER}/federated-ids"
mkdir -p "/scratch/${USER}/results/paper_bulyan20"

declare -A DATASET_SOURCE_PATHS=(
  [cic]="/scratch/${USER}/datasets/cic/cic_ids2017_multiclass.csv"
  [unsw]="/scratch/${USER}/datasets/unsw/UNSW_NB15_training-set.csv"
  [edge-iiotset-full]="/scratch/${USER}/datasets/edge-iiotset/edge_iiotset_full.csv"
)

declare -A DATASET_LINK_PATHS=(
  [cic]="data/cic/cic_ids2017_multiclass.csv"
  [unsw]="data/unsw/UNSW_NB15_training-set.csv"
  [edge-iiotset-full]="data/edge-iiotset/edge_iiotset_full.csv"
)

for dataset in ${DATASETS}; do
  source_path="${DATASET_SOURCE_PATHS[$dataset]:-}"
  link_path="${DATASET_LINK_PATHS[$dataset]:-}"
  if [[ -z "${source_path}" || -z "${link_path}" ]]; then
    echo "Unknown dataset '${dataset}'" >&2
    exit 1
  fi
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing dataset source for ${dataset}: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${link_path}")"
  ln -sf "${source_path}" "${link_path}"
done

echo "Submitting paper Bulyan 20% confirmatory campaign"
echo "max_concurrent=${MAX_CONCURRENT}"
echo "split_total=${SPLIT_TOTAL}"
echo "datasets=${DATASETS}"

for dataset in ${DATASETS}; do
  job_id="$(
    sbatch --parsable \
      --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
      --export=ALL,DATASET="${dataset}",SPLIT_TOTAL="${SPLIT_TOTAL}" \
      scripts/slurm/paper_bulyan_adv20_confirmatory_array.sbatch
  )"
  echo "${dataset}_job_id=${job_id}"
done

echo "Done. Monitor with: squeue -u ${USER}"
