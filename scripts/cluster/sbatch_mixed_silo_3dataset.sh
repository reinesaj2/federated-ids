#!/bin/bash
#SBATCH --job-name=fedids-128-mixed-silo-3dataset
#SBATCH --output=/scratch/%u/federated-ids-128/logs/slurm-%A_%a.out
#SBATCH --error=/scratch/%u/federated-ids-128/logs/slurm-%A_%a.err
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --partition=normal

# JMU CS470 Cluster: Mixed-Silo 3-Dataset Federation (Issue #128)
#
# Experiment configuration:
# - 12 clients: 4 CIC-IDS2017 + 4 UNSW-NB15 + 4 Edge-IIoTset
# - 360 total configs: 4 agg × 3 mu × 3 adv × 10 seeds
# - 6 array tasks: 60 configs each (~15-20 hours per task)
#
# Usage:
#   sbatch scripts/cluster/sbatch_mixed_silo_3dataset.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/$USER/federated-ids-128/logs/slurm-*.out
#
# Cancel:
#   scancel <job-id>
#   scancel -u $USER  # Cancel all your jobs

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=== JMU CS470 Cluster: Mixed-Silo 3-Dataset Federation (Issue #128) ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Environment setup
SCRATCH_DIR="/scratch/${USER}/federated-ids-128"
REPO_DIR="${SCRATCH_DIR}/repo"
DATA_DIR="${SCRATCH_DIR}/datasets"
RESULTS_DIR="${SCRATCH_DIR}/results"
LOGS_DIR="${SCRATCH_DIR}/logs"

# Create directories
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOGS_DIR}"

echo "Scratch directory: ${SCRATCH_DIR}"
echo "Repository directory: ${REPO_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Clone or update repository
if [ ! -d "${REPO_DIR}" ]; then
    echo "Cloning federated-ids repository..."
    cd "${SCRATCH_DIR}"
    git clone https://github.com/reinesaj2/federated-ids.git repo
    cd "${REPO_DIR}"
    git checkout feat/issue-128-mixed-silo-3dataset
else
    echo "Updating existing repository..."
    cd "${REPO_DIR}"
    git fetch origin
    git checkout feat/issue-128-mixed-silo-3dataset
    git pull origin feat/issue-128-mixed-silo-3dataset
fi

# Record git commit hash
GIT_COMMIT=$(git rev-parse HEAD)
echo "Git commit: ${GIT_COMMIT}" | tee "${RESULTS_DIR}/git_commit_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
echo ""

# Load Python environment
echo "Loading Python environment..."
module load python/3.11  # Adjust to available module on JMU cluster

# Create or activate virtual environment
VENV_DIR="${SCRATCH_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "${REPO_DIR}/requirements.txt"
echo ""

# Verify datasets exist
echo "Verifying datasets..."
CIC_PATH="${DATA_DIR}/cic/cic_ids2017_multiclass.csv"
UNSW_PATH="${DATA_DIR}/unsw/UNSW_NB15_training-set.csv"
EDGE_PATH="${DATA_DIR}/edge-iiotset/edge_iiotset_quick.csv"

if [ ! -f "${CIC_PATH}" ]; then
    echo "ERROR: CIC dataset not found at ${CIC_PATH}"
    echo "Please upload datasets to ${DATA_DIR} before running:"
    echo "  mkdir -p ${DATA_DIR}/{cic,unsw,edge-iiotset}"
    echo "  scp -r data/cic/* login02.cluster.cs.jmu.edu:${DATA_DIR}/cic/"
    echo "  scp -r data/unsw/* login02.cluster.cs.jmu.edu:${DATA_DIR}/unsw/"
    echo "  scp -r data/edge-iiotset/* login02.cluster.cs.jmu.edu:${DATA_DIR}/edge-iiotset/"
    exit 1
fi

echo "CIC dataset: ${CIC_PATH} ($(wc -l < ${CIC_PATH}) lines)"
echo "UNSW dataset: ${UNSW_PATH} ($(wc -l < ${UNSW_PATH}) lines)"
echo "Edge-IIoTset dataset: ${EDGE_PATH} ($(wc -l < ${EDGE_PATH}) lines)"
echo ""

# Set matplotlib backend (headless)
export MPLCONFIGDIR="${SCRATCH_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

# Run experiments
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}
SPLIT_TOTAL=6

echo "Running experiments (split ${SPLIT_INDEX}/${SPLIT_TOTAL})..."
echo "Estimated configs: $((360 / SPLIT_TOTAL)) configs"
echo "Estimated runtime: 15-20 hours"
echo ""

cd "${REPO_DIR}"

python scripts/comparative_analysis.py \
    --dimension mixed_silo_3dataset \
    --num_clients 12 \
    --num_rounds 15 \
    --split-index ${SPLIT_INDEX} \
    --split-total ${SPLIT_TOTAL} \
    --server_timeout 600 \
    --client_timeout 1800 \
    --output_dir "${RESULTS_DIR}"

# Record completion
echo ""
echo "End time: $(date)"
echo "=== Experiment array task ${SPLIT_INDEX} completed ==="

# Generate summary
MANIFEST_FILE="${RESULTS_DIR}/experiment_manifest_mixed_silo_3dataset_split$((SPLIT_INDEX + 1))of${SPLIT_TOTAL}.json"
if [ -f "${MANIFEST_FILE}" ]; then
    echo ""
    echo "Experiment manifest:"
    cat "${MANIFEST_FILE}"
    echo ""
    echo "Total experiments in this split: $(jq '.total_experiments' ${MANIFEST_FILE})"
    echo "Successful experiments: $(jq '[.results[] | select(.metrics_exist == true)] | length' ${MANIFEST_FILE})"
fi

echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Logs saved to: ${LOGS_DIR}"
echo "Next steps:"
echo "  1. Download results: scp -r login02.cluster.cs.jmu.edu:${RESULTS_DIR} ."
echo "  2. Analyze metrics: python scripts/plot_results.py --results_dir ${RESULTS_DIR}"
echo "  3. Generate thesis plots: python scripts/generate_thesis_plots.py"
