#!/bin/bash
#SBATCH --job-name=adni-tests
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --output=recognition/ConvNeXt_ADNI_s4985908/logs/adni-tests-%j.out
#SBATCH --error=recognition/ConvNeXt_ADNI_s4985908/logs/adni-tests-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}/recognition/ConvNeXt_ADNI_s4985908"

cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/miniconda3/bin/activate"
conda activate torch

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"

python -m unittest discover -s "${PROJECT_DIR}/tests" -v
