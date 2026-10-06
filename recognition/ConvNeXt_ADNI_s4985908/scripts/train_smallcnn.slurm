#!/bin/bash
#SBATCH --job-name=adni-smallcnn
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=recognition/ConvNeXt_ADNI_s4985908/logs/smallcnn-train-%j.out
#SBATCH --error=recognition/ConvNeXt_ADNI_s4985908/logs/smallcnn-train-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}/recognition/ConvNeXt_ADNI_s4985908"
SEED="${SEED:-3710}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/smallcnn-seed${SEED}-job${SLURM_JOB_ID}"

cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/miniconda3/bin/activate"
conda activate torch

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "git_commit=$(git rev-parse HEAD)"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi

python "${PROJECT_DIR}/train.py" \
    --model small_cnn \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --dropout 0.30 \
    --seed "${SEED}" \
    --device cuda \
    --output-dir "${OUTPUT_DIR}"

echo "finished=$(date --iso-8601=seconds)"
