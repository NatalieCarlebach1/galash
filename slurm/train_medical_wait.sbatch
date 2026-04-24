#!/bin/bash
#SBATCH --job-name=g-med-wait
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/train_%j_%x.out
#SBATCH --error=logs/slurm/train_%j_%x.err

# Wait-and-fire SLURM job for a gated medical CD dataset.
#
# Env vars:
#   DATASET    required: one of {msseg2, isbi_ms, lumiere}
#   CONFIG     defaults to configs/datasets/${DATASET}.yaml
#   ENCODER    default dinov2_rs_base
#   DECODER    default sam2_base_plus
#   RUN_NAME   defaults to fair_${DATASET}_<timestamp>
#   MAX_WAIT   seconds to wait for the _tmp zip; default 36 h
#
# The job holds the GPU allocation and polls every 60 s until
# data/<DATASET>/_tmp/ has a .zip / .tar.* drop.  It then runs
# download_medical.py to extract and slice, and proceeds to train.

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

DATASET="${DATASET:?DATASET env var is required (msseg2 / isbi_ms / lumiere)}"
CONFIG="${CONFIG:-configs/datasets/${DATASET}.yaml}"
ENCODER="${ENCODER:-dinov2_rs_base}"
DECODER="${DECODER:-sam2_base_plus}"
RUN_NAME="${RUN_NAME:-fair_${DATASET}_$(date +%Y%m%d_%H%M)}"
MAX_WAIT="${MAX_WAIT:-129600}"   # 36 h

STAGE_DIR="data/${DATASET}/_tmp"
TRAIN_A_DIR="data/${DATASET}/train/A"

echo "=========================================="
echo "Node:         $(hostname)"
echo "JobID:        ${SLURM_JOB_ID}"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-?}"
echo "Dataset:      ${DATASET}"
echo "Config:       ${CONFIG}"
echo "Encoder:      ${ENCODER}"
echo "Decoder:      ${DECODER}"
echo "Run name:     ${RUN_NAME}"
echo "Stage dir:    ${STAGE_DIR}"
echo "Max wait:     ${MAX_WAIT} s"
echo "=========================================="
nvidia-smi | head -10 || true
echo ""

# 1) Wait until training data is populated OR a stage archive lands.
START=$(date +%s)
READY=0
while true; do
  ELAPSED=$(( $(date +%s) - START ))
  if (( ELAPSED > MAX_WAIT )); then
    echo "Timed out after ${MAX_WAIT} s waiting for ${DATASET} data."
    exit 2
  fi

  # Already prepared?
  if [[ -d "${TRAIN_A_DIR}" ]] && (( $(ls "${TRAIN_A_DIR}" 2>/dev/null | wc -l) > 0 )); then
    echo "Found prepared data at ${TRAIN_A_DIR} (elapsed ${ELAPSED} s)."
    READY=1
    break
  fi

  # Zip / tar staged but not yet extracted?
  if ls "${STAGE_DIR}"/*.zip "${STAGE_DIR}"/*.tar "${STAGE_DIR}"/*.tar.gz 2>/dev/null | grep -q .; then
    echo "Found stage archive at ${STAGE_DIR} (elapsed ${ELAPSED} s) — preparing …"
    python download_medical.py --datasets "${DATASET}"
    if (( $(ls "${TRAIN_A_DIR}" 2>/dev/null | wc -l) > 0 )); then
      READY=1
      break
    else
      echo "Preparer ran but data/${DATASET}/train/A is still empty. Sleeping."
    fi
  fi

  if (( ELAPSED % 600 == 0 )); then
    echo "Still waiting for ${DATASET} data (elapsed ${ELAPSED} s)…"
  fi
  sleep 60
done

(( READY == 1 )) || { echo "Data not ready — aborting."; exit 3; }

echo ""
echo "=== Data counts ==="
for split in train val test; do
  n=$(ls "data/${DATASET}/${split}/A/" 2>/dev/null | wc -l)
  echo "  ${split}: ${n} pairs"
done
echo ""

# 2) Train with the per-dataset fair-comparison YAML.
python train.py \
    --encoder "${ENCODER}" \
    --decoder "${DECODER}" \
    --datasets "${DATASET}" \
    --config "${CONFIG}" \
    --save_dir "runs/${RUN_NAME}"
