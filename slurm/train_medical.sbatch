#!/bin/bash
#SBATCH --job-name=g-med
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/train_%j_%x.out
#SBATCH --error=logs/slurm/train_%j_%x.err

# Train galash on medical change-detection datasets.
#
# Env vars (all optional):
#   DATASETS   space-separated medical keys (default: auto-detect what's ready)
#   ENCODER    default dinov2_rs_base
#   DECODER    default sam2_base_plus
#   EPOCHS     default 50
#   BATCH      default 4
#   LR         default 1e-4
#   FINETUNE   default 1 (fine-tune decoder)
#   TTA        default 1
#   RUN_NAME   default medical_<encoder>_<decoder>
#
# Also waits for any local `download_medical.py` process to finish so
# the loader sees the full slice set at start time.

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

ENCODER="${ENCODER:-dinov2_rs_base}"
DECODER="${DECODER:-sam2_base_plus}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-3}"
PATIENCE="${PATIENCE:-10}"
FINETUNE="${FINETUNE:-1}"
TTA="${TTA:-1}"
RUN_NAME="${RUN_NAME:-medical_${ENCODER}_${DECODER}}"

# Wait for any running download_medical.py preparer to finish.
if pgrep -f "download_medical.py" > /dev/null 2>&1; then
  echo "Waiting for download_medical.py preparer to finish …"
  while pgrep -f "download_medical.py" > /dev/null 2>&1; do
    sleep 30
  done
  echo "Preparer finished."
fi

# Auto-detect which medical datasets have populated train/A and test/A.
MEDICAL_KEYS=(shifts_ms brats_reg msseg2 isbi_ms lumiere chest_imagenome)
if [[ -z "${DATASETS:-}" ]]; then
  READY=()
  for k in "${MEDICAL_KEYS[@]}"; do
    if [[ -d "data/${k}/train/A" ]] && \
       [[ $(ls "data/${k}/train/A" 2>/dev/null | wc -l) -gt 0 ]]; then
      READY+=("$k")
    fi
  done
  if (( ${#READY[@]} == 0 )); then
    echo "No medical datasets have data/<ds>/train/A populated."
    echo "Run:  python download_medical.py --datasets brats_reg --pseudo-labels"
    exit 1
  fi
  DATASETS="${READY[*]}"
fi

FT_FLAG=""
[[ "$FINETUNE" = "1" ]] && FT_FLAG="--finetune_decoder"
TTA_FLAG=""
[[ "$TTA" = "1" ]] && TTA_FLAG="--tta"

echo "=========================================="
echo "Node:        $(hostname)"
echo "JobID:       $SLURM_JOB_ID"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-?}"
echo "Encoder:     $ENCODER"
echo "Decoder:     $DECODER"
echo "Datasets:    $DATASETS"
echo "Epochs:      $EPOCHS   Batch: $BATCH   LR: $LR"
echo "Finetune:    $FINETUNE   TTA: $TTA"
echo "Save dir:    runs/${RUN_NAME}"
echo "=========================================="
nvidia-smi | head -20 || true
echo ""

python train.py \
    --encoder "$ENCODER" \
    --decoder "$DECODER" \
    --datasets $DATASETS \
    --batch $BATCH \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup $WARMUP \
    --patience $PATIENCE \
    --save_dir "runs/${RUN_NAME}" \
    $FT_FLAG \
    $TTA_FLAG
