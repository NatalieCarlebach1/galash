#!/bin/bash
#SBATCH --job-name=galash-tr
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/train_%j_%x.out
#SBATCH --error=logs/slurm/train_%j_%x.err

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

# Configuration via env vars (with defaults):
#   ENCODER   default dinov2_rs_base
#   DECODER   default sam2_base_plus
#   DATASETS  default (auto = all under data/)
#   EPOCHS    default 50
#   BATCH     default 4
#   LR        default 1e-4
#   FINETUNE  default off (set FINETUNE=1 to enable --finetune_decoder)
#   TTA       default on  (set TTA=0 to disable)
#   RUN_NAME  optional, passed as --save_dir suffix

ENCODER="${ENCODER:-dinov2_rs_base}"
DECODER="${DECODER:-sam2_base_plus}"
DATASETS="${DATASETS:-}"
EPOCHS="${EPOCHS:-300}"        # effectively infinite; relies on early-stop
BATCH="${BATCH:-16}"           # VRAM is ample (RTX 6000 Ada, 48 GB)
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-3}"
PATIENCE="${PATIENCE:-15}"
FINETUNE="${FINETUNE:-0}"
TTA="${TTA:-1}"
CONFIG="${CONFIG:-}"           # optional per-dataset YAML; overrides BATCH/EPOCHS/LR/etc.
RUN_NAME="${RUN_NAME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SAVE_DIR="runs"
[[ -n "$RUN_NAME" ]] && SAVE_DIR="runs/${RUN_NAME}"

DS_FLAG=""
[[ -n "$DATASETS" ]] && DS_FLAG="--datasets $DATASETS"
FT_FLAG=""
[[ "$FINETUNE" = "1" ]] && FT_FLAG="--finetune_decoder"
TTA_FLAG=""
[[ "$TTA" = "1" ]] && TTA_FLAG="--tta"
CFG_FLAG=""
[[ -n "$CONFIG" ]] && CFG_FLAG="--config $CONFIG"

echo "=========================================="
echo "Node:        $(hostname)"
echo "JobID:       $SLURM_JOB_ID"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-?}"
echo "Encoder:     $ENCODER"
echo "Decoder:     $DECODER"
echo "Datasets:    ${DATASETS:-<auto>}"
echo "Epochs:      $EPOCHS   Batch: $BATCH   LR: $LR"
echo "Finetune:    $FINETUNE   TTA: $TTA"
echo "Save dir:    $SAVE_DIR"
echo "Extra args:  $EXTRA_ARGS"
echo "=========================================="
nvidia-smi | head -20 || true
echo ""

python train.py \
    --encoder "$ENCODER" \
    --decoder "$DECODER" \
    $CFG_FLAG \
    $DS_FLAG \
    --batch $BATCH \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup $WARMUP \
    --patience $PATIENCE \
    --save_dir "$SAVE_DIR" \
    $FT_FLAG \
    $TTA_FLAG \
    $EXTRA_ARGS
