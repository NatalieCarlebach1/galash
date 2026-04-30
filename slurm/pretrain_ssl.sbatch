#!/bin/bash
#SBATCH --job-name=ssl-pretrain
#SBATCH --partition=dgx
#SBATCH --qos=main-normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/ssl_%j.out
#SBATCH --error=logs/slurm/ssl_%j.err

set -uo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

ENCODER="${ENCODER:?set ENCODER}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-8}"
TEMP="${TEMP:-0.1}"
SAVE_DIR="${SAVE_DIR:-runs/ssl_pretrain/${ENCODER}}"
NO_AMP="${NO_AMP:---no_amp}"

echo "Encoder:   $ENCODER"
echo "Epochs:    $EPOCHS  Batch: $BATCH  LR: $LR  LoRA rank: $LORA_RANK"
echo "Save dir:  $SAVE_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 pretrain_ssl.py \
    --encoder "$ENCODER" \
    --epochs $EPOCHS --batch $BATCH --lr $LR \
    --lora_rank $LORA_RANK --temperature $TEMP \
    --save_dir "$SAVE_DIR" \
    $NO_AMP
