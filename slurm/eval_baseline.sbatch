#!/bin/bash
#SBATCH --job-name=galash-eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

# Which checkpoint / what datasets
CKPT="${CKPT:-runs/20260325_015312/best.pt}"
DATASETS="${DATASETS:-}"
TAG="${TAG:-baseline}"
TTA_FLAG="${TTA_FLAG:---tta}"

OUT_JSON="runs/20260325_015312/test_per_dataset_${TAG}.json"

echo "Node: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi || true
echo ""
echo "Checkpoint: $CKPT"
echo "Datasets:   ${DATASETS:-<auto>}"
echo "TTA:        $TTA_FLAG"
echo "Output:     $OUT_JSON"

DS_FLAG=""
if [[ -n "$DATASETS" ]]; then
  DS_FLAG="--datasets $DATASETS"
fi

# Mirror training config (SAM2, not 2.1)
python eval.py \
    --dino "KevinCha/dinov2-vit-base-remote-sensing" \
    --sam2_ckpt "checkpoints/sam2_hiera_tiny.pt" \
    --sam2_cfg "sam2_hiera_t.yaml" \
    --checkpoint "$CKPT" \
    --data data \
    --split test \
    --batch 4 \
    --workers 4 \
    $DS_FLAG \
    $TTA_FLAG \
    2>&1 | tee "logs/slurm/eval_${SLURM_JOB_ID}_${TAG}.txt"
