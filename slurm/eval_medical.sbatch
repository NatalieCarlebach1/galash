#!/bin/bash
#SBATCH --job-name=galash-med
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/eval_medical_%j.out
#SBATCH --error=logs/slurm/eval_medical_%j.err

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

CKPT="${CKPT:-runs/20260325_015312/best.pt}"
# By default, evaluate every medical benchmark that has a test/ split.
DATASETS="${DATASETS:-shifts_ms brats_reg msseg2 isbi_ms lumiere}"
TAG="${TAG:-medical}"
TTA_FLAG="${TTA_FLAG:---tta}"

OUT_JSON="runs/20260325_015312/test_medical_${TAG}.json"

echo "Node: $(hostname)"
echo "Checkpoint: $CKPT"
echo "Medical datasets: $DATASETS"

# Filter to datasets that actually have data/<ds>/test/A populated.
READY=()
for d in $DATASETS; do
  if [[ -d "data/${d}/test/A" ]] && \
     [[ $(ls "data/${d}/test/A" 2>/dev/null | wc -l) -gt 0 ]]; then
    READY+=("$d")
  else
    echo "  ⚠ skipping ${d}: data/${d}/test/A is empty"
  fi
done

if (( ${#READY[@]} == 0 )); then
  echo "No medical datasets are staged. Run:"
  echo "    python download_medical.py --datasets shifts_ms brats_reg"
  exit 1
fi

echo "Evaluating: ${READY[*]}"

python eval.py \
    --encoder dinov2_rs_base \
    --decoder sam2_tiny \
    --checkpoint "$CKPT" \
    --data data \
    --split test \
    --batch 4 \
    --workers 4 \
    --datasets ${READY[@]} \
    $TTA_FLAG \
    2>&1 | tee "logs/slurm/eval_${SLURM_JOB_ID}_${TAG}.txt"
