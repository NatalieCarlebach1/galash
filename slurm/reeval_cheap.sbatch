#!/bin/bash
#SBATCH --job-name=galash-reeval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/reeval_%j.out
#SBATCH --error=logs/slurm/reeval_%j.err

# Post-hoc re-evaluation of a finished best.pt with Tier-1 cheap wins:
#   - threshold search on val
#   - multi-resolution test (256 / 384 / 512)
#   - optional EMA weights (--use_ema if ckpt has ema_shadow)
# Writes test_results_cheap.json alongside the original test_results.json.
#
# Usage:
#   CKPT=runs/fair_levir_cd_.../20260424_.../best.pt \
#   DS=levir_cd  ENCODER=dinov2_rs_base  DECODER=sam2_base_plus \
#   sbatch slurm/reeval_cheap.sbatch

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

CKPT="${CKPT:?set CKPT=path/to/best.pt}"
DS="${DS:?set DS=<dataset name>}"
ENCODER="${ENCODER:-dinov2_rs_base}"
DECODER="${DECODER:-sam2_base_plus}"
FINETUNE_FLAG="${FINETUNE_FLAG:-}"
USE_EMA_FLAG="${USE_EMA_FLAG:-}"

# Pick test sizes based on dataset (default set is what the dataset's YAML supports)
case "$DS" in
    levir_cd|cdd|dsifn_cd)
        TEST_SIZES="256 384 512"
        VAL_SPLIT=val
        ;;
    second)
        TEST_SIZES="384 512 640"
        VAL_SPLIT=val
        ;;
    levir_cd_plus|s2looking)
        TEST_SIZES="512 768 1024"
        VAL_SPLIT=val
        ;;
    *)
        TEST_SIZES="256 384 512"
        VAL_SPLIT=val
        ;;
esac

RUN_DIR=$(dirname "$CKPT")
OUT_NAME="test_results_cheap_${DS}"
OUT_JSON="${RUN_DIR}/${OUT_NAME}.json"

echo "Checkpoint:    $CKPT"
echo "Dataset:       $DS"
echo "Test sizes:    $TEST_SIZES"
echo "Val split:     $VAL_SPLIT"
echo "Encoder:       $ENCODER"
echo "Decoder:       $DECODER"
echo "Finetune flag: $FINETUNE_FLAG"
echo "EMA flag:      $USE_EMA_FLAG"
echo "Output:        $OUT_JSON"
nvidia-smi | head -10 || true

# shellcheck disable=SC2086
python eval.py \
    --encoder "$ENCODER" \
    --decoder "$DECODER" \
    --checkpoint "$CKPT" \
    --datasets "$DS" \
    --split test \
    --val_split "$VAL_SPLIT" \
    --test_img_sizes $TEST_SIZES \
    --search_threshold \
    --tta \
    --batch 4 --workers 4 \
    $FINETUNE_FLAG \
    $USE_EMA_FLAG \
    2>&1 | tee "logs/slurm/reeval_${SLURM_JOB_ID}.txt"

# Copy the aggregated JSON (eval.py prints per-dataset rows but doesn't save)
# ↓ Simpler: dump the parsed table from the log
python3 - "$OUT_JSON" "logs/slurm/reeval_${SLURM_JOB_ID}.txt" << 'PY'
import sys, re, json
out_json, log_path = sys.argv[1], sys.argv[2]
with open(log_path) as f:
    text = f.read()

# Parse the "picked:" and final summary lines
rec = {}
m_pick = re.search(r'picked: size=(\d+) threshold=([\d.]+)', text)
if m_pick:
    rec['picked_size'] = int(m_pick.group(1))
    rec['picked_threshold'] = float(m_pick.group(2))

# Grab the final summary line — last "<dataset> F1=... IoU=..." line
m_final = None
for m in re.finditer(r'(\S+)\s+F1=([\d.]+)\s+IoU=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)\s+OA=([\d.]+)\s+K=([\d.]+)', text):
    m_final = m
if m_final:
    rec['dataset']   = m_final.group(1)
    rec['f1']        = float(m_final.group(2))
    rec['iou']       = float(m_final.group(3))
    rec['precision'] = float(m_final.group(4))
    rec['recall']    = float(m_final.group(5))
    rec['oa']        = float(m_final.group(6))
    rec['kappa']     = float(m_final.group(7))

with open(out_json, 'w') as f:
    json.dump(rec, f, indent=2)
print(f'Saved: {out_json}  → {rec}')
PY
