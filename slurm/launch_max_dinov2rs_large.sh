#!/bin/bash
# "Max push" sweep with DINOv2-RS-Large encoder + SAM2.1 Large mask decoder.
# Mirror of launch_max_dinov2.sh — same recipe, only the encoder differs
# (dinov2_base -> dinov2_rs_large: 1024-dim, 24 layers, patch 14).
#
# Usage:
#   bash slurm/launch_max_dinov2rs_large.sh
#
set -euo pipefail
cd /home/nfs/tals/galash

TAG="$(date +%Y%m%d_%H%M)"
EXTRA="--no_amp --no_early_stop --ema 0.99 --cnn_skip --finetune_decoder --search_threshold"

ROWS=(
    "cdd            configs/datasets/cdd_heavy.yaml"
    "dsifn_cd       configs/datasets/dsifn_cd_heavy.yaml"
    "levir_cd       configs/datasets/levir_cd_heavy.yaml"
    "second         configs/datasets/second_heavy.yaml"
    "levir_cd_plus  configs/datasets/levir_cd_plus_1024.yaml"
    "s2looking      configs/datasets/s2looking_1024.yaml"
)

mkdir -p logs/slurm
submitted=()
for row in "${ROWS[@]}"; do
    read -r DS CFG <<< "$row"
    NAME="maxdinov2rslarge_${DS}_${TAG}"
    echo "--- $NAME  (dgx, encoder=dinov2_rs_large  decoder=sam2_large  config=$CFG)"
    JID=$(env \
        ENCODER=dinov2_rs_large DECODER=sam2_large \
        CONFIG="$CFG" \
        RUN_NAME="$NAME" \
        EXTRA_ARGS="$EXTRA" \
        sbatch \
            --partition=dgx \
            --gres=gpu:1 \
            --cpus-per-task=16 \
            --mem=128G \
            --time=48:00:00 \
            --job-name="g-${DS}-rslarge" \
            --parsable \
            slurm/train.sbatch)
    echo "    job $JID  -> logs/slurm/train_${JID}_g-${DS}-rslarge.out"
    submitted+=("$JID")
done

echo
echo "Submitted ${#submitted[@]} jobs:"
printf '  %s\n' "${submitted[@]}"
echo
echo "Watch:    JOBS=\"${submitted[*]}\" bash scripts/watch_sweep.sh"
