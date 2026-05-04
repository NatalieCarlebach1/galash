#!/bin/bash
# "Max push" sweep with DINOv2-base encoder + SAM2.1 Large mask decoder.
# Mirror of launch_max_sam2.sh — same recipe (cnn_skip, EMA 0.99,
# finetune_decoder, threshold-search, heavy aug, no early stop) on each dataset
# at native resolution; only the encoder differs (sam2_enc_large -> dinov2_base).
#
# Usage:
#   bash slurm/launch_max_dinov2.sh
#
set -euo pipefail
cd /home/nfs/tals/galash

TAG="$(date +%Y%m%d_%H%M)"
# --no_amp added 2026-05-04 for backbone-fair comparison (see launch_max_sam2.sh).
EXTRA="--no_amp --no_early_stop --ema 0.99 --cnn_skip --finetune_decoder --search_threshold"

# Heavy aug yamls for the small/medium datasets where regularization helps;
# existing 1024² yamls (already minimal-aug) for the big ones.
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
    NAME="maxdinov2_${DS}_${TAG}"
    echo "--- $NAME  (dgx, encoder=dinov2_base  decoder=sam2_large  config=$CFG)"
    JID=$(env \
        ENCODER=dinov2_base DECODER=sam2_large \
        CONFIG="$CFG" \
        RUN_NAME="$NAME" \
        EXTRA_ARGS="$EXTRA" \
        sbatch \
            --partition=dgx \
            --gres=gpu:1 \
            --cpus-per-task=16 \
            --mem=128G \
            --time=48:00:00 \
            --job-name="g-${DS}-dinov2" \
            --parsable \
            slurm/train.sbatch)
    echo "    job $JID  -> logs/slurm/train_${JID}_g-${DS}-dinov2.out"
    submitted+=("$JID")
done

echo
echo "Submitted ${#submitted[@]} jobs:"
printf '  %s\n' "${submitted[@]}"
echo
echo "Watch:    JOBS=\"${submitted[*]}\" bash scripts/watch_sweep.sh"
