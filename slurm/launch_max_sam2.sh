#!/bin/bash
# "Max push" sweep: SAM2.1 Hiera-Large encoder + SAM2.1 Large mask decoder
# (unfrozen) + champion recipe (cnn_skip, EMA, threshold-search, TTA, no early
# stop) on each dataset at its native resolution.
#
# Submits to dgx for the heavy 1024² jobs (need B200 VRAM) and to main for
# the smaller 256/512 jobs (fit on RTX 6000 Ada).
#
# Usage:
#   bash slurm/launch_max_sam2.sh
#
set -euo pipefail
cd /home/nfs/tals/galash

TAG="$(date +%Y%m%d_%H%M)"
# --no_amp added 2026-05-04 for backbone-fair comparison: dinov3 needs it to
# avoid BCE-NaN; sam2/dinov2 also use it now so the recipe is identical across
# all backbones (only the encoder name differs).
EXTRA="--no_amp --no_early_stop --ema 0.99 --cnn_skip --finetune_decoder --search_threshold"

# All on dgx (B200, 183GB VRAM each). Heavy aug yamls for the small/medium
# datasets where regularization helps; existing 1024² yamls (already minimal-aug)
# for the big ones — heavy photometric is less impactful at high res.
# (dataset, yaml)
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
    NAME="maxsam2_${DS}_${TAG}"
    echo "--- $NAME  (dgx, encoder=sam2_enc_large  decoder=sam2_large  config=$CFG)"
    JID=$(env \
        ENCODER=sam2_enc_large DECODER=sam2_large \
        CONFIG="$CFG" \
        RUN_NAME="$NAME" \
        EXTRA_ARGS="$EXTRA" \
        sbatch \
            --partition=dgx \
            --gres=gpu:1 \
            --cpus-per-task=16 \
            --mem=128G \
            --time=48:00:00 \
            --job-name="g-${DS}-max" \
            --parsable \
            slurm/train.sbatch)
    echo "    job $JID  -> logs/slurm/train_${JID}_g-${DS}-max.out"
    submitted+=("$JID")
done

echo
echo "Submitted ${#submitted[@]} jobs:"
printf '  %s\n' "${submitted[@]}"
echo
echo "Watch:    bash scripts/watch_sweep.sh   (set JOBS env to these IDs)"
echo "Monitor:  tail -F logs/slurm/train_<jobid>_g-*-max.out"
