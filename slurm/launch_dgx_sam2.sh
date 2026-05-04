#!/bin/bash
# DGX sweep: SAM2.1 image encoder + SAM2.1 mask decoder, one job per dataset
# at native resolution. 6 datasets -> 6 of the 8 B200 GPUs on mntdgxb200-01.
#
# Usage:
#   bash slurm/launch_dgx_sam2.sh                       # default base_plus on both sides
#   SAM2_VARIANT=large    bash slurm/launch_dgx_sam2.sh
#   SAM2_VARIANT=tiny     bash slurm/launch_dgx_sam2.sh
#
# Encoder uses sam2_enc_<variant> (Hiera image encoder via SAM2 wrapper)
# Decoder uses sam2_<variant>     (SAM2 mask decoder)
#
set -euo pipefail
cd /home/nfs/tals/galash

# Encoder uses sam2_enc_* registry names; decoder uses sam2_* names.
# Pick the variant suffix and we'll set both sides consistently.
SAM2_VARIANT="${SAM2_VARIANT:-base_plus}"   # tiny | small | base_plus | large
SAM2_ENC="sam2_enc_${SAM2_VARIANT}"
SAM2_DEC="sam2_${SAM2_VARIANT}"
TAG="$(date +%Y%m%d_%H%M)"

# (dataset, native-resolution yaml)
PAIRS=(
    "cdd            configs/datasets/cdd.yaml"
    "dsifn_cd       configs/datasets/dsifn_cd.yaml"
    "levir_cd       configs/datasets/levir_cd.yaml"
    "levir_cd_plus  configs/datasets/levir_cd_plus_1024.yaml"
    "s2looking      configs/datasets/s2looking_1024.yaml"
    "second         configs/datasets/second.yaml"
)

mkdir -p logs/slurm

submitted=()
for pair in "${PAIRS[@]}"; do
    read -r DS CFG <<< "$pair"
    NAME="dgxsam2_${DS}_${TAG}"
    echo "--- submitting: $NAME  (encoder=$SAM2_ENC  decoder=$SAM2_DEC  config=$CFG)"
    JID=$(env \
        ENCODER="$SAM2_ENC" DECODER="$SAM2_DEC" \
        CONFIG="$CFG" \
        RUN_NAME="$NAME" \
        sbatch \
            --partition=dgx \
            --gres=gpu:1 \
            --cpus-per-task=16 \
            --mem=128G \
            --time=24:00:00 \
            --job-name="g-${DS}-sam2" \
            --parsable \
            slurm/train.sbatch)
    echo "    job $JID  -> logs/slurm/train_${JID}_g-${DS}-sam2.out"
    submitted+=("$JID")
done

echo ""
echo "Submitted ${#submitted[@]} jobs:"
printf '  %s\n' "${submitted[@]}"
echo ""
echo "Watch: squeue -u \$USER -p dgx"
echo "Tail:  tail -f logs/slurm/train_<jobid>_*.out"
