#!/bin/bash
# Launch the experiment matrix for the galash paper on SLURM.
#
# Tiers (pass tier name as 1st arg, default 'baseline'):
#   baseline      — reproduce the current baseline on fixed dataset corpus
#   per_dataset   — one training run per benchmark (LEVIR-CD, LEVIR-CD+, S2Looking, CDD, DSIFN, SECOND)
#   encoder_sweep — same dataset, multiple encoders (LEVIR-CD only)
#   decoder_sweep — same encoder, multiple decoders (LEVIR-CD only)
#   finetune      — fine-tune SAM decoder on LEVIR-CD
#   ablations     — architectural ablations on LEVIR-CD

set -euo pipefail
cd /home/nfs/tals/galash

TIER="${1:-baseline}"
TAG_SUFFIX="$(date +%Y%m%d_%H%M)"

submit() {
    local name="$1"; shift
    echo "--- submitting: $name ---"
    # shellcheck disable=SC2068
    env "$@" RUN_NAME="${name}_${TAG_SUFFIX}" \
        sbatch --job-name="g-${name}" slurm/train.sbatch
}

case "$TIER" in
    baseline)
        # One pooled run with the canonical config, all 6 production datasets
        submit pooled_baseline \
            ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
            DATASETS="levir_cd levir_cd_plus s2looking cdd dsifn_cd second" \
            EPOCHS=50 BATCH=4 LR=1e-4 TTA=1
        ;;

    per_dataset)
        # One training run per benchmark using the canonical config (no YAML)
        for DS in levir_cd levir_cd_plus s2looking cdd dsifn_cd second; do
            submit "pd_${DS}" \
                ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
                DATASETS="$DS" EPOCHS=50 BATCH=4 LR=1e-4 TTA=1
        done
        ;;

    fair_per_dataset)
        # Fair-comparison runs: each benchmark uses its published SOTA's augmentation
        # pipeline (see configs/datasets/<ds>.yaml). Epochs=300 + early-stop patience=15.
        for DS in levir_cd levir_cd_plus s2looking cdd dsifn_cd second; do
            submit "fair_${DS}" \
                ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
                CONFIG="configs/datasets/${DS}.yaml"
        done
        ;;

    encoder_sweep)
        # Encoder sweep on LEVIR-CD (fair config). 14 encoders × 1 decoder.
        # DINOv1 / DINOv2 / DINOv2-registers / DINOv2-RS / DINOv3 / DINOv3-sat.
        # Skip dinov3_sat_large + dinov3_huge if HF access is not available (they are gated).
        for ENC in \
            dinov1_vits16 dinov1_vitb16 \
            dinov2_small dinov2_base dinov2_large \
            dinov2_small_reg dinov2_base_reg dinov2_large_reg \
            dinov2_rs_small dinov2_rs_base dinov2_rs_large \
            dinov3_small dinov3_base dinov3_large \
            dinov3_sat_large; do
            submit "enc_${ENC}_levir" \
                ENCODER="$ENC" DECODER=sam2_base_plus \
                CONFIG="configs/datasets/levir_cd.yaml"
        done
        ;;

    decoder_sweep)
        # Decoder sweep on LEVIR-CD (fair config). 1 encoder × 7 decoders.
        # SAM1 (3) + SAM2.1 (4). SAM3 excluded — not installed in this env;
        # add back once `pip install segment-anything-3` lands.
        for DEC in sam1_vit_b sam1_vit_l sam1_vit_h \
                   sam2_tiny sam2_small sam2_base_plus sam2_large; do
            submit "dec_${DEC}_levir" \
                ENCODER=dinov2_rs_base DECODER="$DEC" \
                CONFIG="configs/datasets/levir_cd.yaml"
        done
        ;;

    encoder_sweep_lite)
        # Lighter version — 1 representative encoder per family (7 jobs).
        for ENC in dinov1_vitb16 dinov2_base dinov2_base_reg \
                   dinov2_rs_base dinov2_rs_large \
                   dinov3_base dinov3_sat_large; do
            submit "encL_${ENC}_levir" \
                ENCODER="$ENC" DECODER=sam2_base_plus \
                CONFIG="configs/datasets/levir_cd.yaml"
        done
        ;;

    decoder_sweep_lite)
        # Lighter version — 4 representative decoders (4 jobs).
        for DEC in sam1_vit_b sam2_tiny sam2_base_plus sam2_large; do
            submit "decL_${DEC}_levir" \
                ENCODER=dinov2_rs_base DECODER="$DEC" \
                CONFIG="configs/datasets/levir_cd.yaml"
        done
        ;;

    finetune)
        # Frozen vs fine-tuned decoder (the paper's headline comparison)
        submit "ft_on_levir" \
            ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
            DATASETS=levir_cd EPOCHS=50 BATCH=4 LR=1e-4 FINETUNE=1 TTA=1
        ;;

    cheap_wins)
        # Tier-1 improvements: fine-tuned decoder + EMA + longer patience +
        # threshold search. On the 3 datasets where we're within striking
        # distance of SOTA (LEVIR-CD, SECOND, DSIFN). Uses the fair YAML for aug.
        for DS in levir_cd second dsifn_cd; do
            submit "cheap_${DS}" \
                ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
                CONFIG="configs/datasets/${DS}.yaml" \
                FINETUNE=1 TTA=1 PATIENCE=30 \
                EXTRA_ARGS="--ema 0.999 --search_threshold"
        done
        ;;

    cheap_wins_v3sat)
        # Same Tier-1 recipe but with DINOv3-sat (aerial pretraining).
        # Runs on DGX with --no_amp for stability.
        for DS in levir_cd second dsifn_cd s2looking; do
            echo "--- submitting: cheap_v3sat_${DS} ---"
            env RUN_NAME="cheap_v3sat_${DS}_${TAG_SUFFIX}" \
                ENCODER=dinov3_sat_large DECODER=sam2_base_plus \
                CONFIG="configs/datasets/${DS}.yaml" \
                FINETUNE=1 TTA=1 PATIENCE=30 \
                EXTRA_ARGS="--no_amp --ema 0.999 --search_threshold" \
                sbatch --partition=dgx --qos=main-normal \
                       --job-name="g-cheap_v3sat_${DS}" slurm/train.sbatch
        done
        ;;

    medical)
        # Train on the currently-ready medical CD datasets. Uses
        # train_medical.sbatch which first blocks on any running
        # download_medical.py preparer, then auto-detects populated
        # data/<ds>/train/A folders (overridable via DATASETS env).
        echo "--- submitting: medical_pseudo ---"
        env RUN_NAME="medical_pseudo_${TAG_SUFFIX}" \
            ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
            FINETUNE=1 TTA=1 EPOCHS=40 BATCH=4 LR=1e-4 \
            DATASETS="brats_reg" \
            sbatch --job-name="g-med_pseudo" slurm/train_medical.sbatch
        ;;

    fair_medical)
        # Fair-comparison medical CD runs. Each job holds a GPU and
        # polls data/<ds>/_tmp/ every 60 s for the user-staged zip,
        # then prepares and trains with the per-dataset fair YAML.
        # Matches nnU-Net SOTA recipe (LR, WD, epochs, patience,
        # oversampling, no TTA) — see configs/datasets/{msseg2,isbi_ms}.yaml.
        for DS in msseg2 isbi_ms; do
            echo "--- submitting: fair_${DS} ---"
            env DATASET="${DS}" \
                CONFIG="configs/datasets/${DS}.yaml" \
                ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
                RUN_NAME="fair_${DS}_${TAG_SUFFIX}" \
                sbatch --job-name="g-fair_${DS}" slurm/train_medical_wait.sbatch
        done
        ;;

    ablations)
        # Ablations on LEVIR-CD — all isolate one component
        # 1. No TTA
        submit "abl_nottta_levir" \
            ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
            DATASETS=levir_cd EPOCHS=40 BATCH=4 LR=1e-4 TTA=0
        # 2. No latent loss
        submit "abl_nolatent_levir" \
            ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
            DATASETS=levir_cd EPOCHS=40 BATCH=4 LR=1e-4 TTA=1 \
            EXTRA_ARGS="--w_latent 0.0"
        # 3. No OHEM
        submit "abl_noohem_levir" \
            ENCODER=dinov2_rs_base DECODER=sam2_base_plus \
            DATASETS=levir_cd EPOCHS=40 BATCH=4 LR=1e-4 TTA=1 \
            EXTRA_ARGS="--no_ohem"
        ;;

    all)
        # Chain everything (this is a LOT of jobs — adds up to ~30)
        "$0" baseline
        "$0" per_dataset
        "$0" encoder_sweep
        "$0" decoder_sweep
        "$0" finetune
        "$0" ablations
        ;;

    *)
        echo "Unknown tier: $TIER"
        echo "Available: baseline per_dataset encoder_sweep decoder_sweep finetune ablations all"
        exit 1
        ;;
esac

echo ""
echo "Queue after submission:"
squeue -u "$USER" -n "g-*" 2>/dev/null || squeue -u "$USER"
