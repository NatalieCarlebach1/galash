#!/bin/bash
# Watcher v11 — smart priority queue
#
# Phase 1: levir_cd remaining ideas (rs_base) — most informative, understand direction
# Phase 2: levir_cd sam2_large — scale up the decoder
# Phase 3: Other datasets baseline (C) — establish baseline everywhere
# Phase 4: Other datasets experimental — D, E, L, large
#
# Configs (all use no_amp + ema=0.99 for fair comparison with Config C):
#   C     = sam2_base_plus  (baseline)
#   D     = sam2_base_plus + spatial_refine
#   E     = sam2_base_plus + multi_scale_cross_attn
#   L     = sam2_base_plus + attn_supervise(w=1.0, diag_init, unchanged-only mask)
#   N     = sam2_base_plus + no_diff_bypass + attn_supervise(decay=50) + attn_diag_init
#   large = sam2_large      (decoder scale-up)

GALASH=/home/tal/natalie/galat/galash
PYTHONPATH_EXTRA=/home/tal/natalie/galat/sam2
LOG=/tmp/gpu_watcher11.log
FREE_THRESH_MIB=16000

ENCODER=dinov2_rs_base
CKPT=$GALASH/checkpoints
COMMON="--cnn_skip --finetune_decoder --search_threshold --no_early_stop --data $GALASH/data --ckpt_dir $CKPT --no_amp --ema 0.99"

EXTRA_C="$COMMON"
EXTRA_D="$COMMON --spatial_refine"
EXTRA_E="$COMMON --multi_scale_cross_attn"
EXTRA_L="$COMMON --attn_supervise --w_attn_supervise 1.0 --attn_diag_init"
EXTRA_N="$COMMON --no_diff_bypass --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init"
EXTRA_M="$COMMON --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 100 --attn_diag_init"
EXTRA_P="$COMMON --multi_scale_cross_attn --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init"
EXTRA_Q="$COMMON --multi_scale_cross_attn --no_diff_bypass"
EXTRA_R="$COMMON --multi_scale_cross_attn --no_diff_bypass --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init"
EXTRA_S="$COMMON --no_diff_bypass --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init --attn_diag_change_map"
EXTRA_LARGE="$COMMON"

SAVE_C=$GALASH/runs/cosim_base_plus_noamp
SAVE_D=$GALASH/runs/cosim_spatial_refine
SAVE_E=$GALASH/runs/cosim_ms_cross_attn
SAVE_L=$GALASH/runs/cosim_attn_supervise_diaginit
SAVE_N=$GALASH/runs/cosim_no_bypass
SAVE_M=$GALASH/runs/cosim_attn_supervise_diaginit_decay
SAVE_P=$GALASH/runs/cosim_ms_cross_attn_supervise
SAVE_Q=$GALASH/runs/cosim_ms_no_bypass
SAVE_R=$GALASH/runs/cosim_ms_no_bypass_supervise
SAVE_S=$GALASH/runs/cosim_no_bypass_attn_changemap
SAVE_LARGE=$GALASH/runs/cosim_large

declare -A DATASET_CONFIGS=(
    [levir_cd]="$GALASH/configs/datasets/levir_cd_heavy.yaml"
    [cdd]="$GALASH/configs/datasets/cdd_heavy.yaml"
    [dsifn_cd]="$GALASH/configs/datasets/dsifn_cd_heavy.yaml"
    [second]="$GALASH/configs/datasets/second_heavy.yaml"
    [levir_cd_plus]="$GALASH/configs/datasets/levir_cd_plus_1024.yaml"
    [sysu_cd]="$GALASH/configs/datasets/sysu_cd_heavy.yaml"
)

QUEUE=(
    # ── Phase 1: levir_cd remaining ideas (rs_base) ──────────────────
    # S=GPU0 running, M=GPU1 resuming, N=GPU2 resuming — started manually
    levir_cd:R
    levir_cd:P
    levir_cd:Q
    levir_cd:E
    levir_cd:large

    # ── Phase 3: Other datasets — baseline first (C) ─────────────────
    cdd:C
    dsifn_cd:C
    second:C
    sysu_cd:C
    levir_cd_plus:C

    # ── Phase 4: Other datasets — experimental (interleaved by dataset) ─
    cdd:D         cdd:E         cdd:L         cdd:large
    dsifn_cd:D    dsifn_cd:E    dsifn_cd:L    dsifn_cd:large
    second:D      second:E      second:L      second:large
    sysu_cd:D     sysu_cd:E     sysu_cd:L     sysu_cd:large
    levir_cd_plus:D levir_cd_plus:E levir_cd_plus:L levir_cd_plus:large
)

gpu_free_mib() {
    nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

echo "[$(date '+%H:%M:%S')] Watcher v11 started. Queue: ${#QUEUE[@]} jobs." | tee -a $LOG
echo "  Phase 1: levir_cd remaining (N, E, large)" | tee -a $LOG
echo "  Phase 2: Other datasets baseline C" | tee -a $LOG
echo "  Phase 3: Other datasets D, E, L, large" | tee -a $LOG

while [ ${#QUEUE[@]} -gt 0 ]; do
    for gpu in 0 1 2; do
        [ ${#QUEUE[@]} -eq 0 ] && break

        free=$(gpu_free_mib $gpu)
        if [ -n "$free" ] && [ "$free" -gt "$FREE_THRESH_MIB" ]; then
            item="${QUEUE[0]}"
            QUEUE=("${QUEUE[@]:1}")
            ds="${item%%:*}"
            cfg="${item##*:}"
            yaml="${DATASET_CONFIGS[$ds]}"

            case "$cfg" in
                C)     decoder=sam2_base_plus; extra="$EXTRA_C";     save_dir="$SAVE_C"     ;;
                D)     decoder=sam2_base_plus; extra="$EXTRA_D";     save_dir="$SAVE_D"     ;;
                E)     decoder=sam2_base_plus; extra="$EXTRA_E";     save_dir="$SAVE_E"     ;;
                L)     decoder=sam2_base_plus; extra="$EXTRA_L";     save_dir="$SAVE_L"     ;;
                M)     decoder=sam2_base_plus; extra="$EXTRA_M";     save_dir="$SAVE_M"     ;;
                N)     decoder=sam2_base_plus; extra="$EXTRA_N";     save_dir="$SAVE_N"     ;;
                P)     decoder=sam2_base_plus; extra="$EXTRA_P";     save_dir="$SAVE_P"     ;;
                Q)     decoder=sam2_base_plus; extra="$EXTRA_Q";     save_dir="$SAVE_Q"     ;;
                R)     decoder=sam2_base_plus; extra="$EXTRA_R";     save_dir="$SAVE_R"     ;;
                S)     decoder=sam2_base_plus; extra="$EXTRA_S";     save_dir="$SAVE_S"     ;;
                large) decoder=sam2_large;     extra="$EXTRA_LARGE"; save_dir="$SAVE_LARGE" ;;
            esac

            ds_log="/tmp/w11_${cfg}_${ds}.log"
            echo "[$(date '+%H:%M:%S')] GPU $gpu free (${free} MiB). Launching ${ds}:${cfg} (${decoder}) → $ds_log" | tee -a $LOG

            CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PYTHONPATH_EXTRA:$PYTHONPATH \
                nohup python3 $GALASH/train.py \
                    --encoder $ENCODER \
                    --decoder $decoder \
                    --config $yaml \
                    --save_dir $save_dir \
                    $extra \
                > $ds_log 2>&1 &

            echo "[$(date '+%H:%M:%S')] ${ds}:${cfg} PID=$! on GPU $gpu" | tee -a $LOG
            sleep 30  # wait for GPU memory to be claimed before checking next GPU
        fi
    done

    [ ${#QUEUE[@]} -gt 0 ] && sleep 60
done

echo "[$(date '+%H:%M:%S')] All jobs queued." | tee -a $LOG
