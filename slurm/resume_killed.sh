#!/bin/bash
# Resume one or more killed/cancelled training runs from their last.pt.
#
# Usage:
#   bash slurm/resume_killed.sh <run_dir> [<run_dir> ...]
#
# Each <run_dir> is the OUTER dir under runs/, e.g.
#   runs/maxdinov2rslarge_cdd_20260505_0616
# (not the inner timestamped dir). The script picks the newest inner dir,
# uses its last.pt as --resume, reads config.json to reconstruct the original
# encoder/decoder/yaml/recipe flags, and re-submits via slurm/train.sbatch
# with the SAME outer dir as RUN_NAME so the new inner timestamp lands
# alongside the killed one.
#
# train.py picks up at ckpt["epoch"]+1 (line 681) and restores cross_attn,
# bridge, sam_decoder (if finetuned), optimizer, scheduler, best_f1.
#
# Caveat: EMA shadow is written to last.pt (line 785) but NOT restored on
# resume — long-trained jobs lose accumulated EMA averaging. Acceptable for
# rs_large jobs killed at ep 22-34 (warmup), small cost for s2looking@ep182.
set -euo pipefail
cd /home/nfs/tals/galash

if [ $# -eq 0 ]; then
    echo "usage: $0 <run_dir> [<run_dir> ...]" >&2
    exit 1
fi

mkdir -p logs/slurm
submitted=()

for outer in "$@"; do
    outer="${outer%/}"
    if [ ! -d "$outer" ]; then
        echo "skip: $outer does not exist" >&2
        continue
    fi
    name=$(basename "$outer")
    # Walk inner dirs newest-first and pick the first one with both a usable
    # last.pt and a config.json. The newest dir may belong to a freshly-killed
    # re-resume that died before writing its first checkpoint (common at 1024²
    # where one epoch takes ~10 min); in that case we fall back to the older
    # inner whose last.pt has the prior training.
    inner=""
    for inner_candidate in $(ls -td "$outer"/2*/ 2>/dev/null); do
        inner_candidate="${inner_candidate%/}"
        if [ -f "$inner_candidate/last.pt" ] && [ -f "$inner_candidate/config.json" ]; then
            inner="$inner_candidate"
            break
        fi
    done
    if [ -z "$inner" ]; then
        echo "skip: $name has no inner dir with both last.pt and config.json" >&2
        continue
    fi
    last_pt="$inner/last.pt"
    cfg="$inner/config.json"

    # Reconstruct the recipe flags from config.json. Only the booleans + ema
    # value matter — the rest is implied by ENCODER/DECODER/CONFIG env vars
    # passed to train.sbatch.
    read -r ENCODER DECODER CONFIG NO_AMP NO_ES EMA CNN_SKIP FT SEARCH_TH \
        < <(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("encoder","?"), c.get("decoder","?"), c.get("config","") or "",
      "1" if c.get("no_amp") else "0",
      "1" if c.get("no_early_stop") else "0",
      c.get("ema",0) or 0,
      "1" if c.get("cnn_skip") else "0",
      "1" if c.get("finetune_decoder") else "0",
      "1" if c.get("search_threshold") else "0")
' "$cfg")

    EXTRA="--resume $last_pt"
    [ "$NO_AMP" = "1" ]    && EXTRA="$EXTRA --no_amp"
    [ "$NO_ES" = "1" ]     && EXTRA="$EXTRA --no_early_stop"
    [ "$CNN_SKIP" = "1" ]  && EXTRA="$EXTRA --cnn_skip"
    [ "$SEARCH_TH" = "1" ] && EXTRA="$EXTRA --search_threshold"
    case "$EMA" in
        0|0.0|"") : ;;  # no EMA flag
        *)         EXTRA="$EXTRA --ema $EMA" ;;
    esac
    FT_FLAG=""
    [ "$FT" = "1" ] && FT_FLAG="1"

    # Parse outer dir name "max<backbone>_<dataset>_<ts>" into backbone +
    # dataset so the resumed slurm job name matches the original
    # `g-<dataset>-<backbone>` pattern (so watch_sweep.sh auto-discovers it
    # and disambiguates by backbone).
    backbone=""; dataset=""
    for bb in dinov2rslarge dinov2rs dinov3sat dinov3 dinov2 sam2; do
        if [[ "$name" == max${bb}_* ]]; then
            backbone="$bb"
            dataset=$(echo "$name" | sed -E "s/^max${bb}_//; s/_2026[0-9]{4}_[0-9]{4}\$//")
            break
        fi
    done
    if [ -z "$backbone" ]; then
        echo "skip: cannot parse backbone from $name" >&2
        continue
    fi

    # Map backbone-dir-prefix to the slurm-job-name suffix used by the
    # original launchers (so watch_sweep.sh's name parser strips it).
    case "$backbone" in
        dinov2rslarge) suffix="rslarge" ;;
        sam2)          suffix="max" ;;
        *)             suffix="$backbone" ;;
    esac
    job_name="g-${dataset}-${suffix}"

    echo "--- resume $name (from $(basename $inner)/last.pt)"
    echo "    enc=$ENCODER dec=$DECODER cfg=$CONFIG  EXTRA=$EXTRA"
    JID=$(env \
        ENCODER="$ENCODER" DECODER="$DECODER" \
        CONFIG="$CONFIG" \
        RUN_NAME="$name" \
        FINETUNE="$FT_FLAG" \
        EXTRA_ARGS="$EXTRA" \
        sbatch \
            --partition=dgx \
            --gres=gpu:1 \
            --cpus-per-task=16 \
            --mem=128G \
            --time=48:00:00 \
            --job-name="$job_name" \
            --parsable \
            slurm/train.sbatch)
    echo "    job $JID"
    submitted+=("$JID")
done

echo
echo "Submitted ${#submitted[@]} resume jobs:"
printf '  %s\n' "${submitted[@]}"
echo
[ ${#submitted[@]} -gt 0 ] && echo "Watch: bash scripts/watch_sweep.sh"
