#!/bin/bash
# Snapshot of the current SAM2 DGX sweep: shows test_f1 at the epoch with best
# val_f1, and the gap to the published SOTA for each dataset.
#
# Usage:
#   bash scripts/watch_sweep.sh                     # auto: all running g-* jobs
#   JOBS="54479 54480 ..." bash scripts/watch_sweep.sh
#
# Default: pull every currently-queued/running galash job (job names beginning
# with "g-") from squeue. Override with JOBS=... if you want a specific subset.
if [ -z "${JOBS:-}" ]; then
    JOBS=$(squeue -h -u "$USER" -o '%i %j' 2>/dev/null \
           | awk '$2 ~ /^g-/{print $1}' | tr '\n' ' ')
fi
if [ -z "${JOBS// }" ]; then
    echo "No running galash jobs (squeue empty for g-* names)."
    exit 0
fi

# Side effect: rebuild backbone.md from all max*_* runs.
python3 "$(dirname "$0")/update_backbone_md.py" >/dev/null 2>&1 || true

# Per-dataset FAIR SOTA F1 — single-dataset training, no extra CD-labeled data
# and no synthetic CD pairs. Excludes SChanger (extra CD pretraining),
# UniChange / ChangeStar2 (joint multi-dataset training), and ChangeStar+Changen
# (synthetic data). Numbers from all_results.md "Published SOTA Baselines (fair)".
declare -A SOTA=(
    [cdd]=96.12             # RFL-CDNet (2024)
    [dsifn_cd]=96.65        # DDPM-CD  (foundation-model SSL only)
    [levir_cd]=90.24        # ChangeFormer (2022)
    [levir_cd_plus]=87.71   # DDCDNet  (2024)
    [s2looking]=68.60       # FIBTNet  (2024)
    [second]=72.46          # SAM-SCD  (2025, binary collapse)
)
declare -A SOTA_NAME=(
    [cdd]=RFL-CDNet [dsifn_cd]=DDPM-CD [levir_cd]=ChangeFormer
    [levir_cd_plus]=DDCDNet [s2looking]=FIBTNet [second]=SAM-SCD
)

printf "%-15s  %-10s  %-7s  %-7s  %-8s  %-8s  %-8s  %-7s  %-8s  %s\n" \
       "dataset" "backbone" "ep_now" "ep_bv" "best_val" "test@bv" "test+TTA" "SOTA" "gap" "(state, plateau)"
printf "%-15s  %-10s  %-7s  %-7s  %-8s  %-8s  %-8s  %-7s  %-8s  %s\n" \
       "---------------" "----------" "-------" "-------" "--------" "--------" "--------" "-------" "--------" "----------------"

for j in $JOBS; do
    f=$(ls logs/slurm/train_${j}_*.out 2>/dev/null | head -1)
    [ -z "$f" ] && { printf "%-15s  (no log)\n" "job=$j"; continue; }

    raw=$(basename "$f" .out | sed -E 's/train_[0-9]+_g-//')
    # Split job-name into <dataset>-<backbone>. Backbone is the trailing token
    # after the LAST `-` and tells us which sweep prefix to look under.
    backbone=$(echo "$raw" | sed -E 's/.*-([^-]+)$/\1/')
    name=$(echo "$raw" | sed -E 's/-[^-]+$//')
    case "$backbone" in
        rslarge)        prefixes="maxdinov2rslarge" ;;
        dinov2rs)       prefixes="maxdinov2rs" ;;
        dinov3sat)      prefixes="maxdinov3sat" ;;
        dinov3)         prefixes="maxdinov3" ;;
        dinov2)         prefixes="maxdinov2" ;;
        sam2|max)       prefixes="maxsam2 dgxsam2" ;;
        *)              # Unknown suffix — treat the whole thing as dataset and
                        # fall back to scanning all known prefixes (legacy).
                        name="$raw"
                        prefixes="maxdinov2rslarge maxdinov2rs maxdinov3sat maxdinov3 maxdinov2 maxsam2 dgxsam2" ;;
    esac

    state=$(squeue -j $j -h -o '%t' 2>/dev/null || echo "?")
    [ -z "$state" ] && state="DONE"

    # Find the run dir's log.csv (newest) under the matching backbone prefix.
    globs=""
    for p in $prefixes; do
        globs="$globs runs/${p}_${name}_2026*/2*/log.csv"
    done
    runcsv=$(ls -td $globs 2>/dev/null | head -1)
    if [ -z "$runcsv" ] || [ ! -f "$runcsv" ]; then
        printf "%-15s  %-10s  no log.csv (%s)\n" "$name" "$backbone" "$state"
        continue
    fi

    # Pick the row with max val_f1 + report current epoch and plateau length
    read -r ep_bv best_val test_at_bv ep_now plateau < <(python3 - "$runcsv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("- - - - -"); sys.exit(0)
best = max(rows, key=lambda r: float(r.get("val_f1", 0) or 0))
last = rows[-1]
plateau = int(last["epoch"]) - int(best["epoch"])
print(best["epoch"], best["val_f1"], best.get("test_f1", "-"),
      last["epoch"], plateau)
PY
)

    sota=${SOTA[$name]:-?}
    sota_who=${SOTA_NAME[$name]:-?}

    # Final test with TTA, if the run wrote test_results.json
    rundir=$(dirname "$runcsv")
    if [ -f "$rundir/test_results.json" ]; then
        tta_f1=$(python3 -c "import json; d=json.load(open('$rundir/test_results.json')); print(f'{d[\"f1\"]*100:.2f}')" 2>/dev/null || echo "-")
    else
        tta_f1="-"
    fi

    # F1 to %
    bv_pct=$(python3 -c "print(f'{float(\"$best_val\")*100:.2f}')" 2>/dev/null || echo "?")
    tt_pct=$(python3 -c "print(f'{float(\"$test_at_bv\")*100:.2f}')" 2>/dev/null || echo "?")

    # Gap uses TTA test if available (that's what we actually report); else
    # fall back to per-epoch test.
    cmp="$tta_f1"
    [ "$cmp" = "-" ] && cmp="$tt_pct"
    if [ "$sota" != "?" ] && [ "$cmp" != "?" ] && [ "$cmp" != "-" ]; then
        gap=$(python3 -c "print(f'{($cmp - $sota):+.2f}')")
    else
        gap="?"
    fi

    printf "%-15s  %-10s  %-7s  %-7s  %-8s  %-8s  %-8s  %-7s  %-8s  %s\n" \
           "$name" "$backbone" "$ep_now" "$ep_bv" "$bv_pct" "$tt_pct" "$tta_f1" "$sota" "$gap" \
           "$state ($sota_who, +${plateau}ep)"
done
