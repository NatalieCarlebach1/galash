#!/bin/bash
# Per-dataset comparison table for the GALASH experiment grid.
# Usage:
#   bash scripts/results_table.sh
#
# Reads log.csv from every relevant run dir, picks the test_f1 at the
# best-val epoch (i.e. what the saved best.pt produces), and prints one
# table per dataset with config columns:
#   recipe   encoder  SSL  EMA  cnn_skip  img  aug  TTA  test@bestVal  ΔSOTA
# Datasets are separated by double horizontal lines.
#
# Notes on the columns:
#   • test column is from per-epoch eval = NO TTA. Final eval after a run
#     completes adds TTA + threshold-search and typically gains +0.5-1.5pp.
#   • aug intensity is read from the YAML name (light/medium/med-heavy/minimal).
#   • EMA is whatever was passed as --ema; default in train.py is 0.0.
cd /home/nfs/tals/galash

declare -A SOTA=(
    [levir_cd]=0.9287
    [levir_cd_plus]=0.8771
    [s2looking]=0.6895
    [cdd]=0.9762
    [dsifn_cd]=0.9665
    [second]=0.7246
)
# Self-described aug intensity per yaml
declare -A AUG=(
    [levir_cd]=medium
    [levir_cd_plus]=light
    [levir_cd_plus_1024]=light
    [levir_cd_plus_1024_main]=light
    [s2looking]=medium
    [s2looking_1024]=medium
    [s2looking_1024_main]=medium
    [cdd]=med-heavy
    [dsifn_cd]=med-heavy
    [second]=minimal
)

for ds in levir_cd levir_cd_plus s2looking cdd dsifn_cd second; do
    sota=${SOTA[$ds]}
    aug=${AUG[$ds]}
    echo "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    printf '%-90s SOTA = %s\n' "$(echo $ds | tr '[:lower:]' '[:upper:]') (paper bar — assumes TTA)" "$sota"
    echo "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    printf '%-7s %-9s %-4s %-6s %-9s %-5s %-10s %-4s %-13s %s\n' \
        'recipe' 'encoder' 'SSL' 'EMA' 'cnn_skip' 'img' 'aug' 'TTA' 'test@bestVal' 'ΔSOTA'
    echo "─────────────────────────────────────────────────────────────────────────────────────────────────────────────────"

    # collect candidate run dirs containing this dataset name
    rows=()
    for d in $(ls -td runs/abl/abl_${ds}_*/ runs/abl/abl_${ds}_*/ 2>/dev/null | sort -u); do
        name=$(basename "$d")
        sub=""
        for s in $(ls -t "$d" 2>/dev/null); do
            [ -f "$d/$s/log.csv" ] && { sub="$s"; break; }
        done
        [ -z "$sub" ] && continue
        csv="$d/$sub/log.csv"
        # parse run name → encoder + SSL + ema label
        if [[ "$name" =~ _v3l_ssl ]]; then enc="v3l"; ssl="yes✗"
        elif [[ "$name" =~ _v3l ]]; then  enc="v3l"; ssl="no"
        elif [[ "$name" =~ _(rsbase|rs_base|rsb)_ssl ]]; then enc="rs_base"; ssl="yes✗"
        elif [[ "$name" =~ _(rsbase|rs_base|rsb) ]]; then enc="rs_base"; ssl="no"
        else enc="?"; ssl="?"
        fi
        # ema label
        if [[ "$name" =~ _ema099 ]]; then ema="0.99"
        else ema="0.999"
        fi
        # img size — 1024 if name says so, else use yaml-default
        if [[ "$name" =~ _1024 ]]; then img="1024"
        elif [[ "$ds" == "second" ]]; then img="512"
        elif [[ "$ds" == "levir_cd_plus" || "$ds" == "s2looking" ]]; then img="512"
        else img="256"
        fi

        line=$(python3 -c "
import csv
with open('$csv') as f: r=list(csv.reader(f))
h,rows=r[0],r[1:]
if not rows: exit()
iep=h.index('epoch'); ivf=h.index('val_f1') if 'val_f1' in h else -1
itf=h.index('test_f1') if 'test_f1' in h else -1
if ivf<0: exit()
fs=[(int(row[iep]),
     float(row[ivf]) if row[ivf] not in('','nan') else None,
     float(row[itf]) if itf>=0 and row[itf] not in('','nan') else None) for row in rows]
val_only=[(e,v,t) for e,v,t in fs if v is not None]
if not val_only: exit()
last=fs[-1]
be,bv,bt=max(val_only, key=lambda x:x[1])
# Filter out runs that haven't produced anything meaningful yet
# (best val < 0.3 AND last epoch < 5).
if bv < 0.3 and last[0] < 5: exit()
if bt is None:
    th=[(e,t) for e,_,t in fs if t is not None]
    bt=min(th, key=lambda x: abs(x[0]-be))[1] if th else None
sota=$sota
gap = f'{(bt - sota)*100:+.2f}' if bt is not None else '-'
test_str = f'{bt:.4f}@{be}' if bt is not None else '-'
print(f'{test_str}|{gap}|{last[0]}|{bt or 0}')
" 2>/dev/null)
        [ -z "$line" ] && continue
        IFS='|' read -ra parts <<< "$line"
        test_str=${parts[0]}; gap=${parts[1]}; last_ep=${parts[2]}; bt=${parts[3]}
        # cnn_skip on iff name says so
        if [[ "$name" =~ cnn_skip ]]; then cs="yes"; else cs="no"; fi

        rows+=("$bt|$enc|$ssl|$ema|$cs|$img|$aug|no|$test_str|$gap")
    done

    # sort rows by test descending
    if [ ${#rows[@]} -gt 0 ]; then
        printf '%s\n' "${rows[@]}" | sort -t'|' -k1 -nr | while IFS='|' read -ra r; do
            printf '%-7s %-9s %-4s %-6s %-9s %-5s %-10s %-4s %-13s %s\n' \
                'natFT' "${r[1]}" "${r[2]}" "${r[3]}" "${r[4]}" "${r[5]}" "${r[6]}" "${r[7]}" "${r[8]}" "${r[9]}"
        done
    fi
    echo ""
done

echo ""
echo "Notes:"
echo "  • test column = test@bestVal (per-epoch eval, NO TTA). Final eval w/ TTA + threshold typically adds +0.5-1.5pp."
echo "  • SSL 'yes✗' = SSL ckpt was passed but the LoRA-merge fix was missing, so SSL signal didn't actually load."
echo "  • aug labels: minimal = just crop+flip; light = + scale + D4; medium = + photometric + temporal_swap;"
echo "    med-heavy = + color_jitter + blur (no CutMix anywhere — heavy CutMix lives on Natalie_features_Sam branch)."
