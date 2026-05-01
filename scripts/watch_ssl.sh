#!/bin/bash
# Live monitor for SSL pretraining + downstream fine-tune runs.
# Shows only currently-running jobs:
#   - row appears iff its log.csv was modified in the last $FRESH_SECS
#     (default 600s = 10 min), i.e. there is a process actively writing
#     to it. Finished or killed runs drop off automatically.
# Usage:
#   watch -n 30 ./scripts/watch_ssl.sh
cd /home/nfs/tals/galash

SOTA_LEVIR="0.9287"
SOTA_LEVIR_PLUS="0.8771"
SOTA_S2L="0.6895"
SOTA_CDD="0.9762"
SOTA_DSIFN="0.9665"
SOTA_SECOND="0.7246"

NOW=$(date +%s)
# log.csv is only appended at end of epoch (after train + val + test). At
# 1024-px resolution one epoch can take 25-40 min, so the row gap exceeds
# 10 min — bump to 45 min so live 1024-px runs don't get filtered out.
FRESH_SECS=2700

echo "════════════════════════════════════════════════════════════════════════════════════════"
echo " QUEUE  ($(date +'%Y-%m-%d %H:%M:%S'))"
echo "════════════════════════════════════════════════════════════════════════════════════════"
squeue -u "$USER" --format='%.10i %.30j %.8T %.10M %.6P' 2>&1

echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════"
echo " SSL PRETRAIN PROGRESS  (live only)"
echo "════════════════════════════════════════════════════════════════════════════════════════"
printf '%-22s %-7s %-6s %-12s %-10s %s\n' 'encoder' 'jid' 'last_ep' 'avg_loss' 'time/ep' 'latest_step'
echo "----------------------------------------------------------------------------------------"
for jid in $(squeue -u "$USER" -h -o '%i %j' 2>/dev/null | awk '/g-ssl_/ {print $1}'); do
    f="logs/slurm/ssl_${jid}.out"
    [ ! -f "$f" ] && continue
    enc=$(grep -m1 'Encoder:' "$f" 2>/dev/null | awk '{print $2}')
    last_ep_line=$(grep -hE "^\[ep [0-9]+\] avg_loss=" "$f" 2>/dev/null | tail -1)
    last_step_line=$(grep -hE "^\s*\[ep [0-9]+ step" "$f" 2>/dev/null | tail -1 | sed 's/^\s*//')
    if [ -n "$last_ep_line" ]; then
        ep=$(echo "$last_ep_line" | sed -nE 's/^\[ep ([0-9]+)\].*/\1/p')
        loss=$(echo "$last_ep_line" | sed -nE 's/.*avg_loss=([0-9.]+).*/\1/p')
        tt=$(echo "$last_ep_line" | sed -nE 's/.*time=([0-9]+)s/\1s/p')
    else
        ep="-"; loss="-"; tt="-"
    fi
    step=$(echo "$last_step_line" | sed -nE 's/.*step ([0-9]+\/[0-9]+).*loss=([0-9.]+).*/step \1 loss=\2/p')
    printf '%-22s %-7s ep%-4s %-12s %-10s %s\n' "$enc" "$jid" "$ep" "$loss" "$tt" "$step"
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════"
echo " DOWNSTREAM PROGRESS  (only runs with log.csv updated in last $((FRESH_SECS / 60)) min)"
echo "════════════════════════════════════════════════════════════════════════════════════════"
printf '%-7s %-12s %-5s %-10s %-9s %-14s %-7s %s\n' 'recipe' 'dataset' 'ep' 'best_val' 'b_ep/stale' 'test@bestEp' 'SOTA' 'gap'
echo "----------------------------------------------------------------------------------------"

declare -A DS_SOTA=(
    [levir_cd]=$SOTA_LEVIR
    [levir_cd_plus]=$SOTA_LEVIR_PLUS
    [s2looking]=$SOTA_S2L
    [cdd]=$SOTA_CDD
    [dsifn_cd]=$SOTA_DSIFN
    [second]=$SOTA_SECOND
)

# Walk all known run-dir layouts. The mtime filter on log.csv below
# drops anything that isn't being actively written to.
for d in $(ls -td runs/ssl_ft/sslFT_*/ \
                  runs/ssl_ft/satFT_*/ \
                  runs/abl_*_cnn_skip_v3l_*/ \
                  runs/abl/abl_*_cnn_skip_v3l_*/ \
                  runs/final_*/ \
                  2>/dev/null); do
    name=$(basename "$d")
    # pick the most-recent timestamped subdir that actually has a log.csv
    sub=""
    for s in $(ls -t "$d" 2>/dev/null); do
        if [ -f "$d/$s/log.csv" ]; then sub="$s"; break; fi
    done
    [ -z "$sub" ] && continue
    csv="$d/$sub/log.csv"
    # Only show if log.csv was touched in the last FRESH_SECS — i.e.
    # a job is actively writing to it.
    mtime=$(stat -c %Y "$csv" 2>/dev/null || echo 0)
    [ $((NOW - mtime)) -gt $FRESH_SECS ] && continue

    # extract dataset + recipe from run name. Layouts:
    #   sslFT_<DS>_v2rsbase_TAG               -> recipe=sslFT, ds=<DS>
    #   satFT_<DS>_v3sat_TAG                  -> recipe=satFT, ds=<DS>
    #   abl_<DS>_cnn_skip_v3l[_ssl]_TAG       -> recipe=natFT, ds=<DS>
    #   final_<DS>_v3l_ssl_skip_TAG           -> recipe=v3l, ds=<DS>
    #   final_<DS>_sat_ssl_skip_TAG           -> recipe=v3sat, ds=<DS>
    if [[ "$name" =~ ^abl_(.+)_cnn_skip_v3l ]]; then
        recipe="natFT"
        ds=$(echo "$name" | sed -nE 's/^abl_([a-z0-9_]+)_cnn_skip_v3l.*/\1/p')
    elif [[ "$name" =~ ^final_ ]]; then
        if [[ "$name" =~ _v3l_ ]]; then recipe="v3l"; else recipe="v3sat"; fi
        ds=$(echo "$name" | sed -nE 's/^final_([a-z0-9_]+)_(v3l|sat)_ssl_skip.*/\1/p')
    else
        recipe=$(echo "$name" | sed -nE 's/^([a-zA-Z]+FT)_.*/\1/p')
        ds=$(echo "$name" | sed -nE "s/^${recipe}_([a-z0-9_]+)_v[a-z0-9_]+_[0-9_]+/\1/p")
    fi
    [ -z "$ds" ] && ds="unknown"
    sota=${DS_SOTA[$ds]:--}

    python3 -c "
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
val_only=[(e,v) for e,v,_ in fs if v is not None]
if not val_only: exit()
last=fs[-1]; best=max(val_only, key=lambda x:x[1]); stale=last[0]-best[0]
# Carry forward the last non-null test_f1 (which only fires on best-val
# improvements in the per-best-val variant; runs with test-after-every-val
# will have one per row).
test_hist=[(e,t) for e,_,t in fs if t is not None]
last_test_ep, last_test = test_hist[-1] if test_hist else (None, None)
test_str=f'{last_test:.4f}@{last_test_ep}' if last_test is not None else '-'
sota = $sota if str('$sota') != '-' else None
gap = f'{(last_test - sota)*100:+.2f}' if (last_test is not None and sota is not None) else '-'
print(f'{\"$recipe\":<7s} {\"$ds\":<12s} {last[0]:<5d} {best[1]:<10.4f} {best[0]:<3d}/{stale:<5d} {test_str:<14s} {sota if sota is not None else \"-\":<7} {gap}')
" 2>/dev/null
done

echo ""
echo "Watch:  watch -n 30 ./scripts/watch_ssl.sh"
