#!/bin/bash
# Live monitor for SSL pretraining + downstream fine-tune runs.
# Usage:
#   watch -n 30 ./scripts/watch_ssl.sh
cd /home/nfs/tals/galash

SOTA_LEVIR="0.9287"
SOTA_LEVIR_PLUS="0.8771"
SOTA_S2L="0.6895"
SOTA_CDD="0.9762"
SOTA_DSIFN="0.9665"
SOTA_SECOND="0.7246"

echo "════════════════════════════════════════════════════════════════════════════════════════"
echo " QUEUE  ($(date +'%Y-%m-%d %H:%M:%S'))"
echo "════════════════════════════════════════════════════════════════════════════════════════"
squeue -u "$USER" --format='%.10i %.30j %.8T %.10M %.6P' 2>&1 | grep -E "JOBID|ssl_|sslFT_"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════"
echo " SSL PRETRAIN PROGRESS"
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
echo " DOWNSTREAM SSL-FINETUNE PROGRESS"
echo "════════════════════════════════════════════════════════════════════════════════════════"
printf '%-12s %-7s %-5s %-10s %-9s %-10s %-7s %s\n' 'dataset' 'jid' 'ep' 'best_val' 'b_ep/stale' 'test_now' 'SOTA' 'gap'
echo "----------------------------------------------------------------------------------------"

# Detect dataset from job name → match log dir
declare -A DS_SOTA=(
    [levir_cd]=$SOTA_LEVIR
    [levir_cd_plus]=$SOTA_LEVIR_PLUS
    [s2looking]=$SOTA_S2L
    [cdd]=$SOTA_CDD
    [dsifn_cd]=$SOTA_DSIFN
    [second]=$SOTA_SECOND
)

# Look at all sslFT_* run dirs
for d in $(ls -td runs/ssl_ft/sslFT_*/ 2>/dev/null); do
    name=$(basename "$d")
    sub=$(ls "$d" 2>/dev/null | head -1); [ -z "$sub" ] && continue
    csv="$d/$sub/log.csv"
    [ ! -f "$csv" ] && continue
    # extract dataset from run name (sslFT_<DS>_v2rsbase_TAG)
    ds=$(echo "$name" | sed -nE 's/^sslFT_([a-z_]+)_v[a-z0-9_]+_[0-9_]+/\1/p')
    [ -z "$ds" ] && ds="unknown"
    sota=${DS_SOTA[$ds]:--}
    # parse log.csv
    python3 -c "
import csv, json, os
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
test_str=f'{last[2]:.4f}' if last[2] is not None else '-'
sota = $sota if str('$sota') != '-' else None
gap = f'{(last[2] - sota)*100:+.2f}' if (last[2] is not None and sota is not None) else '-'
print(f'{\"$ds\":<12s} -       {last[0]:<5d} {best[1]:<10.4f} {best[0]:<3d}/{stale:<5d} {test_str:<10s} {sota if sota is not None else \"-\":<7} {gap}')
" 2>/dev/null
done

echo ""
echo "Watch:  watch -n 30 ./scripts/watch_ssl.sh"
