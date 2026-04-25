#!/usr/bin/env python3
"""Monitoring view for galash training runs.

For each active training run under `runs/`, shows:
  - slurm job state (running/failed/completed)
  - epoch count
  - last train/val F1
  - best val F1 so far
  - per-dataset SOTA bar and GAP (best - SOTA)

Usage:
  python scripts/monitor.py               # one-shot
  python scripts/monitor.py --watch       # tail, refresh every 60s
  python scripts/monitor.py --tag 1757    # only runs with this timestamp tag
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
LOGS = os.path.join(ROOT, 'logs/slurm')

# Per-dataset SOTA (F1) bars, late 2025. Source: memory/project_sota_targets.md
SOTA = {
    'levir_cd':       (0.9287, 'SChanger (2025)'),
    'levir_cd_plus':  (0.9150, 'ChangeStar+Changen (2023)'),
    's2looking':      (0.6932, 'UniChange (2025)'),
    'cdd':            (0.9762, 'SChanger (2025)'),
    'dsifn_cd':       (0.9665, 'DDPM-CD (2024)'),
    'second':         (0.7312, 'UniChange (2025)'),  # binary collapse
    'sysu_cd':        (0.8263, 'CDNeXt (2023)'),
    'oscd':           (0.6000, 'DS-UNet (2022)'),
    'pooled':         (None,   None),
}


def squeue_rows():
    """Return {jobid: (state, elapsed, node)} for $USER."""
    try:
        out = subprocess.run(['squeue', '-u', os.environ['USER'], '-h',
                              '-o', '%i|%T|%M|%N'],
                             capture_output=True, text=True, timeout=10)
        rows = {}
        for line in out.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 4:
                rows[parts[0]] = (parts[1], parts[2], parts[3])
        return rows
    except Exception as e:
        return {}


def parse_log_csv(csv_path):
    """Return (epochs_done, train_f1_last, val_f1_last, val_f1_best)."""
    if not os.path.isfile(csv_path):
        return (0, None, None, None)
    with open(csv_path) as f:
        lines = f.read().strip().split('\n')
    if len(lines) < 2:
        return (0, None, None, None)
    header = lines[0].split(',')
    cols = {c: i for i, c in enumerate(header)}
    rows = [line.split(',') for line in lines[1:] if line.strip()]
    if not rows:
        return (0, None, None, None)

    def ffield(row, name):
        try:
            return float(row[cols[name]])
        except (KeyError, ValueError, IndexError):
            return None

    last = rows[-1]
    val_f1_best = max((ffield(r, 'val_f1') or -1) for r in rows)
    val_f1_best = val_f1_best if val_f1_best >= 0 else None
    return (len(rows), ffield(last, 'train_f1'),
            ffield(last, 'val_f1'), val_f1_best)


def guess_dataset(run_name):
    """Map run_name → SOTA key. e.g. 'pd_levir_cd_20260424_1757' → 'levir_cd'."""
    name = run_name.lower()
    if 'pooled' in name:
        return 'pooled'
    # prefix like 'pd_XXX_timestamp' or 'enc_XXX_levir_timestamp'
    # Match against known datasets, preferring longest match first
    for ds in sorted(SOTA.keys(), key=len, reverse=True):
        if ds in name:
            return ds
    return None


def find_jobid(run_name):
    """Scan logs/slurm/*.out for the jobid whose RUN_NAME matches."""
    for f in glob.glob(os.path.join(LOGS, 'train_*_g-*.out')):
        m = re.match(r'train_(\d+)_g-(.+)\.out', os.path.basename(f))
        if not m:
            continue
        jobid, tag = m.group(1), m.group(2)
        # run_name example: pd_cdd_20260424_1757; jobname: g-pd_cdd
        # tag from filename: "pd_cdd" (truncated); run_name starts with tag+"_"
        if run_name.startswith(tag + '_') or run_name == tag:
            return jobid
    return None


def test_results_f1(run_dir):
    """Look for test_results.json in any subdir and return its f1."""
    for js in glob.glob(os.path.join(run_dir, '**/test_results.json'), recursive=True):
        try:
            with open(js) as f:
                d = json.load(f)
            return d.get('f1')
        except Exception:
            pass
    return None


def fmt_gap(best, sota, is_preview=False):
    """Format gap vs SOTA. `is_preview=True` adds '~' to signal val-based estimate."""
    if best is None or sota is None:
        return '          '
    gap = best - sota
    prefix = '~' if is_preview else ' '
    if gap >= 0:
        return f'{prefix}+{gap*100:5.2f}pp'
    else:
        return f'{prefix}{gap*100:6.2f}pp'


def fmt_f1(v):
    return f'{v:.4f}' if v is not None else '  -   '


def snapshot(tag_filter=None):
    squeue = squeue_rows()
    # find run directories
    rows = []
    all_csv = glob.glob(os.path.join(RUNS, '**', 'log.csv'), recursive=True)
    # Build a set of "run dirs" — the parent of each log.csv or grandparent if needed
    seen_run_dirs = set()
    for csv in all_csv:
        parent = os.path.dirname(csv)          # e.g. runs/seeds/seed42_levir_cd/20260425_.../
        grandparent = os.path.dirname(parent)  # e.g. runs/seeds/seed42_levir_cd/
        # Use grandparent if it looks like a named run, otherwise parent
        candidate = grandparent if re.search(r'seed|pd_|enc_|dec_|cheap|abl|partner', os.path.basename(grandparent)) else parent
        seen_run_dirs.add(candidate)
    for run_dir in sorted(seen_run_dirs):
        run_name = os.path.basename(run_dir)
        if tag_filter and tag_filter not in run_name:
            continue
        # find the inner log.csv
        csv_paths = glob.glob(os.path.join(run_dir, '*/log.csv'))
        if not csv_paths:
            csv_paths = glob.glob(os.path.join(run_dir, 'log.csv'))
        csv = csv_paths[0] if csv_paths else None
        epochs, tf1, vf1, bf1 = parse_log_csv(csv) if csv else (0, None, None, None)

        jobid = find_jobid(run_name)
        squeue_info = squeue.get(jobid) if jobid else None
        state = squeue_info[0] if squeue_info else 'DONE'
        elapsed = squeue_info[1] if squeue_info else ''

        test_f1 = test_results_f1(run_dir)

        ds = guess_dataset(run_name)
        sota_f1, sota_method = SOTA.get(ds, (None, None)) if ds else (None, None)

        rows.append({
            'name': run_name, 'dataset': ds or '?',
            'jobid': jobid or '-', 'state': state, 'elapsed': elapsed,
            'epochs': epochs, 'train_f1': tf1, 'val_f1': vf1,
            'best_val_f1': bf1, 'test_f1': test_f1,
            'sota_f1': sota_f1, 'sota_method': sota_method,
        })
    return rows


def print_snapshot(rows):
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n=== galash training monitor — {now} ===\n')
    if not rows:
        print('  (no matching runs)')
        return

    header = f'{"run":<30s} {"job":>6s} {"state":>5s} {"ep":>4s}  {"train":>6s}  {"val":>6s}  {"best_v":>6s}  {"TEST":>6s}  {"SOTA":>6s}  {"gap":>10s}'
    print(header)
    print('-' * len(header))
    for r in rows:
        # Prefer test F1 for the SOTA gap (apples-to-apples with published SOTA).
        # If test not yet written, fall back to best val F1 and mark preview.
        if r['test_f1'] is not None:
            gap_str = fmt_gap(r['test_f1'], r['sota_f1'], is_preview=False)
        else:
            gap_str = fmt_gap(r['best_val_f1'], r['sota_f1'], is_preview=True)
        state = r['state'][:5]
        print(
            f'{r["name"]:<30s} '
            f'{str(r["jobid"]):>6s} '
            f'{state:>5s} '
            f'{r["epochs"]:>4d}  '
            f'{fmt_f1(r["train_f1"]):>6s}  '
            f'{fmt_f1(r["val_f1"]):>6s}  '
            f'{fmt_f1(r["best_val_f1"]):>6s}  '
            f'{fmt_f1(r["test_f1"]):>6s}  '
            f'{fmt_f1(r["sota_f1"]):>6s}  '
            f'{gap_str:>10s}'
        )

    # SOTA info footer
    print('\nSOTA reference (late 2025, all TEST-set F1):')
    seen = set()
    for r in rows:
        if r['sota_f1'] is not None and r['dataset'] not in seen:
            print(f'  {r["dataset"]:<18s}  F1={r["sota_f1"]:.4f}  [{r["sota_method"]}]')
            seen.add(r['dataset'])
    print('\nLegend:')
    print('  best_v  = best val F1 over training so far')
    print('  TEST    = test-set F1 from test_results.json (only after run finishes)')
    print('  SOTA    = published test-set F1 (fair comparison target)')
    print('  gap     = TEST − SOTA when TEST available; else ~ best_v − SOTA (val-based preview)')
    print('  A tilde (~) on the gap means the number is a val-based preview, NOT an SOTA claim.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--tag', default=None, help='filter runs whose name contains this substring')
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                rows = snapshot(args.tag)
                # clear-ish
                print('\033[2J\033[H', end='')
                print_snapshot(rows)
                sys.stdout.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('\n(stopped)')
    else:
        print_snapshot(snapshot(args.tag))


if __name__ == '__main__':
    main()
