#!/usr/bin/env python3
"""Extract DSIFN-CD Real/subset from the RAR archive and arrange into data/dsifn_cd/{train,val,test}/{A,B,label}."""
import os, sys, shutil
import rarfile
from pathlib import Path

ROOT = '/home/nfs/tals/galash'
RAR = os.path.join(ROOT, 'data/dsifn_cd/_tmp/dsifn.zip')
OUT = os.path.join(ROOT, 'data/dsifn_cd')

EXTRACT_DIR = os.path.join(ROOT, 'data/dsifn_cd/_tmp/extracted')
os.makedirs(EXTRACT_DIR, exist_ok=True)

print(f"Opening {RAR}")
r = rarfile.RarFile(RAR)
names = r.namelist()
# Extract only Real/subset entries (train/val/test with A/B/label/OUT)
prefix = 'ChangeDetectionDataset/Real/subset/'
sub_names = [n for n in names if n.startswith(prefix)]
print(f"  {len(sub_names)} entries match Real/subset")

# Check: what subdirs exist?
from collections import defaultdict
struct = defaultdict(int)
for n in sub_names:
    parts = n.split('/')
    if len(parts) >= 6:
        key = '/'.join(parts[3:5])  # e.g. "train/A", "test/label"
        struct[key] += 1
print(f"  sub-structure:")
for k, v in sorted(struct.items()):
    print(f"    {k}: {v}")

# Extract
print("Extracting (this may take a few minutes)...")
r.extractall(path=EXTRACT_DIR, members=sub_names)
print("  extracted")

# Now map to A/B/label
src_root = os.path.join(EXTRACT_DIR, prefix.rstrip('/'))
for split in ['train', 'val', 'test']:
    for src_name, tgt in [('t1', 'A'), ('t2', 'B'), ('mask', 'label')]:
        src = os.path.join(src_root, split, src_name)
        dst = os.path.join(OUT, split, tgt)
        os.makedirs(dst, exist_ok=True)
        if os.path.isdir(src):
            n = 0
            for f in sorted(os.listdir(src)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                    n += 1
            print(f"  {split}/{tgt} ← {src_name} ({n} files)")
        else:
            print(f"  {src} missing — trying alternate names")
            # Try capitalized or other variants
            for alt in ['A' if src_name == 't1' else 'B' if src_name == 't2' else 'OUT',
                        src_name.upper()]:
                src2 = os.path.join(src_root, split, alt)
                if os.path.isdir(src2):
                    n = 0
                    for f in sorted(os.listdir(src2)):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                            shutil.copy2(os.path.join(src2, f), os.path.join(dst, f))
                            n += 1
                    print(f"  {split}/{tgt} ← {alt} ({n} files)")
                    break

# Cleanup _tmp
print("\nFinal counts:")
for split in ['train', 'val', 'test']:
    for t in ['A', 'B', 'label']:
        d = os.path.join(OUT, split, t)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if not f.startswith('.')])
            print(f"  {split}/{t}: {n}")

print("\nCleaning up extraction...")
shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
print("Done.")
