#!/usr/bin/env python3
"""Extract LEVIR-CD-256 zip and arrange into data/levir_cd/{train,val,test}/{A,B,label}.

The Dropbox LEVIR-CD256.zip has a FLAT structure:
  LEVIR-CD256/A/{train,val,test}_*.png
  LEVIR-CD256/B/{train,val,test}_*.png
  LEVIR-CD256/label/{train,val,test}_*.png
  LEVIR-CD256/list/{train,val,test}.txt

We split by the filename prefix (train_*, val_*, test_*).
"""
import os, sys, shutil, zipfile

ROOT = '/home/nfs/tals/galash'
TMP = os.path.join(ROOT, 'data/levir_cd/_tmp')
OUT = os.path.join(ROOT, 'data/levir_cd')
ARC = os.path.join(TMP, 'LEVIR-CD256.zip')
EXTRACT = os.path.join(TMP, 'extracted')

os.makedirs(EXTRACT, exist_ok=True)

print(f"Extracting {ARC}...")
with zipfile.ZipFile(ARC, 'r') as z:
    z.extractall(EXTRACT)
print("  extracted")

# Source dirs
base = os.path.join(EXTRACT, 'LEVIR-CD256')
a_src = os.path.join(base, 'A')
b_src = os.path.join(base, 'B')
l_src = os.path.join(base, 'label')

for d in (a_src, b_src, l_src):
    assert os.path.isdir(d), f"missing {d}"

# Clear and recreate output dirs
for split in ('train', 'val', 'test'):
    for t in ('A', 'B', 'label'):
        d = os.path.join(OUT, split, t)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if not f.startswith('.'):
                os.remove(os.path.join(d, f))

def split_of(fname):
    low = fname.lower()
    if low.startswith('train_'):
        return 'train'
    if low.startswith('val_'):
        return 'val'
    if low.startswith('test_'):
        return 'test'
    return None

# Process A/B/label — same filenames should exist in all three
counts = {'train': 0, 'val': 0, 'test': 0}
for fname in sorted(os.listdir(a_src)):
    split = split_of(fname)
    if split is None:
        continue
    for src_dir, tgt in [(a_src, 'A'), (b_src, 'B'), (l_src, 'label')]:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(OUT, split, tgt, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    counts[split] += 1

print(f"\nSplit counts (A images): {counts}")

# Final counts
print("\nFinal LEVIR-CD counts:")
for split in ['train', 'val', 'test']:
    for t in ['A', 'B', 'label']:
        d = os.path.join(OUT, split, t)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if not f.startswith('.')])
            print(f"  {split}/{t}: {n}")

print("\nCleaning _tmp...")
shutil.rmtree(TMP, ignore_errors=True)
print("Done.")
