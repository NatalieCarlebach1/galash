#!/usr/bin/env python3
"""Extract SECOND nested archives and arrange into data/second/{train,val,test}/{A,B,label}.

SECOND ships as:
  _tmp/SECOND_train_set.rar  → train set (im1/im2/label1/label2 at root)
  _tmp/SECOND_total_test.zip → test set under test/
"""
import os, sys, shutil, zipfile
import rarfile
import numpy as np
from PIL import Image

ROOT = '/home/nfs/tals/galash'
TMP = os.path.join(ROOT, 'data/second/_tmp')
OUT = os.path.join(ROOT, 'data/second')
EXTRACT = os.path.join(TMP, 'extracted')
os.makedirs(EXTRACT, exist_ok=True)

train_rar = os.path.join(TMP, 'SECOND_train_set.rar')
test_zip = os.path.join(TMP, 'SECOND_total_test.zip')

# Re-extract if missing
need_train_extract = not os.path.isdir(os.path.join(EXTRACT, 'im1'))
need_test_extract = not os.path.isdir(os.path.join(EXTRACT, 'test'))

if need_train_extract and os.path.exists(train_rar):
    print(f"Extracting {train_rar}...")
    r = rarfile.RarFile(train_rar)
    r.extractall(EXTRACT)
    print("  train extracted")

if need_test_extract and os.path.exists(test_zip):
    print(f"Extracting {test_zip}...")
    with zipfile.ZipFile(test_zip, 'r') as z:
        z.extractall(EXTRACT)
    print("  test extracted")

# Detect each split's source dir.
# After extraction:
#   EXTRACT/im1, im2, label1, label2  → train
#   EXTRACT/test/im1, im2, label1, label2  → test
# No val split in SECOND standard release.
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

SPLITS = {}
if os.path.isdir(os.path.join(EXTRACT, 'im1')):
    SPLITS['train'] = EXTRACT
if os.path.isdir(os.path.join(EXTRACT, 'test', 'im1')):
    SPLITS['test'] = os.path.join(EXTRACT, 'test')

print(f"\nDetected splits: {list(SPLITS.keys())}")

for split_name, split_dir in SPLITS.items():
    print(f"\nProcessing {split_name}: {split_dir}")
    im1 = os.path.join(split_dir, 'im1')
    im2 = os.path.join(split_dir, 'im2')
    l1 = os.path.join(split_dir, 'label1')
    l2 = os.path.join(split_dir, 'label2')

    out_A = os.path.join(OUT, split_name, 'A')
    out_B = os.path.join(OUT, split_name, 'B')
    out_L = os.path.join(OUT, split_name, 'label')
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)
    os.makedirs(out_L, exist_ok=True)

    # Clear previous partial content
    for d in (out_A, out_B, out_L):
        for f in os.listdir(d):
            if not f.startswith('.'):
                os.remove(os.path.join(d, f))

    nA = nB = nL = 0
    for f in sorted(os.listdir(im1)):
        if not f.lower().endswith(IMG_EXTS):
            continue
        shutil.copy2(os.path.join(im1, f), os.path.join(out_A, f))
        nA += 1
    for f in sorted(os.listdir(im2)):
        if not f.lower().endswith(IMG_EXTS):
            continue
        shutil.copy2(os.path.join(im2, f), os.path.join(out_B, f))
        nB += 1

    if os.path.isdir(l1) and os.path.isdir(l2):
        for f in sorted(os.listdir(l1)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            p1 = os.path.join(l1, f)
            p2 = os.path.join(l2, f)
            if not os.path.exists(p2):
                continue
            a = np.array(Image.open(p1))
            b = np.array(Image.open(p2))
            if a.ndim == 3:
                change = np.any(a != b, axis=-1).astype(np.uint8) * 255
            else:
                change = (a != b).astype(np.uint8) * 255
            out_name = os.path.splitext(f)[0] + '.png'
            Image.fromarray(change).save(os.path.join(out_L, out_name))
            nL += 1

    print(f"  A={nA}  B={nB}  label={nL}")

# Final counts
print("\nFinal SECOND counts:")
for split in ['train', 'val', 'test']:
    for t in ['A', 'B', 'label']:
        d = os.path.join(OUT, split, t)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if not f.startswith('.')])
            print(f"  {split}/{t}: {n}")

print("\nCleaning up _tmp...")
shutil.rmtree(TMP, ignore_errors=True)
print("Done.")
