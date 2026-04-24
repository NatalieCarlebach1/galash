#!/usr/bin/env python3
"""Tile OSCD city-level images into 512x512 crops with 256-stride overlap.

OSCD is only 14 train + 10 test city pairs at ~500x500 px (RGB version).
At 512 native resolution with no tiling, we'd train on 13 images — useless.
This script tiles them with overlap so we get ~50-100 crops per city
(~700 train / ~500 test crops after tiling).

Input:  data/oscd/{train,test}/{A,B,label}/*.png
Output: data/oscd_tiled/{train,test}/{A,B,label}/*.png  (with tile indices in filename)

The original `oscd` dir is kept intact for reference.
"""
import os, sys, shutil
import numpy as np
from PIL import Image

ROOT = '/home/nfs/tals/galash'
SRC = os.path.join(ROOT, 'data/oscd')
DST = os.path.join(ROOT, 'data/oscd_tiled')

TILE = 512
STRIDE = 256  # 50% overlap

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def tile_image(img_arr, label_arr):
    """Yield (tile_a, tile_b, tile_l, r, c) for every valid crop, padding at edges if needed."""
    H, W = img_arr.shape[:2]
    if H < TILE or W < TILE:
        # pad up to tile size
        pad_h = max(0, TILE - H)
        pad_w = max(0, TILE - W)
        if img_arr.ndim == 3:
            img_arr = np.pad(img_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            img_arr = np.pad(img_arr, ((0, pad_h), (0, pad_w)), mode='reflect')
        if label_arr is not None:
            label_arr = np.pad(label_arr, ((0, pad_h), (0, pad_w)), mode='constant')
        H, W = img_arr.shape[:2]

    rows = list(range(0, H - TILE + 1, STRIDE))
    cols = list(range(0, W - TILE + 1, STRIDE))
    if rows[-1] + TILE < H:
        rows.append(H - TILE)
    if cols[-1] + TILE < W:
        cols.append(W - TILE)

    for r in rows:
        for c in cols:
            yield (r, c, img_arr[r:r+TILE, c:c+TILE], label_arr[r:r+TILE, c:c+TILE] if label_arr is not None else None)


def process_split(split):
    a_src = os.path.join(SRC, split, 'A')
    b_src = os.path.join(SRC, split, 'B')
    l_src = os.path.join(SRC, split, 'label')
    if not os.path.isdir(a_src):
        return

    a_dst = os.path.join(DST, split, 'A')
    b_dst = os.path.join(DST, split, 'B')
    l_dst = os.path.join(DST, split, 'label')
    for d in (a_dst, b_dst, l_dst):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if not f.startswith('.'):
                os.remove(os.path.join(d, f))

    n_tiles = 0
    n_kept = 0  # tiles with at least one changed pixel (useful for training balance)
    for fname in sorted(os.listdir(a_src)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(fname)[0]

        img_a = np.array(Image.open(os.path.join(a_src, fname)).convert('RGB'))
        img_b = np.array(Image.open(os.path.join(b_src, fname)).convert('RGB'))
        lbl = None
        lp = os.path.join(l_src, fname)
        if os.path.exists(lp):
            # OSCD labels are I;16 — normalize to uint8 binary
            im = Image.open(lp)
            arr = np.array(im)
            lbl = ((arr > 0).astype(np.uint8) * 255)

        # Align sizes (use min dims)
        Ha, Wa = img_a.shape[:2]
        Hb, Wb = img_b.shape[:2]
        Hl, Wl = lbl.shape[:2] if lbl is not None else (Ha, Wa)
        H = min(Ha, Hb, Hl)
        W = min(Wa, Wb, Wl)
        img_a = img_a[:H, :W]
        img_b = img_b[:H, :W]
        if lbl is not None:
            lbl = lbl[:H, :W]

        # Create aligned pairs: zip A and B together
        gen_a = list(tile_image(img_a, lbl))
        gen_b = list(tile_image(img_b, None))
        # gen_b has same rows/cols as gen_a (same H,W) so lengths match
        for (ra, ca, tile_a, tile_l), (rb, cb, tile_b, _) in zip(gen_a, gen_b):
            out_name = f"{stem}_r{ra:04d}_c{ca:04d}.png"
            Image.fromarray(tile_a).save(os.path.join(a_dst, out_name))
            Image.fromarray(tile_b).save(os.path.join(b_dst, out_name))
            if tile_l is not None:
                Image.fromarray(tile_l).save(os.path.join(l_dst, out_name))
            n_tiles += 1
            if tile_l is not None and tile_l.max() > 0:
                n_kept += 1

    print(f"  {split}: {n_tiles} tiles ({n_kept} with change > 0)")


print(f"Tiling OSCD → {DST}  (tile={TILE}, stride={STRIDE})")
for split in ('train', 'test'):
    process_split(split)

print("\nFinal OSCD_tiled counts:")
for split in ('train', 'test'):
    for t in ('A', 'B', 'label'):
        d = os.path.join(DST, split, t)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if not f.startswith('.')])
            print(f"  {split}/{t}: {n}")
print("Done.")
