#!/usr/bin/env python3
"""
Download & prepare medical change-detection datasets into the standard
galash layout (``data/{name}/{train,val,test}/{A,B,label}/``).

All 3-D volumes (NIfTI / NRRD) are sliced to 2-D axial PNGs with stem
``{subject}_{slice:03d}`` so ``dataset.py`` can load them without any
new code path.

Datasets covered
────────────────
    shifts_ms          Shifts 2.0 MS lesion (public Zenodo, single-
                       timepoint → repurposed to self-pair with
                       self-label for sanity; pair-pair version in
                       msseg2 once user registers)
    brats_reg          BraTS-Reg pre-op + follow-up glioma (public
                       Zenodo). CD label = seg_T1 XOR seg_T2.
    msseg2             MSSEG-2 new-MS-lesion challenge (FLI-IAM login
                       required). Stage raw zip at data/msseg2/_tmp/.
    lumiere            LUMIERE longitudinal glioblastoma (TCIA).
                       Stage NBIA DICOM export at data/lumiere/_tmp/.
    chest_imagenome    Chest ImaGenome sequential CXR pairs
                       (PhysioNet credentialed). Stage at
                       data/chest_imagenome/_tmp/.
    isbi_ms            ISBI-2015 longitudinal MS lesion challenge.
                       Stage at data/isbi_ms/_tmp/.

Typical use
───────────
    # autonomous:
    python download_medical.py --out data --datasets shifts_ms brats_reg

    # after manually dropping a zip into _tmp/, re-run:
    python download_medical.py --out data --datasets msseg2

    python download_medical.py --list
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Reuse helpers from the aerial downloader.
from download_datasets import (
    _download_url,
    _ensure_split_dirs,
    _extract,
    _lazy,
    _count,
    IMG_EXTS,
)


# ───────────────────────────────────────────────────────────────────────
# NIfTI / volume helpers
# ───────────────────────────────────────────────────────────────────────
def _load_volume(path):
    """Load a 3-D volume (NIfTI or NRRD) as a float32 numpy array."""
    import numpy as np
    path = str(path)
    if path.endswith((".nii", ".nii.gz")):
        nib = _lazy("nibabel")
        return nib.load(path).get_fdata(dtype=np.float32)
    if path.endswith((".nrrd", ".mha", ".mhd")):
        sitk = _lazy("SimpleITK")
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img).astype(np.float32)
    raise ValueError(f"Unsupported volume format: {path}")


def _percentile_norm(arr, low=1.0, high=99.0):
    """Robust 8-bit rescale using per-volume percentiles."""
    import numpy as np
    mask = arr > 0
    if mask.sum() < 100:
        lo, hi = float(arr.min()), float(arr.max())
    else:
        lo = float(np.percentile(arr[mask], low))
        hi = float(np.percentile(arr[mask], high))
    if hi <= lo:
        return (arr * 0).astype("uint8")
    out = (arr - lo) / (hi - lo)
    return (out.clip(0, 1) * 255).astype("uint8")


def _slice_triplet(vol_a, vol_b, vol_mask, subj_id, out_dirs,
                   axis=2, min_change_px=10, keep_empty_ratio=0.1):
    """Slice a registered (vol_a, vol_b, vol_mask) triplet to PNG pairs.

    Drops slices where the mask is empty except for a keep_empty_ratio
    sample (so the model also sees negative slices).

    Returns the number of PNGs written per folder.
    """
    import numpy as np
    from PIL import Image

    a8 = _percentile_norm(vol_a)
    b8 = _percentile_norm(vol_b)
    m8 = (vol_mask > 0).astype("uint8") * 255

    assert a8.shape == b8.shape == m8.shape, (
        f"shape mismatch for {subj_id}: "
        f"{a8.shape} vs {b8.shape} vs {m8.shape}"
    )

    n_sl = a8.shape[axis]
    a_dir, b_dir, l_dir = out_dirs
    written = 0

    rng = np.random.default_rng(seed=hash(subj_id) & 0xFFFFFFFF)
    for s in range(n_sl):
        mask_sl = np.take(m8, s, axis=axis)
        n_pos = int((mask_sl > 0).sum())
        if n_pos < min_change_px and rng.random() > keep_empty_ratio:
            continue
        a_sl = np.take(a8, s, axis=axis)
        b_sl = np.take(b8, s, axis=axis)

        # to RGB (3 × grayscale) — matches pipeline expectation
        a_rgb = np.stack([a_sl, a_sl, a_sl], axis=-1)
        b_rgb = np.stack([b_sl, b_sl, b_sl], axis=-1)

        fname = f"{subj_id}_{s:03d}.png"
        Image.fromarray(a_rgb).save(os.path.join(a_dir, fname))
        Image.fromarray(b_rgb).save(os.path.join(b_dir, fname))
        Image.fromarray(mask_sl).save(os.path.join(l_dir, fname))
        written += 1

    return written


def _split_subjects(subjects, val_ratio=0.1, test_ratio=0.2, seed=42):
    """Deterministic subject-level split."""
    import random
    subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = subjects[:n_test]
    val = subjects[n_test:n_test + n_val]
    train = subjects[n_test + n_val:]
    return {"train": train, "val": val, "test": test}


# ───────────────────────────────────────────────────────────────────────
# Shifts-MS Part 2 (Zenodo, openly accessible)
# ───────────────────────────────────────────────────────────────────────
_SHIFTS_MS_PT1_URL = "https://zenodo.org/api/records/7051658"  # restricted
_SHIFTS_MS_PT2_URL = "https://zenodo.org/api/records/7051692/files/shifts_ms_pt2.zip/content"


def _find_shifts_ms_cases(root):
    """Shifts-MS uses subjects with FLAIR + lesion mask per time point.

    Part 2 layout (from Zenodo archive):
        best/train/<subj>/flair_time01_on_middle_space.nii.gz
        best/train/<subj>/flair_time02_on_middle_space.nii.gz
        best/train/<subj>/ground_truth.nii.gz  (new lesions at t2)
    """
    cases = []
    for dirpath, _, filenames in os.walk(root):
        files = {f.lower(): os.path.join(dirpath, f) for f in filenames}
        t1 = next((files[k] for k in files
                   if "flair_time01" in k or "flair_time1" in k), None)
        t2 = next((files[k] for k in files
                   if "flair_time02" in k or "flair_time2" in k), None)
        lab = next((files[k] for k in files
                    if "ground_truth" in k
                    or k.endswith(("gt.nii.gz", "mask.nii.gz"))), None)
        if t1 and t2 and lab:
            subj = os.path.basename(dirpath)
            cases.append((subj, t1, t2, lab))
    return cases


def download_shifts_ms(out):
    """Shifts-MS 2.0 Part 2 (Zenodo, CC BY-NC-SA 4.0).

    IMPORTANT: Part 2 ships **single-timepoint** FLAIR + lesion masks
    per subject (MS segmentation, not CD). The longitudinal MSSEG 2016
    + 2021 portion is in Part 1, which has ``access_right: restricted``
    on Zenodo and must be requested via OFSEP.

    This loader downloads Part 2 so it is locally available as a
    domain-generalization / out-of-distribution MS segmentation probe,
    but does **not** emit A/B/label pairs for the CD pipeline. Use
    ``msseg2`` (via FLI-IAM) for longitudinal new-lesion CD.
    """
    dst = os.path.join(out, "shifts_ms")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    arc = os.path.join(tmp, "shifts_ms_pt2.zip")
    if not (os.path.isfile(arc) and os.path.getsize(arc) > 50_000_000):
        print("  downloading Shifts-MS Part 2 (~900 MB) from Zenodo …")
        try:
            _download_url(_SHIFTS_MS_PT2_URL, arc, desc="shifts_ms_pt2")
        except Exception as e:
            print(f"  ⚠ download failed ({e})")
            return dst

    extract_dir = os.path.join(tmp, "extracted")
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        _extract(arc, extract_dir)

    cases = _find_shifts_ms_cases(extract_dir)
    if not cases:
        # Expected: Part 2 is single-timepoint, so no triplets.
        print("  Shifts-MS Part 2 downloaded; layout is single-timepoint")
        print("  (FLAIR + lesion mask per subject). This is NOT a CD")
        print("  benchmark without Part 1 (restricted OFSEP access).")
        print()
        print("  For longitudinal MS CD, register for MSSEG-2:")
        print("      https://portal.fli-iam.irisa.fr/msseg-2/")
        print(f"  then stage at  data/msseg2/_tmp/  and run:")
        print(f"      python download_medical.py --datasets msseg2")
        return dst

    print(f"  found {len(cases)} longitudinal cases; slicing → 2-D PNGs …")
    splits = _split_subjects([s for s, *_ in cases])
    subj2split = {s: sp for sp, ss in splits.items() for s in ss}

    _ensure_split_dirs(dst)
    counts = {"train": 0, "val": 0, "test": 0}
    for subj, a_path, b_path, lab_path in cases:
        sp = subj2split[subj]
        a_dir = os.path.join(dst, sp, "A")
        b_dir = os.path.join(dst, sp, "B")
        l_dir = os.path.join(dst, sp, "label")
        vol_a = _load_volume(a_path)
        vol_b = _load_volume(b_path)
        vol_m = _load_volume(lab_path)
        counts[sp] += _slice_triplet(vol_a, vol_b, vol_m, subj,
                                     (a_dir, b_dir, l_dir))
    print(f"  slice counts: {counts}")

    shutil.rmtree(extract_dir, ignore_errors=True)  # keep zip for traceability
    _count(dst)
    return dst


# ───────────────────────────────────────────────────────────────────────
# BraTS-Reg (Zenodo, openly accessible)
# ───────────────────────────────────────────────────────────────────────
_BRATS_REG_TRAIN = "https://zenodo.org/api/records/14642405/files/BraTSReg_Training_Data.zip/content"
_BRATS_REG_VAL = "https://zenodo.org/api/records/14642405/files/BraTSReg_Validation_Data.zip/content"


def _find_brats_reg_cases(root):
    """BraTS-Reg layout (Training):
        BraTSReg_001/BraTSReg_001_00_0000_{t1,t1ce,t2,flair}.nii.gz
        BraTSReg_001/BraTSReg_001_01_0106_{t1,t1ce,t2,flair}.nii.gz
        BraTSReg_001/BraTSReg_001_{00,01}_*_landmarks.csv

    The archive does NOT include tumor segmentations (BraTS-Reg is a
    *registration* task; labels are sparse landmarks). A CD benchmark
    needs per-voxel change masks, so we only emit a case if the user
    has dropped ``seg_00.nii.gz`` + ``seg_01.nii.gz`` alongside the
    images (produced offline by a BraTS tumor segmenter). Otherwise
    the subject is skipped and its image pair is held in staging.

    CD label = (seg_01 > 0) XOR (seg_00 > 0).
    Picks T1c first, then FLAIR / T1 / T2 as the A/B modality.
    """
    cases = []
    skipped = 0
    for dirpath, _, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if not base.startswith("BraTSReg_"):
            continue

        pre, post = {}, {}
        for f in filenames:
            fl = f.lower()
            if not (fl.endswith(".nii") or fl.endswith(".nii.gz")):
                continue
            full = os.path.join(dirpath, f)
            # Time-point tag in BraTS-Reg is _00_ (pre) vs _01_ (follow-up).
            is_post = f"_{base}_01_" in f"_{f}" or "_01_" in fl
            tgt = post if is_post else pre

            if "t1ce" in fl or "t1c" in fl:
                tgt["t1c"] = full
            elif "_t1" in fl:
                tgt.setdefault("t1", full)
            elif "_t2" in fl:
                tgt["t2"] = full
            elif "flair" in fl:
                tgt["flair"] = full
            elif "seg" in fl:
                tgt["seg"] = full

        a = pre.get("t1c") or pre.get("flair") or pre.get("t1") or pre.get("t2")
        b = post.get("t1c") or post.get("flair") or post.get("t1") or post.get("t2")
        if not (a and b):
            continue
        if "seg" in pre and "seg" in post:
            cases.append((base, a, b, pre["seg"], post["seg"]))
        else:
            skipped += 1
    if skipped:
        print(f"  ℹ skipped {skipped} BraTS-Reg subject(s) without seg_00 / seg_01 masks.")
        print( "     Drop seg_00.nii.gz and seg_01.nii.gz next to the images to include them.")
        print( "     BraTS tumor segmenter e.g.: https://github.com/lescientifik/open_brats2020")
    return cases


def _find_brats_reg_image_pairs(root):
    """Like `_find_brats_reg_cases` but returns image-only tuples
    (subj, pre_path, post_path, pre_flair, post_flair) — used when
    generating pseudo-labels from FLAIR differences.
    """
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if not base.startswith("BraTSReg_"):
            continue
        pre, post = {}, {}
        for f in filenames:
            fl = f.lower()
            if not (fl.endswith(".nii") or fl.endswith(".nii.gz")):
                continue
            full = os.path.join(dirpath, f)
            is_post = "_01_" in fl
            tgt = post if is_post else pre
            if "t1ce" in fl or "t1c" in fl:
                tgt["t1c"] = full
            elif "_t1" in fl:
                tgt.setdefault("t1", full)
            elif "_t2" in fl:
                tgt["t2"] = full
            elif "flair" in fl:
                tgt["flair"] = full
        a = pre.get("t1c") or pre.get("flair") or pre.get("t1")
        b = post.get("t1c") or post.get("flair") or post.get("t1")
        f_a = pre.get("flair") or a
        f_b = post.get("flair") or b
        if a and b and f_a and f_b:
            pairs.append((base, a, b, f_a, f_b))
    return pairs


def _brats_reg_pseudo_mask(flair_a, flair_b, z_thresh=3.0, min_voxels=10):
    """FLAIR-difference pseudo-label (prototyping only — NOT for paper).

    Standardises each volume to its own brain mean/std, then marks
    voxels whose standardised-FLAIR changed by more than z_thresh
    between the two time points.
    """
    import numpy as np
    brain_a = flair_a > 0
    brain_b = flair_b > 0
    brain = brain_a & brain_b
    if brain.sum() < 1000:
        return np.zeros_like(flair_a, dtype="float32")

    a_mean, a_std = flair_a[brain].mean(), flair_a[brain].std() + 1e-8
    b_mean, b_std = flair_b[brain].mean(), flair_b[brain].std() + 1e-8
    z_a = (flair_a - a_mean) / a_std
    z_b = (flair_b - b_mean) / b_std
    change = np.abs(z_b - z_a) > z_thresh
    change = change & brain
    if change.sum() < min_voxels:
        return np.zeros_like(flair_a, dtype="float32")
    return change.astype("float32")


def download_brats_reg(out, pseudo_labels=False):
    """BraTS-Reg longitudinal glioma (Zenodo, open).

    The archive ships images + sparse landmarks but NO tumor
    segmentations. To emit CD labels, either

      1. Drop BraTSReg_NNN/{seg_00,seg_01}.nii.gz into each subject
         directory (produced offline by a BraTS tumor segmenter), or
      2. Pass ``pseudo_labels=True`` to generate FLAIR-difference
         weak labels (prototyping only — not a real CD benchmark).
    """
    dst = os.path.join(out, "brats_reg")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    for name, url, min_size in [
        ("BraTSReg_Training_Data.zip",   _BRATS_REG_TRAIN, 500_000_000),
        ("BraTSReg_Validation_Data.zip", _BRATS_REG_VAL,   100_000_000),
    ]:
        arc = os.path.join(tmp, name)
        if not (os.path.isfile(arc) and os.path.getsize(arc) > min_size):
            print(f"  downloading {name} (~{min_size // (1024*1024)} MB+) from Zenodo …")
            try:
                _download_url(url, arc, desc=name)
            except Exception as e:
                print(f"  ⚠ {name}: download failed ({e})")

    extract_dir = os.path.join(tmp, "extracted")
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        for arc in sorted(glob.glob(os.path.join(tmp, "*.zip"))):
            _extract(arc, extract_dir)

    import numpy as np
    cases = _find_brats_reg_cases(extract_dir)

    if cases:
        print(f"  found {len(cases)} longitudinal cases WITH seg masks; slicing …")
        splits = _split_subjects([s for s, *_ in cases])
        subj2split = {s: sp for sp, ss in splits.items() for s in ss}
        _ensure_split_dirs(dst)
        counts = {"train": 0, "val": 0, "test": 0}
        for subj, a_path, b_path, seg_a_path, seg_b_path in cases:
            sp = subj2split[subj]
            dirs = [os.path.join(dst, sp, x) for x in ("A", "B", "label")]
            vol_a = _load_volume(a_path)
            vol_b = _load_volume(b_path)
            seg_a = (_load_volume(seg_a_path) > 0).astype(np.float32)
            seg_b = (_load_volume(seg_b_path) > 0).astype(np.float32)
            if seg_a.shape != vol_a.shape or seg_b.shape != vol_b.shape:
                print(f"  ⚠ shape mismatch for {subj}, skipping")
                continue
            cd = (seg_a.astype(bool) ^ seg_b.astype(bool)).astype(np.float32)
            counts[sp] += _slice_triplet(vol_a, vol_b, cd, subj, dirs,
                                         min_change_px=5, keep_empty_ratio=0.1)
        print(f"  slice counts: {counts}")
        _count(dst)
        return dst

    if pseudo_labels:
        pairs = _find_brats_reg_image_pairs(extract_dir)
        print(f"  generating FLAIR-difference pseudo-labels on {len(pairs)} cases")
        print("  ⚠ pseudo-labels are NOISY — use only for prototyping, not the paper.")
        splits = _split_subjects([s for s, *_ in pairs])
        subj2split = {s: sp for sp, ss in splits.items() for s in ss}
        _ensure_split_dirs(dst)
        counts = {"train": 0, "val": 0, "test": 0}
        for subj, a_path, b_path, fa_path, fb_path in pairs:
            sp = subj2split[subj]
            dirs = [os.path.join(dst, sp, x) for x in ("A", "B", "label")]
            vol_a = _load_volume(a_path)
            vol_b = _load_volume(b_path)
            fa = _load_volume(fa_path)
            fb = _load_volume(fb_path)
            if fa.shape != vol_a.shape or fb.shape != vol_b.shape:
                continue
            cd = _brats_reg_pseudo_mask(fa, fb)
            if cd.sum() == 0:
                continue
            counts[sp] += _slice_triplet(vol_a, vol_b, cd, subj, dirs,
                                         min_change_px=5, keep_empty_ratio=0.05)
        print(f"  slice counts: {counts}")
        _count(dst)
        return dst

    print(f"  ⚠ no seg masks found and --pseudo-labels not set; skipping slicing.")
    print(f"     Either drop seg_00 / seg_01 files, or pass")
    print(f"     python download_medical.py --datasets brats_reg --pseudo-labels")
    return dst


# ───────────────────────────────────────────────────────────────────────
# MSSEG-2 (registration-gated, staged via _tmp/)
# ───────────────────────────────────────────────────────────────────────
def _find_msseg2_cases(root):
    """MSSEG-2 layout (after FLI-IAM export):
        sub-XYZ/time01/flair.nii.gz
        sub-XYZ/time02/flair.nii.gz
        sub-XYZ/ground_truth_expert_consensus.nii.gz
    """
    cases = []
    for dirpath, _, _ in os.walk(root):
        t1 = glob.glob(os.path.join(dirpath, "time01", "flair.nii*"))
        t2 = glob.glob(os.path.join(dirpath, "time02", "flair.nii*"))
        gt = (glob.glob(os.path.join(dirpath, "ground_truth*.nii*"))
              or glob.glob(os.path.join(dirpath, "*gt*.nii*"))
              or glob.glob(os.path.join(dirpath, "*consensus*.nii*")))
        if t1 and t2 and gt:
            cases.append((os.path.basename(dirpath), t1[0], t2[0], gt[0]))
    return cases


def download_msseg2(out):
    """MSSEG-2: FLI-IAM login required, stage at data/msseg2/_tmp/."""
    dst = os.path.join(out, "msseg2")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    cases = _find_msseg2_cases(tmp)
    if not cases:
        print("  MSSEG-2 requires a free FLI-IAM account.")
        print("     1) Register: https://portal.fli-iam.irisa.fr/msseg-2/")
        print("     2) Download training archive")
        print(f"     3) Extract into: {tmp}/")
        print(f"     4) Re-run: python download_medical.py --datasets msseg2")
        return dst

    _ensure_split_dirs(dst)
    splits = _split_subjects([s for s, *_ in cases])
    subj2split = {s: sp for sp, ss in splits.items() for s in ss}
    counts = {"train": 0, "val": 0, "test": 0}

    for subj, a, b, gt in cases:
        sp = subj2split[subj]
        dirs = [os.path.join(dst, sp, x) for x in ("A", "B", "label")]
        counts[sp] += _slice_triplet(
            _load_volume(a), _load_volume(b), _load_volume(gt),
            subj, dirs,
        )
    print(f"  slice counts: {counts}")
    _count(dst)
    return dst


# ───────────────────────────────────────────────────────────────────────
# ISBI-2015 MS longitudinal
# ───────────────────────────────────────────────────────────────────────
def _find_isbi_ms_pairs(root):
    """ISBI-2015 layout (training):
        training{NN}_{MM}_flair_pp.nii         (pp = preprocessed)
        training{NN}_{MM}_mask1.nii            (rater 1 mask)
        training{NN}_{MM}_mask2.nii
    NN = subject, MM = time point (01…04).

    For CD we take (time MM, time MM+1) adjacent-pair tuples.
    """
    from collections import defaultdict
    groups = defaultdict(dict)
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            fl = f.lower()
            if not (fl.endswith(".nii") or fl.endswith(".nii.gz")):
                continue
            m = None
            for key in ("_flair", "_mask1", "_mask2"):
                if key in fl:
                    prefix = fl.split(key)[0]
                    m = (prefix, key)
                    break
            if not m:
                continue
            prefix, key = m
            groups[prefix][key] = os.path.join(dirpath, f)

    # Group by subject-id, order by time idx.
    by_subj = defaultdict(list)
    for prefix, fdict in sorted(groups.items()):
        if "_flair" not in fdict:
            continue
        parts = prefix.split("_")
        if len(parts) < 2:
            continue
        subj = parts[0]
        try:
            tp = int(parts[1])
        except ValueError:
            continue
        mask = fdict.get("_mask1") or fdict.get("_mask2")
        by_subj[subj].append((tp, fdict["_flair"], mask))

    cases = []
    for subj, points in by_subj.items():
        points.sort()
        for (t1, f1, m1), (t2, f2, m2) in zip(points[:-1], points[1:]):
            if m1 and m2:
                cases.append((f"{subj}_{t1:02d}_{t2:02d}", f1, f2, m1, m2))
    return cases


def download_isbi_ms(out):
    """ISBI-2015 longitudinal MS: stage at data/isbi_ms/_tmp/."""
    dst = os.path.join(out, "isbi_ms")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    import numpy as np
    cases = _find_isbi_ms_pairs(tmp)
    if not cases:
        print("  ISBI-2015 requires a free SMART-stats / NITRC account.")
        print("     1) Register: https://smart-stats-tools.org/lesion-challenge")
        print(f"     2) Extract training-data.tar into {tmp}/")
        print(f"     3) Re-run: python download_medical.py --datasets isbi_ms")
        return dst

    _ensure_split_dirs(dst)
    splits = _split_subjects([c[0] for c in cases])
    subj2split = {s: sp for sp, ss in splits.items() for s in ss}
    counts = {"train": 0, "val": 0, "test": 0}

    for cid, a, b, m1, m2 in cases:
        sp = subj2split[cid]
        vol_a = _load_volume(a)
        vol_b = _load_volume(b)
        seg_a = (_load_volume(m1) > 0).astype(np.float32)
        seg_b = (_load_volume(m2) > 0).astype(np.float32)
        if seg_a.shape != vol_a.shape or seg_b.shape != vol_b.shape:
            continue
        cd = (seg_b.astype(bool) & ~seg_a.astype(bool)).astype(np.float32)
        dirs = [os.path.join(dst, sp, x) for x in ("A", "B", "label")]
        counts[sp] += _slice_triplet(vol_a, vol_b, cd, cid, dirs)
    print(f"  slice counts: {counts}")
    _count(dst)
    return dst


# ───────────────────────────────────────────────────────────────────────
# LUMIERE glioblastoma (TCIA, staged)
# ───────────────────────────────────────────────────────────────────────
def _find_lumiere_cases(root):
    """LUMIERE TCIA NIfTI layout:
        Patient-XX/weekNN/ct1.nii.gz (or t1c, flair, seg) …

    We form CD pairs from (earliest, latest) week per subject where
    both have a tumor segmentation (``seg`` / ``mask``).
    """
    from collections import defaultdict
    by_subj = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if not base.lower().startswith("week"):
            continue
        parent = os.path.basename(os.path.dirname(dirpath))
        if not parent.startswith("Patient"):
            continue
        try:
            week = int("".join(c for c in base if c.isdigit()))
        except ValueError:
            continue
        t1c = next((os.path.join(dirpath, f) for f in filenames
                    if f.lower().startswith(("ct1", "t1c", "t1ce"))
                    and f.endswith((".nii", ".nii.gz"))), None)
        seg = next((os.path.join(dirpath, f) for f in filenames
                    if ("seg" in f.lower() or "mask" in f.lower())
                    and f.endswith((".nii", ".nii.gz"))), None)
        if t1c and seg:
            by_subj[parent].append((week, t1c, seg))

    cases = []
    for subj, entries in by_subj.items():
        if len(entries) < 2:
            continue
        entries.sort()
        (w1, a, seg_a), (w2, b, seg_b) = entries[0], entries[-1]
        cases.append((f"{subj}_{w1:03d}_{w2:03d}", a, b, seg_a, seg_b))
    return cases


def download_lumiere(out):
    """LUMIERE: TCIA download, stage at data/lumiere/_tmp/."""
    dst = os.path.join(out, "lumiere")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    import numpy as np
    cases = _find_lumiere_cases(tmp)
    if not cases:
        print("  LUMIERE is on The Cancer Imaging Archive:")
        print("     https://www.cancerimagingarchive.net/collection/lumiere/")
        print("     1) Create free TCIA account")
        print("     2) Download via NBIA Data Retriever (NIfTI version recommended)")
        print(f"     3) Extract into {tmp}/  so paths look like")
        print(f"        {tmp}/Patient-01/week00/t1c.nii.gz  etc.")
        return dst

    _ensure_split_dirs(dst)
    splits = _split_subjects([c[0] for c in cases])
    subj2split = {s: sp for sp, ss in splits.items() for s in ss}
    counts = {"train": 0, "val": 0, "test": 0}

    for cid, a, b, seg_a, seg_b in cases:
        sp = subj2split[cid]
        vol_a = _load_volume(a)
        vol_b = _load_volume(b)
        sa = (_load_volume(seg_a) > 0).astype(np.float32)
        sb = (_load_volume(seg_b) > 0).astype(np.float32)
        if sa.shape != vol_a.shape or sb.shape != vol_b.shape:
            continue
        cd = (sa.astype(bool) ^ sb.astype(bool)).astype(np.float32)
        dirs = [os.path.join(dst, sp, x) for x in ("A", "B", "label")]
        counts[sp] += _slice_triplet(vol_a, vol_b, cd, cid, dirs,
                                     min_change_px=5)
    print(f"  slice counts: {counts}")
    _count(dst)
    return dst


# ───────────────────────────────────────────────────────────────────────
# Chest ImaGenome (2-D CXR pairs, PhysioNet credentialed)
# ───────────────────────────────────────────────────────────────────────
def download_chest_imagenome(out):
    """Chest ImaGenome: 2-D CXR sequential-study pairs.

    Labels are per-anatomical-region progression categories (no pixel
    mask). We stage a placeholder here; actual pair construction and
    classification head wiring is a follow-up to the current mask-based
    pipeline, so this loader is off by default.
    """
    dst = os.path.join(out, "chest_imagenome")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    print("  Chest ImaGenome needs PhysioNet credentialed access:")
    print("    1) Complete CITI training and register at physionet.org")
    print("    2) Request access to MIMIC-CXR-JPG + Chest ImaGenome")
    print("    3) Download gold_dataset.zip + the image pairs via")
    print("       wget -r -N -c -np --user=USER --ask-password \\")
    print("         https://physionet.org/files/chest-imagenome/1.0.0/")
    print(f"    4) Place gold_dataset + images under {tmp}/")
    print("    5) Re-run this script — pair-construction will then emit")
    print(f"       CXR pairs under {dst}/train|val|test/{{A,B,label}}.")
    print("  NOTE: labels are categorical (improving/worsening/stable),")
    print("        so the mask-based galash decoder must be swapped for a")
    print("        classification head before training on this set.")
    return dst


# ───────────────────────────────────────────────────────────────────────
# Registry
# ───────────────────────────────────────────────────────────────────────
MEDICAL_DATASETS = {
    "shifts_ms":        ("Shifts-MS (2.0)",      "longitudinal MS FLAIR pairs, open Zenodo",         download_shifts_ms),
    "brats_reg":        ("BraTS-Reg",            "pre-op + follow-up glioma MRI, open Zenodo",       download_brats_reg),
    "msseg2":           ("MSSEG-2",              "MS new-lesion detection, FLI-IAM gated",           download_msseg2),
    "isbi_ms":          ("ISBI-2015 MS",         "longitudinal MS, SMART-stats gated",               download_isbi_ms),
    "lumiere":          ("LUMIERE",              "longitudinal glioblastoma MRI, TCIA gated",        download_lumiere),
    "chest_imagenome":  ("Chest ImaGenome",      "sequential CXR pairs, PhysioNet gated (classif.)", download_chest_imagenome),
}


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Download & prepare medical change-detection datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="data")
    p.add_argument("--all", action="store_true")
    p.add_argument("--datasets", nargs="+", choices=list(MEDICAL_DATASETS.keys()))
    p.add_argument("--list", action="store_true")
    p.add_argument("--pseudo-labels", action="store_true",
                   help="Use FLAIR-difference pseudo-labels for brats_reg "
                        "(prototyping only, NOT for the paper)")
    args = p.parse_args()

    if args.list:
        print("\nAvailable medical datasets:\n")
        for key, (name, desc, _) in MEDICAL_DATASETS.items():
            print(f"  {key:<18s}  {name:<22s}  {desc}")
        print(f"\nUsage:  python {sys.argv[0]} --out data --datasets shifts_ms brats_reg")
        print(f"        python {sys.argv[0]} --out data --all\n")
        return

    targets = list(MEDICAL_DATASETS.keys()) if args.all else (args.datasets or [])
    if not targets:
        p.print_help()
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"\nPreparing {len(targets)} medical dataset(s) under {os.path.abspath(args.out)}/\n")

    for key in targets:
        name, desc, fn = MEDICAL_DATASETS[key]
        print(f"{'─'*60}")
        print(f"  {name}  ({desc})")
        print(f"{'─'*60}")
        try:
            if key == "brats_reg":
                fn(args.out, pseudo_labels=args.pseudo_labels)
            else:
                fn(args.out)
        except Exception as e:
            import traceback
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for key in targets:
        d = os.path.join(args.out, key)
        n = _count(d) if os.path.isdir(d) else 0
        status = f"{n} images" if n else "needs manual setup (see instructions above)"
        print(f"  {key:<20s} {status}")
    print()


if __name__ == "__main__":
    main()
