#!/usr/bin/env python3
"""
Download & prepare change-detection datasets into a unified layout:

    data/{dataset_name}/
        train/  val/  test/
            A/        ← reference (time-1) images
            B/        ← target   (time-2) images
            label/    ← binary change masks

Usage
─────
    python download_datasets.py --out data --all
    python download_datasets.py --out data --datasets levir_cd cdd s2looking
    python download_datasets.py --list

Requirements
────────────
    pip install gdown requests tqdm
    Optional: pip install kaggle   (needs ~/.kaggle/kaggle.json)
              pip install rarfile  (for .rar archives)
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────
# Lazy imports
# ───────────────────────────────────────────────────────────────────────
def _lazy(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        return __import__(pkg)


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────
def _download_url(url, dst, desc=None):
    requests = _lazy("requests")
    tqdm_mod = _lazy("tqdm")
    r = requests.get(url, stream=True, allow_redirects=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dst, "wb") as f, tqdm_mod.tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or os.path.basename(dst)
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def _gdown(file_id, dst):
    gdown = _lazy("gdown")
    gdown.download(id=file_id, output=str(dst), quiet=False)


def _gdown_folder(folder_id, dst):
    gdown = _lazy("gdown")
    gdown.download_folder(id=folder_id, output=str(dst), quiet=False)


def _kaggle_download(dataset_slug, dst):
    """Download from Kaggle.  Needs ~/.kaggle/kaggle.json."""
    print(f"  downloading from Kaggle: {dataset_slug}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", dst, "--unzip"],
        check=True,
    )


def _extract(archive, dst):
    print(f"  extracting {os.path.basename(archive)} …")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dst)
        return True
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as t:
            t.extractall(dst)
        return True
    # try .rar
    if archive.lower().endswith(".rar"):
        try:
            rarfile = _lazy("rarfile")
            with rarfile.RarFile(archive, "r") as rf:
                rf.extractall(dst)
            return True
        except Exception:
            # try system unrar
            try:
                subprocess.run(["unrar", "x", "-o+", archive, dst], check=True)
                return True
            except Exception:
                pass
    print(f"  ⚠ could not extract {archive}")
    return False


def _ensure_split_dirs(base):
    for split in ("train", "val", "test"):
        for sub in ("A", "B", "label"):
            os.makedirs(os.path.join(base, split, sub), exist_ok=True)


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _copy_images(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src_dir)):
        if any(f.lower().endswith(e) for e in IMG_EXTS):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            n += 1
    return n


def _count(base):
    """Count images in A/ per split.  Returns total."""
    total = 0
    for split in ("train", "val", "test"):
        a = os.path.join(base, split, "A")
        if os.path.isdir(a):
            n = len([f for f in os.listdir(a) if not f.startswith(".")])
            total += n
            if n:
                print(f"    {split:>5s}/A : {n} images")
    return total


def _find_dirs_named(root, target_name):
    """Case-insensitive recursive find of directories named target_name."""
    results = []
    target_lower = target_name.lower()
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath).lower() == target_lower:
            results.append(dirpath)
    return results


def _arrange_standard(extracted, out):
    """
    Walk the extracted tree and copy images into out/{split}/{A,B,label}/.
    Handles many naming conventions case-insensitively.
    """
    split_aliases = {
        "train": "train", "training": "train",
        "val": "val", "valid": "val", "validation": "val",
        "test": "test", "testing": "test",
    }
    a_aliases = {"a", "t1", "time1", "image1", "im1", "ref", "before", "img1", "pre"}
    b_aliases = {"b", "t2", "time2", "image2", "im2", "tgt", "tar", "after", "img2", "post"}
    label_aliases = {"label", "labels", "gt", "mask", "masks", "out", "change", "annotation"}

    _ensure_split_dirs(out)
    found = False

    # Collect all directories
    for dirpath, dirnames, _ in os.walk(extracted):
        dirname_lower = os.path.basename(dirpath).lower()

        # Check if this dir is a split
        if dirname_lower in split_aliases:
            dst_split = split_aliases[dirname_lower]

            # Look for A/B/label children
            for child in os.listdir(dirpath):
                child_path = os.path.join(dirpath, child)
                if not os.path.isdir(child_path):
                    continue
                child_lower = child.lower()

                target = None
                if child_lower in a_aliases:
                    target = "A"
                elif child_lower in b_aliases:
                    target = "B"
                elif child_lower in label_aliases:
                    target = "label"

                if target:
                    n = _copy_images(child_path, os.path.join(out, dst_split, target))
                    if n:
                        found = True
                        print(f"    {dst_split}/{target} ← {child} ({n} images)")

    return found


# ───────────────────────────────────────────────────────────────────────
# Per-dataset functions
# ───────────────────────────────────────────────────────────────────────

# ── LEVIR-CD ─────────────────────────────────────────────────────────
def download_levir_cd(out):
    """LEVIR-CD via Kaggle (Google Drive is rate-limited)."""
    dst = os.path.join(out, "levir_cd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    ok = False
    # Method 1: Kaggle
    try:
        _kaggle_download("mdrifaturrahman33/levir-cd", tmp)
        ok = True
    except Exception as e:
        print(f"  ⚠ Kaggle failed ({e}), trying gdown …")

    # Method 2: gdown (often rate-limited)
    if not ok:
        arc = os.path.join(tmp, "levir_cd.zip")
        try:
            _gdown("1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim", arc)
            _extract(arc, tmp)
            ok = True
        except Exception as e:
            print(f"  ⚠ gdown failed ({e})")

    # Method 3: Dropbox 256×256 version
    if not ok:
        arc = os.path.join(tmp, "LEVIR-CD256.zip")
        try:
            print("  trying Dropbox (256×256 version) …")
            _download_url(
                "https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=1",
                arc, desc="LEVIR-CD-256"
            )
            _extract(arc, tmp)
            ok = True
        except Exception as e:
            print(f"  ⚠ Dropbox failed ({e})")

    _ensure_split_dirs(dst)
    if ok:
        _arrange_standard(tmp, dst)

    if not _count(dst):
        print("  ⚠ Auto-download failed. Manual options:")
        print("    Kaggle : https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd")
        print("    Official: https://justchenhao.github.io/LEVIR/")

    shutil.rmtree(tmp, ignore_errors=True)
    return dst


# ── LEVIR-CD+ ────────────────────────────────────────────────────────
def download_levir_cd_plus(out):
    """LEVIR-CD+: already downloaded successfully last run."""
    dst = os.path.join(out, "levir_cd_plus")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    arc = os.path.join(tmp, "levir_cd_plus.zip")

    _gdown("1JamSsxiytXdzAIk6VDVWfc-OsX-81U81", arc)
    _extract(arc, tmp)

    _ensure_split_dirs(dst)
    _arrange_standard(tmp, dst)

    shutil.rmtree(tmp, ignore_errors=True)
    _count(dst)
    return dst


# ── WHU-CD ───────────────────────────────────────────────────────────
def download_whu_cd(out):
    """WHU-CD via Kaggle (official site gives HTML, not a zip)."""
    dst = os.path.join(out, "whu_cd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    ok = False
    # Kaggle (pre-processed into standard CD layout)
    try:
        _kaggle_download("clannaspark/whu-cd", tmp)
        ok = True
    except Exception as e:
        print(f"  ⚠ Kaggle failed ({e})")

    # Fallback: Kaggle unmodified version
    if not ok:
        try:
            _kaggle_download("mdrifaturrahman33/whu-cd-datasetunmodified-from-website", tmp)
            ok = True
        except Exception as e:
            print(f"  ⚠ Kaggle fallback also failed ({e})")

    _ensure_split_dirs(dst)
    if ok:
        _arrange_standard(tmp, dst)

    if not _count(dst):
        print("  ⚠ Auto-download failed. Manual options:")
        print("    Kaggle : https://www.kaggle.com/datasets/clannaspark/whu-cd")
        print("    Official: https://gpcv.whu.edu.cn/data/building_dataset.html")

    shutil.rmtree(tmp, ignore_errors=True)
    return dst


# ── DSIFN-CD ─────────────────────────────────────────────────────────
def download_dsifn_cd(out):
    """DSIFN-CD: try Dropbox 256×256 version first (most reliable)."""
    dst = os.path.join(out, "dsifn_cd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    ok = False
    # Dropbox 256×256 (from ChangeFormer, zip, reliable)
    arc = os.path.join(tmp, "dsifn256.zip")
    try:
        print("  downloading DSIFN-CD-256 from Dropbox …")
        _download_url(
            "https://www.dropbox.com/sh/i54h8kkpgar1s07/AAA0rBAFl9UZ3U3Z1_o46UT0a/DSIFN-CD-256.zip?dl=1",
            arc, desc="DSIFN-CD-256"
        )
        _extract(arc, tmp)
        ok = True
    except Exception as e:
        print(f"  ⚠ Dropbox failed ({e})")

    # Fallback: Google Drive
    if not ok:
        arc2 = os.path.join(tmp, "dsifn.zip")
        try:
            print("  trying Google Drive …")
            _gdown("1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9", arc2)
            _extract(arc2, tmp)
            ok = True
        except Exception as e:
            print(f"  ⚠ gdown failed ({e})")

    _ensure_split_dirs(dst)
    if ok:
        _arrange_standard(tmp, dst)

    if not _count(dst):
        print("  ⚠ Auto-download failed. Manual options:")
        print("    OpenDataLab: https://opendatalab.com/OpenDataLab/DSIFN-CD")
        print("    GitHub: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images")

    shutil.rmtree(tmp, ignore_errors=True)
    return dst


# ── S2Looking ────────────────────────────────────────────────────────
def _arrange_s2looking(extracted, out):
    """S2Looking uses Image1/Image2/label1/label2.  Merge labels."""
    import numpy as np
    from PIL import Image

    _ensure_split_dirs(out)
    found = False

    for split in ("train", "val", "test", "Train", "Val", "Test"):
        dst_split = split.lower()
        split_dir = os.path.join(extracted, split)
        if not os.path.isdir(split_dir):
            # search nested
            for child in os.listdir(extracted):
                cand = os.path.join(extracted, child, split)
                if os.path.isdir(cand):
                    split_dir = cand
                    break
            else:
                continue

        # Map: Image1→A, Image2→B
        for src_name, tgt in [("Image1", "A"), ("Image2", "B")]:
            src = os.path.join(split_dir, src_name)
            if os.path.isdir(src):
                n = _copy_images(src, os.path.join(out, dst_split, tgt))
                if n:
                    found = True
                    print(f"    {dst_split}/{tgt} ← {src_name} ({n} images)")

        # Merge label1 (new buildings) + label2 (demolished) → label
        l1_dir = os.path.join(split_dir, "label1")
        l2_dir = os.path.join(split_dir, "label2")
        label_dst = os.path.join(out, dst_split, "label")
        os.makedirs(label_dst, exist_ok=True)

        if os.path.isdir(l1_dir):
            n = 0
            for f in sorted(os.listdir(l1_dir)):
                if not any(f.lower().endswith(e) for e in IMG_EXTS):
                    continue
                l1_path = os.path.join(l1_dir, f)
                l2_path = os.path.join(l2_dir, f) if os.path.isdir(l2_dir) else None

                l1 = np.array(Image.open(l1_path).convert("L"))
                if l2_path and os.path.exists(l2_path):
                    l2 = np.array(Image.open(l2_path).convert("L"))
                    merged = np.maximum(l1, l2)  # union of new + demolished
                else:
                    merged = l1

                Image.fromarray(merged).save(os.path.join(label_dst, f))
                n += 1

            if n:
                found = True
                print(f"    {dst_split}/label ← label1+label2 merged ({n} masks)")

    return found


def download_s2looking(out):
    """S2Looking: 5000 pairs, 1024×1024."""
    dst = os.path.join(out, "s2looking")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    # Download from Google Drive
    print("  downloading S2Looking from Google Drive (11 GB) …")
    try:
        _gdown_folder("1zzb6hif2hwWx4z8UIMLpMAInkhbmmrFY", tmp)
    except Exception as e:
        print(f"  ⚠ folder download failed ({e})")

    # Extract any zips
    for zf in glob.glob(os.path.join(tmp, "**", "*.zip"), recursive=True):
        _extract(zf, tmp)

    _arrange_s2looking(tmp, dst)

    if not _count(dst):
        print("  ⚠ Arrangement failed.  Manual download:")
        print("    https://github.com/S2Looking/Dataset")
        print(f"    Extract, then: python download_datasets.py --arrange_s2looking <path> --out {out}")
        # keep _tmp so user can arrange manually
    else:
        shutil.rmtree(tmp, ignore_errors=True)

    return dst


# ── SECOND ───────────────────────────────────────────────────────────
def _arrange_second(extracted, out):
    """SECOND uses im1/im2/label1/label2 (semantic labels, .tif).
    Change mask = pixels where label1 != label2."""
    import numpy as np
    from PIL import Image

    _ensure_split_dirs(out)
    found = False

    for split in ("train", "val", "test", "Train", "Val", "Test"):
        dst_split = split.lower()
        split_dir = os.path.join(extracted, split)
        if not os.path.isdir(split_dir):
            for child in os.listdir(extracted):
                cand = os.path.join(extracted, child, split)
                if os.path.isdir(cand):
                    split_dir = cand
                    break
            else:
                continue

        # im1→A, im2→B
        for src_name, tgt in [("im1", "A"), ("im2", "B")]:
            src = os.path.join(split_dir, src_name)
            if os.path.isdir(src):
                n = _copy_images(src, os.path.join(out, dst_split, tgt))
                if n:
                    found = True
                    print(f"    {dst_split}/{tgt} ← {src_name} ({n} images)")

        # Generate binary change mask from label1 vs label2
        l1_dir = os.path.join(split_dir, "label1")
        l2_dir = os.path.join(split_dir, "label2")
        label_dst = os.path.join(out, dst_split, "label")
        os.makedirs(label_dst, exist_ok=True)

        if os.path.isdir(l1_dir) and os.path.isdir(l2_dir):
            n = 0
            for f in sorted(os.listdir(l1_dir)):
                if not any(f.lower().endswith(e) for e in IMG_EXTS):
                    continue
                l1_path = os.path.join(l1_dir, f)
                l2_path = os.path.join(l2_dir, f)
                if not os.path.exists(l2_path):
                    continue

                l1 = np.array(Image.open(l1_path))
                l2 = np.array(Image.open(l2_path))
                # Binary change mask: any pixel where semantic class differs
                if l1.ndim == 3:  # RGB semantic maps
                    change = np.any(l1 != l2, axis=-1).astype(np.uint8) * 255
                else:
                    change = (l1 != l2).astype(np.uint8) * 255

                out_name = os.path.splitext(f)[0] + ".png"
                Image.fromarray(change).save(os.path.join(label_dst, out_name))
                n += 1

            if n:
                found = True
                print(f"    {dst_split}/label ← label1!=label2 ({n} change masks)")

    return found


def download_second(out):
    """SECOND: 4662 pairs, 512×512, multi-class."""
    dst = os.path.join(out, "second")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    arc = os.path.join(tmp, "second.zip")

    print("  downloading SECOND from Google Drive (3.8 GB) …")
    try:
        _gdown("1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u", arc)
        _extract(arc, tmp)
    except Exception as e:
        print(f"  ⚠ download failed ({e})")
        print("  Manual: https://captain-whu.github.io/SCD/")
        shutil.rmtree(tmp, ignore_errors=True)
        return dst

    _arrange_second(tmp, dst)

    if not _count(dst):
        print("  ⚠ Arrangement failed.")
        print("  Manual: https://captain-whu.github.io/SCD/")
    else:
        shutil.rmtree(tmp, ignore_errors=True)

    return dst


# ── CDD ──────────────────────────────────────────────────────────────
def download_cdd(out):
    """CDD (Season-Varying): 16000 pairs, 256×256."""
    dst = os.path.join(out, "cdd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    tmp = os.path.join(dst, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    # Correct Zenodo API URL
    urls = [
        "https://zenodo.org/api/records/13290067/files/CDD.zip/content",
        "https://zenodo.org/records/13290067/files/CDD.zip?download=1",
    ]
    arc = os.path.join(tmp, "cdd.zip")

    ok = False
    for url in urls:
        try:
            print(f"  trying {url[:60]}…")
            _download_url(url, arc, desc="CDD")
            if os.path.getsize(arc) > 1000:
                _extract(arc, tmp)
                ok = True
                break
        except Exception as e:
            print(f"  ⚠ failed ({e})")

    _ensure_split_dirs(dst)
    if ok:
        _arrange_standard(tmp, dst)

    # CDD may use "OUT" for labels
    if ok and not _count(dst):
        # deep search
        for root, dirs, _ in os.walk(tmp):
            for d in dirs:
                dl = d.lower()
                full = os.path.join(root, d)
                # find split parent
                parts = full.lower().split(os.sep)
                split = None
                for p in parts:
                    if p in ("train", "val", "test"):
                        split = p
                        break
                if not split:
                    continue
                if dl in ("a",):
                    _copy_images(full, os.path.join(dst, split, "A"))
                elif dl in ("b",):
                    _copy_images(full, os.path.join(dst, split, "B"))
                elif dl in ("out", "label", "gt"):
                    _copy_images(full, os.path.join(dst, split, "label"))

    if not _count(dst):
        print("  ⚠ Auto-download failed.")
        print("    Zenodo: https://zenodo.org/records/13290067")

    shutil.rmtree(tmp, ignore_errors=True)
    return dst


# ── SYSU-CD ──────────────────────────────────────────────────────────
def download_sysu_cd(out):
    """SYSU-CD: only available on BaiduYun / OneDrive (no GitHub releases)."""
    dst = os.path.join(out, "sysu_cd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    _ensure_split_dirs(dst)

    # Check if user manually placed files
    tmp = os.path.join(dst, "_tmp")
    if os.path.isdir(tmp):
        _arrange_standard(tmp, dst)
        if _count(dst):
            shutil.rmtree(tmp, ignore_errors=True)
            return dst

    print("  SYSU-CD is only available via BaiduYun or OneDrive (no direct links).")
    print()
    print("  Download options:")
    print("    BaiduYun : https://pan.baidu.com/s/15lQPG_hXZbLp91VywwcT7Q  (code: mlls)")
    print("    OneDrive : https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/liumx23_mail2_sysu_edu_cn/Emgc0jtEcshAnRkgq1ZTE9AB-kfXzSEzU_PAQ-5YF8Neaw")
    print()
    print(f"  After downloading, extract into: {dst}/_tmp/")
    print(f"  Then re-run: python download_datasets.py --out {out} --datasets sysu_cd")

    return dst


# ── OSCD ─────────────────────────────────────────────────────────────
def download_oscd(out):
    """OSCD: available on HuggingFace."""
    dst = os.path.join(out, "oscd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    _ensure_split_dirs(dst)

    print("  OSCD download options:")
    print("    HuggingFace: pip install datasets")
    print('                 from datasets import load_dataset')
    print('                 ds = load_dataset("blanchon/OSCD_RGB")')
    print("    IEEE DataPort: https://rcdaudt.github.io/oscd/")
    print(f"  Arrange into: {dst}/train/{{A,B,label}}")

    return dst


# ── CLCD ─────────────────────────────────────────────────────────────
def download_clcd(out):
    """CLCD (CropLand): data is linked in the repo README, not in the repo itself."""
    dst = os.path.join(out, "clcd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    _ensure_split_dirs(dst)

    # Check if user placed data
    tmp = os.path.join(dst, "_tmp")
    if os.path.isdir(tmp):
        _arrange_standard(tmp, dst)
        if _count(dst):
            shutil.rmtree(tmp, ignore_errors=True)
            return dst

    print("  CLCD data is not in the GitHub repo itself.")
    print("  Check the README for download links:")
    print("    https://github.com/liumency/CropLand-CD")
    print(f"  Extract into: {dst}/_tmp/  then re-run this script.")

    return dst


# ── xBD / xView2 ────────────────────────────────────────────────────
def download_xbd(out):
    """xBD: requires registration."""
    dst = os.path.join(out, "xbd")
    if _count(dst):
        print("  already present, skipping")
        return dst

    _ensure_split_dirs(dst)

    print("  xBD requires registration at https://xview2.org/dataset")
    print("  After downloading, use: python download_datasets.py --arrange_xbd /path/to/xbd")

    return dst


def arrange_xbd(raw_dir, out):
    """Convert raw xBD into standard A/B/label layout."""
    dst = os.path.join(out, "xbd")
    _ensure_split_dirs(dst)

    for split in ("train", "test", "hold"):
        split_dir = None
        for cand in (split, f"tier3_{split}"):
            p = os.path.join(raw_dir, cand)
            if os.path.isdir(p):
                split_dir = p
                break
        if not split_dir:
            continue

        dst_split = {"train": "train", "test": "test", "hold": "val"}[split]
        images_dir = os.path.join(split_dir, "images")
        if not os.path.isdir(images_dir):
            continue

        for fname in sorted(os.listdir(images_dir)):
            if "_pre_disaster" in fname:
                base = fname.replace("_pre_disaster", "")
                post = fname.replace("_pre_disaster", "_post_disaster")
                if os.path.exists(os.path.join(images_dir, post)):
                    shutil.copy2(os.path.join(images_dir, fname),
                                 os.path.join(dst, dst_split, "A", base))
                    shutil.copy2(os.path.join(images_dir, post),
                                 os.path.join(dst, dst_split, "B", base))

    print("  ⚠ xBD labels are JSON polygons → rasterize with https://github.com/DIUx-xView/xView2_scoring")
    _count(dst)


# ── HRSCD ────────────────────────────────────────────────────────────
def download_hrscd(out):
    dst = os.path.join(out, "hrscd")
    _ensure_split_dirs(dst)
    if _count(dst):
        print("  already present, skipping")
        return dst
    print("  HRSCD requires IEEE DataPort (free account):")
    print("    https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset")
    return dst


# ── DynamicEarthNet ──────────────────────────────────────────────────
def download_dynamicearthnet(out):
    dst = os.path.join(out, "dynamicearthnet")
    _ensure_split_dirs(dst)
    if _count(dst):
        print("  already present, skipping")
        return dst
    print("  DynamicEarthNet (~1 TB): https://mediatum.ub.tum.de/1650201")
    print("  Multi-temporal — pick two dates per site for bi-temporal CD.")
    return dst


# ───────────────────────────────────────────────────────────────────────
# Registry
# ───────────────────────────────────────────────────────────────────────
DATASETS = {
    "levir_cd":         ("LEVIR-CD",         "637 pairs, 1024², 0.5m, buildings",         download_levir_cd),
    "levir_cd_plus":    ("LEVIR-CD+",        "985 pairs, 1024², 0.5m, buildings",         download_levir_cd_plus),
    "whu_cd":           ("WHU-CD",           "~7k patches, 256², 0.2m, buildings",        download_whu_cd),
    "dsifn_cd":         ("DSIFN-CD",         "3940 pairs, 512², sub-m, 6-class",          download_dsifn_cd),
    "s2looking":        ("S2Looking",        "5000 pairs, 1024², 0.5-0.8m, buildings",    download_s2looking),
    "second":           ("SECOND",           "4662 pairs, 512², 0.5-3m, multi-class",     download_second),
    "cdd":              ("CDD",              "16000 pairs, 256², 3-100cm, binary",         download_cdd),
    "sysu_cd":          ("SYSU-CD",          "20000 pairs, 256², 0.5m, urban",             download_sysu_cd),
    "oscd":             ("OSCD",             "24 pairs, ~600², 10m, Sentinel-2",           download_oscd),
    "clcd":             ("CLCD",             "600 pairs, 512², 0.5-2m, cropland",          download_clcd),
    "xbd":              ("xBD / xView2",     "11034 pairs, 1024², sub-m, damage",          download_xbd),
    "hrscd":            ("HRSCD",            "291 pairs, 10000², 0.5m, multi-class",       download_hrscd),
    "dynamicearthnet":  ("DynamicEarthNet",  "daily multi-temporal, 1024², 3m",             download_dynamicearthnet),
}

# Medical change-detection datasets (see download_medical.py for details).
try:
    from download_medical import MEDICAL_DATASETS
    DATASETS.update(MEDICAL_DATASETS)
except Exception:
    pass


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Download & prepare change-detection datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="data", help="Root output directory")
    p.add_argument("--all", action="store_true", help="Download ALL datasets")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()))
    p.add_argument("--list", action="store_true", help="List available datasets")
    p.add_argument("--arrange_xbd", default=None, help="Path to raw xBD data")
    p.add_argument("--arrange_s2looking", default=None, help="Path to raw S2Looking data")
    p.add_argument("--arrange_second", default=None, help="Path to raw SECOND data")
    args = p.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for key, (name, desc, _) in DATASETS.items():
            print(f"  {key:<20s}  {name:<20s}  {desc}")
        print(f"\nUsage:  python {sys.argv[0]} --out data --datasets levir_cd cdd s2looking")
        print(f"        python {sys.argv[0]} --out data --all\n")
        return

    if args.arrange_xbd:
        arrange_xbd(args.arrange_xbd, args.out)
        return
    if args.arrange_s2looking:
        _arrange_s2looking(args.arrange_s2looking, os.path.join(args.out, "s2looking"))
        _count(os.path.join(args.out, "s2looking"))
        return
    if args.arrange_second:
        _arrange_second(args.arrange_second, os.path.join(args.out, "second"))
        _count(os.path.join(args.out, "second"))
        return

    targets = list(DATASETS.keys()) if args.all else (args.datasets or [])
    if not targets:
        p.print_help()
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"\nDownloading {len(targets)} dataset(s) into {os.path.abspath(args.out)}/\n")

    for key in targets:
        name, desc, fn = DATASETS[key]
        print(f"{'─'*60}")
        print(f"  {name}  ({desc})")
        print(f"{'─'*60}")
        try:
            fn(args.out)
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for key in targets:
        d = os.path.join(args.out, key)
        if os.path.isdir(d):
            n = _count(d)
            status = f"{n} images" if n else "needs manual setup (see instructions above)"
            print(f"  {key:<20s} {status}")
    print()


if __name__ == "__main__":
    main()
