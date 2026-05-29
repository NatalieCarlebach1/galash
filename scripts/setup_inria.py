#!/usr/bin/env python3
"""Extract Inria Aerial Image Labeling Dataset from zip.

Produces:
  data/inria/train/images/<city><N>.tif   — RGB, 5000×5000
  data/inria/train/gt/<city><N>.tif       — binary mask, 5000×5000

Run once before pre-training:
  python3 scripts/setup_inria.py
"""
import argparse, zipfile, pathlib, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip",  default="data/inria/inria-aerial-image-labeling-dataset.zip")
    p.add_argument("--out",  default="data/inria")
    args = p.parse_args()

    zip_path = pathlib.Path(args.zip)
    out_dir  = pathlib.Path(args.out)
    if not zip_path.exists():
        sys.exit(f"Zip not found: {zip_path}")

    print(f"Extracting {zip_path} → {out_dir} ...")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist()
                   if "train/" in m and m.endswith(".tif")]
        print(f"  {len(members)} train files to extract")
        for i, m in enumerate(members, 1):
            dest = out_dir / pathlib.Path(m).relative_to("AerialImageDataset")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            with z.open(m) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if i % 20 == 0:
                print(f"  {i}/{len(members)}")

    imgs = list((out_dir / "train/images").glob("*.tif"))
    gts  = list((out_dir / "train/gt").glob("*.tif"))
    print(f"Done. {len(imgs)} images, {len(gts)} GT masks.")

if __name__ == "__main__":
    main()
