"""
Overfit sanity-check: train on a tiny subset (8-16 images) and verify
that the loss drops to ~0 and F1 → 1.0.

If this does NOT overfit → architecture / gradient bug.
If it does overfit        → arch is correct, go train on full data.

Usage:
    python overfit_test.py                         # auto-detect data
    python overfit_test.py --n 8 --steps 200       # 8 images, 200 steps
    python overfit_test.py --dataset cdd            # use a specific dataset
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model import ChangeDetector
from dataset import CDDataset, Augmenter, ValTransform


# ── losses (inlined for self-contained script) ──
def dice_loss(logits, target, smooth=1.0):
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()


def compute_loss(masks, iou_pred, change_map, gt_mask, ph, pw):
    pred = F.interpolate(masks, gt_mask.shape[-2:], mode="bilinear", align_corners=False)
    loss_bce = F.binary_cross_entropy_with_logits(pred, gt_mask)
    loss_dice = dice_loss(pred, gt_mask)

    gt_small = F.adaptive_avg_pool2d(gt_mask, (ph, pw))
    gt_small = (gt_small.view(gt_mask.size(0), -1) > 0.3).float()
    loss_latent = F.binary_cross_entropy(change_map.clamp(1e-6, 1 - 1e-6), gt_small)

    total = loss_bce + loss_dice + 0.5 * loss_latent
    return total, dict(bce=loss_bce.item(), dice=loss_dice.item(), latent=loss_latent.item())


def compute_metrics(masks, gt_mask):
    pred = F.interpolate(masks, gt_mask.shape[-2:], mode="bilinear", align_corners=False)
    pred_bin = (pred.sigmoid() > 0.5).float()
    tp = (pred_bin * gt_mask).sum().item()
    fp = (pred_bin * (1 - gt_mask)).sum().item()
    fn = ((1 - pred_bin) * gt_mask).sum().item()
    eps = 1e-8
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    iou = tp / (tp + fp + fn + eps)
    return dict(f1=f1, iou=iou, precision=p, recall=r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--dataset", default=None, help="single dataset name (auto-detect if omitted)")
    p.add_argument("--n", type=int, default=8, help="number of images to overfit on")
    p.add_argument("--steps", type=int, default=300, help="gradient steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--dino", default="facebook/dinov2-base")
    p.add_argument("--sam2_ckpt", default="checkpoints/sam2_hiera_tiny.pt")
    p.add_argument("--sam2_cfg", default="sam2_hiera_t.yaml")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── find a dataset with enough images ──
    if args.dataset:
        ds_name = args.dataset
    else:
        for name in ["cdd", "levir_cd", "s2looking", "second", "levir_cd_plus"]:
            d = os.path.join(args.data, name, "train", "A")
            if os.path.isdir(d) and len(os.listdir(d)) >= args.n:
                ds_name = name
                break
        else:
            raise RuntimeError(f"No dataset with >= {args.n} train images found in {args.data}/")

    print(f"Overfit test: {args.n} images from '{ds_name}', {args.steps} steps, lr={args.lr}")
    print(f"  DINOv2: {args.dino}")
    print(f"  SAM2:   {args.sam2_ckpt} ({args.sam2_cfg})")
    print(f"  Device: {device}\n")

    # ── tiny dataset (no augmentation — we WANT to memorise) ──
    split_dir = os.path.join(args.data, ds_name, "train")
    tf = ValTransform(img_size=args.img_size)
    full_ds = CDDataset(split_dir, transform=tf, name=ds_name)
    subset = Subset(full_ds, list(range(min(args.n, len(full_ds)))))
    loader = DataLoader(subset, batch_size=min(args.n, 4), shuffle=True, num_workers=0)

    # preload all batches into memory for speed
    batches = [(r.to(device), t.to(device), m.to(device)) for r, t, m in loader]
    print(f"  loaded {len(subset)} images in {len(batches)} batch(es)\n")

    # ── model ──
    print("Building model …")
    model = ChangeDetector(
        dino_model_name=args.dino,
        sam2_checkpoint=args.sam2_ckpt,
        sam2_config=args.sam2_cfg,
    ).to(device)

    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"  trainable params: {n_train:,}\n")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0)
    model.train()

    # ── overfit loop ──
    print(f"{'step':>5s}  {'loss':>7s}  {'bce':>7s}  {'dice':>7s}  {'latent':>7s}  {'F1':>6s}  {'IoU':>6s}")
    print("-" * 60)

    t0 = time.time()
    batch_idx = 0

    for step in range(1, args.steps + 1):
        ref, tgt, mask = batches[batch_idx % len(batches)]
        batch_idx += 1

        ph = ref.shape[2] // model.patch_size
        pw = ref.shape[3] // model.patch_size

        masks_out, iou_pred, cmap = model(ref, tgt)
        loss, metrics = compute_loss(masks_out, iou_pred, cmap, mask, ph, pw)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.steps:
            with torch.no_grad():
                m = compute_metrics(masks_out, mask)
            print(
                f"{step:5d}  {loss.item():7.4f}  {metrics['bce']:7.4f}  "
                f"{metrics['dice']:7.4f}  {metrics['latent']:7.4f}  "
                f"{m['f1']:6.4f}  {m['iou']:6.4f}"
            )

    elapsed = time.time() - t0
    print(f"\n{args.steps} steps in {elapsed:.1f}s ({elapsed/args.steps:.2f}s/step)")

    # ── final eval on the same data ──
    print("\nFinal evaluation on overfit set:")
    model.eval()
    with torch.no_grad():
        for ref, tgt, mask in batches:
            masks_out, iou_pred, cmap = model(ref, tgt)
            m = compute_metrics(masks_out, mask)
            loss, metrics = compute_loss(masks_out, iou_pred, cmap, mask,
                                         ref.shape[2] // model.patch_size,
                                         ref.shape[3] // model.patch_size)

    print(f"  loss={loss.item():.4f}  F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}")

    if m["f1"] > 0.8:
        print("\n  PASS — model can overfit. Architecture looks correct.")
    elif m["f1"] > 0.5:
        print("\n  PARTIAL — learning but slowly. Try more steps or higher lr.")
    else:
        print("\n  FAIL — F1 still low. Check gradient flow / architecture.")


if __name__ == "__main__":
    main()
