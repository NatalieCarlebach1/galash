"""
Evaluation script for the Change Detection pipeline v2.

Computes per-dataset and aggregate metrics: Precision, Recall, F1, IoU, OA, Kappa.
Optionally saves predicted change masks. Supports TTA (test-time augmentation).

Usage:
    python eval.py \
        --dino KevinCha/dinov2-vit-base-remote-sensing \
        --sam2_variant base_plus \
        --checkpoint runs/YYYYMMDD_HHMMSS/best.pt \
        --data data \
        --split test \
        --tta \
        --save_masks predictions/
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from model import ChangeDetector, SAM2_VARIANTS
from dataset import CDDataset, ValTransform, IMG_EXTS


# ---------------------------------------------------------------------------
# Metrics accumulator
# ---------------------------------------------------------------------------
class ChangeMetrics:
    """Accumulates TP/FP/FN/TN across batches for binary change detection."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5):
        pred = F.interpolate(pred_logits, gt_mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = (pred.sigmoid() > threshold).float()
        gt = gt_mask.float()

        self.tp += (pred * gt).sum().item()
        self.fp += (pred * (1 - gt)).sum().item()
        self.fn += ((1 - pred) * gt).sum().item()
        self.tn += ((1 - pred) * (1 - gt)).sum().item()

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        oa = (tp + tn) / (tp + tn + fp + fn + eps)

        total = tp + tn + fp + fn + eps
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total ** 2)
        kappa = (oa - pe) / (1 - pe + eps)

        return dict(
            precision=precision, recall=recall, f1=f1,
            iou=iou, oa=oa, kappa=kappa,
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        )


# ---------------------------------------------------------------------------
# Evaluate one dataset split
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_dataset(model, ds_name, split_dir, img_size, batch_size, num_workers,
                     device, threshold, save_dir=None, use_tta=False):
    """Evaluate a single dataset split and optionally save predicted masks."""
    tf = ValTransform(img_size=img_size)
    ds = CDDataset(split_dir, transform=tf, name=ds_name)

    if len(ds) == 0:
        return None

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    if save_dir:
        out_dir = os.path.join(save_dir, ds_name)
        os.makedirs(out_dir, exist_ok=True)

    metrics = ChangeMetrics()
    sample_idx = 0

    for ref, tgt, gt in loader:
        ref, tgt, gt = ref.to(device), tgt.to(device), gt.to(device)

        with autocast(device_type="cuda", enabled=True):
            if use_tta:
                masks, iou_pred, _ = model.forward_tta(ref, tgt)
            else:
                masks, iou_pred, _ = model(ref, tgt)

        metrics.update(masks, gt, threshold=threshold)

        # save per-image masks
        if save_dir:
            pred = F.interpolate(masks, gt.shape[-2:], mode="bilinear", align_corners=False)
            pred_np = (pred.sigmoid() > threshold).cpu().numpy()
            for i in range(pred_np.shape[0]):
                if sample_idx < len(ds.triplets):
                    fname = os.path.basename(ds.triplets[sample_idx][0])
                    stem = os.path.splitext(fname)[0]
                    out_path = os.path.join(out_dir, f"{stem}.png")
                    from PIL import Image
                    Image.fromarray((pred_np[i, 0] * 255).astype(np.uint8)).save(out_path)
                sample_idx += 1

    return metrics.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--dino", default="KevinCha/dinov2-vit-base-remote-sensing")
    p.add_argument("--sam2_variant", default=None, choices=list(SAM2_VARIANTS.keys()),
                   help="SAM2.1 variant: tiny, small, base_plus, large")
    p.add_argument("--sam2_ckpt", default=None)
    p.add_argument("--sam2_cfg", default=None)
    p.add_argument("--sam2_ckpt_dir", default="checkpoints")
    p.add_argument("--finetune_decoder", action="store_true",
                   help="Load model with finetune_decoder=True (for loading finetuned decoder weights)")
    p.add_argument("--checkpoint", required=True, help="best.pt from training")
    # data
    p.add_argument("--data", default="data")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="which datasets to evaluate (auto-detect if omitted)")
    p.add_argument("--split", default="test")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    # TTA
    p.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    # output
    p.add_argument("--save_masks", default=None, help="directory to save predicted masks")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── resolve SAM2 checkpoint + config ──
    if args.sam2_ckpt is not None and args.sam2_cfg is not None:
        sam2_ckpt = args.sam2_ckpt
        sam2_cfg = args.sam2_cfg
    elif args.sam2_variant is not None:
        variant = SAM2_VARIANTS[args.sam2_variant]
        sam2_ckpt = os.path.join(args.sam2_ckpt_dir, variant["ckpt"])
        sam2_cfg = variant["cfg"]
    else:
        # Try to auto-detect from checkpoint config
        variant = SAM2_VARIANTS["base_plus"]
        sam2_ckpt = os.path.join(args.sam2_ckpt_dir, variant["ckpt"])
        sam2_cfg = variant["cfg"]

    # ── model ──
    print("Loading model …")
    model = ChangeDetector(
        dino_model_name=args.dino,
        sam2_checkpoint=sam2_ckpt,
        sam2_config=sam2_cfg,
        finetune_decoder=args.finetune_decoder,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.bridge.load_state_dict(ckpt["bridge"], strict=False)
    model.cross_attn.load_state_dict(ckpt["cross_attn"])
    if args.finetune_decoder and "sam_decoder" in ckpt:
        model.sam_decoder.load_state_dict(ckpt["sam_decoder"])
    epoch = ckpt.get("epoch", "?")
    best_f1 = ckpt.get("best_f1", ckpt.get("val_loss", "?"))
    print(f"  loaded epoch={epoch}  best_f1={best_f1}")
    if args.tta:
        print(f"  TTA: enabled (flips + rotations)")
    model.eval()

    # ── discover datasets ──
    if args.datasets is None:
        args.datasets = []
        for name in sorted(os.listdir(args.data)):
            split_dir = os.path.join(args.data, name, args.split, "A")
            if os.path.isdir(split_dir) and len(os.listdir(split_dir)) > 0:
                args.datasets.append(name)
    print(f"  evaluating: {args.datasets}  split={args.split}\n")

    # ── evaluate each dataset ──
    all_results = {}
    agg_metrics = ChangeMetrics()

    for ds_name in args.datasets:
        split_dir = os.path.join(args.data, ds_name, args.split)
        if not os.path.isdir(os.path.join(split_dir, "A")):
            print(f"  {ds_name}: no {args.split} split, skipping")
            continue

        print(f"Evaluating {ds_name} …")
        result = evaluate_dataset(
            model, ds_name, split_dir, args.img_size, args.batch, args.workers,
            device, args.threshold, args.save_masks, use_tta=args.tta,
        )
        if result is None:
            print(f"  {ds_name}: 0 matched pairs, skipping")
            continue

        all_results[ds_name] = result

        # accumulate into aggregate
        agg_metrics.tp += result["tp"]
        agg_metrics.fp += result["fp"]
        agg_metrics.fn += result["fn"]
        agg_metrics.tn += result["tn"]

        print(
            f"  {ds_name:<18s}  F1={result['f1']:.4f}  IoU={result['iou']:.4f}  "
            f"P={result['precision']:.4f}  R={result['recall']:.4f}  "
            f"OA={result['oa']:.4f}  K={result['kappa']:.4f}"
        )

    # ── aggregate ──
    if len(all_results) > 1:
        agg = agg_metrics.compute()
        all_results["AGGREGATE"] = agg
        print(
            f"\n  {'AGGREGATE':<18s}  F1={agg['f1']:.4f}  IoU={agg['iou']:.4f}  "
            f"P={agg['precision']:.4f}  R={agg['recall']:.4f}  "
            f"OA={agg['oa']:.4f}  K={agg['kappa']:.4f}"
        )

    # ── summary table ──
    print(f"\n{'='*70}")
    print(f"{'Dataset':<18s} {'F1':>7s} {'IoU':>7s} {'Prec':>7s} {'Rec':>7s} {'OA':>7s} {'Kappa':>7s}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        print(
            f"{name:<18s} {r['f1']:7.4f} {r['iou']:7.4f} "
            f"{r['precision']:7.4f} {r['recall']:7.4f} "
            f"{r['oa']:7.4f} {r['kappa']:7.4f}"
        )
    print(f"{'='*70}")

    if args.save_masks:
        print(f"\nPredicted masks saved to {args.save_masks}/")


if __name__ == "__main__":
    main()
