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
from torch.amp import autocast
from torch.utils.data import DataLoader

from model import ChangeDetector, ENCODERS, DECODERS, SAM2_VARIANTS
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
                masks, iou_pred, _, _ = model.forward_tta(ref, tgt)
            else:
                masks, iou_pred, _, _ = model(ref, tgt)

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
# Threshold search on val
# ---------------------------------------------------------------------------
@torch.no_grad()
def search_threshold_on_val(model, val_dir, img_size, batch_size, num_workers,
                            device, use_tta=False,
                            thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)):
    tf = ValTransform(img_size=img_size)
    ds = CDDataset(val_dir, transform=tf, name="val")
    if len(ds) == 0:
        return 0.5
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds, gts = [], []
    for ref, tgt, gt in loader:
        ref, tgt, gt = ref.to(device), tgt.to(device), gt.to(device)
        with autocast(device_type="cuda", enabled=True):
            if use_tta:
                masks, _, _, _ = model.forward_tta(ref, tgt)
            else:
                masks, _, _, _ = model(ref, tgt)
        pred = F.interpolate(masks, gt.shape[-2:], mode="bilinear", align_corners=False)
        preds.append(pred.sigmoid().cpu())
        gts.append(gt.cpu())
    P = torch.cat(preds); G = torch.cat(gts)
    best_t, best_f1, eps = 0.5, -1.0, 1e-8
    for t in thresholds:
        b = (P > t).float()
        tp = (b * G).sum().item(); fp = (b * (1 - G)).sum().item()
        fn = ((1 - b) * G).sum().item()
        prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--encoder", default="dinov2_rs_base", help="Encoder name or HuggingFace ID")
    p.add_argument("--decoder", default="sam2_base_plus", help="Decoder name from registry")
    p.add_argument("--ckpt_dir", default="checkpoints", help="Directory containing decoder checkpoints")
    p.add_argument("--finetune_decoder", action="store_true",
                   help="Load model with finetune_decoder=True (for loading finetuned decoder weights)")
    p.add_argument("--checkpoint", required=True, help="best.pt from training")
    # backward compat
    p.add_argument("--dino", default=None, help="(deprecated) Use --encoder")
    p.add_argument("--sam2_variant", default=None, help="(deprecated) Use --decoder")
    p.add_argument("--sam2_ckpt", default=None, help="(deprecated)")
    p.add_argument("--sam2_cfg", default=None, help="(deprecated)")
    p.add_argument("--sam2_ckpt_dir", default=None, help="(deprecated)")
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
    # Tier-1 post-hoc cheap wins:
    p.add_argument("--search_threshold", action="store_true",
                   help="Sweep threshold on val split, apply best to test split.")
    p.add_argument("--val_split", default="val",
                   help="Which split to use for threshold search (default: val).")
    p.add_argument("--test_img_sizes", nargs="+", type=int, default=None,
                   help="List of test resolutions to evaluate (e.g. 256 384 512). "
                        "Picks the best via val; falls back to --img_size if not set.")
    p.add_argument("--use_ema", action="store_true",
                   help="If checkpoint has ema_shadow, load EMA weights.")
    # output
    p.add_argument("--save_masks", default=None, help="directory to save predicted masks")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── resolve backward compat ──
    ckpt_dir = args.sam2_ckpt_dir or args.ckpt_dir
    encoder_name = args.dino or args.encoder
    decoder_name = args.decoder
    if args.sam2_variant:
        decoder_name = f"sam2_{args.sam2_variant}"

    # ── model ──
    print("Loading model …")
    # Tier-1 latent-space + LoRA args may be embedded in the saved ckpt['args']
    # — auto-pick them up so eval matches training architecture.
    bidir_attn = False
    local_window = 1
    lora_rank = 0
    lora_target = "none"
    lora_alpha = 16.0
    saved_ckpt = (torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                  if os.path.isfile(args.checkpoint) else {})
    saved_args = saved_ckpt.get('args', {}) if isinstance(saved_ckpt, dict) else {}
    if isinstance(saved_args, dict):
        bidir_attn = bool(saved_args.get('bidir_attn', False))
        local_window = int(saved_args.get('local_window', 1))
        lora_rank = int(saved_args.get('lora_rank', 0))
        lora_target = saved_args.get('lora_target', 'none')
        lora_alpha = float(saved_args.get('lora_alpha', 16.0))

    model = ChangeDetector(
        encoder=encoder_name,
        decoder=decoder_name,
        ckpt_dir=ckpt_dir,
        finetune_decoder=args.finetune_decoder,
        bidir_attn=bidir_attn,
        local_window=local_window,
        lora_rank=lora_rank,
        lora_target=lora_target,
        lora_alpha=lora_alpha,
        sam2_checkpoint=args.sam2_ckpt,
        sam2_config=args.sam2_cfg,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.bridge.load_state_dict(ckpt["bridge"], strict=False)
    model.cross_attn.load_state_dict(ckpt["cross_attn"])
    if args.finetune_decoder and "sam_decoder" in ckpt:
        model.sam_decoder.load_state_dict(ckpt["sam_decoder"])
    # Restore LoRA params (lora_A / lora_B) into their host modules.
    if "lora_state" in ckpt:
        named = dict(model.named_parameters())
        n_loaded = 0
        for n, t in ckpt["lora_state"].items():
            if n in named and named[n].shape == t.shape:
                named[n].data.copy_(t.to(named[n].device))
                n_loaded += 1
        print(f"  loaded LoRA params: {n_loaded}/{len(ckpt['lora_state'])}")
    # Optional EMA swap-in
    if args.use_ema and "ema_shadow" in ckpt:
        print("  loading EMA shadow weights")
        for name, p in model.named_parameters():
            if p.requires_grad and name in ckpt["ema_shadow"]:
                p.data.copy_(ckpt["ema_shadow"][name].to(p.dtype).to(p.device))
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

        # Determine image-size sweep for this dataset
        sizes = args.test_img_sizes if args.test_img_sizes else [args.img_size]
        val_dir = os.path.join(args.data, ds_name, args.val_split)
        has_val_for_search = (args.search_threshold or len(sizes) > 1) and \
            os.path.isdir(os.path.join(val_dir, "A"))

        # Pick best (size, threshold) by max val F1
        best_cfg = {"size": sizes[0], "threshold": args.threshold, "val_f1": None}
        if has_val_for_search:
            for sz in sizes:
                t = args.threshold
                if args.search_threshold:
                    t = search_threshold_on_val(
                        model, val_dir, sz, args.batch, args.workers,
                        device, use_tta=args.tta,
                    )
                val_res = evaluate_dataset(
                    model, ds_name, val_dir, sz, args.batch, args.workers,
                    device, t, None, use_tta=args.tta,
                )
                if val_res is None:
                    continue
                print(f"  val@size={sz} threshold={t:.2f}  F1={val_res['f1']:.4f}")
                if best_cfg["val_f1"] is None or val_res["f1"] > best_cfg["val_f1"]:
                    best_cfg = {"size": sz, "threshold": t, "val_f1": val_res["f1"]}
            print(f"  picked: size={best_cfg['size']} threshold={best_cfg['threshold']:.2f}  "
                  f"(val_f1={best_cfg['val_f1']:.4f})")

        result = evaluate_dataset(
            model, ds_name, split_dir, best_cfg["size"], args.batch, args.workers,
            device, best_cfg["threshold"], args.save_masks, use_tta=args.tta,
        )
        if result is None:
            print(f"  {ds_name}: 0 matched pairs, skipping")
            continue
        result["picked_size"] = best_cfg["size"]
        result["picked_threshold"] = best_cfg["threshold"]

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
