"""
Training script for the Change Detection pipeline v2.

Only the Bridge + CrossChangeAttention are optimised (+ optionally SAM decoder).
Losses: BCE + Dice on SAM2 mask output  +  temperature-scaled latent change map.
        Optional OHEM (online hard example mining) for mask losses.

Pipeline:
    1. Train on all available datasets (auto-splits val from train if missing)
    2. Select best model by val F1
    3. Final evaluation on held-out test set (with optional TTA)
    4. Save checkpoints + CSV log

Usage:
    python train.py \
        --dino KevinCha/dinov2-vit-base-remote-sensing \
        --sam2_variant base_plus \
        --sam2_ckpt_dir checkpoints \
        --finetune_decoder \
        --batch 4 --workers 4 --epochs 50

    # Or with explicit checkpoint/config (backward compatible):
    python train.py \
        --dino KevinCha/dinov2-vit-base-remote-sensing \
        --sam2_ckpt checkpoints/sam2.1_hiera_base_plus.pt \
        --sam2_cfg sam2.1/sam2.1_hiera_b+.yaml \
        --batch 4 --workers 4 --epochs 50
"""

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from model import ChangeDetector, ENCODERS, DECODERS, SAM2_VARIANTS, list_encoders, list_decoders
from dataset import build_loaders


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    prob = logits.sigmoid().flatten(1)
    tgt = target.flatten(1)
    inter = (prob * tgt).sum(dim=1)
    union = prob.sum(dim=1) + tgt.sum(dim=1)
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


def ohem_bce_loss(logits: torch.Tensor, target: torch.Tensor, top_k_ratio: float = 0.7):
    """Online hard example mining: only backprop through hardest pixels."""
    loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    # Flatten per-sample, sort, keep top-k
    B = loss_map.size(0)
    loss_flat = loss_map.view(B, -1)
    k = max(1, int(loss_flat.size(1) * top_k_ratio))
    topk_loss, _ = loss_flat.topk(k, dim=1)
    return topk_loss.mean()


def compute_loss(
    masks, iou_pred, change_map, gt_mask, patch_h, patch_w,
    w_bce=1.0, w_dice=1.0, w_latent=0.5, w_iou=0.5,
    latent_temperature=0.03, use_ohem=True, ohem_ratio=0.7,
):
    pred = F.interpolate(masks, gt_mask.shape[-2:], mode="bilinear", align_corners=False)

    if use_ohem:
        loss_bce = ohem_bce_loss(pred, gt_mask, top_k_ratio=ohem_ratio)
    else:
        loss_bce = F.binary_cross_entropy_with_logits(pred, gt_mask)
    loss_dice = dice_loss(pred, gt_mask)

    with torch.no_grad():
        pred_bin = (pred.sigmoid() > 0.5).float()
        inter = (pred_bin * gt_mask).flatten(1).sum(1)
        union = (pred_bin + gt_mask).clamp(0, 1).flatten(1).sum(1)
        gt_iou = (inter / (union + 1e-6)).unsqueeze(1)
    loss_iou = F.mse_loss(iou_pred, gt_iou)

    # Latent change map loss with temperature scaling
    gt_small = F.adaptive_avg_pool2d(gt_mask, (patch_h, patch_w))
    gt_small = (gt_small.view(gt_mask.size(0), -1) > 0.3).float()
    with torch.amp.autocast("cuda", enabled=False):
        # Scale change_map logits by 1/temperature before BCE
        scaled_map = (change_map.float() / latent_temperature).sigmoid()
        scaled_map = scaled_map.clamp(1e-6, 1 - 1e-6)
        loss_latent = F.binary_cross_entropy(scaled_map, gt_small)

    total = w_bce * loss_bce + w_dice * loss_dice + w_latent * loss_latent + w_iou * loss_iou
    return total, dict(
        loss=total.item(), bce=loss_bce.item(), dice=loss_dice.item(),
        iou_loss=loss_iou.item(), latent=loss_latent.item(),
    )


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, log_every=50,
                    latent_temperature=0.03, use_ohem=True, ohem_ratio=0.7):
    model.train()
    running = {}
    tp = fp = fn = 0
    t0 = time.time()

    for step, (ref, tgt, mask) in enumerate(loader):
        ref, tgt, mask = ref.to(device), tgt.to(device), mask.to(device)
        ph = ref.shape[2] // model.patch_size
        pw = ref.shape[3] // model.patch_size

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            masks, iou_pred, cmap = model(ref, tgt)
            loss, metrics = compute_loss(
                masks, iou_pred, cmap, mask, ph, pw,
                latent_temperature=latent_temperature,
                use_ohem=use_ohem, ohem_ratio=ohem_ratio,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + v

        with torch.no_grad():
            pred = F.interpolate(masks, mask.shape[-2:], mode="bilinear", align_corners=False)
            pred_bin = (pred.sigmoid() > 0.5).float()
            tp += (pred_bin * mask).sum().item()
            fp += (pred_bin * (1 - mask)).sum().item()
            fn += ((1 - pred_bin) * mask).sum().item()

        if (step + 1) % log_every == 0:
            avg = {k: v / (step + 1) for k, v in running.items()}
            elapsed = time.time() - t0
            print(
                f"  [epoch {epoch}  step {step+1}/{len(loader)}]  "
                f"loss={avg['loss']:.4f}  bce={avg['bce']:.4f}  "
                f"dice={avg['dice']:.4f}  latent={avg['latent']:.4f}  "
                f"({elapsed:.0f}s)"
            )

    n = max(len(loader), 1)
    avg = {k: v / n for k, v in running.items()}
    eps = 1e-8
    avg["precision"] = tp / (tp + fp + eps)
    avg["recall"] = tp / (tp + fn + eps)
    avg["f1"] = 2 * avg["precision"] * avg["recall"] / (avg["precision"] + avg["recall"] + eps)
    avg["iou"] = tp / (tp + fp + fn + eps)
    avg["time"] = time.time() - t0
    return avg


@torch.no_grad()
def evaluate(model, loader, device, scaler, latent_temperature=0.03, use_tta=False):
    model.eval()
    running = {}
    tp = fp = fn = tn = 0

    for ref, tgt, mask in loader:
        ref, tgt, mask = ref.to(device), tgt.to(device), mask.to(device)
        ph = ref.shape[2] // model.patch_size
        pw = ref.shape[3] // model.patch_size

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            if use_tta:
                masks, iou_pred, cmap = model.forward_tta(ref, tgt)
                # For loss computation with TTA, use None change_map
                # We skip latent loss during TTA eval
                cmap_for_loss = torch.zeros(ref.size(0), ph * pw, device=device)
            else:
                masks, iou_pred, cmap = model(ref, tgt)
                cmap_for_loss = cmap

            _, metrics = compute_loss(
                masks, iou_pred, cmap_for_loss, mask, ph, pw,
                latent_temperature=latent_temperature,
            )

        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + v

        pred = F.interpolate(masks, mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_bin = (pred.sigmoid() > 0.5).float()
        tp += (pred_bin * mask).sum().item()
        fp += (pred_bin * (1 - mask)).sum().item()
        fn += ((1 - pred_bin) * mask).sum().item()
        tn += ((1 - pred_bin) * (1 - mask)).sum().item()

    n = max(len(loader), 1)
    avg = {k: v / n for k, v in running.items()}
    eps = 1e-8
    avg["precision"] = tp / (tp + fp + eps)
    avg["recall"] = tp / (tp + fn + eps)
    avg["f1"] = 2 * avg["precision"] * avg["recall"] / (avg["precision"] + avg["recall"] + eps)
    avg["iou"] = tp / (tp + fp + fn + eps)
    avg["oa"] = (tp + tn) / (tp + tn + fp + fn + eps)
    return avg


def print_metrics(prefix, m):
    print(
        f"  {prefix:<6s} loss={m['loss']:.4f}  F1={m['f1']:.4f}  "
        f"IoU={m['iou']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"
    )


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------
class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.fieldnames = None

    def log(self, row: dict):
        self.rows.append(row)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    # Model — new unified interface
    p.add_argument("--encoder", default="dinov2_rs_base", help="Encoder name from registry or HuggingFace ID")
    p.add_argument("--decoder", default="sam2_base_plus", help="Decoder name from registry")
    p.add_argument("--ckpt_dir", default="checkpoints", help="Directory containing decoder checkpoints")
    p.add_argument("--finetune_decoder", action="store_true",
                   help="Fine-tune decoder with lower LR")
    p.add_argument("--decoder_lr_scale", type=float, default=0.1,
                   help="LR multiplier for decoder fine-tuning (default: 0.1)")
    p.add_argument("--list_models", action="store_true", help="List all available encoders/decoders and exit")
    # Backward compat (still work if provided)
    p.add_argument("--dino", default=None, help="(deprecated) Use --encoder instead")
    p.add_argument("--sam2_variant", default=None, help="(deprecated) Use --decoder instead")
    p.add_argument("--sam2_ckpt", default=None, help="(deprecated) Explicit SAM2 checkpoint path")
    p.add_argument("--sam2_cfg", default=None, help="(deprecated) Explicit SAM2 config")
    p.add_argument("--sam2_ckpt_dir", default=None, help="(deprecated) Use --ckpt_dir instead")
    # Data
    p.add_argument("--data", default="data")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience: stop after N epochs without val F1 improvement")
    # Loss
    p.add_argument("--latent_temp", type=float, default=0.03,
                   help="Temperature for latent change map loss (default: 0.03)")
    p.add_argument("--w_latent", type=float, default=0.2,
                   help="Weight for latent loss (default: 0.2, was 0.5)")
    p.add_argument("--no_ohem", action="store_true", help="Disable OHEM for BCE loss")
    p.add_argument("--ohem_ratio", type=float, default=0.7,
                   help="OHEM: fraction of hardest pixels to keep (default: 0.7)")
    # Eval
    p.add_argument("--tta", action="store_true", help="Use test-time augmentation for final test")
    # Output
    p.add_argument("--save_dir", default="runs")
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    use_amp = not args.no_amp

    # ── list models and exit ──
    if args.list_models:
        print("\n=== ENCODERS ===")
        list_encoders()
        print("\n=== DECODERS ===")
        list_decoders()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── resolve backward-compatible args ──
    ckpt_dir = args.sam2_ckpt_dir or args.ckpt_dir
    encoder_name = args.dino or args.encoder
    sam2_ckpt = args.sam2_ckpt
    sam2_cfg = args.sam2_cfg

    # Map old --sam2_variant to new --decoder
    decoder_name = args.decoder
    if args.sam2_variant is not None:
        variant_to_decoder = {"tiny": "sam2_tiny", "small": "sam2_small",
                              "base_plus": "sam2_base_plus", "large": "sam2_large"}
        decoder_name = variant_to_decoder[args.sam2_variant]

    print(f"Encoder: {encoder_name}")
    print(f"Decoder: {decoder_name}")
    if args.finetune_decoder:
        print(f"Decoder fine-tuning: ON (lr_scale={args.decoder_lr_scale})")
    print()

    # ── run directory with timestamp ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # save config
    full_config = vars(args).copy()
    full_config["encoder_resolved"] = encoder_name
    full_config["decoder_resolved"] = decoder_name
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)

    logger = CSVLogger(os.path.join(run_dir, "log.csv"))
    print(f"Run directory: {run_dir}\n")

    # ── data ──
    print("Building dataloaders …")
    loaders = build_loaders(
        data_root=args.data,
        datasets=args.datasets,
        img_size=args.img_size,
        batch_size=args.batch,
        num_workers=args.workers,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    test_loader = loaders.get("test")

    if val_loader is None:
        print("\n⚠ No val data — will select best model by train loss\n")

    # ── model ──
    print("\nBuilding model …")
    model = ChangeDetector(
        encoder=encoder_name,
        decoder=decoder_name,
        ckpt_dir=ckpt_dir,
        finetune_decoder=args.finetune_decoder,
        decoder_lr_scale=args.decoder_lr_scale,
        sam2_checkpoint=sam2_ckpt,
        sam2_config=sam2_cfg,
    ).to(device)

    n_train_p = sum(p.numel() for p in model.trainable_parameters())
    n_all_p = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_train_p/1e6:.1f}M / {n_all_p/1e6:.1f}M total\n")

    # ── optimiser with differential LR ──
    param_groups = model.param_groups(args.lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    start_epoch = 0
    best_f1 = 0.0

    # ── resume ──
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.cross_attn.load_state_dict(ckpt["cross_attn"])
        model.bridge.load_state_dict(ckpt["bridge"], strict=False)
        if args.finetune_decoder and "sam_decoder" in ckpt:
            model.sam_decoder.load_state_dict(ckpt["sam_decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_f1={best_f1:.4f}\n")

    # ══════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════
    print(f"Training for {args.epochs} epochs …")
    print(f"  latent_temp={args.latent_temp}  w_latent={args.w_latent}  "
          f"OHEM={'ON' if not args.no_ohem else 'OFF'}")
    print(f"  early stopping: patience={args.patience}")
    if args.tta:
        print(f"  TTA: enabled for final test evaluation")
    print()

    total_t0 = time.time()
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{args.epochs-1}  lr={lr_now:.2e}")
        print(f"{'='*60}")

        # --- train ---
        train_m = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            latent_temperature=args.latent_temp,
            use_ohem=not args.no_ohem,
            ohem_ratio=args.ohem_ratio,
        )
        scheduler.step()
        print_metrics("train", train_m)

        # --- validate ---
        row = {"epoch": epoch, "lr": lr_now}
        for k, v in train_m.items():
            row[f"train_{k}"] = round(v, 6)

        is_best = False
        if val_loader is not None:
            val_m = evaluate(
                model, val_loader, device, scaler,
                latent_temperature=args.latent_temp,
            )
            print_metrics("val", val_m)
            for k, v in val_m.items():
                row[f"val_{k}"] = round(v, 6)
            is_best = val_m["f1"] > best_f1
            if is_best:
                best_f1 = val_m["f1"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            is_best = train_m["f1"] > best_f1
            if is_best:
                best_f1 = train_m["f1"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        row["best_f1"] = round(best_f1, 6)
        logger.log(row)

        # --- save checkpoints ---
        ckpt = dict(
            epoch=epoch,
            cross_attn=model.cross_attn.state_dict(),
            bridge=model.bridge.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best_f1=best_f1,
            args=full_config,
        )
        if args.finetune_decoder:
            ckpt["sam_decoder"] = model.sam_decoder.state_dict()
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            print(f"  ** new best F1={best_f1:.4f} saved **")

        # --- early stopping ---
        if epochs_without_improvement >= args.patience:
            print(f"\n  Early stopping: no improvement for {args.patience} epochs (best F1={best_f1:.4f})")
            break

    total_time = time.time() - total_t0
    print(f"\nTraining complete in {total_time/60:.1f} min.  Best val F1 = {best_f1:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL TEST EVALUATION
    # ══════════════════════════════════════════════════════════════════
    if test_loader is not None:
        print(f"\n{'='*60}")
        print("Final evaluation on TEST set (using best checkpoint)")
        if args.tta:
            print("  TTA: enabled (flips + rotations)")
        print(f"{'='*60}")

        best_path = os.path.join(run_dir, "best.pt")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.cross_attn.load_state_dict(ckpt["cross_attn"])
            model.bridge.load_state_dict(ckpt["bridge"])
            if args.finetune_decoder and "sam_decoder" in ckpt:
                model.sam_decoder.load_state_dict(ckpt["sam_decoder"])
            print(f"  loaded best.pt (epoch {ckpt['epoch']}, val_f1={ckpt['best_f1']:.4f})")

        test_m = evaluate(
            model, test_loader, device, scaler,
            latent_temperature=args.latent_temp,
            use_tta=args.tta,
        )

        print(f"\n  {'Metric':<12s}  Value")
        print(f"  {'-'*25}")
        for k in ["f1", "iou", "precision", "recall", "oa", "loss"]:
            if k in test_m:
                print(f"  {k:<12s}  {test_m[k]:.4f}")

        # save test results
        test_results = {k: round(v, 6) for k, v in test_m.items()}
        test_results["tta"] = args.tta
        with open(os.path.join(run_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"\n  Results saved to {run_dir}/test_results.json")

    # ── summary ──
    print(f"\n{'='*60}")
    print(f"Run complete: {run_dir}")
    print(f"  config.json       — hyperparameters")
    print(f"  log.csv           — per-epoch train/val metrics")
    print(f"  best.pt           — best model (val F1={best_f1:.4f})")
    print(f"  last.pt           — last epoch checkpoint")
    if test_loader:
        print(f"  test_results.json — final test metrics")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
