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


def _lovasz_grad(gt_sorted):
    """Compute the Lovász gradient for sorted ground-truth labels."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Lovász hinge for binary classification, flat tensors. Maximises IoU."""
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm.detach()]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad.detach())


def lovasz_hinge(logits, labels):
    """Per-sample Lovász hinge then mean. logits, labels: [B, *]."""
    losses = []
    for log, lab in zip(logits.flatten(1), labels.flatten(1)):
        losses.append(lovasz_hinge_flat(log, lab))
    return torch.stack(losses).mean()


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
    latent_soft=False,
    aux_change_maps=None, w_aux_latent=0.0,
    latent_loss_type: str = "bce",
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

    # Latent change map loss with temperature scaling.
    # When latent_soft=True, the per-patch ground-truth is the *fractional*
    # change density in [0, 1] (no thresholding), preserving boundary
    # ambiguity as a soft signal. BCE handles soft targets natively.
    gt_small = F.adaptive_avg_pool2d(gt_mask, (patch_h, patch_w))
    gt_small = gt_small.view(gt_mask.size(0), -1)            # [B, S], in [0, 1]
    if not latent_soft:
        gt_small = (gt_small > 0.3).float()                  # legacy hard target
    with torch.amp.autocast("cuda", enabled=False):
        # change_map is in [0, 1] (it's `1 - similarity` after softmax averaging).
        # We scale by 1/temperature and pass through sigmoid for BCE/MSE.
        # For Lovász we treat the scaled value as a logit-style score and need
        # binary labels.
        scaled_logit = change_map.float() / latent_temperature
        scaled_map = scaled_logit.sigmoid().clamp(1e-6, 1 - 1e-6)
        if latent_loss_type == "lovasz":
            # Lovász requires binary GT — threshold at 0.5 of the soft target.
            gt_bin = (gt_small > 0.5).float() if latent_soft else gt_small
            # Centered logit in [-1, 1] for hinge; (scaled_map - 0.5) * 2 ∈ (-1, 1).
            centered = (scaled_map - 0.5) * 2.0
            loss_latent = lovasz_hinge(centered, gt_bin)
        elif latent_loss_type == "mse":
            loss_latent = F.mse_loss(scaled_map, gt_small)
        else:  # 'bce' default
            loss_latent = F.binary_cross_entropy(scaled_map, gt_small)

    # Multi-scale auxiliary latent loss (deep supervision).
    # aux_change_maps: list of [B, 1, H_i, W_i] logit maps from coarsest → finest.
    # Weights decay 1.0, 0.5, 0.25, ... so finest scale dominates.
    loss_aux = torch.tensor(0.0, device=masks.device)
    if aux_change_maps is not None and len(aux_change_maps) > 0 and w_aux_latent > 0:
        with torch.amp.autocast("cuda", enabled=False):
            for k, aux_logits in enumerate(aux_change_maps):
                Hk, Wk = aux_logits.shape[-2:]
                tgt_k = F.adaptive_avg_pool2d(gt_mask.float(), (Hk, Wk))
                if not latent_soft:
                    tgt_k = (tgt_k > 0.3).float()
                # logit -> prob, clamped, then BCE on (prob, tgt) since tgt is in [0,1].
                prob = aux_logits.float().sigmoid().clamp(1e-6, 1 - 1e-6)
                wk = 0.5 ** k        # 1.0, 0.5, 0.25, 0.125, ...
                loss_aux = loss_aux + wk * F.binary_cross_entropy(prob, tgt_k)

    total = (w_bce * loss_bce + w_dice * loss_dice + w_latent * loss_latent
             + w_iou * loss_iou + w_aux_latent * loss_aux)
    return total, dict(
        loss=total.item(), bce=loss_bce.item(), dice=loss_dice.item(),
        iou_loss=loss_iou.item(), latent=loss_latent.item(),
        aux_latent=float(loss_aux.item()) if aux_change_maps else 0.0,
    )


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, log_every=50,
                    latent_temperature=0.03, use_ohem=True, ohem_ratio=0.7,
                    latent_soft=False, w_aux_latent=0.0, latent_loss_type="bce",
                    ema=None):
    model.train()
    running = {}
    tp = fp = fn = 0
    t0 = time.time()

    for step, (ref, tgt, mask) in enumerate(loader):
        ref, tgt, mask = ref.to(device), tgt.to(device), mask.to(device)
        ph = ref.shape[2] // model.patch_size
        pw = ref.shape[3] // model.patch_size

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            masks, iou_pred, cmap, aux_maps = model(ref, tgt)
            loss, metrics = compute_loss(
                masks, iou_pred, cmap, mask, ph, pw,
                latent_temperature=latent_temperature,
                use_ohem=use_ohem, ohem_ratio=ohem_ratio,
                latent_soft=latent_soft,
                aux_change_maps=aux_maps, w_aux_latent=w_aux_latent,
                latent_loss_type=latent_loss_type,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # EMA update after optimiser step
        if ema is not None:
            ema.update(model)

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
def evaluate(model, loader, device, scaler, latent_temperature=0.03, use_tta=False,
             threshold: float = 0.5, latent_soft: bool = False):
    model.eval()
    running = {}
    tp = fp = fn = tn = 0

    for ref, tgt, mask in loader:
        ref, tgt, mask = ref.to(device), tgt.to(device), mask.to(device)
        ph = ref.shape[2] // model.patch_size
        pw = ref.shape[3] // model.patch_size

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            if use_tta:
                masks, iou_pred, cmap, _ = model.forward_tta(ref, tgt)
                # For loss computation with TTA, use None change_map
                # We skip latent loss during TTA eval
                cmap_for_loss = torch.zeros(ref.size(0), ph * pw, device=device)
            else:
                masks, iou_pred, cmap, aux_maps = model(ref, tgt)
                cmap_for_loss = cmap

            _, metrics = compute_loss(
                masks, iou_pred, cmap_for_loss, mask, ph, pw,
                latent_temperature=latent_temperature,
                latent_soft=latent_soft,
            )

        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + v

        pred = F.interpolate(masks, mask.shape[-2:], mode="bilinear", align_corners=False)
        pred_bin = (pred.sigmoid() > threshold).float()
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


@torch.no_grad()
def _search_threshold(model, val_loader, device, scaler, use_tta=False,
                      latent_temperature=0.03,
                      thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)):
    """Run once over val, accumulate soft predictions + GT, pick threshold with max F1."""
    model.eval()
    all_preds = []
    all_gts = []
    for ref, tgt, mask in val_loader:
        ref, tgt, mask = ref.to(device), tgt.to(device), mask.to(device)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            if use_tta:
                masks, _, _, _ = model.forward_tta(ref, tgt)
            else:
                masks, _, _, _ = model(ref, tgt)
        pred = F.interpolate(masks, mask.shape[-2:], mode="bilinear", align_corners=False)
        all_preds.append(pred.sigmoid().cpu())
        all_gts.append(mask.cpu())
    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)

    best_t = 0.5
    best_f1 = -1.0
    eps = 1e-8
    for t in thresholds:
        b = (preds > t).float()
        tp = (b * gts).sum().item()
        fp = (b * (1 - gts)).sum().item()
        fn = ((1 - b) * gts).sum().item()
        prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        print(f"    threshold={t:.2f}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    return best_t


def print_metrics(prefix, m):
    print(
        f"  {prefix:<6s} loss={m['loss']:.4f}  F1={m['f1']:.4f}  "
        f"IoU={m['iou']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"
    )


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of ONLY the trainable parameters (cross_attn + bridge,
    plus sam_decoder if fine-tuned) in full fp32 precision, updated at every step
    with `shadow = decay * shadow + (1 - decay) * param`. The frozen DINO encoder
    is kept on the original model (no point EMAing frozen weights).

    Used for val + test evaluation via `apply_to(model)` / `restore(model)`
    context-style swaps of the trainable tensors.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().float()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                s = self.shadow[name]
                s.mul_(d).add_(p.detach().float(), alpha=(1 - d))

    @torch.no_grad()
    def apply_to(self, model):
        """Swap the EMA weights into the model; return a backup for restore()."""
        backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.dtype).to(p.device))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.clone().float()


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
    p.add_argument("--latent_soft", action="store_true",
                   help="Use soft (fractional) per-patch GT density in [0,1] for "
                        "the latent change-map BCE, instead of binary >0.3 threshold. "
                        "Preserves boundary ambiguity. Expected +0.2-0.4 F1.")
    p.add_argument("--w_aux_latent", type=float, default=0.0,
                   help="Weight on multi-scale auxiliary latent loss (deep supervision "
                        "from bridge intermediates at 64/128/256). 0 disables. "
                        "Try 0.2 first. Expected +0.3-0.5 F1.")
    # Tier-1 latent-space ablations
    p.add_argument("--bidir_attn", action="store_true",
                   help="Use bidirectional cross-attention (avg of ref→tgt and tgt→ref). "
                        "Enforces symmetry of binary CD. Expected +0.1-0.3 F1.")
    p.add_argument("--local_window", type=int, default=1,
                   help="Local-window similarity in CrossChangeAttention. window=1 (default) "
                        "is diagonal-only (current behaviour). window=3 takes max over a "
                        "3x3 spatial window — robust to small registration shifts. "
                        "Expected +0.3-0.5 F1, esp on S2Looking.")
    p.add_argument("--latent_loss", default="bce", choices=["bce", "mse", "lovasz"],
                   help="Loss type for the patch-level latent change map. 'bce' (default) "
                        "is the legacy binary cross-entropy. 'mse' is patch-density "
                        "regression. 'lovasz' directly optimises IoU.")
    # LoRA — Low-Rank Adaptation
    p.add_argument("--lora_rank", type=int, default=0,
                   help="LoRA rank. 0 = disabled. Try 8 or 16 (TTP uses 8 on SAM ViT-H).")
    p.add_argument("--lora_target", default="none",
                   choices=["none", "dino", "sam", "both"],
                   help="Where to inject LoRA. 'dino' = encoder Q/K/V/O attention; "
                        "'sam' = SAM decoder attention projections; 'both'.")
    p.add_argument("--lora_alpha", type=float, default=16.0,
                   help="LoRA alpha (scaling). Effective scaling = alpha / rank.")
    p.add_argument("--no_ohem", action="store_true", help="Disable OHEM for BCE loss")
    p.add_argument("--ohem_ratio", type=float, default=0.7,
                   help="OHEM: fraction of hardest pixels to keep (default: 0.7)")
    # Eval
    p.add_argument("--tta", action="store_true", help="Use test-time augmentation for final test")
    # Output
    p.add_argument("--save_dir", default="runs")
    p.add_argument("--resume", default=None)
    # Per-dataset YAML config (overrides img_size / batch / epochs / patience / lr /
    # warmup / finetune_decoder / tta / augmentations / normalization / dataset selection).
    p.add_argument("--config", default=None,
                   help="Path to a dataset YAML (configs/datasets/<name>.yaml). "
                        "When set, reads augmentations + training hparams from the YAML.")
    # Cheap-wins flags
    p.add_argument("--ema", type=float, default=None,
                   help="EMA decay for a shadow model (e.g. 0.999). "
                        "When set, maintains an exponential-moving-average copy of trainable params, "
                        "evaluates/saves from it. Add ~+0.3-0.5 F1 typically.")
    p.add_argument("--search_threshold", action="store_true",
                   help="At final test eval, sweep threshold in [0.30, 0.70] on val and "
                        "apply the best to test. Typically +0.2-0.4 F1 on LEVIR/CDD.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility (sets torch, numpy, random).")
    args = p.parse_args()

    # ── apply YAML config BEFORE anything downstream uses the args ──
    cfg = None
    if args.config:
        from configurable_aug import load_config
        cfg = load_config(args.config)
        print(f"[config] loaded {args.config}")
        # dataset selection
        if cfg.get("dataset"):
            args.datasets = [cfg["dataset"]]
        # sizes + hparams
        if "img_size" in cfg: args.img_size = int(cfg["img_size"])
        tr = cfg.get("training") or {}
        for k_yaml, k_arg in [
            ("batch", "batch"), ("epochs", "epochs"), ("patience", "patience"),
            ("lr", "lr"), ("warmup", "warmup"),
            ("decoder_lr_scale", "decoder_lr_scale"),
        ]:
            if k_yaml in tr:
                setattr(args, k_arg, tr[k_yaml])
        if tr.get("finetune_decoder") is True:
            args.finetune_decoder = True
        if tr.get("tta") is True:
            args.tta = True
        print(f"[config] img_size={args.img_size} batch={args.batch} epochs={args.epochs} "
              f"lr={args.lr} patience={args.patience} finetune={args.finetune_decoder} tta={args.tta}")

    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

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
        config=cfg,
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
        bidir_attn=args.bidir_attn,
        local_window=args.local_window,
        lora_rank=args.lora_rank,
        lora_target=args.lora_target,
        lora_alpha=args.lora_alpha,
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

    # ── EMA (optional) ──
    ema = ModelEMA(model, decay=args.ema) if args.ema else None
    if ema is not None:
        print(f"  EMA: enabled (decay={args.ema})")

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
            latent_soft=args.latent_soft,
            w_aux_latent=args.w_aux_latent,
            latent_loss_type=args.latent_loss,
            ema=ema,
        )
        scheduler.step()
        print_metrics("train", train_m)

        # --- validate ---
        row = {"epoch": epoch, "lr": lr_now}
        for k, v in train_m.items():
            row[f"train_{k}"] = round(v, 6)

        is_best = False
        if val_loader is not None:
            # Evaluate using EMA weights (if enabled) — model selection should track EMA
            ema_backup = ema.apply_to(model) if ema is not None else None
            val_m = evaluate(
                model, val_loader, device, scaler,
                latent_temperature=args.latent_temp,
                latent_soft=args.latent_soft,
            )
            if ema is not None:
                ema.restore(model, ema_backup)
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
        # When EMA is on, SAVE the EMA weights (that's what we val-evaluate on and
        # what we want to test with). We briefly swap in EMA, copy state, then restore.
        if ema is not None:
            ema_backup = ema.apply_to(model)

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
        if ema is not None:
            ckpt["ema_shadow"] = ema.state_dict()
        # LoRA: save only the trainable lora_A / lora_B params (small).
        if args.lora_rank > 0:
            ckpt["lora_state"] = {
                n: p.detach().cpu().clone()
                for n, p in model.named_parameters()
                if p.requires_grad and ("lora_A" in n or "lora_B" in n)
            }
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            print(f"  ** new best F1={best_f1:.4f} saved **")

        if ema is not None:
            ema.restore(model, ema_backup)

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

        # --- Optional: sweep threshold on val, apply to test (Tier-1 cheap win) ---
        best_threshold = 0.5
        if args.search_threshold and val_loader is not None:
            print("\n  Searching best threshold on val …")
            best_threshold = _search_threshold(
                model, val_loader, device, scaler,
                use_tta=args.tta, latent_temperature=args.latent_temp,
            )
            print(f"  picked threshold = {best_threshold:.2f}")

        test_m = evaluate(
            model, test_loader, device, scaler,
            latent_temperature=args.latent_temp,
            use_tta=args.tta,
            threshold=best_threshold,
            latent_soft=args.latent_soft,
        )

        print(f"\n  {'Metric':<12s}  Value")
        print(f"  {'-'*25}")
        for k in ["f1", "iou", "precision", "recall", "oa", "loss"]:
            if k in test_m:
                print(f"  {k:<12s}  {test_m[k]:.4f}")

        # save test results
        test_results = {k: round(v, 6) for k, v in test_m.items()}
        test_results["tta"] = args.tta
        test_results["threshold"] = best_threshold
        test_results["ema_decay"] = args.ema
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
