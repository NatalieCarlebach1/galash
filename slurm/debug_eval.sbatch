#!/bin/bash
#SBATCH --job-name=galash-debug
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/debug_%j.out
#SBATCH --error=logs/slurm/debug_%j.err

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

python3 << 'PY'
import torch
import sys, os
sys.path.insert(0, '.')
from model import ChangeDetector
from dataset import CDDataset, ValTransform
import torch.nn.functional as F

device = torch.device('cuda')

print('Building model (same as training config)...')
model = ChangeDetector(
    encoder='KevinCha/dinov2-vit-base-remote-sensing',
    decoder='sam2_base_plus',  # will be overridden
    sam2_checkpoint='checkpoints/sam2_hiera_tiny.pt',
    sam2_config='sam2_hiera_t.yaml',
    finetune_decoder=False,
).to(device)

ckpt = torch.load('runs/20260325_015312/best.pt', map_location=device, weights_only=False)

# Diagnose bridge key mismatch
model_bridge_keys = set(model.bridge.state_dict().keys())
ckpt_bridge_keys = set(ckpt['bridge'].keys())
missing = model_bridge_keys - ckpt_bridge_keys
unexpected = ckpt_bridge_keys - model_bridge_keys
print(f'Bridge keys — model={len(model_bridge_keys)} ckpt={len(ckpt_bridge_keys)} common={len(model_bridge_keys & ckpt_bridge_keys)}')
print(f'  missing (in model, not ckpt): {len(missing)} — first 10: {list(missing)[:10]}')
print(f'  unexpected (in ckpt, not model): {len(unexpected)} — first 10: {list(unexpected)[:10]}')

model.bridge.load_state_dict(ckpt['bridge'], strict=False)
model.cross_attn.load_state_dict(ckpt['cross_attn'], strict=True)
print(f'loaded epoch={ckpt["epoch"]} best_f1={ckpt["best_f1"]}')

model.eval()
print('Model in eval mode?', not model.training)

# Grab a single batch from levir_cd_plus test
tf = ValTransform(img_size=512)
ds = CDDataset('data/levir_cd_plus/test', transform=tf, name='levir_cd_plus')
print(f'levir_cd_plus test size: {len(ds)}')

# Debug one sample
ref, tgt, gt = ds[0]
ref = ref.unsqueeze(0).to(device)
tgt = tgt.unsqueeze(0).to(device)
gt = gt.unsqueeze(0).to(device)
print(f'ref: {ref.shape}, gt unique: {gt.unique()}, gt positive %: {(gt > 0.5).float().mean().item():.4f}')

# Test 1: Direct forward (no autocast, no TTA)
with torch.no_grad():
    masks, iou_pred, change_map = model(ref, tgt)
print(f'\\n[TEST 1: no-autocast no-TTA]')
print(f'  masks: {masks.shape}, min={masks.min().item():.3f}, max={masks.max().item():.3f}')
print(f'  sigmoid(masks): min={masks.sigmoid().min().item():.3f}, max={masks.sigmoid().max().item():.3f}')
pred = F.interpolate(masks, gt.shape[-2:], mode='bilinear', align_corners=False)
pred_bin = (pred.sigmoid() > 0.5).float()
print(f'  pred_bin positive %: {pred_bin.mean().item():.4f}')
print(f'  gt positive %: {gt.mean().item():.4f}')
# Compute F1 for this sample
tp = (pred_bin * gt).sum().item()
fp = (pred_bin * (1-gt)).sum().item()
fn = ((1-pred_bin) * gt).sum().item()
prec = tp / (tp+fp+1e-8); rec = tp / (tp+fn+1e-8); f1 = 2*prec*rec/(prec+rec+1e-8)
print(f'  sample F1={f1:.4f} P={prec:.4f} R={rec:.4f}')

# Test 2: With autocast
from torch.amp import autocast
with torch.no_grad(), autocast('cuda', enabled=True):
    masks2, iou2, _ = model(ref, tgt)
print(f'\\n[TEST 2: autocast-on no-TTA]')
print(f'  masks: {masks2.shape}, dtype={masks2.dtype}, min={masks2.min().item():.3f}, max={masks2.max().item():.3f}')

# Test 3: TTA
with torch.no_grad():
    masks3, iou3, _ = model.forward_tta(ref, tgt)
print(f'\\n[TEST 3: TTA no-autocast]')
print(f'  masks: {masks3.shape}, dtype={masks3.dtype}, min={masks3.min().item():.3f}, max={masks3.max().item():.3f}')
pred3 = F.interpolate(masks3, gt.shape[-2:], mode='bilinear', align_corners=False)
pred3_bin = (pred3.sigmoid() > 0.5).float()
print(f'  pred_bin positive %: {pred3_bin.mean().item():.4f}')

# Test 4: Batched (evaluate full levir_cd_plus quickly without TTA, without autocast)
print(f'\\n[TEST 4: full levir_cd_plus test, no-TTA, no-autocast]')
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
tp=fp=fn=tn=0
with torch.no_grad():
    for i, (r,t,g) in enumerate(loader):
        r, t, g = r.to(device), t.to(device), g.to(device)
        m, _, _ = model(r, t)
        p = F.interpolate(m, g.shape[-2:], mode='bilinear', align_corners=False)
        pb = (p.sigmoid() > 0.5).float()
        tp += (pb*g).sum().item()
        fp += (pb*(1-g)).sum().item()
        fn += ((1-pb)*g).sum().item()
        tn += ((1-pb)*(1-g)).sum().item()
        if i == 0:
            print(f'  batch 0: pb% = {pb.mean().item():.4f}, g% = {g.mean().item():.4f}')
prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8); f1 = 2*prec*rec/(prec+rec+1e-8); iou=tp/(tp+fp+fn+1e-8)
print(f'  levir_cd_plus test: F1={f1:.4f} IoU={iou:.4f} P={prec:.4f} R={rec:.4f}')
PY
