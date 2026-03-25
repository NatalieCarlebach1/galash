# Galash — Change Detection with DINOv2 + SAM2.1

A change detection pipeline for aerial/satellite imagery that combines frozen DINOv2 features with SAM2.1 mask decoding.

## Architecture

```
ref image ─┐                               ┌─ Bridge v2 (trainable) ──┐
           ├→ DINOv2/v3 (frozen) ──────────┤  FPN + residual + xformer├→ SAM2.1 Decoder → mask + IoU
tgt image ─┘   multi-scale features        └─ CrossChangeAttn ────────┘
                                               (trainable)           ↗ change map (aux loss)
```

**Trainable components:**
- `CrossChangeAttention` — cross-attention between ref/tgt patch tokens with learnable temperature
- `Bridge v2` — residual blocks, FPN top-down fusion, transformer refinement, high-res feature projection
- SAM2.1 decoder (optionally fine-tuned with differential LR)

**Frozen components:**
- DINOv2 backbone (remote-sensing variant)
- SAM2.1 prompt encoder

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (GPU required for training)
- Git

### 1. Clone the repository

```bash
git clone git@github.com:talshaharabany/galash.git
cd galash
```

### 2. Create a virtual environment (recommended)

```bash
conda create -n galash python=3.12 -y
conda activate galash
```

### 3. Install PyTorch

Install PyTorch with CUDA support for your system from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# Example for CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Download SAM2.1 checkpoints

```bash
mkdir -p checkpoints && cd checkpoints

# Download the variant(s) you need:
# Tiny (149M) — fastest, good for prototyping
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Small (176M)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Base+ (309M) — recommended
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Large (857M) — best quality
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

cd ..
```

### 7. Verify installation

```bash
python -c "
from model import ChangeDetector, SAM2_VARIANTS
from dataset import build_loaders
print('All imports OK')
"
```

## Data

Place datasets under `data/` following this structure:

```
data/{dataset_name}/
  train/
    A/       (reference images, time-1)
    B/       (target images, time-2)
    label/   (binary change masks)
  val/       (optional — auto-split from train if missing)
  test/
```

Supported datasets: LEVIR-CD, LEVIR-CD+, S2Looking, SECOND, CDD, DSIFN-CD, OSCD.

Use `python download_datasets.py` to download them automatically.

## Training

```bash
# List all available encoders and decoders
python train.py --list_models

# Recommended: DINOv2-RS base + SAM2.1 base+ with fine-tuning
python train.py \
    --encoder dinov2_rs_base \
    --decoder sam2_base_plus \
    --finetune_decoder \
    --img_size 512 --batch 4 --epochs 50 --tta

# DINOv3 satellite encoder (needs HF access) + SAM2.1 large
python train.py \
    --encoder dinov3_sat_large \
    --decoder sam2_large \
    --finetune_decoder

# SAM1 decoder (no high-res features)
python train.py --encoder dinov2_base --decoder sam1_vit_l

# SAM3 decoder (auto-downloads from HuggingFace)
python train.py --encoder dinov2_rs_base --decoder sam3 --finetune_decoder

# Lighter variant
python train.py --encoder dinov2_rs_small --decoder sam2_tiny --epochs 30
```

### Available Encoders

| Name | HuggingFace ID | Dim | Layers | Patch | Notes |
|------|---------------|-----|--------|-------|-------|
| `dinov2_small` | facebook/dinov2-small | 384 | 12 | 14 | |
| `dinov2_base` | facebook/dinov2-base | 768 | 12 | 14 | |
| `dinov2_large` | facebook/dinov2-large | 1024 | 24 | 14 | |
| `dinov2_giant` | facebook/dinov2-giant | 1536 | 40 | 14 | |
| `dinov2_small_reg` | facebook/dinov2-with-registers-small | 384 | 12 | 14 | With register tokens |
| `dinov2_base_reg` | facebook/dinov2-with-registers-base | 768 | 12 | 14 | With register tokens |
| `dinov2_large_reg` | facebook/dinov2-with-registers-large | 1024 | 24 | 14 | With register tokens |
| `dinov2_giant_reg` | facebook/dinov2-with-registers-giant | 1536 | 40 | 14 | With register tokens |
| `dinov2_rs_small` | KevinCha/dinov2-vit-small-remote-sensing | 384 | 12 | 16 | RS fine-tuned |
| `dinov2_rs_base` | KevinCha/dinov2-vit-base-remote-sensing | 768 | 12 | 16 | RS fine-tuned (default) |
| `dinov2_rs_large` | KevinCha/dinov2-vit-large-remote-sensing | 1024 | 24 | 14 | RS fine-tuned |
| `dinov3_small` | facebook/dinov3-vits16-pretrain-lvd1689m | 384 | 12 | 16 | Gated, needs HF access |
| `dinov3_base` | facebook/dinov3-vitb16-pretrain-lvd1689m | 768 | 12 | 16 | Gated |
| `dinov3_large` | facebook/dinov3-vitl16-pretrain-lvd1689m | 1024 | 24 | 16 | Gated |
| `dinov3_huge` | facebook/dinov3-vith16plus-pretrain-lvd1689m | 1280 | 32 | 16 | Gated |
| `dinov3_sat_large` | facebook/dinov3-vitl16-pretrain-sat493m | 1024 | 24 | 16 | Satellite, gated |

### Available Decoders

| Name | Family | Checkpoint | High-Res | Notes |
|------|--------|-----------|----------|-------|
| `sam1_vit_b` | SAM1 | sam_vit_b.pth | No | Lightest |
| `sam1_vit_l` | SAM1 | sam_vit_l.pth | No | |
| `sam1_vit_h` | SAM1 | sam_vit_h.pth | No | |
| `sam2_tiny` | SAM2.1 | sam2.1_hiera_tiny.pt | Yes | Fastest SAM2 |
| `sam2_small` | SAM2.1 | sam2.1_hiera_small.pt | Yes | |
| `sam2_base_plus` | SAM2.1 | sam2.1_hiera_base_plus.pt | Yes | Recommended |
| `sam2_large` | SAM2.1 | sam2.1_hiera_large.pt | Yes | Best SAM2 quality |
| `sam3` | SAM3 | Auto-download from HF | Yes | 848M, single variant |

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--encoder` | dinov2_rs_base | Encoder from registry or HuggingFace ID |
| `--decoder` | sam2_base_plus | Decoder from registry |
| `--finetune_decoder` | off | Fine-tune decoder with lower LR |
| `--decoder_lr_scale` | 0.1 | LR multiplier for decoder |
| `--latent_temp` | 0.03 | Temperature for latent change map loss |
| `--w_latent` | 0.2 | Weight for latent auxiliary loss |
| `--no_ohem` | off | Disable online hard example mining |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--tta` | off | Test-time augmentation on final test |
| `--img_size` | 512 | Input image size (supports 512, 768, 1024) |

## Evaluation

```bash
python eval.py \
    --encoder dinov2_rs_base \
    --decoder sam2_base_plus \
    --finetune_decoder \
    --checkpoint runs/YYYYMMDD_HHMMSS/best.pt \
    --tta \
    --save_masks predictions/
```

## Data Augmentation

Training augmentations (all configurable):
- Random scale (0.75–1.25x) + crop
- Horizontal/vertical flip, 90° rotation
- Independent color jitter, Gaussian blur, grayscale per image
- Random erasing / cutout (same region on both images)
- Elastic distortion (consistent across ref/tgt/mask)
- Ref/tgt swap (for symmetric change labels)

## Results

Training on LEVIR-CD + LEVIR-CD+ + S2Looking + SECOND + CDD (~17k pairs):

### Baseline comparison table

| Encoder | Decoder | Fine-tune | Val F1 | Test F1 | Test IoU |
|---------|---------|-----------|--------|---------|----------|
| dinov2_rs_base | sam2_tiny (frozen) | No | 0.907 | 0.871 | 0.772 |
| dinov2_rs_base | sam2_base_plus | Yes | TBD | TBD | TBD |
| dinov2_rs_base | sam2_large | Yes | TBD | TBD | TBD |
| dinov2_rs_base | sam1_vit_b | No | TBD | TBD | TBD |
| dinov2_rs_base | sam1_vit_l | No | TBD | TBD | TBD |
| dinov2_rs_base | sam3 | Yes | TBD | TBD | TBD |
| dinov2_rs_large | sam2_base_plus | Yes | TBD | TBD | TBD |
| dinov2_base | sam2_base_plus | Yes | TBD | TBD | TBD |
| dinov2_large | sam2_base_plus | Yes | TBD | TBD | TBD |
| dinov3_sat_large | sam2_base_plus | Yes | TBD | TBD | TBD |
| dinov3_base | sam2_base_plus | Yes | TBD | TBD | TBD |
