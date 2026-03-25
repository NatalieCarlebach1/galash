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

## Setup

```bash
# Install SAM2 (from https://github.com/facebookresearch/sam2)
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..

# Install dependencies
pip install -r requirements.txt

# Download SAM2.1 checkpoints
cd checkpoints
# tiny (149M), small (176M), base_plus (309M), large (857M)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..
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
# Recommended: base_plus decoder with fine-tuning
python train.py \
    --dino KevinCha/dinov2-vit-base-remote-sensing \
    --sam2_variant base_plus \
    --finetune_decoder \
    --decoder_lr_scale 0.1 \
    --img_size 512 \
    --batch 4 --epochs 50 \
    --latent_temp 0.03 --w_latent 0.2 \
    --tta

# Lighter variant
python train.py --sam2_variant tiny --epochs 30

# Explicit checkpoint paths (backward compatible)
python train.py \
    --sam2_ckpt checkpoints/sam2.1_hiera_base_plus.pt \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml
```

### SAM2.1 variants

| Variant      | Flag              | Params | Notes                  |
|-------------|-------------------|--------|------------------------|
| Tiny        | `--sam2_variant tiny`      | 39M  | Fastest, baseline      |
| Small       | `--sam2_variant small`     | 46M  | Good speed/quality     |
| Base+       | `--sam2_variant base_plus` | 81M  | Recommended            |
| Large       | `--sam2_variant large`     | 224M | Best quality           |

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--finetune_decoder` | off | Fine-tune SAM2 decoder (recommended) |
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
    --sam2_variant base_plus \
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

| Config | Val F1 | Test F1 | Test IoU |
|--------|--------|---------|----------|
| v1: tiny decoder (frozen) | 0.907 | 0.871 | 0.772 |
| v2: base+ decoder (fine-tuned) + augmentation | TBD | TBD | TBD |
