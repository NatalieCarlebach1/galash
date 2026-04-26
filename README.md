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

Supported aerial datasets: LEVIR-CD, LEVIR-CD+, S2Looking, SECOND, CDD, DSIFN-CD, OSCD.

```bash
python download_datasets.py --list                                  # all entries
python download_datasets.py --out data --datasets levir_cd cdd      # specific
python download_datasets.py --out data --all                        # everything
```

### Medical change-detection benchmarks

Medical CD uses the same `A/B/label` layout. 3-D volumes (NIfTI `.nii.gz`)
are sliced to 2-D axial PNGs with stem `{subject}_{slice:03d}` by the
preparer.

| Key | Source | Access | Task |
|-----|--------|--------|------|
| `brats_reg` | BraTS-Reg (Zenodo) | **open** (~4 GB) | Pre-op vs follow-up glioma MRI. Archive ships images only (no seg) — pass `--pseudo-labels` for FLAIR-difference weak labels, or drop `seg_00.nii.gz` + `seg_01.nii.gz` next to each subject |
| `shifts_ms` | Shifts 2.0 MS (Zenodo Part 2) | **open**, single-timepoint only (no CD labels) | MS domain-shift segmentation probe |
| `msseg2` | FLI-IAM / OFSEP | free account | Longitudinal new-MS-lesion detection |
| `isbi_ms` | SMART-stats | free account | Longitudinal MS over 4 time points |
| `lumiere` | TCIA | free account | Longitudinal glioblastoma MRI |
| `chest_imagenome` | PhysioNet | credentialed (CITI) | Sequential CXR pairs (categorical labels) |

```bash
python download_medical.py --list
python download_medical.py --out data --datasets brats_reg --pseudo-labels   # open, ~4 GB, FLAIR-diff labels
python download_medical.py --out data --datasets msseg2                      # after staging at data/msseg2/_tmp/
```

Evaluate on the medical benchmarks (reuses the aerial checkpoint):

```bash
sbatch slurm/eval_medical.sbatch
# or
DATASETS="shifts_ms brats_reg" sbatch slurm/eval_medical.sbatch
```

#### Fair-comparison medical training (MSSEG-2, ISBI-2015)

SOTA methods on MSSEG-2 (DEFUSE-MS, F1 0.65 / Dice 0.55) and ISBI-2015
(Temporal Difference Weighting, Dice 0.75 / F1 0.74) are all 3-D nnU-Net
variants. We can't match the 3-D architecture, but we **can** match the
training recipe — see `configs/datasets/{msseg2,isbi_ms}.yaml` which port
LR, weight-decay, epoch budget, early-stop patience, foreground-oversampling,
augmentation set, and (importantly) **no TTA** from the nnU-Net SOTA
defaults.

Both datasets are gated; stage the zips manually, then the queued jobs
auto-fire:

```bash
# submit two wait-and-fire training jobs (will hold a GPU and poll every 60 s)
bash slurm/launch_sweep.sh fair_medical
# → submits  g-fair_msseg2  and  g-fair_isbi_ms

# then, whenever you've downloaded the data:
mv ~/Downloads/msseg2_training.zip  data/msseg2/_tmp/
mv ~/Downloads/isbi_training.tar    data/isbi_ms/_tmp/
# → preparer runs, slicing fires, training starts — no further action needed
```

Data sources (one-click DUA per portal, free accounts):
- MSSEG-2: https://portal.fli-iam.irisa.fr/msseg-2/
- ISBI-2015: https://smart-stats-tools.org/lesion-challenge  (alt mirror: https://iacl.ece.jhu.edu/index.php/MSChallenge/data)

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

Per-dataset best test F1 vs published SOTA (late 2025).

| Dataset | GALASH best | SOTA | Method | Gap |
|---|---|---|---|---|
| **SECOND** | **74.30** | 73.12 | UniChange (2025) | **+1.18 pp** ✅ |
| **DSIFN-CD** | **96.74** | 96.65 | DDPM-CD (2024) | **+0.09 pp** ✅ |
| CDD | 96.76 | 97.62 | SChanger (2025) | −0.86 pp |
| LEVIR-CD | 90.89 | 92.87 | SChanger (2025) | −1.98 pp |
| S2Looking | 65.71 | 69.32 | UniChange (2025) | −3.61 pp |
| LEVIR-CD+ | 83.20 | 91.50 | ChangeStar+Changen (2023) | −8.30 pp |

GALASH trains only ~16.5 M parameters on top of frozen backbones — 3× fewer
than ChangeFormer, 27× fewer than DDPM-CD.

### What works (validated across many runs)

| Trick | Impact |
|---|---|
| `dinov3_large` encoder | +1.5 to +3.2 F1 over DINOv2-RS-base on hard datasets |
| Cheap-wins recipe (FT decoder + EMA + threshold + patience 30) | +0.85 to +1.42 F1 on SECOND, big on S2Looking |
| Multi-resolution test (256/384/512, val-picked) | +0.4 to +0.7 F1 with no retraining |
| Per-dataset fair-aug YAML configs | matches each SOTA method's pipeline |

### What doesn't help (clean negative results)

LoRA (rank 4/8/16 on DINO/SAM/both), multi-scale auxiliary supervision,
bidirectional / local-window cross-attention, Lovász latent loss, MSE
patch-density regression, soft-target latent BCE — all null at convergence
on LEVIR-CD. The current architecture is near-locally-optimal at this scale
of data; gains come from better backbones, not extra complexity.

### Architecture details

| Component | Trainable params | Notes |
|---|---|---|
| DINO encoder | 0 (frozen) | one of 16 registry variants |
| CrossChangeAttention | ~50 K | learnable temperature |
| Bridge v2 | ~11 M | FPN + transformer refinement, dense-prompt repurposing |
| SAM mask decoder | 0 (frozen) or ~80 M (FT at 0.1× LR) | one of 7 variants |
| **Total trainable (frozen decoder)** | **~16.5 M** | |
