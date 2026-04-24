# Per-dataset training configs

Each `<dataset>.yaml` in this directory encodes the augmentation pipeline,
input resolution, and training hyperparameters used by top published methods
on that benchmark. This lets GALASH claim a **fair comparison** — we only
use augmentations the reference method uses, no cherry-picked extras.

## Schema

```yaml
dataset: <name matching data/<dataset>/ dir>
img_size: <train resolution>
test_img_size: <test resolution>

normalization:
  mean: [R, G, B]          # e.g. ImageNet [0.485, 0.456, 0.406] or ChangeFormer [0.5, 0.5, 0.5]
  std:  [R, G, B]

augmentations:
  - {op: <name>, <op-specific params>, prob: 0.5, independent: false}
  - ...                     # applied in order; any op NOT listed is NOT applied

training:
  batch: 16
  epochs: 300               # "effectively infinite" + early-stop
  patience: 15
  lr: 1.0e-4
  warmup: 3
  decoder_lr_scale: 0.1
  finetune_decoder: false
  tta: true

reference_methods:
  - {name: SChanger, year: 2025, f1: 0.9287, url: "https://arxiv.org/..."}
  - ...
```

## Supported augmentation ops

| `op` | Parameters | Notes |
|---|---|---|
| `random_flip` | `direction: {horizontal, vertical}`, `prob` | shared across ref/tgt/mask |
| `random_rotate` | `degrees: int`, `prob`, `discrete: bool` (if true → {0,90,180,270}) | shared |
| `random_crop` | `size: int`, `cat_max_ratio: float = 1.0` | shared |
| `scale_random_crop` | `size: int`, `scale_range: [lo, hi]` | BIT/ChangeFormer style |
| `random_scale` | `scales: [list]`, `prob` | ChangeStar-style discrete scale |
| `temporal_swap` | `prob` | swaps ref↔tgt (ChangerEx, SChanger) |
| `photometric_distortion` | `brightness_delta`, `contrast_range`, `saturation_range`, `hue_delta` | mmcv-compatible |
| `color_jitter` | `brightness`, `contrast`, `saturation`, `hue`, `independent: bool` | torchvision-style |
| `gaussian_blur` | `radius_range: [lo, hi]`, `prob`, `independent: bool` | |
| `one_of` | `ops: [sub-list]`, `prob` | apply one sub-op with the given prob |

## How to run a fair per-dataset baseline

```bash
cd /home/nfs/tals/galash
python train.py --config configs/datasets/levir_cd.yaml \
    --encoder dinov2_rs_base --decoder sam2_base_plus \
    --save_dir runs/fair_levir_cd
```

Or via SLURM:

```bash
CONFIG=configs/datasets/levir_cd.yaml \
    sbatch --job-name=fair-levir slurm/train.sbatch
```

## Matrix of design choices

| Dataset | Style | Normalisation | Img size | Temporal swap | Photometric |
|---|---|---|---|---|---|
| LEVIR-CD | ChangerEx / open-cd | ImageNet | 256 | ✓ | ✓ (mmcv) |
| LEVIR-CD+ | ChangeStar / Changen | ImageNet | 512 crop from 1024 | ✗ | ✗ |
| S2Looking | ChangerEx / open-cd | ImageNet | 512 crop from 1024 | ✓ | ✓ (mmcv) |
| CDD | ChangeFormer | [0.5]×3 → [-1,1] | 256 | ✗ | ✓ (torch ColorJitter, independent) |
| DSIFN-CD | ChangeFormer | [0.5]×3 → [-1,1] | 256 | ✗ | ✓ (torch ColorJitter, independent) |
| SECOND | SAM-CD | ImageNet | 512 | ✗ | ✗ |

## Augmentations we explicitly DO NOT use (fair-comparison promise)

We exclude augmentations that no top method on that benchmark uses:
- **Elastic distortion** — only SChanger on LEVIR-CD has it (paper-only claim, code unreleased)
- **Cutout / random erasing** — only SChanger
- **CutMix / Mixup** — no top method on the 6 benchmarks
- **Independent photometric on datasets that don't use it** (e.g. LEVIR-CD+, SECOND)

See `paper/sections/04_experiments.tex` for the fair-comparison claim.
