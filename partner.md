# GALASH — partner collaboration guide

Hi! 💙 We're working together on this change-detection paper. I'm running on a SLURM cluster; **you have 3 GPUs**, and there's plenty of useful work we can split between us.

This file brings you up to speed on the project, tells you how to set up, and gives you a concrete list of experiments you can run on your machine that **don't overlap** with mine.

---

## 1. TL;DR — what is this project?

**GALASH** = a change-detection pipeline for aerial / satellite image pairs:

```
  ref image ─┐
             ├─→ DINOv2/v3 (frozen)  ──→  Bridge v2        ──┐
  tgt image ─┘   multi-scale features    (trainable 11M)     ├→ SAM2.1 decoder → change mask
                                         CrossChangeAttn ────┘    (frozen or fine-tuned)
```

**Our pitch:** compose two frozen foundation models — a DINO encoder and a SAM decoder — with only a tiny trainable bridge. Only ~16.5 M trainable params on top of 300 M+ frozen backbones. Aiming for a top-tier venue (targeting **BMVC**, paper scaffold in `paper/`).

**Novel pieces:**
1. **Unified encoder/decoder registry** (16 encoders × 8 decoders in `model.py`) — systematic study of foundation-model pairings for CD.
2. **Cross-change attention** with learnable temperature.
3. **Bridge v2** — FPN + transformer refinement; repurposes SAM's normally-unused dense-prompt slot as a change conditioning signal.
4. Fair per-dataset augmentation configs in `configs/datasets/*.yaml` — matched to each benchmark's published SOTA.

---

## 2. Status as of 2026-04-24 evening

### Datasets — all 6 benchmarks ready

| Dataset | Train / Val / Test | Tile | Status |
|---|---|---|---|
| LEVIR-CD | 7120 / 1024 / 2048 | 256² | ready |
| LEVIR-CD+ | 637 / 0 / 348 | 1024² | ready (auto-holdout 10%) |
| S2Looking | 3500 / 500 / 1000 | 1024² | ready |
| CDD | 10000 / 2998 / 3000 | 256² | ready |
| DSIFN-CD | 10000 / 2998 / 3000 | 256² | ready |
| SECOND | 2968 / 0 / 1694 | 512² | ready (binary-collapse) |

If any of these are missing on your machine, see §4.

### Current SOTA gaps (live as of this commit)

| Dataset | Our best (test) | SOTA | Gap |
|---|---|---|---|
| LEVIR-CD | 0.9000 | 0.9287 (SChanger) | −2.87 pp |
| LEVIR-CD+ | 0.8172 | 0.9150 (ChangeStar+Changen) | −9.78 pp |
| S2Looking | live ~0.60 val | 0.6932 (UniChange) | ~−9 pp (still running) |
| CDD | live 0.9484 val | 0.9762 (SChanger) | ~−2.78 pp (running) |
| DSIFN-CD | live 0.9481 val | 0.9665 (DDPM-CD) | **~−1.84 pp** — closest running |
| SECOND | **0.7320** ⬆ | 0.7312 (UniChange) | **+0.08 pp above SOTA** |

The closest gaps to SOTA are on **LEVIR-CD, CDD, DSIFN-CD, SECOND**. The two stubborn ones are **LEVIR-CD+** and **S2Looking** (both likely need Changen-style synthetic-pair pretraining to close).

### What's been run / is running on my end

- **~30 jobs on the SLURM cluster** covering:
  - 6 fair per-dataset baselines (dinov2_rs_base + sam2_base_plus, running)
  - Encoder sweep: DINOv1 / DINOv2 / DINOv2-reg / DINOv2-RS / DINOv3 / DINOv3-sat × 1 decoder on LEVIR-CD
  - Decoder sweep: dinov2_rs_base × SAM1 (3 sizes) × SAM2.1 (4 sizes) on LEVIR-CD
  - Full DINOv3-sat per-dataset sweep (aerial pretraining, 6 datasets)
  - "Cheap wins" sweep: fine-tuned decoder + EMA + threshold search + longer patience on LEVIR-CD / SECOND / DSIFN

See `EXPERIMENTS.md` for the full live plan.

---

## 3. Architecture in 5 minutes

If you want the full story read `paper/sections/03_method.tex`. Short version:

1. **Encoder** (frozen DINO family) — takes ref and target, returns 4 layers of multi-scale patch tokens.
2. **CrossChangeAttention** ([model.py:102-143](model.py#L102-L143)) — per-patch cross-attention between ref and target tokens with **learned temperature**. Outputs: `change_tokens` (feed to bridge) + `change_map` (aux patch-level loss).
3. **Bridge v2** ([model.py:187-282](model.py#L187-L282)) — multi-scale FPN, transformer refinement, emits:
   - `image_emb` (256×64×64) → SAM's `image_embeddings`
   - `dense_prompt` (256×64×64) → SAM's `dense_prompt_embeddings` (**this slot is normally unused; we fill it with change tokens**)
   - `high_res_features` → SAM2.1's hierarchical head
4. **SAM decoder** (frozen or optionally fine-tuned at 0.1× LR) emits mask + IoU.

**Loss**: BCE(OHEM 70%) + Dice + IoU(MSE) + Latent (patch-level BCE) with weights 1 / 1 / 0.5 / 0.2.

**Trainable:** ~0.05 M (cross-attn) + ~11 M (bridge) + optional ~10-200 M (SAM decoder if fine-tuned). Default: ~16.5 M trainable on top of frozen DINO + SAM.

---

## 4. Setup on your machine

### Prerequisites

```bash
# Clone + branch
git clone git@github.com:talshaharabany/galash.git
cd galash
git checkout feat/encoder-decoder-registry
```

### Conda environment

On my cluster I use `ns-sam3`. For you, replicate with:

```bash
conda create -n galash python=3.12 -y
conda activate galash

# PyTorch — pick the CUDA version matching your GPUs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# (or cu121/cu118 if your GPUs are older — RTX 30xx typically cu121; RTX 40xx/Ada cu124)

# SAM2 (must be installed from the Meta repo)
git clone https://github.com/facebookresearch/sam2.git ../sam2
cd ../sam2 && pip install -e . && cd ../galash

# SAM1 (optional, for decoder sweep)
pip install segment-anything

# Everything else
pip install -r requirements.txt
pip install gdown rarfile py7zr pyyaml
```

### SAM2.1 checkpoints

I have these at `checkpoints/`:
```
sam2.1_hiera_tiny.pt       # 156 MB
sam2.1_hiera_small.pt      # 184 MB
sam2.1_hiera_base_plus.pt  # 324 MB
sam2.1_hiera_large.pt      # 898 MB
```

Download them from [facebookresearch/sam2](https://github.com/facebookresearch/sam2#download-checkpoints) into your `checkpoints/` folder. (SAM1 vit-b/l/h if you want to try those too.)

### Datasets

```bash
# Try auto-downloads first
python download_datasets.py --out data --datasets cdd s2looking second

# If any fail (Dropbox/GDrive being flaky), the repaired versions are in scripts/:
python scripts/fix_dsifn.py         # needs dsifn.zip (actually a RAR) in data/dsifn_cd/_tmp/
python scripts/fix_second.py        # needs SECOND archives in data/second/_tmp/
python scripts/fix_levir_cd.py      # needs LEVIR-CD256.zip in data/levir_cd/_tmp/
```

For this project the **must-haves** are LEVIR-CD, CDD, DSIFN-CD, SECOND (256²-pixel datasets — fit easily in 24 GB VRAM). **Nice to have**: S2Looking, LEVIR-CD+ (1024² tiles — need more memory).

### HuggingFace auth (for gated DINOv3)

The DINOv3 variants (small/base/large/sat_large) are behind a gated HF license. Request access at [facebook/dinov3-vitl16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m). Once granted:

```bash
huggingface-cli login   # paste your HF token
```

### Smoke test

```bash
# Verify model builds
python train.py --list_models

# Overfit test (30 sec on 1 GPU) — catches most integration issues
python overfit_test.py --n 8 --steps 300
# Expected: F1 > 0.8 on the 8-image overfit
```

---

## 5. Experiments I'd suggest for your 3 GPUs

**Ground rules so we don't duplicate compute:**

1. I'm already running the full per-dataset fair sweep + encoder sweep + decoder sweep on LEVIR-CD.
2. I'm about to run Changen-style synthetic pretraining (Tier 3) — that's on my side.
3. You'd have the most impact on the things I **haven't** queued: multi-seed stability, per-dataset ablations, and larger-backbone runs for the benchmarks I can't comfortably fit.

Here are three priority tiers. **Each item fits on a single 24 GB GPU**, so with 3 GPUs you can run 3 concurrent.

### Tier A — multi-seed statistical stability (highest ROI)

Reviewers love seeing variance across seeds. My runs are single-seed. You could fill this gap.

**Idea:** run the **fair per-dataset baseline** (`dinov2_rs_base + sam2_base_plus`) with 3 seeds on each of LEVIR-CD, CDD, DSIFN-CD, SECOND. 12 runs, ~3 hours each.

```bash
# Add --seed flag to train.py if missing (it's not there yet — I'll add it, or you can)
for SEED in 42 43 44; do
  for DS in levir_cd cdd dsifn_cd second; do
    python train.py \
      --config configs/datasets/${DS}.yaml \
      --encoder dinov2_rs_base \
      --decoder sam2_base_plus \
      --save_dir runs/seeds/seed${SEED}_${DS} \
      --seed $SEED
  done
done
```

Outputs go into `runs/seeds/...` so we can collect mean±std per dataset.

### Tier B — decoder fine-tuning on the datasets I haven't touched

My "cheap_wins" sweep fine-tunes the decoder only on LEVIR-CD / SECOND / DSIFN. You could cover **CDD and LEVIR-CD+**:

```bash
# FT on CDD
python train.py \
  --config configs/datasets/cdd.yaml \
  --encoder dinov2_rs_base --decoder sam2_base_plus \
  --finetune_decoder --ema 0.999 --search_threshold \
  --patience 30 \
  --save_dir runs/partner/cheap_cdd

# FT on LEVIR-CD+
python train.py \
  --config configs/datasets/levir_cd_plus.yaml \
  --encoder dinov2_rs_base --decoder sam2_base_plus \
  --finetune_decoder --ema 0.999 --search_threshold \
  --patience 30 \
  --save_dir runs/partner/cheap_levir_cd_plus
```

### Tier C — bigger encoders I can't comfortably fit

If your GPUs are 24 GB, **dinov2_rs_large** (patch 14, 300 M params) and **dinov3_large** (patch 16) barely fit at bs=4. Very valuable for the paper if you can get final numbers:

```bash
# dinov2_rs_large on LEVIR-CD
python train.py \
  --config configs/datasets/levir_cd.yaml \
  --encoder dinov2_rs_large --decoder sam2_base_plus \
  --batch 4 \
  --save_dir runs/partner/dinov2_rs_large_levir

# NOTE: the KevinCha loader for dinov2_rs_large has a weight-key bug
# (load_dino_rs.py:72 KeyError on 'blocks.0.mlp.fc1.weight').
# It looks fixable — the large checkpoint uses a different key scheme.
# If you're up for debugging that file, I'd owe you several coffees.
```

### Tier D — ablation runs (small, fast)

These are single-config ablations on LEVIR-CD, ~2 hours each on 1 GPU:

```bash
# ablation: no TTA
python train.py --config configs/datasets/levir_cd.yaml \
  --encoder dinov2_rs_base --decoder sam2_base_plus \
  --save_dir runs/partner/abl_notta   # (no --tta flag)

# ablation: no latent loss
python train.py --config configs/datasets/levir_cd.yaml \
  --encoder dinov2_rs_base --decoder sam2_base_plus \
  --w_latent 0.0 --tta \
  --save_dir runs/partner/abl_nolatent

# ablation: no OHEM
python train.py --config configs/datasets/levir_cd.yaml \
  --encoder dinov2_rs_base --decoder sam2_base_plus \
  --no_ohem --tta \
  --save_dir runs/partner/abl_noohem
```

### Recommended sequence for your 3 GPUs (Tier A first)

Starts the work most likely to make the paper. All 12 seed runs fit in about **3 GPU-days** (4 hours × 12 runs / 3 GPUs):

```
GPU 0: seed42 × [levir_cd, cdd, dsifn_cd, second]
GPU 1: seed43 × [levir_cd, cdd, dsifn_cd, second]
GPU 2: seed44 × [levir_cd, cdd, dsifn_cd, second]
```

Then move to Tier B + Tier D. Tier C is a stretch.

---

## 6. How to sync results with me

Two options:

### Option A — git-based (for small results)

Push your `runs/partner/*/test_results.json` files to the branch:

```bash
cd ~/galash
git pull origin feat/encoder-decoder-registry
# after your runs finish
git add runs/partner/*/test_results.json runs/partner/*/log.csv runs/partner/*/config.json
git commit -m "partner: add seed{42,43,44} baselines for LEVIR/CDD/DSIFN/SECOND"
git push origin feat/encoder-decoder-registry
```

(`.gitignore` currently ignores `runs/` — we'll need to force-add these files or loosen the ignore for `runs/partner/*.json`.)

### Option B — cloud sync for full checkpoints

Checkpoints are big (~80 MB each for base, ~900 MB for large). If we want to share `best.pt` files, a Dropbox / Google Drive shared folder is easier. Up to you.

---

## 7. Monitoring + tools

I wrote a monitor that shows live SOTA-gap across all runs:

```bash
python scripts/monitor.py           # one-shot
python scripts/monitor.py --watch   # refresh every 60 sec
```

It shows each run's current val F1, the final test F1 (when done), the published SOTA for that dataset, and the gap. Very handy for seeing progress at a glance.

`scripts/monitor.py` reads `runs/*/log.csv` + `runs/*/test_results.json`, so it works on your `runs/partner/*` too.

---

## 8. Project roadmap + what comes next

Rough plan (subject to what results come back):

1. **This week** — finish the current sweep. Collect per-dataset best numbers, fill in `paper/sections/04_experiments.tex` Table 1. Your multi-seed runs (Tier A) slot here too.
2. **Next week** — Tier 3: Changen-style synthetic pretraining for the LEVIR-CD+ / S2Looking gaps. That's on my side, needs heavier infra.
3. **End of month** — write the paper. Current target venue: BMVC 2026 (deadline usually April–May).

If anything in the plan doesn't work for you, or you'd rather do something else (e.g., you have an idea for a new architectural piece), just say the word. 🙂

---

## 9. Files to know

| Path | What it is |
|---|---|
| `model.py` | Encoder/decoder registry + ChangeDetector + Bridge v2 + CrossChangeAttention |
| `train.py` | Training loop. Has EMA + threshold search. Accepts `--config` YAML |
| `eval.py` | Eval. Has `--search_threshold` + `--test_img_sizes` + `--use_ema` |
| `dataset.py` | Dataset loaders. `build_loaders(config=...)` wires into YAML |
| `configurable_aug.py` | YAML-driven augmentation pipeline (matches SOTA papers per-dataset) |
| `configs/datasets/*.yaml` | Per-benchmark fair-comparison configs (aug list + hyperparams) |
| `configs/datasets/README.md` | Schema and what-matches-what |
| `scripts/fix_*.py` | Dataset repair scripts (DSIFN, SECOND, LEVIR-CD, OSCD tiling) |
| `scripts/monitor.py` | Live SOTA-gap monitor |
| `slurm/train.sbatch` | SLURM training template. Configurable via env vars |
| `slurm/launch_sweep.sh <tier>` | Submit experiment tiers. `fair_per_dataset`, `encoder_sweep`, `decoder_sweep`, `cheap_wins`, ... |
| `paper/` | BMVC LaTeX source. `main.tex` compiles with `cd paper && latexmk -pdf main` |
| `EXPERIMENTS.md` | Top-level experiment plan + where numbers come from |
| `partner.md` | This file! |

---

Love you ❤️ — let's publish this together.

— Tal (via Claude)
