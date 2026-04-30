# Update from Natalie — 2026-04-29

Hey Tal! Major update — lots of new results and new features since last time.

---

## 0. Branch

Still on `Natalie_features_Sam` (branched off `feat/encoder-decoder-registry`).

---

## 1. Environment & Setup (unchanged)

- Conda env: `/home/tal/natalie/conda_env` (PyTorch 2.5.1+cu121)
- SAM2 installed at `/home/tal/natalie/galat/sam2`
- All 4 SAM2.1 checkpoints in `checkpoints/`
- All 5 datasets in `data/` (cdd, dsifn_cd, levir_cd, s2looking, second — no levir_cd_plus)
- `python` on this machine = Python 2.7 — always use `python3` or the conda env binary

---

## 2. New features implemented in model.py / train.py

Three new flags, all ablation-ready:

### `--cnn_skip`
Adds a lightweight Siamese CNN branch (~66K params) that replaces the Bridge's
bilinearly-upsampled `high_res_features` with real pixel-level change signals:
- `feat_s0`: |CNN(ref) − CNN(tgt)| at 256×256 (replaces fake upsampled slot)
- `feat_s1`: |CNN(ref) − CNN(tgt)| at 128×128 (replaces fake upsampled slot)

Result on LEVIR-CD: **+1.32pp** (0.9018 → 0.9150). Cuts the gap vs SOTA in half.

### `--simple_diff`
Ablation: replaces CrossChangeAttention with elementwise `(ref−tgt)` difference.
CrossChangeAttention params are excluded from the optimizer. Used to quantify
how much CrossChangeAttention actually contributes vs plain subtraction.

### `--learnable_offset` / `--max_offset`
A small MLP (~148K params) predicts (dx, dy) per patch from ref tokens and warps
tgt tokens before CrossChangeAttention. Designed for misregistered datasets.
**Result on S2Looking: HURT by −3.13pp** (0.5956 → 0.5643). The S2Looking
parallax is 3D geometry-dependent (per-building height) — a single MLP can't
learn it from patch tokens. Do not use on S2Looking.

---

## 3. Complete results — all finished local runs

### 3a. Fair config (frozen decoder, seed 42/43/44)

| Run | Epochs | TEST F1 | SOTA | Gap |
|-----|--------|---------|------|-----|
| seed42 × LEVIR-CD | 113 | 0.9018 | 0.9287 | −2.69pp |
| seed43 × LEVIR-CD | 75 | 0.8994 | 0.9287 | −2.93pp |
| seed44 × LEVIR-CD | 71 | 0.8987 | 0.9287 | −3.00pp |
| seed42 × CDD | 300 | 0.9678 | 0.9762 | −0.84pp |
| seed42 × DSIFN-CD | 300 | **0.9676** | 0.9665 | **+0.11pp ✅** |

LEVIR-CD: mean=0.9000, std=0.0016 across 3 seeds — tight variance.

### 3b. Cheap-wins config (FT decoder + EMA 0.999 + threshold search + patience 30)

| Run | Epochs | TEST F1 | SOTA | Gap |
|-----|--------|---------|------|-----|
| cheap_levir_cd | 166 | 0.9051 | 0.9287 | −2.36pp |
| cheap_cdd | 300 | 0.9676 | 0.9762 | −0.86pp |
| cheap_dsifn_cd | 300 | **0.9675** | 0.9665 | **+0.10pp ✅** |
| cheap_second | 62 | 0.7147 | 0.7246 | −1.65pp |
| cheap_s2looking | 122 | 0.5956 | 0.6895 | −9.76pp |

Note: cheap-wins = no improvement on CDD (0.9676 vs 0.9678 fair). Matches cluster pattern.

### 3c. Ablation runs (new)

| Run | Flags | Epochs | TEST F1 | SOTA | Gap | vs baseline | Verdict |
|-----|-------|--------|---------|------|-----|-------------|---------|
| abl_levir_cnn_skip | `--cnn_skip` cheap | 121 | **0.9150** | 0.9287 | −1.37pp | **+1.32pp** | ✅ keeps |
| abl_s2looking_offset | `--learnable_offset` cheap | 65 | 0.5643 | 0.6895 | −12.89pp | −3.13pp | ❌ hurts |

---

## 4. Currently running (2026-04-29)

| GPU | Run | Flags | Question |
|-----|-----|-------|---------|
| 0 | abl_cdd_cnn_skip | `--cnn_skip` cheap | Does CNN skip close the −0.84pp CDD gap? |
| 1 | abl_s2looking_cnn_skip | `--cnn_skip` cheap | Does CNN skip help S2Looking at all? |
| 2 | abl_levir_simple_diff | `--simple_diff` cheap | How much does CrossChangeAttention contribute? |

The `simple_diff` run on GPU 2 is the most important for the paper — if it lands
around 0.900 while `abl_levir_cnn_skip` got 0.9150, CrossChangeAttention is worth ~1.5pp.

---

## 5. Best local results per dataset

| Dataset | Best local F1 | Config | SOTA | Gap |
|---------|--------------|--------|------|-----|
| LEVIR-CD | **0.9150** | cheap + cnn_skip | 0.9287 | −1.37pp |
| CDD | **0.9678** | fair seed42 | 0.9762 | −0.84pp |
| DSIFN-CD | **0.9676** | fair seed42 | 0.9665 | **+0.11pp ✅** |
| SECOND | 0.7147 | cheap | 0.7246 | −1.65pp |
| S2Looking | 0.5956 | cheap | 0.6895 | −9.76pp |

---

## 6. Paper strategy discussion

We discussed paper strategy at length. Key points:

**Venue target:** CVPR/ICCV (needs 3+ SOTA wins). Currently 2/6 datasets, one barely (+0.11pp).

**What needs to happen:**
1. `simple_diff` ablation proves CrossChangeAttention matters (+1pp expected)
2. CNN skip closes LEVIR-CD to −1.37pp (done), need to extend to CDD + S2Looking
3. 3 seeds on DSIFN-CD to confirm SOTA win is statistically robust
4. CrossChangeAttention vs simple diff — the core novelty claim

**Key insight from architecture discussion:**
- DINO and SAM1 encoder are both plain ViT — no spatial hierarchy
- The Bridge upsample (16×16 → 64×64) is bilinear interpolation: inventing pixels
- CNN skip replaces fake high_res_features with real boundary information
- SAM2 Hiera encoder is the right long-term direction (real spatial hierarchy → real skip connections)
- Using SAM2 as both encoder AND decoder with CrossChangeAttention at the bottleneck
  is novel — nobody does exactly this combination

---

## 7. Infrastructure notes (unchanged from before)

**OOM fix:** Always launch 512px datasets with:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Zombie fix:** Use `pkill -9 -f train.py` and verify with nvidia-smi before new runs.

**dinov3_large is gated** — needs HF token. All local runs use dinov2_rs_base.

**Canonical launch command:**
```bash
cd /home/tal/natalie/galat/galash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=<N> nohup \
  /home/tal/natalie/conda_env/bin/python3 train.py \
  --config configs/datasets/<dataset>.yaml \
  --encoder dinov2_rs_base --decoder sam2_base_plus \
  --finetune_decoder --ema 0.999 --search_threshold --patience 30 \
  [--cnn_skip] [--simple_diff] [--learnable_offset] \
  --save_dir runs/abl/abl_<name> > /tmp/abl_<name>.log 2>&1 &
```

---

## 8. What's next when GPUs free

1. **Wait for 3 running ablations** — results in a few hours
2. **Run CNN skip on LEVIR-CD+** — once data is downloaded (biggest gap dataset)
3. **3 seeds on DSIFN-CD with cnn_skip** — confirm SOTA win is robust
4. **CNN skip + simple_diff together** — does CrossChangeAttention still matter WITH CNN skip?

---

Love you! — Natalie 💙
(written with help from Claude on 2026-04-29)
