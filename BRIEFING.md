# GALASH — Full Briefing for New Claude Session
_Last updated: 2026-05-29. Written by Natalie's Claude._

---

## 1. What This Project Is

Change-detection model for satellite imagery. Architecture:

```
Frozen DINOv2-RS-base encoder
        ↓  (ref tokens, tgt tokens)
CrossChangeAttention   ← trainable
        ↓  (change_tokens, change_map, attn_diag)
Bridge v2              ← trainable
        ↓  (multi-scale features)
SAM2.1 mask decoder    ← optionally finetuned
        ↓
Binary change mask
```

**Trainable params**: ~13–35M depending on flags.  
**Primary test dataset**: LEVIR-CD (256×256 pairs, buildings, 0.5m/px).  
**Fair SOTA target**: SChanger = 92.87% F1 on LEVIR-CD.  
**Our best so far**: 91.83% (Config L / Config M — see §5).

---

## 2. Machine & Environment

- **Machine**: Natalie's local workstation — 3× NVIDIA RTX A5000 (24 GB each)
- **Conda env**: `/home/tal/natalie/conda_env` — always use `python3`
- **SAM2 path**: `/home/tal/natalie/galat/sam2` (must be on PYTHONPATH)
- **Project root**: `/home/tal/natalie/galat/galash` (referred to as `$GALASH`)
- **Key files**:
  - `model.py` — architecture + CrossChangeAttention
  - `train.py` — training loop, all CLI flags
  - `dataset.py` — data loaders incl. InriaPretrainDataset
  - `configs/datasets/levir_cd_heavy.yaml` — heavy aug config (256² training)

---

## 3. Key Concepts

### CrossChangeAttention
Cross-attention between ref and tgt patch tokens. Produces:
- `change_tokens` [B, S, D] — aggregated change context → fed to Bridge
- `change_map` [B, S] — spatial change probability → used in latent loss
- `attn_diag` [B, S] — diagonal of head-averaged attention matrix

**Two bypasses that were hurting us** (identified in session):
1. **change_map bypass**: `change_map = 1 - cosine_similarity(ref, tgt)` — bypasses attention diagonal; cosim does the job instead of attention
2. **(r-t) bypass**: `bridge_input = [(r-t) + change_tokens]` — naive difference redundant with change_tokens; attention doesn't need to learn if bridge always has (r-t)

### Attention diagonal
When attention is working correctly, each ref patch should attend most strongly to the same spatial position in tgt. `attn_diag[i]` = attention weight at position (i,i). 

**Uniform baseline**: 1/256 ≈ 0.00391 for 256-patch sequences.  
**Probed values** (see §5 for config meanings):
- Baseline C: ~1.0x uniform (attention is noise)
- L/M with supervision: **11.31x uniform** (diagonal is very sharp)
- After supervision fully decays in M: still **11.31x** — structure is retained permanently

### attn_supervise
Loss = `-log(attn_diag[unchanged_patches])` — pushes diagonal high for unchanged regions.  
Weight schedule: `w * max(0, 1 - epoch/decay_epochs)`

### attn_diag_init
Initializes Q and K projection matrices as identity → attention starts diagonal from epoch 0.

---

## 4. All CLI Flags Added (in model.py / train.py)

| Flag | Effect |
|------|--------|
| `--cnn_skip` | Lightweight Siamese CNN replaces Bridge's bilinear upsampled features with real pixel-level change signals at 128px and 256px. **+1.32pp on LEVIR-CD** — most important single flag. |
| `--finetune_decoder` | Unfreeze SAM2 mask decoder. Adds ~30M trainable params. Required for good results. |
| `--no_amp` | Disable AMP (bf16). Required — avoids BCE NaN. |
| `--ema 0.99` | EMA model weights. Consistently helps. |
| `--search_threshold` | Search optimal binary threshold on val set per epoch. |
| `--no_early_stop` | Run for full 300 epochs. |
| `--attn_supervise` | Add `-log(attn_diag[unchanged])` loss term. |
| `--w_attn_supervise 1.0` | Weight for attn supervise loss. |
| `--attn_supervise_decay_epochs N` | Linearly decay attn supervise weight to 0 over N epochs. |
| `--attn_diag_init` | Initialize Q/K projections as identity matrices. |
| `--no_diff_bypass` | Remove (r-t) from bridge input; bridge must rely purely on change_tokens. |
| `--attn_diag_change_map` | Use `1 - attn_diag` as change_map instead of cosine similarity. Requires attn_supervise + attn_diag_init. |
| `--multi_scale_cross_attn` | Replaces (r-t) at intermediate bridge scales with a shared CrossChangeAttention. |
| `--spatial_refine` | Adds a spatial refinement layer after CrossChangeAttention. |
| `--resume PATH` | Resume training from checkpoint. Creates new timestamped subdir but restores model+optimizer. |

---

## 5. Config Definitions (used in watcher / experiments)

All configs use: `--cnn_skip --finetune_decoder --search_threshold --no_early_stop --data ... --ckpt_dir ... --no_amp --ema 0.99`

| Config | Extra flags | Save dir | Meaning |
|--------|-------------|----------|---------|
| **C** | (none) | `cosim_base_plus_noamp` | Baseline — best simple config |
| **D** | `--spatial_refine` | `cosim_spatial_refine` | + spatial refinement layer |
| **E** | `--multi_scale_cross_attn` | `cosim_ms_cross_attn` | + multi-scale cross-attn replaces (r-t) |
| **L** | `--attn_supervise --w_attn_supervise 1.0 --attn_diag_init` | `cosim_attn_supervise_diaginit` | + supervision (no decay) + identity init |
| **M** | `--attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 100 --attn_diag_init` | `cosim_attn_supervise_diaginit_decay` | L + decay to 0 over 100 epochs |
| **N** | `--no_diff_bypass --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init` | `cosim_no_bypass` | Remove (r-t) bypass + supervision + decay=50 |
| **P** | `--multi_scale_cross_attn --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init` | `cosim_ms_cross_attn_supervise` | E + supervision + decay |
| **Q** | `--multi_scale_cross_attn --no_diff_bypass` | `cosim_ms_no_bypass` | E + no_diff_bypass |
| **R** | `--multi_scale_cross_attn --no_diff_bypass --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init` | `cosim_ms_no_bypass_supervise` | E + N combined (most complex) |
| **S** | `--no_diff_bypass --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 50 --attn_diag_init --attn_diag_change_map` | `cosim_no_bypass_attn_changemap` | N + use attn diagonal as change_map |
| **large** | (none but sam2_large decoder) | `cosim_large` | Scale up decoder |

---

## 6. All Results (LEVIR-CD, test F1)

### Completed runs (best_f1 across all epochs)

| Config | best_f1 | Notes |
|--------|---------|-------|
| **L** (attn_supervise + diag_init) | **0.9183** | Best overall. Supervision always on. |
| **M** (decay=100) | **0.9183** | Tied with L. Supervision decays to 0 by ep100. |
| **D** (spatial_refine) | 0.9165 | Simple addition, almost as good. |
| **C** (baseline) | 0.9178 | No attention modifications. |
| old L warmup variant | 0.9117 | Earlier experiment. |
| **N** (no_bypass + decay=50) | 0.9066 | Cold start from removing bypass. |
| **S** (no_bypass + attn_changemap) | 0.9034 | Similar to N, still climbing. |
| cosim_base_plus (AMP) | 0.8908 | Earlier baseline with AMP. |

### Key finding on attn_diag supervision
- Supervision works: pushes diagonal to **11.31x uniform** (vs 1.0x without)
- After M's supervision decays to 0 at ep100: diagonal stays at **11.31x** permanently
- But F1 gain is only ~0.05pp over unmodified baseline — the attention is sharp but the improvement is marginal

### Key finding on no_diff_bypass (N, S configs)
- Removing (r-t) from bridge causes **significant cold start** (~1.8pp behind baseline at ep37)
- After 300 epochs N only reaches 0.9066 — never catches up to baseline
- Conclusion: the (r-t) bypass is actually helpful, not a crutch

### Tal's cluster results (dinov2_base + sam2_large, max-push recipe)

| Dataset | Test F1+TTA | vs Fair SOTA |
|---------|------------|--------------|
| LEVIR-CD | 0.9106 | +0.82pp ✅ (vs ChangeFormer) |
| SECOND | 0.7201 | −0.45pp |
| DSIFN-CD | 0.9523 | −1.42pp |
| CDD | 0.9441 | −1.71pp |
| LEVIR-CD+ | 0.8606 | −1.65pp |
| S2Looking | 0.6527 | −3.33pp |

**Note**: Tal's SOTA bars use older references (ChangeFormer for LEVIR-CD). Updated fair SOTA is SChanger at 92.87% — we're actually 1.8pp behind on LEVIR-CD, not +0.82pp.

### SAM-SAM vs DINOv2-SAM
- SAM2-enc-large + SAM2-large = **worse** than DINOv2-base + SAM2-large on every dataset
- DINOv2-large-reg is the best single encoder Tal tested (0.9057 F1)
- Smallest decoder (sam2_tiny) beats larger ones — capacity is inversely correlated

---

## 7. Currently Running (as of 2026-05-29)

| GPU | PID | Config | Epoch | val_f1 | best_f1 | Status |
|-----|-----|--------|-------|--------|---------|--------|
| 0 | 851886 | **S** (no_bypass + attn_changemap) | ~270 | 0.903 | 0.903 | Climbing slowly |
| 1 | 991173 | **M** (decay=100) | ~300 | 0.918 | **0.9183** | Near end / done |
| 2 | 987571 | **N** (no_bypass + decay=50) | ~300 | 0.906 | 0.9066 | Near end / done |

**Watcher**: PID 852414, running `/tmp/gpu_watcher11.sh`  
**Watcher queue** (next to run when GPUs free): R, P, Q, E, large, then cdd:C, dsifn_cd:C, second:C, sysu_cd:C, levir_cd_plus:C, then D/E/L/large for each dataset.

---

## 8. Watcher Setup

**Script**: `/tmp/gpu_watcher11.sh`  
**Log**: `/tmp/gpu_watcher11.log`  
**Per-job logs**: `/tmp/w11_{config}_{dataset}.log`  

**Race condition fix** (already applied): `sleep 30` after each launch so GPU memory is claimed before the next GPU is checked.

**Important**: Only ever run ONE watcher instance. Two watchers = duplicate launches. Check with:
```bash
ps aux | grep gpu_watcher | grep -v grep
```

To restart watcher:
```bash
pkill -f gpu_watcher11.sh
nohup bash /tmp/gpu_watcher11.sh > /tmp/gpu_watcher11.log 2>&1 &
```

To resume a specific run manually:
```bash
CUDA_VISIBLE_DEVICES=X PYTHONPATH=/home/tal/natalie/galat/sam2:$PYTHONPATH \
  nohup python3 $GALASH/train.py \
    --encoder dinov2_rs_base --decoder sam2_base_plus \
    --config $GALASH/configs/datasets/levir_cd_heavy.yaml \
    --save_dir $GALASH/runs/<save_dir_name> \
    --cnn_skip --finetune_decoder --search_threshold --no_early_stop \
    --data $GALASH/data --ckpt_dir $GALASH/checkpoints --no_amp --ema 0.99 \
    <extra flags> \
    --resume $GALASH/runs/<save_dir_name>/<timestamp>/last.pt \
  > /tmp/my_run.log 2>&1 &
```

---

## 9. Attention Diagonal Probe

Script at `/tmp/probe_m.py`. Loads a checkpoint and measures how diagonal the attention matrix is vs uniform.

To run:
```bash
CUDA_VISIBLE_DEVICES=0 python3 /tmp/probe_m.py 2>/dev/null
```

Edit the `probe(ckpt_path, label)` call at the bottom to point at any checkpoint. Key metrics:
- `ratio`: mean diagonal / (1/S). 1.0x = uniform (attention is noise), 11x = very sharp diagonal
- `patches > 2x uniform`: % of patches with clearly above-chance diagonal weight

---

## 10. What to Try Next (Priority Order)

### 10a. Immediate (when M/N/R finish)
- **Wait for R** (ms + no_bypass + decay=50) to finish — R is next in the watcher queue. It reached ep43, best=0.899 before the crash. Will resume fresh. This tests whether multi-scale cross-attn + no-bypass together is better than either alone.
- **Wait for watcher to complete other datasets** — the queue has all 6 datasets × all configs. Most informative: cdd:C, dsifn_cd:C as baselines, then the experimental configs.

### 10b. Architecture — most likely to help
- **Combine L + cnn_skip on other datasets**: L gives +0.05pp on LEVIR-CD but might give more on other datasets. The current experiments only use LEVIR-CD for config exploration.
- **dinov2_large_reg on this machine**: Tal's best encoder (0.9057 vs rs_base's 0.9000). Would give a cleaner comparison base. Requires downloading the checkpoint.
- **Fix (r-t) bypass differently**: Instead of removing it, try making `change_tokens` additive on top of (r-t) with a learnable gate. The cold-start problem with N/S shows the network needs the bypass initially.

### 10c. Pre-training (biggest potential gain)
- **Inria dataset pre-training**: SChanger pre-trains on Inria single-temporal building segmentation before fine-tuning on LEVIR-CD. Our approach: synthetic pairs from two different Inria tiles, XOR of building masks = change GT. Dataset class `InriaPretrainDataset` is already implemented in `dataset.py`.
- Expected gain: ~1pp+ on LEVIR-CD based on SChanger's results.
- Script to write: `pretrain_inria.py` — pre-train model, save checkpoint, then use `--resume` to fine-tune on LEVIR-CD.

### 10d. The gap to SOTA
- Current best: 0.9183 on LEVIR-CD
- Fair SOTA (SChanger): 0.9287
- Gap: **1.04pp**
- SChanger's edge is almost certainly the Inria pre-training — our architecture is otherwise comparable.

---

## 11. Null Results (don't re-run these)

| What | Result | Why |
|------|--------|-----|
| LoRA on DINO encoder | −0.09 to −0.76pp | Encoder features are already excellent frozen |
| Bidirectional cross-attention | −0.31pp | Symmetry doesn't help here |
| Local-window similarity (3×3) | −0.61pp | Hurts even on misregistered datasets |
| Learnable offset per-patch | −3.13pp on S2Looking | 3D parallax not learnable per-patch |
| Soft latent target | null | No gain |
| Lovász latent loss | −0.73pp | |
| SAM2-enc-large as encoder | worse than DINOv2-base | SAM encoder not trained for dense matching |
| Simple diff (no CrossChangeAttention) | ~0.900 | Establishes CrossChangeAttention = +1.5pp |

---

## 12. File Structure Reference

```
galash/
├── model.py                    # Architecture — CrossChangeAttention, Bridge, ChangeDetector
├── train.py                    # Training loop, all CLI args, loss functions
├── dataset.py                  # All dataset loaders incl. InriaPretrainDataset
├── checkpoints/                # SAM2.1 checkpoints (sam2_base_plus, sam2_large, etc.)
├── data/
│   ├── levir_cd/               # train/val/test, A/B/label
│   ├── cdd/
│   ├── dsifn_cd/
│   ├── second/
│   ├── sysu_cd/
│   └── levir_cd_plus/
├── configs/datasets/           # YAML dataset configs (heavy = 256/512² heavy aug)
├── runs/                       # All experiment outputs
│   ├── cosim_base_plus_noamp/  # Config C — baseline
│   ├── cosim_attn_supervise_diaginit/         # Config L
│   ├── cosim_attn_supervise_diaginit_decay/   # Config M ← best attn experiment
│   ├── cosim_no_bypass/                       # Config N
│   ├── cosim_no_bypass_attn_changemap/        # Config S
│   ├── cosim_spatial_refine/                  # Config D
│   ├── cosim_ms_cross_attn/                   # Config E
│   ├── cosim_ms_no_bypass_supervise/          # Config R
│   └── ...
├── scripts/
│   ├── setup_inria.py          # Extract Inria dataset (run once)
│   └── update_backbone_md.py   # Auto-regenerates backbone.md
├── BRIEFING.md                 # ← this file
├── CLAUDE.md                   # Short operational notes
├── all_results.md              # All historical results
├── EXPERIMENTS.md              # Tal's experiment tracking (partially stale)
└── backbone.md                 # Auto-generated backbone comparison table
```

---

## 13. Log CSV Column Reference

```
epoch, lr,
train_loss, train_bce, train_dice, train_iou_loss, train_latent, train_aux_latent, train_attn_loss,
train_precision, train_recall, train_f1, train_iou, train_time,
val_loss, val_bce, val_dice, val_iou_loss, val_latent, val_aux_latent, val_attn_loss,
val_precision, val_recall, val_f1, val_iou, val_oa,
test_loss, test_bce, test_dice, test_iou_loss, test_latent, test_aux_latent, test_attn_loss,
test_precision, test_recall, test_f1, test_iou, test_oa,
best_f1
```

Note: older runs (pre attn_loss column) have 36 columns; newer runs have 39.  
Always find the column by name: `head -1 log.csv | tr ',' '\n' | grep -n '^val_f1$'`

---

## 14. Fair SOTA Reference

| Dataset | Fair SOTA F1 | Method |
|---------|-------------|--------|
| LEVIR-CD | **92.87** | SChanger (2025) — pre-trains on Inria |
| CDD | **97.62** | SChanger (2025) |
| DSIFN-CD | **96.65** | DDPM-CD (2024) |
| SECOND | **72.46** | SAM-SCD (2025) |
| S2Looking | **68.95** | SChanger (2025) |
| LEVIR-CD+ | **87.71** | DDCDNet (2024) |

"Fair" = single-dataset training, no synthetic CD pairs, no joint training.  
SChanger IS fair — only uses Inria single-temporal segmentation for pre-training (not bitemporal CD data).
