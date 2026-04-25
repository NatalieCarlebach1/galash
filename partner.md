# GALASH — partner collaboration guide

Hi 💙 — this file is the master overview. There's also `partner_tal.md` (your
notes back to me) and `EXPERIMENTS.md` (single-source-of-truth progress doc).

**Latest update (Tal): 2026-04-25 mid-day** — read §0 first, it's the catch-up.
Then the rest of the file is the original setup guide; update yourself when
something changes.

---

## 0. What's new since you pushed `partner_tal.md` (read this first!)

### Your work (commit `3c606fa`) — landed cleanly ✅

- `--seed` flag in train.py — used in all my new sweeps too 🙏
- monitor.py local-path fix — I'm using your version now
- partner_tal.md with your environment + Tier-A setup + interim seed-42 results

Your **seed42 × levir_cd** at val 0.8964 (epoch 34) is right on track to match
my fair-config baseline (0.9045 test). Looks great.

### My new commits + code changes you should know about

Branch is now at `93d3b51`+. Pull to get:

| Commit | What |
|---|---|
| `d8e3489` | Paper Tables 1–6 filled with real numbers (`paper/sections/*.tex`) |
| `93d3b51` | `EXPERIMENTS.md` — full progress audit + 19-item TODO table |
| (uncommitted, in flight) | Tier-1 latent-space ablations: bidir / local-window / lovasz / mse / multi-scale aux |

**New CLI flags in `train.py`** (all default to "off" → backward compatible):

| Flag | What |
|---|---|
| `--latent_soft` | Use continuous fractional density instead of `>0.3` binary target for the patch-level latent loss |
| `--w_aux_latent W` | Weight on multi-scale aux deep-supervision heads in the bridge (3 scales: 64/128/256) |
| `--bidir_attn` | Bidirectional cross-attention (avg ref→tgt and tgt→ref) — symmetry of binary CD |
| `--local_window K` | Local-window similarity in CrossChangeAttention (window K×K around diagonal). K=1 = current behaviour |
| `--latent_loss {bce,mse,lovasz}` | Loss on the patch-level latent map. `lovasz` directly optimises IoU |
| `--ema 0.999` | EMA shadow weights, used for val + saved as best.pt |
| `--search_threshold` | At final test eval, sweep threshold ∈ [0.30, 0.70] on val and apply best to test |

**New CLI flags in `eval.py`**:

| Flag | What |
|---|---|
| `--use_ema` | Load `ema_shadow` weights from the checkpoint (if present) |
| `--search_threshold` | Sweep threshold on val, apply best to test |
| `--test_img_sizes 256 384 512` | Multi-resolution test, picks best (size, threshold) by val F1 |

**Architectural changes (model.py):**
- `CrossChangeAttention` now accepts `bidirectional`, `local_window` kwargs.
- `Bridge` now exposes 3 auxiliary 1×1 conv heads (always active in forward; the loss decides whether to use them).
- `ChangeDetector.forward` returns a **4-tuple** `(masks, iou, change_map, aux_change_maps)` — the 4th element is a list of aux logit maps for deep supervision.
- DINOv3 / DINOv2-with-registers now work — fixed register-token stripping.
- DINOv1 added to encoder registry (but pos-embed at 224² needs interpolation fix to train at 256² — known TODO).

### Current SOTA-comparison results

We have **2 SOTA-beating numbers** so far:

| Dataset | Our best | SOTA | Gap |
|---|---|---|---|
| **DSIFN-CD** | **0.9674** | 0.9665 (DDPM-CD) | **+0.09 pp ✅** |
| **SECOND** | **0.7320** | 0.7312 (UniChange) | **+0.08 pp ✅** |
| LEVIR-CD | **0.9089** | 0.9287 (SChanger) | −1.98 pp |
| CDD | 0.9675 | 0.9762 (SChanger) | −0.87 pp |
| LEVIR-CD+ | 0.8172 | 0.9150 (ChangeStar+Changen) | −9.78 pp |
| S2Looking | 0.6014 | 0.6932 (UniChange) | −9.18 pp |

**Best LEVIR-CD recipe so far:** `dinov2_rs_base + sam2_tiny` + post-hoc multi-res
(picks size 384) + val-tuned threshold 0.55 = **0.9089**.

The encoder leader is `dinov2_large_reg` (0.9057 on LEVIR-CD), the runner-up
is `dinov3_large` (0.9025). DINOv2-RS-base is 3rd at 0.9000.

**Surprising decoder finding:** SAM2.1-tiny (39 M) **beats** SAM2.1-large
(225 M) by 0.33 F1. Decoder capacity is essentially irrelevant.

### Currently running on the cluster (~50 jobs, 28 R / 16 PD)

| Sweep | Count | Status |
|---|---|---|
| 10-job latent-loss hyperparam sweep (`w_latent` × `latent_soft`) | 10 | mostly converged ~ep 35-50 |
| 5-job multi-scale aux loss sweep (`w_aux_latent` ∈ {0, 0.1, 0.2, 0.5} + soft combo) | 5 | most converged ~ep 70 |
| **5-job Tier-1 sweep** (bidir / lw3 / lovasz / mse / **combo**) | 5 | just submitted, pending |
| sam2_tiny on remaining 5 datasets | 5 | started |
| dinov3_large on remaining 5 datasets | 5 | pending dgx |
| Largest backbones (dinov2_giant, dinov2_giant_reg, dinov3_huge) on LEVIR-CD | 3 | pending dgx |
| dinov2_large_reg + cheap-wins | 2 | pending dgx |
| SAM1 decoders (vit_b/l/h) with `--no_amp` | 3 | pending dgx |
| Pooled baseline (all 6 datasets) | 1 | pending main |
| Plus running encL_v2_large_reg / cheap_v3sat / medical jobs | ~7 | running |

**Most paper-impactful in this batch**: the Tier-1 combo (52171), the largest
backbones (dinov2_giant, dinov3_huge), and the pooled baseline (unlocks the
cross-dataset generalization Table 4).

### Suggestions for your seeded runs given everything above

You're already running seed 42 on 3 datasets — keep going! When seed 42 finishes
all 4, please launch seeds 43 and 44 with the same config. Once seed 44 lands,
we have variance bars for SECOND and DSIFN — directly addresses reviewer concerns
about whether `+0.09 pp above SOTA` is real or seed noise. **Highest paper-value
work you can do right now.**

If you want to add to it: try `--seed 42 --bidir_attn --local_window 3 --latent_loss lovasz`
on LEVIR-CD once Tier-1 results land (let me confirm which combo wins). That gives
us the seed-42 reproduction of our champion config.

---

## 1. (Original §1) — TL;DR — what is this project?

**GALASH** = a change-detection pipeline for aerial / satellite image pairs:

```
  ref image ─┐
             ├─→ DINOv2/v3 (frozen)  ──→  Bridge v2        ──┐
  tgt image ─┘   multi-scale features    (trainable 11M)     ├→ SAM2.1 decoder → change mask
                                         CrossChangeAttn ────┘    (frozen or fine-tuned)
```

**Our pitch:** compose two frozen foundation models — a DINO encoder and a SAM
decoder — with only a tiny trainable bridge. Only ~16.5 M trainable params on
top of 300 M+ frozen backbones. Aiming for **BMVC** (paper scaffold in `paper/`).

**Novel pieces:**
1. **Unified encoder/decoder registry** (16 encoders × 8 decoders in `model.py`).
2. **Cross-change attention** with learnable temperature.
3. **Bridge v2** — FPN + transformer refinement; repurposes SAM's normally-unused
   dense-prompt slot as a change conditioning signal.
4. **Per-dataset fair augmentation configs** (`configs/datasets/*.yaml`) matching
   each benchmark's published SOTA. Enforced fair comparison.
5. (NEW) **Multi-scale auxiliary deep supervision** at 3 bridge scales
   (`--w_aux_latent`).
6. (NEW) **Bidirectional + local-window cross-attention** (`--bidir_attn --local_window K`)
   — better symmetry, robust to misregistration. *Currently being ablated.*
7. (NEW) **Lovász latent loss** (`--latent_loss lovasz`) — direct IoU optimization
   on the patch-level supervision.

---

## 2. Setup on your machine — *unchanged from §4 of original, see partner_tal.md for actual setup notes*

You've already got everything working. If you ever need to redo:

```bash
git checkout feat/encoder-decoder-registry
git pull
# Already pip-installed: torch, sam2 (-e), segment-anything, gdown, rarfile, py7zr, pyyaml
# Already downloaded: 4 SAM2.1 checkpoints + 5 datasets
# Smoke test:
python overfit_test.py --n 8 --steps 300 \
  --sam2_ckpt checkpoints/sam2.1_hiera_tiny.pt \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_t.yaml
```

(Reminder: `python` on your box is Py2; use `python3` or your conda binary.)

---

## 3. Experiments to run — UPDATED priority list

**Ground rules:** I'm running ~50 concurrent jobs on the cluster covering
encoder/decoder/cheap-wins/Tier-1 sweeps. **Don't duplicate what I'm running.**
The big gaps that only you can fill (because variance + multi-seed needs a
different machine) are below.

### 🔥 Priority A — multi-seed for SOTA-beating runs (highest paper impact)

You're already doing this for seed 42 — just keep going.

**Goal:** 3 seeds × 4 datasets = 12 runs. Gives us mean ± std for the two SOTA
claims (DSIFN +0.09, SECOND +0.08) and our two close-runner-ups (LEVIR-CD,
CDD). Reviewers will ask for variance bars on `+0.09`-style claims and we'll
have them.

```bash
for SEED in 42 43 44; do
  for DS in levir_cd cdd dsifn_cd second; do
    python train.py \
      --config configs/datasets/${DS}.yaml \
      --encoder dinov2_rs_base --decoder sam2_base_plus \
      --seed $SEED \
      --save_dir runs/seeds/seed${SEED}_${DS}
  done
done
```

(You've already got seed 42 × {levir_cd, cdd, dsifn_cd} running. Add SECOND
when GPU 2 frees up, then move to seeds 43 and 44.)

### 🔥 Priority B — Champion config when Tier-1 results land (~3 hours from now)

Once my Tier-1 combo (job 52171) finishes — `--bidir_attn --local_window 3
--latent_loss lovasz` — and assuming it beats baseline, we want a seed run of
the champion config to confirm it's real. **Wait for me to confirm before launching.**

### Priority C — fill in benchmark gaps I haven't touched

Currently no one is running:
- **`dinov2_rs_base + sam2_tiny + cheap-wins` on LEVIR-CD+ + S2Looking** (seed 42, would round out Table 1's "+ cheap wins" row)
- **Tier-D ablations** on LEVIR-CD: `--w_latent 0`, `--no_ohem`, no-tta — though my cluster has these. Check if they've finished test eval before duplicating.

### Skip (I'm covering)

- Encoder sweep (running on dgx)
- Decoder sweep (running on main)
- Pooled baseline (just submitted on main)
- DINOv3-sat per-dataset (running on dgx)
- Latent-loss / multi-scale aux sweeps (running on main)

---

## 4. How to sync results

Same as before:

```bash
cd ~/galash
git pull origin feat/encoder-decoder-registry
# … run training …
git add runs/seeds/*/test_results.json runs/seeds/*/log.csv runs/seeds/*/config.json
git commit -m "partner: seed${SEED} × ${DS} done, test_f1=${F1}"
git push origin feat/encoder-decoder-registry
```

Note `.gitignore` excludes `runs/`. Force-add the JSON+CSV files explicitly with
`git add -f` if needed, OR add a .gitignore exception for `runs/seeds/*.json`.

---

## 5. Tools you'll want

```bash
# Live monitor with SOTA gap (your fixed version, works locally now):
python scripts/monitor.py --watch

# After a run finishes, post-hoc cheap wins (multi-res + threshold search):
python eval.py --encoder dinov2_rs_base --decoder sam2_base_plus \
  --checkpoint runs/seeds/seed42_levir_cd/<timestamp>/best.pt \
  --datasets levir_cd \
  --test_img_sizes 256 384 512 --search_threshold --tta
```

The cheap-wins multi-res lifts LEVIR-CD test F1 by ~+0.4 to +0.7 pp at zero
training cost — apply to every finished checkpoint.

---

## 6. File index — what's where

| Path | What |
|---|---|
| `model.py` | Encoder/decoder registry + ChangeDetector + Bridge v2 (now with aux heads) + CrossChangeAttention (now with bidir + local-window) |
| `train.py` | Training loop. Has `--seed` (you added), `--ema`, `--search_threshold`, `--latent_soft`, `--w_aux_latent`, `--bidir_attn`, `--local_window`, `--latent_loss` |
| `eval.py` | Eval. Has `--search_threshold`, `--test_img_sizes`, `--use_ema` |
| `dataset.py` | Dataset loaders. `build_loaders(config=...)` wires into YAML |
| `configurable_aug.py` | YAML-driven augmentation pipeline (10 ops, matches SOTA papers per-dataset) |
| `configs/datasets/*.yaml` | 6 per-benchmark fair-comparison configs |
| `scripts/monitor.py` | Live SOTA-gap monitor (your fixed version) |
| `scripts/fix_*.py` | Dataset repair scripts (DSIFN, SECOND, LEVIR-CD) |
| `slurm/*.sbatch` | SLURM templates (cluster only) |
| `slurm/launch_sweep.sh <tier>` | Submit experiment tiers |
| `paper/` | BMVC LaTeX source — Tables 1–6 now filled with real numbers |
| **`EXPERIMENTS.md`** | **Full progress + 19-item TODO table — read this for what's next** |
| `partner.md` | This file (master guide) |
| `partner_tal.md` | Your notes from yesterday |

---

## 7. What I'm doing right now

1. Implementing **Tier 2** improvements: difference-feature branch, attention entropy, deeper multi-scale aux. Will queue once Tier 1 lands.
2. **Writing up the paper** — Tables 1–6 are filled, Tables 4 (cross-dataset gen) and 7 (encoder×decoder matrix) still pending data.
3. **Architecture figure** still TODO (`paper/figures/architecture.pdf` is a `\fbox{}` placeholder).

If your seed-42 SECOND finishes before I'm awake, post the test F1 to me — that's
the single most valuable number you can produce right now (it's the second SOTA
claim and currently single-seed).

---

Love you ❤️ — keep going!

— Tal (via Claude Opus)
