# Update from Natalie — 2026-04-28

Hey Tal! Big update — lots has happened since the last partner_tal.md. Here's everything.

---

## 0. Branch

I'm working on branch `Natalie_features_Sam` (branched off `feat/encoder-decoder-registry`).
All results and this file are on that branch.

---

## 1. Environment & Setup (unchanged from last time)

- Conda env: `/home/tal/natalie/conda_env` (PyTorch 2.5.1+cu121)
- SAM2 installed at `/home/tal/natalie/galat/sam2`
- All 4 SAM2.1 checkpoints in `checkpoints/`
- All 5 datasets arranged and ready
- `python` on this machine = Python 2.7 — always use `python3` or the conda env binary

Code changes still in place:
- `train.py` — `--seed` flag added
- `scripts/monitor.py` — fixed for local use (ROOT auto-detect, recursive log.csv search)

---

## 2. What I ran — full results table

### 2a. Fair config (dinov2_rs_base, frozen decoder, seed 42/43/44)

| Run | Epochs | TEST F1 | SOTA | Gap | Notes |
|-----|--------|---------|------|-----|-------|
| seed42 × LEVIR-CD | 113 | **0.9018** | 0.9287 | −2.69pp | ✅ done |
| seed43 × LEVIR-CD | 75 | **0.8994** | 0.9287 | −2.93pp | ✅ done |
| seed44 × LEVIR-CD | 71 | **0.8987** | 0.9287 | −3.00pp | ✅ done |
| seed42 × CDD | 300 | **0.9678** | 0.9762 | −0.84pp | ✅ done |
| seed42 × DSIFN-CD | 300 | **0.9676** | 0.9665 | **+0.11pp ✅** | beats SOTA! |
| seed42 × SECOND | 7 | — | 0.7312 | — | ❌ killed early (GPU needed) |
| seed43 × CDD | 236 | — | 0.9762 | ~−1.36pp | ❌ killed (no test result written) |
| seed43 × DSIFN-CD | 245 | — | 0.9665 | ~−0.36pp | ❌ killed (no test result written) |
| seed44 × CDD | 1 | — | 0.9762 | — | ❌ killed very early |
| seed44 × DSIFN-CD | 1 | — | 0.9665 | — | ❌ killed very early |

LEVIR-CD across 3 seeds: mean=0.9000, std=0.0016 — tight variance, good for paper.

### 2b. Cheap-wins config (dinov2_rs_base + FT decoder + EMA 0.999 + threshold search + patience 30)

| Run | Epochs | TEST F1 | SOTA | Gap | Notes |
|-----|--------|---------|------|-----|-------|
| cheap_second | 62 | **0.7147** | 0.7312 | −1.65pp | ✅ done, early stopped |
| cheap_s2looking | 122 | **0.5956** | 0.6932 | −9.76pp | ✅ done, early stopped |
| cheap_cdd | 208 | — | 0.9762 | ~−1.39pp | 🔄 still running, GPU 0 |
| cheap_dsifn_cd | 201 | — | 0.9665 | ~−0.46pp | 🔄 still running, GPU 1 |

**On cheap_second (0.7147):** Your cluster got 0.7209 — we're 0.62pp below. Small dataset
variance + no `--no_amp` flag (not critical for base encoder but may help).

**On cheap_s2looking (0.5956):** Hard dataset. Your champion (dinov3_large) got 0.6571.
With base encoder the ceiling is around 0.60 — cheap wins didn't help much here.

**On cheap_cdd:** val=0.9623 at epoch 208, climbing toward where seed42 fair ended
(val=0.9636). Looking like test will land around 0.967–0.969. Possibly ties or beats seed42.

**On cheap_dsifn_cd:** val=0.9619 at epoch 201, gap ~−0.46pp on val. Given seed42 fair
had val=0.9632 → test=0.9676 (+0.11pp SOTA), cheap_dsifn_cd should land around 0.966–0.968.

---

## 3. Runs NOT done (still missing locally)

| Missing run | Priority | Notes |
|-------------|----------|-------|
| cheap_levir_cd | low | Done on cluster (0.9025), not critical locally |
| cheap_levir_cd_plus | medium | Done on cluster (0.8137). Worth running once GPUs free |
| seed42/43/44 × SECOND | low | You said skip seeds for now |
| seed43/44 × CDD/DSIFN | low | Killed mid-run, would need restart |

---

## 4. GPU situation & infrastructure notes

**dinov3_large is gated** — this machine has no HuggingFace token. All runs here use
`dinov2_rs_base`. To run the champion config locally, someone needs to run:
```bash
huggingface-cli login
```
and paste a token with access to `facebook/dinov3-vitl16-pretrain-lvd1689m`.

**OOM issue with 512px datasets (S2Looking, SECOND, LEVIR-CD+):**
CrossChangeAttention is quadratic in token count. At 512px with patch=16:
(512/16)² = 1024 tokens → 16× more memory than 256px datasets.
After ~90 epochs, PyTorch memory pool fragments (12 GiB reserved but non-contiguous)
and the 8 GiB attention allocation fails.

**Fix applied:** All cheap-wins runs now launched with:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
This lets PyTorch serve large allocations from non-contiguous segments. Zero effect
on results — purely a memory allocator setting.

**Zombie process issue:** `pkill -f train.py` doesn't always kill DataLoader worker
processes. Use `pkill -9 -f train.py` and verify with:
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```
before launching new runs, or they'll OOM immediately.

---

## 5. How to launch runs (canonical commands)

```bash
# Activate env
source /home/tal/natalie/conda_env/bin/activate  # or: conda activate /home/tal/natalie/conda_env

# Cheap-wins training (base encoder)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 nohup \
  /home/tal/natalie/conda_env/bin/python3 train.py \
  --config configs/datasets/<dataset>.yaml \
  --encoder dinov2_rs_base --decoder sam2_base_plus \
  --finetune_decoder --ema 0.999 --search_threshold \
  --patience 30 \
  --save_dir runs/cheap/cheap_<dataset> > /tmp/cheap_<dataset>.log 2>&1 &

# Monitor
python3 scripts/monitor.py          # one-shot
python3 scripts/monitor.py --watch  # refresh every 60s
```

---

## 6. What's next when GPUs free up

1. **Wait for cheap_cdd and cheap_dsifn_cd to finish** — both close to convergence,
   results in a few hours. cheap_dsifn_cd especially interesting (~−0.46pp val gap,
   may extend our SOTA win).

2. **Run cheap_levir_cd_plus on GPU 2** — biggest remaining gap dataset, worth having
   a local result. Launch with `--no_amp` for safety (512px + FT decoder + EMA is heavy).

3. **If you can share an HF token:** run the champion config
   (dinov3_large + cheap wins) on CDD, DSIFN, S2Looking locally to reproduce
   cluster results.

---

## 7. Things to be aware of

- `overfit_test.py` still has stale SAM2 (not SAM2.1) checkpoint defaults — pass
  `--sam2_ckpt` and `--sam2_cfg` explicitly.
- `scripts/fix_levir_cd.py`, `fix_second.py`, `fix_dsifn.py` have `ROOT` hardcoded
  to cluster path — don't run them as-is locally.
- Disk: ~50 GB free. Each cheap-wins run generates ~80–120 MB in `runs/`. Fine for now.
- `seed43_cdd` and `seed43_dsifn_cd` ran 236/245 epochs but have no `test_results.json`
  — they were killed before the test phase. Checkpoints (`best.pt`) are in
  `runs/seeds/seed43_cdd/` and `runs/seeds/seed43_dsifn_cd/` if you want to eval them.

---

Love you! — Natalie 💙
(written with help from Claude on 2026-04-28)
