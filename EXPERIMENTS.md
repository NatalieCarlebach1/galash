# Galash experiments — state and how to run

## Dataset corpus (as of 2026-04-24)

All 7 benchmarks live under `data/` with `{train,val,test}/{A,B,label}` layout.
`_broken/` holds archived copies of the originally-shipped broken versions for reference.

| Dataset | Train | Val | Test | Native tile | Source |
|---|---|---|---|---|---|
| cdd | 10000 | 2998 | 3000 | 256² | Zenodo 13290067 |
| dsifn_cd | 10000 | 2998 | 3000 | 256² | DSIFN GDrive (ChangeDetectionDataset/Real/subset) |
| levir_cd | 7120 | 1024 | 2048 | 256² | LEVIR-CD256 Dropbox (cropped standard split) |
| levir_cd_plus | 637 | 0 (auto-split) | 348 | 1024² | official GDrive |
| s2looking | 3500 | 500 | 1000 | 1024² | GDrive folder |
| second | 2968 | 0 (auto-split) | 1694 | 512² | GDrive; labels generated from semantic `label1 != label2` |
| oscd | 14 | 0 | 10 | variable | RGB preview only |
| oscd_tiled | 75 | 0 | 29 | 512² | derived by `scripts/tile_oscd.py` |

**Missing but reported in literature:** SYSU-CD (manual download from BaiduYun / OneDrive
— see `download_datasets.py` for the links). Not blocking.

## Fix scripts (one-shots)

- `scripts/fix_dsifn.py` — extracts DSIFN `Real/subset` from the gdown RAR
- `scripts/fix_second.py` — handles the nested `SECOND_train_set.rar` + `SECOND_total_test.zip`
  and generates binary change masks from `label1 != label2`
- `scripts/fix_levir_cd.py` — extracts LEVIR-CD-256 and splits by filename prefix
- `scripts/tile_oscd.py` — tiles OSCD city-level images at stride 256 × tile 512

## SLURM infrastructure

- `slurm/train.sbatch` — single-run training template; all parameters via env vars
- `slurm/eval_baseline.sbatch` — per-dataset eval for a given checkpoint
- `slurm/debug_eval.sbatch` — diagnostic harness
- `slurm/launch_sweep.sh <tier>` — launches experiment tiers:
  - `baseline` — one pooled training run on all 6 datasets
  - `per_dataset` — one run per benchmark
  - `encoder_sweep` — fixed decoder, 7 encoder variants on LEVIR-CD
  - `decoder_sweep` — fixed encoder, 6 decoder variants on LEVIR-CD
  - `finetune` — fine-tuned SAM decoder on LEVIR-CD
  - `ablations` — no-TTA / no-latent / no-OHEM on LEVIR-CD
  - `all` — everything (~30 jobs)

Cluster: main partition has 8 nodes, each with 8 GPUs (RTX 6000 Ada, 48 GB). Plenty of headroom.

## Known bug: architecture mismatch in old checkpoint

`runs/20260325_015312/best.pt` reports val_f1 = 0.907, but evaluating it with
current `model.py` yields near-random F1 (0.001 – 0.08).

Cause: `Bridge` was refactored v1 → v2 in commit `cf772ac` (added FPN laterals +
transformer refinement). Bridge now has 140 params; the checkpoint has 60.
`model.bridge.load_state_dict(ckpt["bridge"], strict=False)` silently keeps
100 params randomly initialized.

**Fix:** re-train from scratch with current code (training pipeline itself is fine —
achieves 0.907 val F1 in the saved log). The old checkpoint is abandoned.

## Current active jobs (2026-04-24 ~13:45)

| JobID | Name | Target | Datasets |
|---|---|---|---|
| 52006 | pooled_baseline | DINOv2-RS base + SAM2.1 base+ | all 6 |
| 52007 | pd_levir_cd | DINOv2-RS base + SAM2.1 base+ | LEVIR-CD |
| 52008 | pd_levir_cd_plus | " | LEVIR-CD+ |
| 52009 | pd_s2looking | " | S2Looking |
| 52010 | pd_cdd | " | CDD |
| 52011 | pd_dsifn_cd | " | DSIFN-CD |
| 52012 | pd_second | " | SECOND |

Outputs: `runs/pooled_baseline_<tag>/` and `runs/pd_<dataset>_<tag>/`.

## Next tiers to launch once baselines settle

```bash
./slurm/launch_sweep.sh encoder_sweep   # 7 jobs
./slurm/launch_sweep.sh decoder_sweep   # 6 jobs
./slurm/launch_sweep.sh finetune        # 1 job
./slurm/launch_sweep.sh ablations       # 3 jobs
```

## Per-dataset SOTA bars (late 2025 — see memory/project_sota_targets.md)

| Dataset | SOTA F1 | Method |
|---|---|---|
| LEVIR-CD | 92.87 (92.10 TTP) | SChanger-base |
| LEVIR-CD+ | 91.5 | ChangeStar+Changen |
| S2Looking | 69.32 | UniChange |
| CDD | 97.62 (IoU 91.62) | SChanger / DDPM-CD |
| DSIFN-CD | 96.65 | DDPM-CD |
| SECOND (binary) | 73.12 | UniChange |
