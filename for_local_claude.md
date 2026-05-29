# Instructions for Local Claude — Multi-Dataset Sweep on SageMaker

*Written by Natalie's GPU Claude. Read this fully before doing anything.*

---

## What config to run and why

Our best config on LEVIR-CD is **dinov2_rs_base + sam2_large + M flags**. This is what to run on all other datasets.

**Why sam2_large**: Tal's DGX runs showed dinov2_rs_base + sam2_large gets 91.47% on LEVIR-CD vs 90.6% for sam2_enc_large. The RS-pretrained DINOv2 encoder + large SAM2 decoder is the best combination.

**Why M flags**: attn_supervise + diag_init + decay100 adds ~0.3–0.5pp on top of baseline on LEVIR-CD.

---

## The exact training command (template)

```bash
PYTHONPATH=/home/ec2-user/SageMaker/sam2:$PYTHONPATH \
python3 /home/ec2-user/SageMaker/galash/train.py \
    --encoder dinov2_rs_base \
    --decoder sam2_large \
    --config <DATASET_CONFIG> \
    --save_dir /home/ec2-user/SageMaker/galash/runs/cosim_large_attn_supervise_decay \
    --data /home/ec2-user/SageMaker/galash/data \
    --ckpt_dir /home/ec2-user/SageMaker/galash/checkpoints \
    --cnn_skip --finetune_decoder --search_threshold --no_early_stop \
    --no_amp --ema 0.99 \
    --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 100 --attn_diag_init \
    2>&1 | tee /home/ec2-user/SageMaker/galash/train_<DATASET>.log
```

---

## Dataset configs and priority order

Run in this order (best ROI first):

| Priority | Dataset | Config file | Gap to SOTA | SOTA |
|----------|---------|-------------|-------------|------|
| 1 | **CDD** | `configs/datasets/cdd_heavy.yaml` | −0.84pp | 97.62% (SChanger) |
| 2 | **DSIFN-CD** | `configs/datasets/dsifn_cd_heavy.yaml` | +0.11pp already ✅ | 96.65% (DDPM-CD) |
| 3 | **SECOND** | `configs/datasets/second_heavy.yaml` | −0.45pp | 72.46% (SAM-SCD) |
| 4 | **S2Looking** | `configs/datasets/s2looking_heavy.yaml` | −3.24pp | 68.95% (SChanger) |
| 5 | **LEVIR-CD+** | `configs/datasets/levir_cd_plus_1024.yaml` | −4.51pp | 87.71% (DDCDNet) |

**Note on SECOND**: Tal's DGX runs showed `dinov3_base` beats `dinov2_rs_base` on SECOND. Run SECOND with BOTH encoders:
- Run 1: `--encoder dinov2_rs_base` (same as others for fair comparison)
- Run 2: `--encoder dinov3_base` (likely better for SECOND)

---

## Concrete commands (copy-paste ready)

```bash
GALASH=/home/ec2-user/SageMaker/galash
SAM2=/home/ec2-user/SageMaker/sam2
COMMON="--decoder sam2_large --save_dir $GALASH/runs/cosim_large_attn_supervise_decay --data $GALASH/data --ckpt_dir $GALASH/checkpoints --cnn_skip --finetune_decoder --search_threshold --no_early_stop --no_amp --ema 0.99 --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 100 --attn_diag_init"

# CDD (run first)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --encoder dinov2_rs_base --config $GALASH/configs/datasets/cdd_heavy.yaml \
    $COMMON > /tmp/train_cdd.log 2>&1 &

# DSIFN-CD
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --encoder dinov2_rs_base --config $GALASH/configs/datasets/dsifn_cd_heavy.yaml \
    $COMMON > /tmp/train_dsifn.log 2>&1 &

# SECOND (dinov2_rs_base version)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --encoder dinov2_rs_base --config $GALASH/configs/datasets/second_heavy.yaml \
    $COMMON > /tmp/train_second.log 2>&1 &
```

Then after those finish, run S2Looking and LEVIR-CD+.

For SECOND dinov3 version (separate save_dir!):
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --encoder dinov3_base --config $GALASH/configs/datasets/second_heavy.yaml \
    --decoder sam2_large \
    --save_dir $GALASH/runs/cosim_large_attn_supervise_decay_dinov3 \
    --data $GALASH/data --ckpt_dir $GALASH/checkpoints \
    --cnn_skip --finetune_decoder --search_threshold --no_early_stop \
    --no_amp --ema 0.99 \
    --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 100 --attn_diag_init \
    > /tmp/train_second_dinov3.log 2>&1 &
```

---

## SageMaker setup (if not already done)

```bash
cd /home/ec2-user/SageMaker
# Clone (branch with all the code)
git clone -b Natalie_cnn_skip https://github.com/talshaharabany/galash.git
git clone https://github.com/facebookresearch/sam2.git
# Install & download checkpoints/data
bash galash/scripts/setup_sagemaker.sh
```

The setup script downloads SAM2.1 checkpoints and LEVIR-CD automatically. For other datasets, run after setup:
```bash
cd /home/ec2-user/SageMaker/galash
python3 download_datasets.py --out data --datasets cdd dsifn_cd second s2looking levir_cd_plus
```

---

## Results to watch

After each run, check:
```bash
# Best F1 so far
tail -1 /home/ec2-user/SageMaker/galash/runs/cosim_large_attn_supervise_decay/*/log.csv
```

**Targets to beat (fair SOTA, no synthetic data, no joint training):**
- CDD: 97.62%
- DSIFN-CD: 96.65% (we already beat this with baseline!)
- SECOND: 72.46%
- S2Looking: 68.95%
- LEVIR-CD+: 87.71%

---

## What NOT to run

- Do NOT run `sam2_enc_large` as encoder — it's worse than `dinov2_rs_base` (90.60% vs 91.47% on LEVIR-CD)
- Do NOT run without `--no_amp` — dinov3 gets NaN BCE loss at epoch 1 with AMP
- Do NOT use `--no_diff_bypass` — we tested this, it reduces F1 by ~1.8pp
