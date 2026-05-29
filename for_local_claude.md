# Instructions for Local Claude — Multi-Dataset Sweep on SageMaker

*Written by Natalie's GPU Claude. Read this fully before doing anything.*

---

## Your job

You are running on a Windows machine with no GPU. Your task is to:
1. Launch (or connect to) a SageMaker notebook instance
2. SSH into it
3. Run the training sweep for all change-detection datasets

---

## Step 1 — Launch / connect to SageMaker instance

**Recommended instance**: `ml.g5.xlarge` (A10G 24GB, ~$1.01/hr) or `ml.g5.2xlarge` for more RAM.

### Option A — AWS console (manual)
Go to AWS Console → SageMaker → Notebook instances → start your instance → open terminal.

### Option B — AWS CLI (automated)
```bash
# Check if instance exists and get status
aws sagemaker describe-notebook-instance --notebook-instance-name galash-training --region us-east-1

# Start it if stopped
aws sagemaker start-notebook-instance --notebook-instance-name galash-training --region us-east-1

# Wait until InService, then get the URL
aws sagemaker describe-notebook-instance --notebook-instance-name galash-training --region us-east-1 --query 'Url'
```

### Option C — SSH directly
If you have the instance public DNS and key file:
```bash
ssh -i ~/.ssh/your-key.pem ec2-user@<instance-public-dns>
```

---

## Step 2 — First-time setup (run once)

Open a terminal on the SageMaker instance (via JupyterLab → Terminal, or SSH) and run:

```bash
cd /home/ec2-user/SageMaker

# Clone the repo (branch with all current code)
git clone -b Natalie_cnn_skip https://github.com/talshaharabany/galash.git
git clone https://github.com/facebookresearch/sam2.git

# Install everything and download checkpoints
bash galash/scripts/setup_sagemaker.sh
```

The setup script installs Python packages, SAM2.1 checkpoints, and LEVIR-CD dataset automatically.

Then download the other datasets:
```bash
cd /home/ec2-user/SageMaker/galash
python3 download_datasets.py --out data --datasets cdd dsifn_cd second s2looking levir_cd_plus sysu_cd
```

---

## Step 3 — Run the training sweep

**Config to use for all datasets**: `dinov2_rs_base + sam2_large + M flags`

This is our best configuration. Details:
- Encoder: `dinov2_rs_base` (remote-sensing pretrained DINOv2)
- Decoder: `sam2_large` (225M param SAM2 mask decoder)
- Flags: `--attn_supervise --attn_supervise_decay_epochs 100 --attn_diag_init` (Config M)
- Standard: `--cnn_skip --finetune_decoder --search_threshold --no_early_stop --no_amp --ema 0.99`

### Run all datasets (one per GPU — g5.xlarge has 1 GPU, run sequentially):

```bash
GALASH=/home/ec2-user/SageMaker/galash
SAM2=/home/ec2-user/SageMaker/sam2
SAVE=$GALASH/runs/cosim_large_attn_supervise_decay
COMMON="--encoder dinov2_rs_base --decoder sam2_large --save_dir $SAVE --data $GALASH/data --ckpt_dir $GALASH/checkpoints --cnn_skip --finetune_decoder --search_threshold --no_early_stop --no_amp --ema 0.99 --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 100 --attn_diag_init"

# 1. CDD
PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --config $GALASH/configs/datasets/cdd_heavy.yaml $COMMON \
    > /tmp/train_cdd.log 2>&1 &
echo "CDD PID: $!"
# wait for it to finish, then run next:
wait

# 2. DSIFN-CD
PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --config $GALASH/configs/datasets/dsifn_cd_heavy.yaml $COMMON \
    > /tmp/train_dsifn.log 2>&1 &
echo "DSIFN PID: $!"
wait

# 3. SECOND (dinov2_rs_base)
PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --config $GALASH/configs/datasets/second_heavy.yaml $COMMON \
    > /tmp/train_second.log 2>&1 &
echo "SECOND PID: $!"
wait

# 4. SECOND again with dinov3_base (likely better for SECOND — run separately)
PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --encoder dinov3_base --decoder sam2_large \
    --config $GALASH/configs/datasets/second_heavy.yaml \
    --save_dir $GALASH/runs/cosim_large_attn_supervise_decay_dinov3 \
    --data $GALASH/data --ckpt_dir $GALASH/checkpoints \
    --cnn_skip --finetune_decoder --search_threshold --no_early_stop \
    --no_amp --ema 0.99 \
    --attn_supervise --w_attn_supervise 1.0 --attn_supervise_decay_epochs 100 --attn_diag_init \
    > /tmp/train_second_dinov3.log 2>&1 &
wait

# 5. S2Looking
PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --config $GALASH/configs/datasets/s2looking_heavy.yaml $COMMON \
    > /tmp/train_s2looking.log 2>&1 &
wait

# 6. LEVIR-CD+ (hardest, largest gap)
PYTHONPATH=$SAM2:$PYTHONPATH nohup python3 $GALASH/train.py \
    --config $GALASH/configs/datasets/levir_cd_plus_1024.yaml $COMMON \
    > /tmp/train_levirplus.log 2>&1 &
wait
```

### To monitor progress of any run:
```bash
tail -f /tmp/train_cdd.log
# or check F1 per epoch:
tail -5 /home/ec2-user/SageMaker/galash/runs/cosim_large_attn_supervise_decay/*/log.csv
```

---

## Priority order

| Priority | Dataset | Expected gap | Note |
|----------|---------|-------------|------|
| 1 | **CDD** | −0.84pp | Most achievable |
| 2 | **DSIFN-CD** | already +0.11pp ✅ | Just validate |
| 3 | **SECOND** | −0.45pp | Run dinov2 AND dinov3 |
| 4 | **S2Looking** | −3.24pp | Harder |
| 5 | **LEVIR-CD+** | −4.51pp | Hardest, run last |

Fair SOTA targets (no synthetic data, no joint training):
- CDD: **97.62%** (SChanger)
- DSIFN-CD: **96.65%** (DDPM-CD) — already beaten
- SECOND: **72.46%** (SAM-SCD)
- S2Looking: **68.95%** (SChanger)
- LEVIR-CD+: **87.71%** (DDCDNet)

---

## What NOT to do

- Do NOT use `--encoder sam2_enc_large` — worse than dinov2_rs_base (90.60% vs 91.47% on LEVIR-CD)
- Do NOT remove `--no_amp` — dinov3 gets NaN loss at epoch 1 with AMP
- Do NOT add `--no_diff_bypass` — tested, reduces F1 by ~1.8pp
- Do NOT forget `PYTHONPATH=$SAM2:$PYTHONPATH` — training will crash without it

---

## Stop the instance when done

```bash
aws sagemaker stop-notebook-instance --notebook-instance-name galash-training --region us-east-1
```
