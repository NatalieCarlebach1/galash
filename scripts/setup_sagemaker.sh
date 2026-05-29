#!/bin/bash
# ============================================================
# GALASH — SageMaker Setup Script
# Run this ONCE in a SageMaker terminal after launching an instance.
#
# Recommended instance: ml.g5.xlarge (A10G 24GB, ~$1.01/hr)
# Or: ml.g4dn.xlarge (T4 16GB, cheaper but tight on memory)
#
# Usage:
#   bash setup_sagemaker.sh
# ============================================================

set -e
GALASH=/home/ec2-user/SageMaker/galash
SAM2_DIR=/home/ec2-user/SageMaker/sam2
CKPT=$GALASH/checkpoints
DATA=$GALASH/data

echo "=============================="
echo "  GALASH SageMaker Setup"
echo "=============================="

# ── 1. Clone repos ────────────────────────────────────────────
echo "[1/6] Cloning repos..."
cd /home/ec2-user/SageMaker

if [ ! -d galash ]; then
    git clone -b Natalie_cnn_skip https://github.com/talshaharabany/galash.git
else
    echo "  galash already cloned, pulling latest..."
    cd galash && git pull && cd ..
fi

if [ ! -d sam2 ]; then
    git clone https://github.com/facebookresearch/sam2.git
else
    echo "  sam2 already cloned"
fi

# ── 2. Install Python dependencies ────────────────────────────
echo "[2/6] Installing Python packages..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers huggingface-hub numpy scipy pillow tqdm hydra-core omegaconf iopath

# Install SAM2 as a package
cd $SAM2_DIR
pip install -q -e ".[demo]" 2>/dev/null || pip install -q -e .
cd $GALASH

# ── 3. Download SAM2.1 checkpoints ────────────────────────────
echo "[3/6] Downloading SAM2.1 checkpoints..."
mkdir -p $CKPT
cd $CKPT

download_if_missing() {
    local fname=$1
    local url=$2
    if [ ! -f "$fname" ]; then
        echo "  Downloading $fname ..."
        wget -q --show-progress "$url" -O "$fname"
    else
        echo "  $fname already exists, skipping."
    fi
}

download_if_missing "sam2.1_hiera_base_plus.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

download_if_missing "sam2.1_hiera_large.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

download_if_missing "sam2.1_hiera_small.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"

download_if_missing "sam2.1_hiera_tiny.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

# ── 4. Download DINOv2-RS-base checkpoint ─────────────────────
echo "[4/6] Downloading DINOv2-RS-base checkpoint..."
cd $CKPT
download_if_missing "dinov2-vit-base-remote-sensing-student-backbone.pth" \
    "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/dinov2-vit-base-remote-sensing-student-backbone.pth"

cd $GALASH

# ── 5. Download datasets ──────────────────────────────────────
echo "[5/6] Downloading datasets..."
pip install -q gdown requests tqdm
mkdir -p $DATA
cd $GALASH

# download_datasets.py handles idempotency (skips if already present)
python3 download_datasets.py --out $DATA --datasets levir_cd \
    || echo "  LEVIR-CD auto-download failed — see manual instructions below."

# ── 6. Write launch script ────────────────────────────────────
echo "[6/6] Writing launch script..."
cat > $GALASH/run_training.sh << 'LAUNCH'
#!/bin/bash
# Run a single training job on SageMaker
# Usage: bash run_training.sh [GPU_ID]
GPU=${1:-0}
GALASH=/home/ec2-user/SageMaker/galash
SAM2=/home/ec2-user/SageMaker/sam2

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$SAM2:$PYTHONPATH \
    python3 $GALASH/train.py \
        --encoder dinov2_rs_base \
        --decoder sam2_base_plus \
        --config $GALASH/configs/datasets/levir_cd_heavy.yaml \
        --save_dir $GALASH/runs/cosim_base_plus_sagemaker \
        --data $GALASH/data \
        --ckpt_dir $GALASH/checkpoints \
        --cnn_skip --finetune_decoder --search_threshold \
        --no_early_stop --no_amp --ema 0.99 \
        2>&1 | tee $GALASH/train.log
LAUNCH
chmod +x $GALASH/run_training.sh

echo ""
echo "=============================="
echo "  Setup complete!"
echo "=============================="
echo ""
echo "To run training:"
echo "  bash $GALASH/run_training.sh"
echo ""
echo "To check results:"
echo "  tail -f $GALASH/train.log"
echo "  cat $GALASH/runs/cosim_base_plus_sagemaker/*/log.csv | tail -5"
echo ""
echo "PYTHONPATH for all scripts:"
echo "  export PYTHONPATH=$SAM2_DIR:\$PYTHONPATH"
echo ""

# ── Manual dataset instructions (if auto-download failed) ────
echo "------------------------------------------------------"
echo "If dataset download failed, run manually:"
echo "  cd $GALASH"
echo "  python3 download_datasets.py --list          # see all available"
echo "  python3 download_datasets.py --out $DATA --datasets levir_cd"
echo "  python3 download_datasets.py --out $DATA --all  # all datasets"
echo ""
echo "Expected structure for each dataset:"
echo "  $DATA/<dataset>/"
echo "  ├── train/  A/  B/  label/"
echo "  ├── val/    A/  B/  label/"
echo "  └── test/   A/  B/  label/"
echo "------------------------------------------------------"
