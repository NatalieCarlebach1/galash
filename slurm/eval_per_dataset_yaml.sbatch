#!/bin/bash
#SBATCH --job-name=galash-evalyaml
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/evalyaml_%j.out
#SBATCH --error=logs/slurm/evalyaml_%j.err

set -euo pipefail
cd /home/nfs/tals/galash
source /home/nfs/tals/miniconda3/etc/profile.d/conda.sh
conda activate ns-sam3

# Evaluate a single checkpoint per-dataset, each with its OWN config
# (matching normalisation + test resolution). Used for:
#   - pooled baseline: one checkpoint → 6 per-dataset test F1 numbers
#     (goes in Table 4 "cross-dataset generalisation")
#   - any encoder/decoder sweep variant: comparable per-dataset numbers
#
# Env vars:
#   CKPT    path to best.pt
#   ENCODER encoder name (default dinov2_rs_base)
#   DECODER decoder name (default sam2_base_plus)
#   CONFIGS space-separated list of YAMLs; default = all 6
#   TAG     label for the aggregated JSON output

CKPT="${CKPT:?set CKPT=path/to/best.pt}"
ENCODER="${ENCODER:-dinov2_rs_base}"
DECODER="${DECODER:-sam2_base_plus}"
CONFIGS="${CONFIGS:-configs/datasets/levir_cd.yaml configs/datasets/levir_cd_plus.yaml configs/datasets/s2looking.yaml configs/datasets/cdd.yaml configs/datasets/dsifn_cd.yaml configs/datasets/second.yaml}"
TAG="${TAG:-per_ds}"
FINETUNE_FLAG="${FINETUNE_FLAG:-}"   # set to "--finetune_decoder" if ckpt includes sam_decoder

RUN_DIR=$(dirname "$CKPT")
OUT_JSON="${RUN_DIR}/test_per_dataset_${TAG}.json"

echo "Checkpoint:   $CKPT"
echo "Encoder:      $ENCODER"
echo "Decoder:      $DECODER"
echo "Configs:      $CONFIGS"
echo "Output JSON:  $OUT_JSON"
nvidia-smi | head -20 || true

python3 - "$CKPT" "$OUT_JSON" "$ENCODER" "$DECODER" "$FINETUNE_FLAG" $CONFIGS << 'PY'
import sys, os, json, torch
sys.path.insert(0, '.')

from configurable_aug import load_config, ConfigurableValTransform
from dataset import CDDataset
from eval import evaluate_dataset, ChangeMetrics
from model import ChangeDetector
from torch.utils.data import DataLoader

ckpt_path = sys.argv[1]
out_json  = sys.argv[2]
encoder   = sys.argv[3]
decoder   = sys.argv[4]
ft_flag   = sys.argv[5]          # "--finetune_decoder" or ""
configs   = sys.argv[6:]

device = torch.device('cuda')
finetune = (ft_flag == '--finetune_decoder')

print('Loading model …')
model = ChangeDetector(encoder=encoder, decoder=decoder,
                       finetune_decoder=finetune).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.bridge.load_state_dict(ckpt['bridge'], strict=False)
model.cross_attn.load_state_dict(ckpt['cross_attn'], strict=True)
if finetune and 'sam_decoder' in ckpt:
    model.sam_decoder.load_state_dict(ckpt['sam_decoder'])
print(f'  loaded epoch={ckpt.get("epoch","?")} best_f1={ckpt.get("best_f1","?")}')
model.eval()

results = {}
for yaml_path in configs:
    cfg = load_config(yaml_path)
    ds_name = cfg['dataset']
    split_dir = os.path.join('data', ds_name, 'test')
    if not os.path.isdir(os.path.join(split_dir, 'A')):
        print(f'  {ds_name}: no test split, skipping')
        continue

    val_tf = ConfigurableValTransform(cfg)
    ds = CDDataset(split_dir, transform=val_tf, name=ds_name)
    if len(ds) == 0:
        continue

    test_size = int(cfg.get('test_img_size', cfg.get('img_size', 512)))
    print(f'\nEvaluating {ds_name}  (test_img_size={test_size}, {len(ds)} pairs) …')

    r = evaluate_dataset(
        model, ds_name, split_dir,
        img_size=test_size, batch_size=4, num_workers=4,
        device=device, threshold=0.5, save_dir=None, use_tta=True,
    )
    results[ds_name] = r
    print(f"  {ds_name}: F1={r['f1']:.4f}  IoU={r['iou']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

# Aggregate
if len(results) > 1:
    agg = ChangeMetrics()
    for r in results.values():
        agg.tp += r['tp']; agg.fp += r['fp']; agg.fn += r['fn']; agg.tn += r['tn']
    results['AGGREGATE'] = agg.compute()

# Save
with open(out_json, 'w') as f:
    json.dump({k: {kk: (int(vv) if isinstance(vv, (int,)) else float(vv))
                   for kk, vv in v.items()} for k, v in results.items()},
              f, indent=2, default=float)
print(f'\nSaved {out_json}')
PY
