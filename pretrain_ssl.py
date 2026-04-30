"""Bitemporal contrastive SSL fine-tuning of DINO encoders.

For each (ref, tgt) pair from the pooled CD train+val splits:
  1. Forward both through DINO (paired augmentation already applied via ValTransform).
  2. Compute per-patch contrastive loss:
     positive: ref[b, i] ↔ tgt[b, i] (same batch element, same spatial patch)
     negatives: tgt patches at OTHER batch elements (cross-image)
  3. Hard negatives within the same image (different patch) are masked out
     so the encoder isn't punished for representing nearby unchanged patches similarly.

Symmetric InfoNCE (ref→tgt and tgt→ref).

LoRA injection on DINO Q/K/V/O attention by default (rank=8). Use --full_ft for
full encoder unfreezing (riskier).

Output: a checkpoint per epoch under save_dir, containing the DINO state_dict.
Load downstream via train.py --dino_pretrained <path>.

Usage (single encoder, all datasets):
  python pretrain_ssl.py --encoder dinov3_large \
      --epochs 10 --batch 16 --lr 1e-5 --lora_rank 8 \
      --save_dir runs/ssl_pretrain/dinov3_large
"""

import argparse, os, time, random
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from model import ChangeDetector, inject_lora

# Cover both HF ViTModel naming AND DINOv3-style naming.
DINO_LORA_TARGETS_ALL = (
    # HF ViT naming
    "attention.query", "attention.key", "attention.value", "attention.output.dense",
    # DINOv3 / DINOv2 naming
    "q_proj", "k_proj", "v_proj", "o_proj",
)
from dataset import CDDataset, ValTransform


class BitemporalSSLDataset(Dataset):
    """Returns (ref, tgt) only — drops mask. Wraps an existing CDDataset."""
    def __init__(self, cd_ds): self.ds = cd_ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        ref, tgt, _ = self.ds[idx]
        return ref, tgt


def info_nce_bitemporal(ref_tok, tgt_tok, temperature=0.1):
    """InfoNCE on patch tokens with same-image hard-negative masking.
    Inputs: [B, S, D] L2-normalized later.
    """
    B, S, D = ref_tok.shape
    ref = F.normalize(ref_tok, dim=-1).reshape(B * S, D)
    tgt = F.normalize(tgt_tok, dim=-1).reshape(B * S, D)

    logits = (ref @ tgt.t()) / temperature                        # [BS, BS]
    labels = torch.arange(B * S, device=ref.device)

    # Mask same-image-different-patch as hard negatives we ignore
    # (block-diagonal True except true positive on diagonal).
    block_idx = torch.arange(B * S, device=ref.device) // S       # [BS]
    same_image = (block_idx.unsqueeze(0) == block_idx.unsqueeze(1))  # [BS, BS]
    same_image.fill_diagonal_(False)
    logits = logits.masked_fill(same_image, float('-inf'))

    loss_r2t = F.cross_entropy(logits, labels)
    loss_t2r = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_r2t + loss_t2r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', required=True)
    p.add_argument('--decoder', default='sam2_base_plus',
                   help='Required to build ChangeDetector but not used for SSL.')
    p.add_argument('--datasets', nargs='+',
                   default=['levir_cd', 'levir_cd_plus', 's2looking', 'cdd', 'dsifn_cd', 'second'])
    p.add_argument('--data_root', default='data')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--lora_rank', type=int, default=8)
    p.add_argument('--lora_alpha', type=float, default=16.0)
    p.add_argument('--full_ft', action='store_true',
                   help='Full DINO unfreeze (no LoRA). Risky.')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--save_dir', default='runs/ssl_pretrain')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Pool bitemporal pairs across all 6 datasets, train+val splits ──
    tf = ValTransform(img_size=args.img_size)
    parts = []
    print('Building pooled bitemporal SSL dataset:')
    for ds_name in args.datasets:
        for split in ['train', 'val']:
            split_dir = os.path.join(args.data_root, ds_name, split)
            if os.path.isdir(os.path.join(split_dir, 'A')):
                cd = CDDataset(split_dir, transform=tf, name=ds_name)
                if len(cd) > 0:
                    parts.append(BitemporalSSLDataset(cd))
                    print(f'  {ds_name}/{split:<5}: {len(cd)} pairs')
    pooled = ConcatDataset(parts)
    print(f'  TOTAL: {len(pooled)} bitemporal pairs')
    loader = DataLoader(pooled, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    # ── Build minimal ChangeDetector to load DINO ──
    print(f'\nBuilding model with encoder={args.encoder} (decoder built but unused)...')
    cd = ChangeDetector(encoder=args.encoder, decoder=args.decoder).to(device)
    # Freeze everything except DINO
    for p_ in cd.parameters():
        p_.requires_grad = False

    if args.full_ft:
        for p_ in cd.dino.parameters():
            p_.requires_grad = True
        n_train = sum(p_.numel() for p_ in cd.dino.parameters())
        print(f'  Full DINO unfreeze: {n_train/1e6:.1f}M trainable params')
    else:
        n_lora = inject_lora(cd.dino, DINO_LORA_TARGETS_ALL,
                             rank=args.lora_rank, alpha=args.lora_alpha)
        cd.dino.to(device)  # newly-added LoRA modules need to be moved to device
        n_train = sum(p_.numel() for p_ in cd.dino.parameters() if p_.requires_grad)
        print(f'  LoRA injected: {n_lora} layers, rank={args.lora_rank}')
        print(f'  LoRA trainable: {n_train/1e6:.2f}M params')

    trainable = [p_ for p_ in cd.dino.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler('cuda', enabled=not args.no_amp and device.type == 'cuda')

    os.makedirs(args.save_dir, exist_ok=True)

    print(f'\nTraining for {args.epochs} epochs, batch={args.batch}, lr={args.lr}, '
          f'temp={args.temperature}\n')

    for ep in range(args.epochs):
        t0 = time.time()
        running = 0.0; n_steps = 0
        cd.train()  # only LoRA / DINO will actually update
        for ref, tgt in loader:
            ref, tgt = ref.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                # Inline DINO forward (model._dino_forward_pair is @torch.no_grad).
                B = ref.size(0)
                both = torch.cat([ref, tgt], dim=0)
                out = cd.dino(pixel_values=both)
                last = out.hidden_states[-1]
                n_skip = 1 + int(getattr(cd.dino.config, 'num_register_tokens', 0))
                last = last[:, n_skip:, :]
                ref_tok, tgt_tok = last[:B], last[B:]
                loss = info_nce_bitemporal(ref_tok, tgt_tok,
                                           temperature=args.temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt); scaler.update()
            running += loss.item(); n_steps += 1
            if n_steps % 50 == 0:
                print(f'  [ep {ep} step {n_steps}/{len(loader)}] loss={running/n_steps:.4f}  '
                      f'({time.time()-t0:.0f}s)')

        avg = running / max(1, n_steps)
        elapsed = time.time() - t0
        print(f'[ep {ep}] avg_loss={avg:.4f}  time={elapsed:.0f}s\n')

        # Save full DINO state_dict (works whether LoRA or full FT)
        ckpt_path = os.path.join(args.save_dir, f'ssl_dino_ep{ep:03d}.pt')
        torch.save({
            'epoch': ep,
            'avg_loss': avg,
            'dino_state': cd.dino.state_dict(),
            'encoder': args.encoder,
            'lora_rank': args.lora_rank,
            'full_ft': args.full_ft,
            'args': vars(args),
        }, ckpt_path)
    # final symlink
    final = os.path.join(args.save_dir, 'ssl_dino_final.pt')
    if os.path.islink(final): os.unlink(final)
    os.symlink(f'ssl_dino_ep{args.epochs-1:03d}.pt', final)
    print(f'Done. Final checkpoint: {final}')


if __name__ == '__main__':
    main()
