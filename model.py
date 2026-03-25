"""
Change Detection Pipeline  v2
──────────────────────────────
ref image ─┐                          ┌─ Bridge v2 (trainable) ──┐
           ├→ DINOv2/v3 (frozen) ─────┤  FPN + residual + xformer├→ SAM2.1 Decoder → mask + IoU
tgt image ─┘   multi-scale features   └─ CrossChangeAttn ────────┘
                                          (trainable, temp.)    ↗ small change map (aux loss)

v2 changes:
  - SAM2.1 family support (t/s/b+/l) with optional decoder fine-tuning
  - Improved Bridge: residual blocks, FPN top-down fusion, transformer refinement
  - TTA (test-time augmentation) support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# SAM2.1 variant registry
# ---------------------------------------------------------------------------
SAM2_VARIANTS = {
    "tiny":      {"ckpt": "sam2.1_hiera_tiny.pt",      "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml"},
    "small":     {"ckpt": "sam2.1_hiera_small.pt",     "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml"},
    "base_plus": {"ckpt": "sam2.1_hiera_base_plus.pt", "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml"},
    "large":     {"ckpt": "sam2.1_hiera_large.pt",     "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml"},
}


# ---------------------------------------------------------------------------
# Cross-Attention with learnable temperature  (TRAINABLE)
# ---------------------------------------------------------------------------
class CrossChangeAttention(nn.Module):
    """Cross-attention between reference and target patch tokens.

    Produces:
        change_tokens – per-patch change representation
        change_map    – [B, S] soft change score (the "small map" for aux loss)
    """

    def __init__(self, dim: int, num_heads: int = 8, init_temperature: float = 0.07):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # learnable inverse-temperature (log-space for stable optimisation)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ref_tokens: Tensor, tgt_tokens: Tensor):
        B, S, D = ref_tokens.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(ref_tokens).view(B, S, H, d).transpose(1, 2)  # [B,H,S,d]
        k = self.k_proj(tgt_tokens).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(tgt_tokens).view(B, S, H, d).transpose(1, 2)

        temp = self.log_temp.exp().clamp(min=1.0)
        attn_logits = (q @ k.transpose(-2, -1)) * (temp / d**0.5)
        attn_weights = attn_logits.softmax(dim=-1)  # [B,H,S,S]

        attended = (attn_weights @ v).transpose(1, 2).reshape(B, S, D)
        attended = self.out_proj(attended)

        # change = how the target *differs* from the reference
        change_tokens = self.norm(attended - ref_tokens)

        # small map: 1 − diagonal-similarity (high → more change)
        avg_attn = attn_weights.mean(dim=1)  # [B,S,S]
        similarity = torch.diagonal(avg_attn, dim1=1, dim2=2)  # [B,S]
        change_map = 1.0 - similarity

        return change_tokens, change_map


# ---------------------------------------------------------------------------
# Building blocks for Bridge v2
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Conv residual block with GroupNorm + GELU."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class TransformerRefinement(nn.Module):
    """Lightweight spatial self-attention for refining bridge output."""
    def __init__(self, dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C, H, W] → [B, C, H, W]"""
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).view(B, C, H, W)


# ---------------------------------------------------------------------------
# Bridge v2: FPN + residual + transformer refinement  (TRAINABLE)
# ---------------------------------------------------------------------------
class Bridge(nn.Module):
    """Projects multi-scale DINOv2 features → SAM2-compatible tensors.

    v2 improvements:
        - Residual blocks in each scale projection
        - FPN-style top-down fusion (high-level semantics inform low-level)
        - Transformer refinement before SAM decoder

    Returns
        image_embeddings : [B, 256, th, tw]    for mask decoder
        dense_prompt     : [B, 256, th, tw]    change-aware prompt
        high_res_features: [feat_s0, feat_s1]  for SAM2 upsampling path
                           feat_s0: [B, 32, 4*th, 4*tw]
                           feat_s1: [B, 64, 2*th, 2*tw]
    """

    def __init__(self, dino_dim: int = 768, sam_dim: int = 256, num_scales: int = 4):
        super().__init__()
        self.sam_dim = sam_dim

        # Per-scale linear projection + residual conv refinement
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dino_dim, sam_dim),
                nn.LayerNorm(sam_dim),
                nn.GELU(),
                nn.Linear(sam_dim, sam_dim),
            )
            for _ in range(num_scales)
        ])
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(sam_dim),
                ResidualBlock(sam_dim),
            )
            for _ in range(num_scales)
        ])

        # FPN top-down lateral connections (from coarse to fine)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sam_dim, sam_dim, 1),
                nn.GroupNorm(32, sam_dim),
            )
            for _ in range(num_scales - 1)  # no lateral for the coarsest level
        ])

        # Final fusion after FPN
        self.fusion = nn.Sequential(
            nn.Conv2d(sam_dim * num_scales, sam_dim, 1),
            ResidualBlock(sam_dim),
        )

        # Transformer refinement on fused features
        self.refine = TransformerRefinement(dim=sam_dim, num_heads=4, num_layers=2)

        # Dense change prompt
        self.change_proj = nn.Sequential(
            nn.Linear(dino_dim, sam_dim),
            nn.LayerNorm(sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, sam_dim),
        )

        # High-res feature projectors for SAM2 decoder upsampling path
        self.hr_proj_s1 = nn.Sequential(
            nn.Linear(dino_dim, 64),
            nn.GELU(),
        )
        self.hr_conv_s1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
        )
        self.hr_proj_s0 = nn.Sequential(
            nn.Linear(dino_dim, 32),
            nn.GELU(),
        )
        self.hr_conv_s0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

    def forward(
        self,
        multi_feats: List[Tensor],
        change_tokens: Tensor,
        h: int, w: int,
        target_h: int = 64, target_w: int = 64,
    ):
        B = multi_feats[0].size(0)

        # Step 1: per-scale projection + residual refinement
        scale_maps = []
        for feat, proj, conv in zip(multi_feats, self.scale_projs, self.scale_convs):
            x = proj(feat)                                             # [B,S,C]
            x = x.view(B, h, w, self.sam_dim).permute(0, 3, 1, 2)     # [B,C,h,w]
            x = F.interpolate(x, (target_h, target_w), mode="bilinear", align_corners=False)
            x = conv(x)
            scale_maps.append(x)

        # Step 2: FPN top-down pathway (coarse → fine)
        # scale_maps[0]=earliest/finest, scale_maps[-1]=latest/coarsest
        for i in range(len(scale_maps) - 1, 0, -1):
            coarse = scale_maps[i]
            lateral = self.lateral_convs[i - 1](scale_maps[i - 1])
            scale_maps[i - 1] = lateral + coarse  # already same spatial size

        # Step 3: fuse + refine
        image_emb = self.fusion(torch.cat(scale_maps, dim=1))  # [B,256,th,tw]
        image_emb = self.refine(image_emb)

        # Dense change prompt
        dense = self.change_proj(change_tokens)
        dense = dense.view(B, h, w, self.sam_dim).permute(0, 3, 1, 2)
        dense = F.interpolate(dense, (target_h, target_w), mode="bilinear", align_corners=False)

        # High-res features from early DINOv2 layers
        feat_s1 = self.hr_proj_s1(multi_feats[0])
        feat_s1 = feat_s1.view(B, h, w, 64).permute(0, 3, 1, 2)
        feat_s1 = F.interpolate(feat_s1, (target_h * 2, target_w * 2), mode="bilinear", align_corners=False)
        feat_s1 = self.hr_conv_s1(feat_s1)

        feat_s0 = self.hr_proj_s0(multi_feats[0])
        feat_s0 = feat_s0.view(B, h, w, 32).permute(0, 3, 1, 2)
        feat_s0 = F.interpolate(feat_s0, (target_h * 4, target_w * 4), mode="bilinear", align_corners=False)
        feat_s0 = self.hr_conv_s0(feat_s0)

        return image_emb, dense, [feat_s0, feat_s1]


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
class ChangeDetector(nn.Module):
    """
    DINOv2/v3 (frozen) → CrossChangeAttention (trainable)
                        → Bridge v2            (trainable)
                        → SAM2.1 MaskDecoder   (frozen or fine-tuned) → masks + IoU
    """

    def __init__(
        self,
        dino_model_name: str,
        sam2_checkpoint: str,
        sam2_config: str = "sam2_hiera_l.yaml",
        num_heads: int = 8,
        temperature: float = 0.07,
        feature_layers: Optional[List[int]] = None,
        sam_target_size: int = 64,
        finetune_decoder: bool = False,
        decoder_lr_scale: float = 0.1,
    ):
        super().__init__()
        self.sam_target_size = sam_target_size
        self.finetune_decoder = finetune_decoder
        self.decoder_lr_scale = decoder_lr_scale

        # ── DINOv2 / v3 backbone  (FROZEN) ──────────────────────────
        self.dino = self._load_dino(dino_model_name)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        cfg = self.dino.config
        self.dino_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        n = cfg.num_hidden_layers

        # layers tapped for multi-scale features
        if feature_layers is None:
            self.feature_layers = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        else:
            self.feature_layers = feature_layers

        # ── Cross-Attention  (TRAINABLE) ─────────────────────────────
        self.cross_attn = CrossChangeAttention(
            self.dino_dim, num_heads, temperature
        )

        # ── Bridge / Projector  (TRAINABLE) ──────────────────────────
        self.bridge = Bridge(
            self.dino_dim, sam_dim=256, num_scales=len(self.feature_layers)
        )

        # ── SAM2 Mask Decoder  (FROZEN or FINE-TUNED) ───────────────
        self._load_sam2(sam2_checkpoint, sam2_config)

    # -------- DINO loading --------
    @staticmethod
    def _load_dino(name_or_path: str):
        try:
            from huggingface_hub import hf_hub_download
            import json
            cfg_path = hf_hub_download(name_or_path, "config.json")
            with open(cfg_path) as f:
                raw = json.load(f)
            if raw.get("architectures") == ["DINOv2"] and "embed_dim" in raw:
                from load_dino_rs import load_dinov2_remote_sensing
                return load_dinov2_remote_sensing(name_or_path)
        except Exception:
            pass

        from transformers import AutoModel
        return AutoModel.from_pretrained(name_or_path, output_hidden_states=True)

    # -------- SAM2 loading helpers --------
    def _load_sam2(self, checkpoint: str, config: str):
        from sam2.build_sam import build_sam2

        sam2 = build_sam2(config, checkpoint)
        self.sam_decoder = sam2.sam_mask_decoder
        self.sam_prompt_enc = sam2.sam_prompt_encoder

        if not self.finetune_decoder:
            for p in self.sam_decoder.parameters():
                p.requires_grad = False
            self.sam_decoder.eval()
        # prompt encoder is always frozen
        for p in self.sam_prompt_enc.parameters():
            p.requires_grad = False
        self.sam_prompt_enc.eval()

    # -------- frozen feature extraction --------
    @torch.no_grad()
    def _dino_forward_pair(self, ref: Tensor, tgt: Tensor):
        B = ref.size(0)
        both = torch.cat([ref, tgt], dim=0)
        out = self.dino(pixel_values=both)
        hs = out.hidden_states

        ref_multi, tgt_multi = [], []
        for i in self.feature_layers:
            h = hs[i + 1][:, 1:, :]
            ref_multi.append(h[:B])
            tgt_multi.append(h[B:])

        last = hs[-1][:, 1:, :]
        ref_tok, tgt_tok = last[:B], last[B:]
        return ref_multi, tgt_multi, ref_tok, tgt_tok

    # -------- forward --------
    def forward(self, ref: Tensor, tgt: Tensor):
        B = ref.size(0)
        ph = ref.shape[2] // self.patch_size
        pw = ref.shape[3] // self.patch_size

        # 1) DINOv2 features (frozen)
        ref_multi, tgt_multi, ref_tok, tgt_tok = self._dino_forward_pair(ref, tgt)

        # 2) Cross-attention (trainable)
        change_tokens, change_map = self.cross_attn(ref_tok, tgt_tok)

        # 3) Multi-scale difference + change residual → bridge input
        bridge_input = [
            (r - t) + change_tokens for r, t in zip(ref_multi, tgt_multi)
        ]

        # 4) Bridge → SAM2 format (trainable)
        th = tw = self.sam_target_size
        image_emb, dense_prompt, high_res_features = self.bridge(
            bridge_input, change_tokens, ph, pw, th, tw
        )

        # 5) SAM2 decoder
        image_pe = self.sam_prompt_enc.get_dense_pe()
        if image_pe.shape[-2:] != (th, tw):
            image_pe = F.interpolate(
                image_pe, (th, tw), mode="bilinear", align_corners=False
            )

        sparse_emb, _ = self.sam_prompt_enc(
            points=None, boxes=None, masks=None
        )

        masks, iou_pred, _sam_tokens, _obj_scores = self.sam_decoder(
            image_embeddings=image_emb,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb.expand(B, -1, -1),
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return masks, iou_pred, change_map

    # -------- TTA forward --------
    @torch.no_grad()
    def forward_tta(self, ref: Tensor, tgt: Tensor, flips: bool = True, rotations: bool = True):
        """Test-time augmentation: average predictions over geometric transforms."""
        transforms = [(False, False, 0)]  # original
        if flips:
            transforms += [(True, False, 0), (False, True, 0), (True, True, 0)]
        if rotations:
            transforms += [(False, False, 1), (False, False, 2), (False, False, 3)]

        mask_sum = None
        iou_sum = None

        for hflip, vflip, rot90k in transforms:
            r, t = ref.clone(), tgt.clone()
            if hflip:
                r = r.flip(-1)
                t = t.flip(-1)
            if vflip:
                r = r.flip(-2)
                t = t.flip(-2)
            if rot90k > 0:
                r = torch.rot90(r, rot90k, [-2, -1])
                t = torch.rot90(t, rot90k, [-2, -1])

            masks, iou_pred, _ = self(r, t)

            # undo transforms on masks
            if rot90k > 0:
                masks = torch.rot90(masks, -rot90k, [-2, -1])
            if vflip:
                masks = masks.flip(-2)
            if hflip:
                masks = masks.flip(-1)

            if mask_sum is None:
                mask_sum = masks.sigmoid()
                iou_sum = iou_pred
            else:
                mask_sum = mask_sum + masks.sigmoid()
                iou_sum = iou_sum + iou_pred

        n = len(transforms)
        # Return logits from averaged probabilities
        avg_prob = mask_sum / n
        avg_logits = torch.logit(avg_prob.clamp(1e-6, 1 - 1e-6))
        return avg_logits, iou_sum / n, None

    # -------- convenience --------
    def trainable_parameters(self):
        """Only the bridge + cross-attention are optimised (+ optionally decoder)."""
        yield from self.cross_attn.parameters()
        yield from self.bridge.parameters()
        if self.finetune_decoder:
            yield from self.sam_decoder.parameters()

    def param_groups(self, lr: float):
        """Return param groups with differential LR for decoder fine-tuning."""
        groups = [
            {"params": list(self.cross_attn.parameters()) + list(self.bridge.parameters()),
             "lr": lr},
        ]
        if self.finetune_decoder:
            groups.append({
                "params": list(self.sam_decoder.parameters()),
                "lr": lr * self.decoder_lr_scale,
            })
        return groups

    def train(self, mode: bool = True):
        """Keep frozen parts in eval regardless of mode."""
        super().train(mode)
        self.dino.eval()
        self.sam_prompt_enc.eval()
        if not self.finetune_decoder:
            self.sam_decoder.eval()
        return self
