"""
Change Detection Pipeline  v3
──────────────────────────────
ref image ─┐                          ┌─ Bridge v2 (trainable) ──┐
           ├→ Encoder (frozen) ───────┤  FPN + residual + xformer├→ Decoder → mask + IoU
tgt image ─┘   multi-scale features   └─ CrossChangeAttn ────────┘
                                          (trainable, temp.)    ↗ small change map (aux loss)

Supported encoders:
  - DINOv2: small/base/large/giant (patch14), with-registers variants
  - DINOv2-RS: KevinCha small/base/large (remote sensing fine-tuned)
  - DINOv3: small/base/large/huge (patch16), satellite variants (gated, needs HF access)

Supported decoders:
  - SAM1: vit_b / vit_l / vit_h
  - SAM2.1: tiny / small / base_plus / large
  - SAM3: single variant (848M, uses HF download)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# ENCODER REGISTRY
# ═══════════════════════════════════════════════════════════════════════
ENCODERS = {
    # --- DINOv1 (original DINO, Caron et al. 2021) ---
    "dinov1_vits16":      {"hf": "facebook/dino-vits16",        "dim": 384,  "layers": 12, "patch": 16},
    "dinov1_vits8":       {"hf": "facebook/dino-vits8",         "dim": 384,  "layers": 12, "patch": 8},
    "dinov1_vitb16":      {"hf": "facebook/dino-vitb16",        "dim": 768,  "layers": 12, "patch": 16},
    "dinov1_vitb8":       {"hf": "facebook/dino-vitb8",         "dim": 768,  "layers": 12, "patch": 8},
    # --- DINOv2 official (patch_size=14) ---
    "dinov2_small":       {"hf": "facebook/dinov2-small",       "dim": 384,  "layers": 12, "patch": 14},
    "dinov2_base":        {"hf": "facebook/dinov2-base",        "dim": 768,  "layers": 12, "patch": 14},
    "dinov2_large":       {"hf": "facebook/dinov2-large",       "dim": 1024, "layers": 24, "patch": 14},
    "dinov2_giant":       {"hf": "facebook/dinov2-giant",       "dim": 1536, "layers": 40, "patch": 14},
    # --- DINOv2 with registers (patch_size=14) ---
    "dinov2_small_reg":   {"hf": "facebook/dinov2-with-registers-small",  "dim": 384,  "layers": 12, "patch": 14},
    "dinov2_base_reg":    {"hf": "facebook/dinov2-with-registers-base",   "dim": 768,  "layers": 12, "patch": 14},
    "dinov2_large_reg":   {"hf": "facebook/dinov2-with-registers-large",  "dim": 1024, "layers": 24, "patch": 14},
    "dinov2_giant_reg":   {"hf": "facebook/dinov2-with-registers-giant",  "dim": 1536, "layers": 40, "patch": 14},
    # --- DINOv2 Remote Sensing (KevinCha) ---
    "dinov2_rs_small":    {"hf": "KevinCha/dinov2-vit-small-remote-sensing",     "dim": 384,  "layers": 12, "patch": 16, "loader": "kevincha"},
    "dinov2_rs_base":     {"hf": "KevinCha/dinov2-vit-base-remote-sensing",      "dim": 768,  "layers": 12, "patch": 16, "loader": "kevincha"},
    "dinov2_rs_large":    {"hf": "KevinCha/dinov2-vit-large-remote-sensing",     "dim": 1024, "layers": 24, "patch": 14, "loader": "kevincha"},
    "dinov2_rs_large_50": {"hf": "KevinCha/dinov2-vit-large-remote-sensing-50ep","dim": 1024, "layers": 24, "patch": 14, "loader": "kevincha"},
    # --- DINOv3 official (patch_size=16, gated — needs HF access) ---
    "dinov3_small":       {"hf": "facebook/dinov3-vits16-pretrain-lvd1689m",      "dim": 384,  "layers": 12, "patch": 16},
    "dinov3_base":        {"hf": "facebook/dinov3-vitb16-pretrain-lvd1689m",      "dim": 768,  "layers": 12, "patch": 16},
    "dinov3_large":       {"hf": "facebook/dinov3-vitl16-pretrain-lvd1689m",      "dim": 1024, "layers": 24, "patch": 16},
    "dinov3_huge":        {"hf": "facebook/dinov3-vith16plus-pretrain-lvd1689m",  "dim": 1280, "layers": 32, "patch": 16},
    # --- DINOv3 satellite (patch_size=16, gated) ---
    "dinov3_sat_large":   {"hf": "facebook/dinov3-vitl16-pretrain-sat493m",       "dim": 1024, "layers": 24, "patch": 16},
}

# ═══════════════════════════════════════════════════════════════════════
# DECODER REGISTRY
# ═══════════════════════════════════════════════════════════════════════
DECODERS = {
    # --- SAM1 (no high_res_features, returns 2 values) ---
    "sam1_vit_b":         {"family": "sam1", "ckpt": "sam_vit_b.pth",              "type": "vit_b"},
    "sam1_vit_l":         {"family": "sam1", "ckpt": "sam_vit_l.pth",              "type": "vit_l"},
    "sam1_vit_h":         {"family": "sam1", "ckpt": "sam_vit_h.pth",              "type": "vit_h"},
    # --- SAM2.1 (high_res_features, returns 4 values) ---
    "sam2_tiny":          {"family": "sam2", "ckpt": "sam2.1_hiera_tiny.pt",       "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml"},
    "sam2_small":         {"family": "sam2", "ckpt": "sam2.1_hiera_small.pt",      "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml"},
    "sam2_base_plus":     {"family": "sam2", "ckpt": "sam2.1_hiera_base_plus.pt",  "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml"},
    "sam2_large":         {"family": "sam2", "ckpt": "sam2.1_hiera_large.pt",      "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml"},
    # --- SAM3 (same interface as SAM2, single variant) ---
    "sam3":               {"family": "sam3"},
}

# Backward compatibility
SAM2_VARIANTS = {
    "tiny":      {"ckpt": "sam2.1_hiera_tiny.pt",      "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml"},
    "small":     {"ckpt": "sam2.1_hiera_small.pt",      "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml"},
    "base_plus": {"ckpt": "sam2.1_hiera_base_plus.pt",  "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml"},
    "large":     {"ckpt": "sam2.1_hiera_large.pt",      "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml"},
}


def list_encoders():
    """Print all available encoders."""
    print(f"{'Name':<22s} {'HuggingFace ID':<55s} {'Dim':>5s} {'Layers':>6s} {'Patch':>5s}")
    print("-" * 100)
    for name, info in ENCODERS.items():
        print(f"{name:<22s} {info['hf']:<55s} {info['dim']:>5d} {info['layers']:>6d} {info['patch']:>5d}")


def list_decoders():
    """Print all available decoders."""
    print(f"{'Name':<18s} {'Family':<6s} {'Checkpoint':<35s}")
    print("-" * 65)
    for name, info in DECODERS.items():
        ckpt = info.get("ckpt", "HuggingFace auto-download")
        print(f"{name:<18s} {info['family']:<6s} {ckpt:<35s}")


# ---------------------------------------------------------------------------
# Cross-Attention with learnable temperature  (TRAINABLE)
# ---------------------------------------------------------------------------
class CrossChangeAttention(nn.Module):
    """Cross-attention between reference and target patch tokens.

    Produces:
        change_tokens – per-patch change representation
        change_map    – [B, S] soft change score (the "small map" for aux loss)

    Tier-1 ablations (controlled by kwargs):
      - bidirectional: average forward (ref→tgt) and backward (tgt→ref)
        attentions. Enforces symmetry of binary CD.
      - local_window: instead of taking only the diagonal of the attention map,
        take max similarity over a (W×W) spatial window around each diagonal
        entry. Robust to small ref/tgt registration shifts. W=1 = current behaviour.
    """

    def __init__(self, dim: int, num_heads: int = 8, init_temperature: float = 0.07,
                 bidirectional: bool = False, local_window: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))
        self.bidirectional = bidirectional
        self.local_window = max(1, int(local_window))

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _local_window_similarity(avg_attn, h, w, window):
        """Replace diag(avg_attn) with max over (window x window) tgt-side neighborhood
        for each ref position. avg_attn: [B, S, S] with S = h*w."""
        if window == 1:
            return torch.diagonal(avg_attn, dim1=1, dim2=2)  # [B, S]
        B = avg_attn.size(0)
        device = avg_attn.device
        half = window // 2
        ri = torch.arange(h, device=device).view(h, 1).expand(h, w)  # [h, w]
        rj = torch.arange(w, device=device).view(1, w).expand(h, w)
        sims = []
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                ti = (ri + di).clamp(0, h - 1)
                tj = (rj + dj).clamp(0, w - 1)
                i_idx = (ri * w + rj).flatten()
                j_idx = (ti * w + tj).flatten()
                # advanced indexing on dim=1 only would require a more complex op;
                # gather over dim=2 with broadcast on dim=1:
                rows = avg_attn[:, i_idx, :]                           # [B, S, S]
                vals = rows.gather(2, j_idx.view(1, -1, 1).expand(B, -1, 1)).squeeze(-1)  # [B, S]
                sims.append(vals)
        return torch.stack(sims, dim=-1).max(dim=-1).values  # [B, S]

    def _attn_block(self, q_in, k_in, v_in):
        """One direction of cross-attention; returns (attended, attn_weights)."""
        B, S, D = q_in.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(q_in).view(B, S, H, d).transpose(1, 2)
        k = self.k_proj(k_in).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(v_in).view(B, S, H, d).transpose(1, 2)
        temp = self.log_temp.exp().clamp(min=1.0)
        attn_logits = (q @ k.transpose(-2, -1)) * (temp / d ** 0.5)
        attn_weights = attn_logits.softmax(dim=-1)
        attended = (attn_weights @ v).transpose(1, 2).reshape(B, S, D)
        attended = self.out_proj(attended)
        return attended, attn_weights

    def forward(self, ref_tokens: Tensor, tgt_tokens: Tensor,
                grid_h: int = None, grid_w: int = None):
        B, S, D = ref_tokens.shape
        # Forward direction: query=ref, key/value=tgt
        attended_fwd, attn_fwd = self._attn_block(ref_tokens, tgt_tokens, tgt_tokens)
        change_tokens = self.norm(attended_fwd - ref_tokens)

        # If grid not given, assume square. Caller typically passes ph, pw.
        if grid_h is None:
            grid_h = grid_w = int(S ** 0.5)
            if grid_h * grid_w != S:
                grid_h = grid_w = 1   # fallback (won't trigger window > 1 cleanly)

        avg_attn_fwd = attn_fwd.mean(dim=1)            # [B, S, S]
        sim_fwd = self._local_window_similarity(avg_attn_fwd, grid_h, grid_w, self.local_window)

        if self.bidirectional:
            _, attn_bwd = self._attn_block(tgt_tokens, ref_tokens, ref_tokens)
            avg_attn_bwd = attn_bwd.mean(dim=1)
            sim_bwd = self._local_window_similarity(avg_attn_bwd, grid_h, grid_w, self.local_window)
            similarity = 0.5 * (sim_fwd + sim_bwd)
        else:
            similarity = sim_fwd

        change_map = 1.0 - similarity
        return change_tokens, change_map


# ---------------------------------------------------------------------------
# Building blocks for Bridge v2
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
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
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).view(B, C, H, W)


# ---------------------------------------------------------------------------
# Bridge v2  (TRAINABLE)
# ---------------------------------------------------------------------------
class Bridge(nn.Module):
    """Projects multi-scale encoder features → decoder-compatible tensors.

    Adapts output based on decoder family:
      - SAM2/SAM3: returns high_res_features [feat_s0, feat_s1]
      - SAM1: returns high_res_features=None
    """

    def __init__(self, dino_dim: int = 768, sam_dim: int = 256, num_scales: int = 4,
                 use_high_res: bool = True):
        super().__init__()
        self.sam_dim = sam_dim
        self.use_high_res = use_high_res

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
            nn.Sequential(ResidualBlock(sam_dim), ResidualBlock(sam_dim))
            for _ in range(num_scales)
        ])

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(sam_dim, sam_dim, 1), nn.GroupNorm(32, sam_dim))
            for _ in range(num_scales - 1)
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(sam_dim * num_scales, sam_dim, 1),
            ResidualBlock(sam_dim),
        )
        self.refine = TransformerRefinement(dim=sam_dim, num_heads=4, num_layers=2)

        self.change_proj = nn.Sequential(
            nn.Linear(dino_dim, sam_dim),
            nn.LayerNorm(sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, sam_dim),
        )

        if use_high_res:
            self.hr_proj_s1 = nn.Sequential(nn.Linear(dino_dim, 64), nn.GELU())
            self.hr_conv_s1 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.GroupNorm(16, 64), nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.GroupNorm(16, 64), nn.GELU(),
            )
            self.hr_proj_s0 = nn.Sequential(nn.Linear(dino_dim, 32), nn.GELU())
            self.hr_conv_s0 = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            )

        # Auxiliary multi-scale change-prediction heads (deep supervision).
        # Output 1-channel logit maps at three resolutions to be supervised
        # against down-pooled GT masks. Cheap (~3K params), only used when
        # the train-loop has multi_scale_latent=True. Always populated so
        # the forward pass is deterministic; loss decides whether to use.
        self.aux_head_fused = nn.Conv2d(sam_dim, 1, kernel_size=1)
        if use_high_res:
            self.aux_head_hr1 = nn.Conv2d(64, 1, kernel_size=1)
            self.aux_head_hr0 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, multi_feats, change_tokens, h, w, target_h=64, target_w=64):
        B = multi_feats[0].size(0)

        scale_maps = []
        for feat, proj, conv in zip(multi_feats, self.scale_projs, self.scale_convs):
            x = proj(feat)
            x = x.view(B, h, w, self.sam_dim).permute(0, 3, 1, 2)
            x = F.interpolate(x, (target_h, target_w), mode="bilinear", align_corners=False)
            x = conv(x)
            scale_maps.append(x)

        for i in range(len(scale_maps) - 1, 0, -1):
            coarse = scale_maps[i]
            lateral = self.lateral_convs[i - 1](scale_maps[i - 1])
            scale_maps[i - 1] = lateral + coarse

        image_emb = self.fusion(torch.cat(scale_maps, dim=1))
        image_emb = self.refine(image_emb)

        dense = self.change_proj(change_tokens)
        dense = dense.view(B, h, w, self.sam_dim).permute(0, 3, 1, 2)
        dense = F.interpolate(dense, (target_h, target_w), mode="bilinear", align_corners=False)

        high_res_features = None
        if self.use_high_res:
            feat_s1 = self.hr_proj_s1(multi_feats[0])
            feat_s1 = feat_s1.view(B, h, w, 64).permute(0, 3, 1, 2)
            feat_s1 = F.interpolate(feat_s1, (target_h * 2, target_w * 2), mode="bilinear", align_corners=False)
            feat_s1 = self.hr_conv_s1(feat_s1)

            feat_s0 = self.hr_proj_s0(multi_feats[0])
            feat_s0 = feat_s0.view(B, h, w, 32).permute(0, 3, 1, 2)
            feat_s0 = F.interpolate(feat_s0, (target_h * 4, target_w * 4), mode="bilinear", align_corners=False)
            feat_s0 = self.hr_conv_s0(feat_s0)

            high_res_features = [feat_s0, feat_s1]

        # Multi-scale auxiliary change predictions (logits, B×1×H×W).
        # Coarsest first → finest last; lets compute_loss apply
        # progressively decaying weights.
        aux_change_maps = [self.aux_head_fused(image_emb)]   # [B,1,64,64]
        if self.use_high_res:
            aux_change_maps.append(self.aux_head_hr1(feat_s1))   # [B,1,128,128]
            aux_change_maps.append(self.aux_head_hr0(feat_s0))   # [B,1,256,256]

        return image_emb, dense, high_res_features, aux_change_maps


# ═══════════════════════════════════════════════════════════════════════
# Encoder / Decoder loading helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_encoder(name_or_hf: str):
    """Load an encoder by registry name or HuggingFace ID.

    Returns a model with:
      - model.config.hidden_size
      - model.config.num_hidden_layers
      - model.config.patch_size
      - model(pixel_values=x).hidden_states  (tuple of [B, 1+S, D])
    """
    # Check registry first
    info = ENCODERS.get(name_or_hf)
    if info is not None:
        hf_id = info["hf"]
        loader = info.get("loader")
    else:
        # Treat as direct HuggingFace ID
        hf_id = name_or_hf
        loader = None
        # Check if it's a KevinCha model
        if "kevincha" in hf_id.lower() or "remote-sensing" in hf_id.lower():
            loader = "kevincha"

    if loader == "kevincha":
        from load_dino_rs import load_dinov2_remote_sensing
        return load_dinov2_remote_sensing(hf_id)

    from transformers import AutoModel
    return AutoModel.from_pretrained(hf_id, output_hidden_states=True, trust_remote_code=True)


def _load_decoder_sam1(checkpoint: str, model_type: str, ckpt_dir: str):
    """Load SAM1 mask decoder + prompt encoder."""
    from segment_anything import sam_model_registry
    ckpt_path = os.path.join(ckpt_dir, checkpoint) if not os.path.isabs(checkpoint) else checkpoint
    sam = sam_model_registry[model_type](checkpoint=ckpt_path if os.path.isfile(ckpt_path) else None)
    return sam.mask_decoder, sam.prompt_encoder, "sam1"


def _load_decoder_sam2(checkpoint: str, config: str, ckpt_dir: str):
    """Load SAM2.1 mask decoder + prompt encoder."""
    from sam2.build_sam import build_sam2
    ckpt_path = os.path.join(ckpt_dir, checkpoint) if not os.path.isabs(checkpoint) else checkpoint
    sam2 = build_sam2(config, ckpt_path)
    return sam2.sam_mask_decoder, sam2.sam_prompt_encoder, "sam2"


def _load_decoder_sam3(ckpt_dir: str):
    """Load SAM3 mask decoder + prompt encoder (from tracker)."""
    from sam3.model_builder import build_tracker
    tracker = build_tracker(apply_temporal_disambiguation=False)
    return tracker.sam_mask_decoder, tracker.sam_prompt_encoder, "sam3"


def load_decoder(name: str, ckpt_dir: str = "checkpoints"):
    """Load a decoder by registry name.

    Returns: (mask_decoder, prompt_encoder, family_str)
    """
    info = DECODERS[name]
    family = info["family"]

    if family == "sam1":
        return _load_decoder_sam1(info["ckpt"], info["type"], ckpt_dir)
    elif family == "sam2":
        return _load_decoder_sam2(info["ckpt"], info["cfg"], ckpt_dir)
    elif family == "sam3":
        return _load_decoder_sam3(ckpt_dir)
    else:
        raise ValueError(f"Unknown decoder family: {family}")


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════
class ChangeDetector(nn.Module):
    """
    Encoder (frozen) → CrossChangeAttention (trainable)
                      → Bridge v2            (trainable)
                      → Decoder              (frozen or fine-tuned) → masks + IoU
    """

    def __init__(
        self,
        encoder: str = "dinov2_rs_base",
        decoder: str = "sam2_base_plus",
        ckpt_dir: str = "checkpoints",
        num_heads: int = 8,
        temperature: float = 0.07,
        feature_layers: Optional[List[int]] = None,
        sam_target_size: int = 64,
        finetune_decoder: bool = False,
        decoder_lr_scale: float = 0.1,
        # Tier-1 latent-space ablations
        bidir_attn: bool = False,
        local_window: int = 1,
        # Backward compat: direct HF/path overrides
        dino_model_name: Optional[str] = None,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None,
    ):
        super().__init__()
        self.sam_target_size = sam_target_size
        self.finetune_decoder = finetune_decoder
        self.decoder_lr_scale = decoder_lr_scale

        # ── Encoder (FROZEN) ─────────────────────────────────────────
        encoder_id = dino_model_name if dino_model_name else encoder
        self.dino = _load_encoder(encoder_id)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        cfg = self.dino.config
        self.dino_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        n = cfg.num_hidden_layers

        if feature_layers is None:
            self.feature_layers = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        else:
            self.feature_layers = feature_layers

        # ── Cross-Attention (TRAINABLE) ──────────────────────────────
        self.cross_attn = CrossChangeAttention(
            self.dino_dim, num_heads, temperature,
            bidirectional=bidir_attn, local_window=local_window,
        )

        # ── Decoder loading ──────────────────────────────────────────
        if sam2_checkpoint and sam2_config:
            # Backward compat: explicit checkpoint + config
            from sam2.build_sam import build_sam2
            sam2 = build_sam2(sam2_config, sam2_checkpoint)
            self.sam_decoder = sam2.sam_mask_decoder
            self.sam_prompt_enc = sam2.sam_prompt_encoder
            self.decoder_family = "sam2"
        else:
            self.sam_decoder, self.sam_prompt_enc, self.decoder_family = load_decoder(
                decoder, ckpt_dir
            )

        # Freeze decoder (optionally)
        if not finetune_decoder:
            for p in self.sam_decoder.parameters():
                p.requires_grad = False
            self.sam_decoder.eval()
        for p in self.sam_prompt_enc.parameters():
            p.requires_grad = False
        self.sam_prompt_enc.eval()

        # ── Bridge (TRAINABLE) — adapts based on decoder family ──────
        use_high_res = self.decoder_family in ("sam2", "sam3")
        self.bridge = Bridge(
            self.dino_dim, sam_dim=256, num_scales=len(self.feature_layers),
            use_high_res=use_high_res,
        )

        self.encoder_name = encoder_id
        self.decoder_name = decoder

    # -------- frozen feature extraction --------
    @torch.no_grad()
    def _dino_forward_pair(self, ref: Tensor, tgt: Tensor):
        B = ref.size(0)
        both = torch.cat([ref, tgt], dim=0)
        out = self.dino(pixel_values=both)
        hs = out.hidden_states

        # Skip CLS (1) plus any register tokens. DINOv2-with-registers and
        # DINOv3 add 4 register tokens between CLS and the patch tokens.
        n_skip = 1 + int(getattr(self.dino.config, "num_register_tokens", 0))

        ref_multi, tgt_multi = [], []
        for i in self.feature_layers:
            h = hs[i + 1][:, n_skip:, :]
            ref_multi.append(h[:B])
            tgt_multi.append(h[B:])

        last = hs[-1][:, n_skip:, :]
        ref_tok, tgt_tok = last[:B], last[B:]
        return ref_multi, tgt_multi, ref_tok, tgt_tok

    # -------- decoder dispatch --------
    def _decode(self, image_emb, dense_prompt, high_res_features, B):
        image_pe = self.sam_prompt_enc.get_dense_pe()
        th, tw = image_emb.shape[-2:]
        if image_pe.shape[-2:] != (th, tw):
            image_pe = F.interpolate(image_pe, (th, tw), mode="bilinear", align_corners=False)

        sparse_emb, _ = self.sam_prompt_enc(points=None, boxes=None, masks=None)

        if self.decoder_family == "sam1":
            masks, iou_pred = self.sam_decoder(
                image_embeddings=image_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb.expand(B, -1, -1),
                dense_prompt_embeddings=dense_prompt,
                multimask_output=False,
            )
        else:
            # SAM2 / SAM3 interface
            masks, iou_pred, _, _ = self.sam_decoder(
                image_embeddings=image_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb.expand(B, -1, -1),
                dense_prompt_embeddings=dense_prompt,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

        return masks, iou_pred

    # -------- forward --------
    def forward(self, ref: Tensor, tgt: Tensor):
        B = ref.size(0)
        ph = ref.shape[2] // self.patch_size
        pw = ref.shape[3] // self.patch_size

        ref_multi, tgt_multi, ref_tok, tgt_tok = self._dino_forward_pair(ref, tgt)
        change_tokens, change_map = self.cross_attn(ref_tok, tgt_tok, grid_h=ph, grid_w=pw)

        bridge_input = [
            (r - t) + change_tokens for r, t in zip(ref_multi, tgt_multi)
        ]

        th = tw = self.sam_target_size
        image_emb, dense_prompt, high_res_features, aux_change_maps = self.bridge(
            bridge_input, change_tokens, ph, pw, th, tw
        )

        masks, iou_pred = self._decode(image_emb, dense_prompt, high_res_features, B)
        # Returns: pixel mask, predicted IoU, patch-level change_map (for legacy
        # latent loss), list of aux change-map logits at multiple scales (for
        # multi-scale latent loss when --multi_scale_latent is set).
        return masks, iou_pred, change_map, aux_change_maps

    # -------- TTA forward --------
    @torch.no_grad()
    def forward_tta(self, ref: Tensor, tgt: Tensor, flips: bool = True, rotations: bool = True):
        transforms = [(False, False, 0)]
        if flips:
            transforms += [(True, False, 0), (False, True, 0), (True, True, 0)]
        if rotations:
            transforms += [(False, False, 1), (False, False, 2), (False, False, 3)]

        mask_sum = None
        iou_sum = None

        for hflip, vflip, rot90k in transforms:
            r, t = ref.clone(), tgt.clone()
            if hflip:
                r, t = r.flip(-1), t.flip(-1)
            if vflip:
                r, t = r.flip(-2), t.flip(-2)
            if rot90k > 0:
                r = torch.rot90(r, rot90k, [-2, -1])
                t = torch.rot90(t, rot90k, [-2, -1])

            masks, iou_pred, _, _ = self(r, t)

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
        avg_prob = mask_sum / n
        avg_logits = torch.logit(avg_prob.clamp(1e-6, 1 - 1e-6))
        # Aux change maps are train-only; TTA doesn't need them.
        return avg_logits, iou_sum / n, None, []

    # -------- convenience --------
    def trainable_parameters(self):
        yield from self.cross_attn.parameters()
        yield from self.bridge.parameters()
        if self.finetune_decoder:
            yield from self.sam_decoder.parameters()

    def param_groups(self, lr: float):
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
        super().train(mode)
        self.dino.eval()
        self.sam_prompt_enc.eval()
        if not self.finetune_decoder:
            self.sam_decoder.eval()
        return self
