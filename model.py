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

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LoRA — Low-Rank Adaptation (Hu et al. 2021)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps an nn.Linear, freezing the original weights and adding a low-rank
    additive delta (lora_B @ lora_A · x · scaling). Memory cost: rank·(in+out)
    extra params; compute cost: tiny extra matmul.

    On init, lora_A is Kaiming-uniform and lora_B is zero — so the LoRA delta
    starts at exactly 0, preserving the frozen network's initial behaviour.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base.in_features
        out_f = base.out_features
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # (rank, in_f) and (out_f, rank) so the delta is lora_B @ lora_A
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialised to zero → delta = 0 at start (identity behaviour)

    def forward(self, x: Tensor) -> Tensor:
        # base(x): the frozen linear; LoRA delta added on top
        out = self.base(x)
        delta = (self.dropout(x) @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return out + self.scaling * delta


# DINOv2 attention layers (HuggingFace ViTModel interface)
DINO_LORA_TARGETS = ("attention.query", "attention.key", "attention.value",
                     "attention.output.dense")
# SAM2/SAM1 mask-decoder + prompt-encoder attention projections
SAM_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


def _matches_suffix(name: str, suffixes: Iterable[str]) -> bool:
    """True if the qualified module name ends with any of the suffixes."""
    return any(name == s or name.endswith("." + s) for s in suffixes)


def inject_lora(root: nn.Module, suffixes: Iterable[str], rank: int = 8,
                alpha: float = 16.0, dropout: float = 0.0) -> int:
    """Walk `root`, replace every nn.Linear whose qualified name ends with one
    of `suffixes` with a LoRALinear wrapper. Returns count of injected layers."""
    to_replace = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and _matches_suffix(name, suffixes):
            to_replace.append(name)

    n = 0
    for qualified in to_replace:
        # navigate to leaf parent
        parts = qualified.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf_name = parts[-1]
        old = getattr(parent, leaf_name)
        if isinstance(old, LoRALinear):
            continue
        setattr(parent, leaf_name, LoRALinear(old, rank=rank, alpha=alpha, dropout=dropout))
        n += 1
    return n


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
    # --- SAM2 image encoder (Hiera). Output: vision_features at H/16, dim=256. ---
    # Wrapped to expose a DINO/HF-like .config + .hidden_states interface.
    "sam2_enc_tiny":      {"hf": "<sam2_enc>", "loader": "sam2_enc", "dim": 256, "layers": 12, "patch": 16,
                           "sam2_cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2_ckpt": "sam2.1_hiera_tiny.pt"},
    "sam2_enc_small":     {"hf": "<sam2_enc>", "loader": "sam2_enc", "dim": 256, "layers": 12, "patch": 16,
                           "sam2_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2_ckpt": "sam2.1_hiera_small.pt"},
    "sam2_enc_base_plus": {"hf": "<sam2_enc>", "loader": "sam2_enc", "dim": 256, "layers": 12, "patch": 16,
                           "sam2_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2_ckpt": "sam2.1_hiera_base_plus.pt"},
    "sam2_enc_large":     {"hf": "<sam2_enc>", "loader": "sam2_enc", "dim": 256, "layers": 24, "patch": 16,
                           "sam2_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2_ckpt": "sam2.1_hiera_large.pt"},
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
        temp = self.log_temp.exp().clamp(min=1.0, max=100.0)
        q_f, k_f, v_f = q.float(), k.float(), v.float()
        attn_logits = (q_f @ k_f.transpose(-2, -1)) * (temp.float() / d ** 0.5)
        attn_logits = attn_logits.clamp(-100.0, 100.0)
        attn_weights = attn_logits.softmax(dim=-1)
        attended = (attn_weights @ v_f).transpose(1, 2).reshape(B, S, D)
        attended = self.out_proj(attended.to(q.dtype))
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
# Learnable deformable alignment  (TRAINABLE)
# ---------------------------------------------------------------------------
class LearnableAlignment(nn.Module):
    """Predicts a per-patch (dx, dy) offset from ref tokens, then warps tgt
    tokens via bilinear grid_sample before CrossChangeAttention.

    Initialized to the identity (zero offsets) so training starts identical
    to the unaligned baseline. max_offset caps the warp in patch units.
    """

    def __init__(self, dim: int, max_offset: float = 2.0):
        super().__init__()
        self.max_offset = max_offset
        self.offset_net = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 2),
        )
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def forward(self, ref_tok: Tensor, tgt_tok: Tensor, h: int, w: int) -> Tensor:
        B, N, D = ref_tok.shape
        # offsets in patch units, bounded by max_offset
        offsets = self.offset_net(ref_tok).tanh() * self.max_offset  # [B, N, 2]
        offsets_hw = offsets.view(B, h, w, 2)  # (dy, dx) in patch units

        # Base sampling grid in [-1, 1] (grid_sample convention: (x, y) = (col, row))
        gy = torch.linspace(-1, 1, h, device=ref_tok.device)
        gx = torch.linspace(-1, 1, w, device=ref_tok.device)
        base_y, base_x = torch.meshgrid(gy, gx, indexing="ij")
        base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Normalize patch-unit offsets to [-1, 1] space
        norm_dx = offsets_hw[..., 0] * (2.0 / max(w - 1, 1))  # col shift
        norm_dy = offsets_hw[..., 1] * (2.0 / max(h - 1, 1))  # row shift
        delta = torch.stack([norm_dx, norm_dy], dim=-1)        # [B, h, w, 2]

        grid = (base_grid + delta).clamp(-1.0, 1.0)

        tgt_2d = tgt_tok.view(B, h, w, D).permute(0, 3, 1, 2)  # [B, D, h, w]
        aligned = F.grid_sample(tgt_2d, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        return aligned.permute(0, 2, 3, 1).view(B, N, D)


# ---------------------------------------------------------------------------
# CNN skip-connection branch  (TRAINABLE)
# ---------------------------------------------------------------------------
class CNNSkip(nn.Module):
    """Lightweight Siamese CNN that produces real high-resolution change
    features to replace the Bridge's bilinearly-upsampled tokens in SAM's
    high_res_features slots.

    Outputs:
        feat_s0  [B, 32, 4*target_h, 4*target_h]  (finest — replaces Bridge feat_s0)
        feat_s1  [B, 64, 2*target_h, 2*target_h]  (replaces Bridge feat_s1)

    Change signal = |CNN(ref) - CNN(tgt)| at each scale.
    """

    def __init__(self):
        super().__init__()
        # Shared Siamese stem — outputs full-res features (32 channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
        )
        # Stride-2 branch — outputs half-res features (64 channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64), nn.GELU(),
        )

    def forward(self, ref: Tensor, tgt: Tensor, target_h: int = 64) -> list:
        ref_full = self.stem(ref)                   # [B, 32, H, W]
        tgt_full = self.stem(tgt)
        diff_full = (ref_full - tgt_full).abs()     # [B, 32, H, W]

        ref_half = self.downsample(ref_full)        # [B, 64, H/2, W/2]
        tgt_half = self.downsample(tgt_full)
        diff_half = (ref_half - tgt_half).abs()     # [B, 64, H/2, W/2]

        # Resize to match what SAM decoder expects
        s0_h, s0_w = target_h * 4, target_h * 4
        s1_h, s1_w = target_h * 2, target_h * 2
        feat_s0 = F.interpolate(diff_full, (s0_h, s0_w), mode="bilinear", align_corners=False)
        feat_s1 = F.interpolate(diff_half, (s1_h, s1_w), mode="bilinear", align_corners=False)
        return [feat_s0, feat_s1]


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

    if loader == "sam2_enc":
        # SAM2 image encoder wrapped to expose the HF-style interface
        # (config + pixel_values + hidden_states).
        return _load_sam2_image_encoder(info)

    from transformers import AutoModel
    return AutoModel.from_pretrained(hf_id, output_hidden_states=True, trust_remote_code=True)


# ─────────────────────────────────────────────────────────────────
# SAM2 image encoder wrapper — exposes a DINO/HF-compatible API so
# ChangeDetector can swap it in for a DINO encoder.
# Output: vision_features (B, 256, H/16, W/16) -> reshape to (B, S, 256)
#         and prepend a fake CLS so .hidden_states[-1] looks like ViT.
# ─────────────────────────────────────────────────────────────────
class _Sam2EncoderConfig:
    def __init__(self, hidden_size, num_hidden_layers, patch_size, num_register_tokens=0):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens


class _Sam2EncoderOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _Sam2ImageEncoderWrapper(nn.Module):
    """Wraps SAM2's image_encoder so it looks like a DINO HF model.
    Forward(pixel_values) -> .hidden_states[-1] = (B, 1+S, 256)
    where the leading token is a fake CLS (mean-pooled features).
    """
    def __init__(self, image_encoder, hidden_size, patch_size, num_hidden_layers):
        super().__init__()
        self.image_encoder = image_encoder
        self.config = _Sam2EncoderConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            patch_size=patch_size,
        )

    def forward(self, pixel_values=None, **kwargs):
        x = pixel_values
        out = self.image_encoder(x)
        # vision_features: (B, 256, H/16, W/16) — flatten to tokens
        vf = out["vision_features"]
        B, D, Hp, Wp = vf.shape
        toks = vf.flatten(2).transpose(1, 2)              # (B, S, D)
        cls = toks.mean(dim=1, keepdim=True)              # (B, 1, D) — fake CLS
        last_hidden = torch.cat([cls, toks], dim=1)       # (B, 1+S, D)
        # ChangeDetector indexes hidden_states[i+1] for i in feature_layers
        # (default = quarter-points of num_hidden_layers). SAM2's Hiera does
        # not naturally expose those intermediate layers, so we just repeat
        # the final feature N+1 times — multi-scale aux supervision then
        # sees the same feature at all scales, which collapses to a single
        # auxiliary loss term (acceptable; not the focus of this experiment).
        N = self.config.num_hidden_layers
        hidden_states = tuple(last_hidden for _ in range(N + 1))
        return _Sam2EncoderOutput(hidden_states=hidden_states)


def _load_sam2_image_encoder(info: dict):
    """Build a SAM2 image_encoder + wrapper."""
    from sam2.build_sam import build_sam2
    cfg = info["sam2_cfg"]
    ckpt = info["sam2_ckpt"]
    ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join("checkpoints", ckpt)
    sam2 = build_sam2(cfg, ckpt_path)
    enc = sam2.image_encoder
    return _Sam2ImageEncoderWrapper(
        image_encoder=enc,
        hidden_size=info["dim"],
        patch_size=info["patch"],
        num_hidden_layers=info["layers"],
    )


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
        # Spatial alignment & high-res skip
        learnable_offset: bool = False,
        max_offset: float = 2.0,
        cnn_skip: bool = False,
        # Ablation: replace CrossChangeAttention with simple elementwise difference
        simple_diff: bool = False,
        # LoRA
        lora_rank: int = 0,
        lora_target: str = "none",
        lora_alpha: float = 16.0,
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

        # ── LoRA injection ────────────────────────────────────────────
        # Adds rank-r low-rank adapters to attention projections of the
        # frozen DINO encoder and/or SAM mask decoder. lora_target ∈
        # {none, dino, sam, both}. Default is none = no change.
        self.lora_rank = lora_rank
        self.lora_target = lora_target
        if lora_rank > 0 and lora_target in ("dino", "both"):
            n_dino = inject_lora(self.dino, DINO_LORA_TARGETS,
                                 rank=lora_rank, alpha=lora_alpha)
            print(f"  LoRA injected into DINO encoder: {n_dino} layers, rank={lora_rank}")
        if lora_rank > 0 and lora_target in ("sam", "both"):
            # If full FT was on, LoRA is redundant — disable to keep
            # parameter count honest. User gets a warning.
            if finetune_decoder:
                print("  ⚠ LoRA on SAM + --finetune_decoder both set; "
                      "disabling full FT to avoid double-training.")
                for p in self.sam_decoder.parameters():
                    p.requires_grad = False
                self.sam_decoder.eval()
            n_sam = inject_lora(self.sam_decoder, SAM_LORA_TARGETS,
                                rank=lora_rank, alpha=lora_alpha)
            print(f"  LoRA injected into SAM decoder: {n_sam} layers, rank={lora_rank}")

        # ── Bridge (TRAINABLE) — adapts based on decoder family ──────
        use_high_res = self.decoder_family in ("sam2", "sam3")
        self.bridge = Bridge(
            self.dino_dim, sam_dim=256, num_scales=len(self.feature_layers),
            use_high_res=use_high_res,
        )

        self.simple_diff = simple_diff

        # ── Learnable alignment (TRAINABLE, optional) ────────────────
        self.alignment = LearnableAlignment(self.dino_dim, max_offset) if learnable_offset else None

        # ── CNN skip (TRAINABLE, optional) ───────────────────────────
        self.cnn_skip_branch = CNNSkip() if (cnn_skip and use_high_res) else None

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

        if self.alignment is not None:
            tgt_tok = self.alignment(ref_tok, tgt_tok, ph, pw)

        if self.simple_diff:
            change_tokens = F.layer_norm(ref_tok - tgt_tok, [self.dino_dim])
            change_map = 1.0 - F.cosine_similarity(ref_tok, tgt_tok, dim=-1)
        else:
            change_tokens, change_map = self.cross_attn(ref_tok, tgt_tok, grid_h=ph, grid_w=pw)

        bridge_input = [
            (r - t) + change_tokens for r, t in zip(ref_multi, tgt_multi)
        ]

        th = tw = self.sam_target_size
        image_emb, dense_prompt, high_res_features, aux_change_maps = self.bridge(
            bridge_input, change_tokens, ph, pw, th, tw
        )

        if self.cnn_skip_branch is not None:
            high_res_features = self.cnn_skip_branch(ref, tgt, th)

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
        """All params with requires_grad=True. LoRA-aware: includes lora_A/B
        deltas that live inside the otherwise-frozen DINO and SAM modules."""
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def _named_lora_params(self):
        for name, p in self.named_parameters():
            if p.requires_grad and ("lora_A" in name or "lora_B" in name):
                yield name, p

    def param_groups(self, lr: float):
        # Bridge + cross_attn (skipped when simple_diff) + optional new modules at full LR.
        main = list(self.bridge.parameters())
        if not self.simple_diff:
            main += list(self.cross_attn.parameters())
        if self.alignment is not None:
            main += list(self.alignment.parameters())
        if self.cnn_skip_branch is not None:
            main += list(self.cnn_skip_branch.parameters())
        groups = [{"params": main, "lr": lr}]

        # SAM decoder full FT at decoder_lr_scale × lr (legacy).
        if self.finetune_decoder:
            sam_decoder_full = [p for n, p in self.sam_decoder.named_parameters()
                                if p.requires_grad and "lora_" not in n]
            if sam_decoder_full:
                groups.append({"params": sam_decoder_full,
                               "lr": lr * self.decoder_lr_scale})

        # LoRA params get their own group at the same LR scale as bridge —
        # standard practice for LoRA fine-tuning.
        lora_params = [p for _, p in self._named_lora_params()]
        if lora_params:
            groups.append({"params": lora_params, "lr": lr})
        return groups

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        self.sam_prompt_enc.eval()
        if not self.finetune_decoder:
            self.sam_decoder.eval()
        return self
