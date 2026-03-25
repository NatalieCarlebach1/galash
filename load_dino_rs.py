"""
Load KevinCha DINOv2 Remote-Sensing weights into HuggingFace Dinov2Model.

Supports:
    dinov2-vit-base-remote-sensing      (ViT-B/16, 768d, 12 layers)
    dinov2-vit-large-remote-sensing*    (ViT-L/14, 1024d, 24 layers)
    dinov2-vit-small-remote-sensing*    (ViT-S/16, 384d, 12 layers)

Usage:
    from load_dino_rs import load_dinov2_remote_sensing
    model = load_dinov2_remote_sensing("KevinCha/dinov2-vit-base-remote-sensing")
    # → standard HF Dinov2Model with output_hidden_states support
"""

import json
import torch
from huggingface_hub import hf_hub_download
from transformers import Dinov2Model, Dinov2Config


# Map from KevinCha config keys to Dinov2Config
_ARCH_MAP = {
    # (embed_dim, depth, num_heads, patch_size)
    (384, 12, 6, 16): "small",
    (768, 12, 12, 16): "base",
    (1024, 24, 16, 14): "large",
}


def _convert_weights(src: dict, num_layers: int, embed_dim: int) -> dict:
    """Convert DINOv2 native checkpoint keys → HuggingFace Dinov2Model keys."""
    dst = {}

    # Embeddings
    dst["embeddings.cls_token"] = src["cls_token"]
    dst["embeddings.mask_token"] = src["mask_token"]
    dst["embeddings.patch_embeddings.projection.weight"] = src["patch_embed.proj.weight"]
    dst["embeddings.patch_embeddings.projection.bias"] = src["patch_embed.proj.bias"]
    dst["embeddings.position_embeddings"] = src["pos_embed"]

    # Register tokens (if present)
    if "register_tokens" in src:
        dst["embeddings.register_tokens"] = src["register_tokens"]

    # Final layernorm
    dst["layernorm.weight"] = src["norm.weight"]
    dst["layernorm.bias"] = src["norm.bias"]

    # Encoder blocks
    for i in range(num_layers):
        sp = f"blocks.{i}"
        dp = f"encoder.layer.{i}"

        # Split fused QKV → separate Q, K, V
        qkv_w = src[f"{sp}.attn.qkv.weight"]  # [3*D, D]
        qkv_b = src[f"{sp}.attn.qkv.bias"]     # [3*D]
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=0)

        dst[f"{dp}.attention.attention.query.weight"] = q_w
        dst[f"{dp}.attention.attention.query.bias"] = q_b
        dst[f"{dp}.attention.attention.key.weight"] = k_w
        dst[f"{dp}.attention.attention.key.bias"] = k_b
        dst[f"{dp}.attention.attention.value.weight"] = v_w
        dst[f"{dp}.attention.attention.value.bias"] = v_b

        # Attention output projection
        dst[f"{dp}.attention.output.dense.weight"] = src[f"{sp}.attn.proj.weight"]
        dst[f"{dp}.attention.output.dense.bias"] = src[f"{sp}.attn.proj.bias"]

        # MLP
        dst[f"{dp}.mlp.fc1.weight"] = src[f"{sp}.mlp.fc1.weight"]
        dst[f"{dp}.mlp.fc1.bias"] = src[f"{sp}.mlp.fc1.bias"]
        dst[f"{dp}.mlp.fc2.weight"] = src[f"{sp}.mlp.fc2.weight"]
        dst[f"{dp}.mlp.fc2.bias"] = src[f"{sp}.mlp.fc2.bias"]

        # Layer norms
        dst[f"{dp}.norm1.weight"] = src[f"{sp}.norm1.weight"]
        dst[f"{dp}.norm1.bias"] = src[f"{sp}.norm1.bias"]
        dst[f"{dp}.norm2.weight"] = src[f"{sp}.norm2.weight"]
        dst[f"{dp}.norm2.bias"] = src[f"{sp}.norm2.bias"]

        # Layer scale (DINOv2-RS may not have these — use ones as default)
        if f"{sp}.ls1.gamma" in src:
            dst[f"{dp}.layer_scale1.lambda1"] = src[f"{sp}.ls1.gamma"]
            dst[f"{dp}.layer_scale2.lambda1"] = src[f"{sp}.ls2.gamma"]

    return dst


def load_dinov2_remote_sensing(
    repo_id: str = "KevinCha/dinov2-vit-base-remote-sensing",
) -> Dinov2Model:
    """Download and load a DINOv2-RS model as a standard HF Dinov2Model.

    Args:
        repo_id: HuggingFace repo, e.g.
            "KevinCha/dinov2-vit-base-remote-sensing"
            "KevinCha/dinov2-vit-large-remote-sensing-50ep"

    Returns:
        Dinov2Model with output_hidden_states support.
    """
    # Load config
    cfg_path = hf_hub_download(repo_id, "config.json")
    with open(cfg_path) as f:
        raw_cfg = json.load(f)

    embed_dim = raw_cfg["embed_dim"]
    depth = raw_cfg["depth"]
    num_heads = raw_cfg["num_heads"]
    patch_size = raw_cfg["patch_size"]
    img_size = raw_cfg.get("img_size", 224)
    num_registers = raw_cfg.get("num_register_tokens", 0)

    arch_key = (embed_dim, depth, num_heads, patch_size)
    arch_name = _ARCH_MAP.get(arch_key, f"custom-{embed_dim}d-{depth}L")
    print(f"Loading DINOv2-RS {arch_name}: dim={embed_dim}, layers={depth}, "
          f"heads={num_heads}, patch={patch_size}, registers={num_registers}")

    # Build HF config
    hf_config = Dinov2Config(
        hidden_size=embed_dim,
        num_hidden_layers=depth,
        num_attention_heads=num_heads,
        intermediate_size=embed_dim * int(raw_cfg.get("mlp_ratio", 4)),
        patch_size=patch_size,
        image_size=img_size,
        num_register_tokens=0,  # HF Dinov2Model may not support registers; skip them
        hidden_act="gelu",
        qkv_bias=raw_cfg.get("qkv_bias", True),
        output_hidden_states=True,
    )

    # Create empty model
    model = Dinov2Model(hf_config)

    # Load weights — try the standalone backbone .pth first, fallback to safetensors
    try:
        # Find the backbone .pth file
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        pth_files = [f for f in files if f.endswith(".pth") and "backbone" in f.lower()]
        if pth_files:
            pth_path = hf_hub_download(repo_id, pth_files[0])
            src_sd = torch.load(pth_path, map_location="cpu", weights_only=True)
            print(f"  loaded backbone from {pth_files[0]}")
        else:
            raise FileNotFoundError("No .pth backbone file")
    except Exception:
        # Fallback: load safetensors and extract student.backbone.*
        from safetensors.torch import load_file
        st_path = hf_hub_download(repo_id, "model.safetensors")
        full_sd = load_file(st_path)
        prefix = "student.backbone."
        src_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
        print(f"  loaded from model.safetensors (extracted {len(src_sd)} backbone keys)")

    # Convert keys
    hf_sd = _convert_weights(src_sd, depth, embed_dim)

    # Handle missing layer_scale (fill with ones if HF model expects them)
    model_sd = model.state_dict()
    for k in model_sd:
        if "layer_scale" in k and k not in hf_sd:
            hf_sd[k] = torch.ones_like(model_sd[k])

    # Load
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    if missing:
        print(f"  missing keys: {missing}")
    if unexpected:
        print(f"  unexpected keys: {unexpected}")
    if not missing and not unexpected:
        print(f"  all {len(hf_sd)} keys loaded successfully")

    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="KevinCha/dinov2-vit-base-remote-sensing")
    p.add_argument("--test", action="store_true", help="run a forward pass test")
    args = p.parse_args()

    model = load_dinov2_remote_sensing(args.repo)
    print(f"\nConfig: hidden_size={model.config.hidden_size}, "
          f"patch_size={model.config.patch_size}, "
          f"layers={model.config.num_hidden_layers}")

    if args.test:
        model.eval()
        x = torch.randn(1, 3, 518, 518)
        out = model(pixel_values=x, output_hidden_states=True)
        print(f"\nForward pass test (input {list(x.shape)}):")
        print(f"  last_hidden_state: {list(out.last_hidden_state.shape)}")
        print(f"  hidden_states: {len(out.hidden_states)} layers, "
              f"each {list(out.hidden_states[0].shape)}")
        n_patches = out.last_hidden_state.shape[1]
        print(f"  {n_patches} output tokens "
              f"(1 CLS + {model.config.num_register_tokens} registers + patches)")
