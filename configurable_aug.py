"""YAML-driven augmentation pipeline for per-dataset fair-comparison training.

Usage
-----
    from configurable_aug import load_config, ConfigurableAugmenter, ConfigurableValTransform

    cfg = load_config("configs/datasets/levir_cd.yaml")
    train_tf = ConfigurableAugmenter(cfg)
    val_tf   = ConfigurableValTransform(cfg)

Each YAML config declares an *ordered* list of augmentation ops. Ops NOT in the
list are NOT applied — so per-dataset configs can enforce a fair match to
each benchmark's published method (see configs/datasets/README.md).
"""

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter, ImageEnhance

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────
def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────
# Tensor helpers
# ─────────────────────────────────────────────────────────────────
def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _normalize(t: torch.Tensor, mean, std) -> torch.Tensor:
    for c, m, s in zip(t, mean, std):
        c.sub_(m).div_(s)
    return t


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────
# Augmenter
# ─────────────────────────────────────────────────────────────────
class ConfigurableAugmenter:
    """Executes a YAML-declared augmentation pipeline on a (ref, tgt, mask) triple."""

    def __init__(self, config: Dict[str, Any]):
        self.img_size = int(config.get("img_size", 512))
        norm = config.get("normalization") or {}
        self.mean = tuple(norm.get("mean", IMAGENET_MEAN))
        self.std = tuple(norm.get("std", IMAGENET_STD))
        self.ops: List[Dict[str, Any]] = list(config.get("augmentations") or [])

    def __call__(
        self, ref: Image.Image, tgt: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure mask is single-channel L
        if mask.mode != "L":
            mask = mask.convert("L")

        for op in self.ops:
            ref, tgt, mask = self._apply(op, ref, tgt, mask)

        # Final fit-to-img_size
        if ref.size != (self.img_size, self.img_size):
            ref = ref.resize((self.img_size, self.img_size), Image.BILINEAR)
            tgt = tgt.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        ref_t = _normalize(_to_tensor(ref), self.mean, self.std)
        tgt_t = _normalize(_to_tensor(tgt), self.mean, self.std)
        mask_t = _mask_to_tensor(mask)
        return ref_t, tgt_t, mask_t

    # ---------- dispatch ----------
    def _apply(self, op, ref, tgt, mask):
        name = op["op"]
        prob = float(op.get("prob", 1.0))
        if random.random() > prob:
            return ref, tgt, mask

        if name == "random_flip":
            return self._flip(op, ref, tgt, mask)
        if name == "random_rotate":
            return self._rotate(op, ref, tgt, mask)
        if name == "random_crop":
            return self._crop(op, ref, tgt, mask)
        if name == "scale_random_crop":
            return self._scale_crop(op, ref, tgt, mask)
        if name == "random_scale":
            return self._scale(op, ref, tgt, mask)
        if name == "temporal_swap":
            return tgt, ref, mask
        if name == "photometric_distortion":
            return self._pmd(op, ref, tgt, mask)
        if name == "color_jitter":
            return self._color_jitter(op, ref, tgt, mask)
        if name == "gaussian_blur":
            return self._blur(op, ref, tgt, mask)
        if name == "one_of":
            chosen = random.choice(op["ops"])
            # if 'prob' was already applied above, pass through with prob=1.0
            sub = dict(chosen)
            sub.setdefault("prob", 1.0)
            return self._apply(sub, ref, tgt, mask)
        raise ValueError(f"Unknown augmentation op: {name!r}")

    # ---------- op implementations ----------
    @staticmethod
    def _flip(op, ref, tgt, mask):
        d = op.get("direction", "horizontal")
        if d == "horizontal":
            return (ref.transpose(Image.FLIP_LEFT_RIGHT),
                    tgt.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT))
        if d == "vertical":
            return (ref.transpose(Image.FLIP_TOP_BOTTOM),
                    tgt.transpose(Image.FLIP_TOP_BOTTOM),
                    mask.transpose(Image.FLIP_TOP_BOTTOM))
        raise ValueError(f"bad flip direction: {d}")

    @staticmethod
    def _rotate(op, ref, tgt, mask):
        deg = float(op.get("degrees", 0))
        if op.get("discrete", False):
            k = random.choice([1, 2, 3])  # 90, 180, 270
            return (ref.rotate(90 * k, expand=False),
                    tgt.rotate(90 * k, expand=False),
                    mask.rotate(90 * k, expand=False))
        angle = random.uniform(-deg, deg)
        return (ref.rotate(angle, resample=Image.BILINEAR, expand=False),
                tgt.rotate(angle, resample=Image.BILINEAR, expand=False),
                mask.rotate(angle, resample=Image.NEAREST, expand=False))

    @staticmethod
    def _crop(op, ref, tgt, mask):
        size = int(op["size"])
        w, h = ref.size
        if w < size or h < size:
            # pad up reflectively
            pad_w = max(0, size - w)
            pad_h = max(0, size - h)
            if pad_w or pad_h:
                def _pad(img, mode):
                    arr = np.array(img)
                    if arr.ndim == 3:
                        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
                    else:
                        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode=mode)
                    return Image.fromarray(arr)
                ref = _pad(ref, "reflect")
                tgt = _pad(tgt, "reflect")
                mask = _pad(mask, "constant")
            w, h = ref.size

        cat_max_ratio = float(op.get("cat_max_ratio", 1.0))
        # Try up to 10 times to get a crop whose dominant-label ratio < cat_max_ratio.
        for _ in range(10):
            i = random.randint(0, max(0, h - size))
            j = random.randint(0, max(0, w - size))
            box = (j, i, j + size, i + size)
            mcrop = mask.crop(box)
            if cat_max_ratio < 1.0:
                arr = np.array(mcrop)
                vals, counts = np.unique(arr, return_counts=True)
                if counts.max() / counts.sum() < cat_max_ratio:
                    break
            else:
                break
        return ref.crop(box), tgt.crop(box), mask.crop(box)

    @classmethod
    def _scale_crop(cls, op, ref, tgt, mask):
        """BIT/ChangeFormer-style: scale (random factor in range) then crop."""
        size = int(op["size"])
        lo, hi = op.get("scale_range", [1.0, 1.2])
        s = random.uniform(lo, hi)
        w, h = ref.size
        new_w, new_h = int(w * s), int(h * s)
        if new_w >= size and new_h >= size:
            ref = ref.resize((new_w, new_h), Image.BILINEAR)
            tgt = tgt.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        return cls._crop({"op": "random_crop", "size": size}, ref, tgt, mask)

    @staticmethod
    def _scale(op, ref, tgt, mask):
        scales = op.get("scales", [1.0])
        s = random.choice(scales)
        w, h = ref.size
        return (ref.resize((int(w * s), int(h * s)), Image.BILINEAR),
                tgt.resize((int(w * s), int(h * s)), Image.BILINEAR),
                mask.resize((int(w * s), int(h * s)), Image.NEAREST))

    @staticmethod
    def _pmd(op, ref, tgt, mask):
        """mmcv-style PhotoMetricDistortion. Applied per-image independently
        (matches mmseg default behaviour)."""
        b_delta = int(op.get("brightness_delta", 32))
        c_lo, c_hi = op.get("contrast_range", [0.5, 1.5])
        s_lo, s_hi = op.get("saturation_range", [0.5, 1.5])
        h_delta = int(op.get("hue_delta", 18))

        def apply_one(img):
            arr = np.array(img).astype(np.float32)
            # brightness
            if random.random() < 0.5:
                arr = arr + random.uniform(-b_delta, b_delta)
            # contrast (pre)
            contrast_first = random.random() < 0.5
            if contrast_first and random.random() < 0.5:
                arr = arr * random.uniform(c_lo, c_hi)
            # saturation
            hsv = _rgb_to_hsv(arr)
            if random.random() < 0.5:
                hsv[..., 1] *= random.uniform(s_lo, s_hi)
            # hue (degrees, 0..360)
            if random.random() < 0.5:
                hsv[..., 0] = (hsv[..., 0] + random.uniform(-h_delta, h_delta)) % 360
            arr = _hsv_to_rgb(hsv)
            if not contrast_first and random.random() < 0.5:
                arr = arr * random.uniform(c_lo, c_hi)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        return apply_one(ref), apply_one(tgt), mask

    @staticmethod
    def _color_jitter(op, ref, tgt, mask):
        """torchvision-style ColorJitter. If independent=True, draws separate
        params for ref and tgt (matches ChangeFormer)."""
        b = float(op.get("brightness", 0))
        c = float(op.get("contrast", 0))
        s = float(op.get("saturation", 0))
        h = float(op.get("hue", 0))
        independent = bool(op.get("independent", True))

        def draw_params():
            return (
                random.uniform(max(0, 1 - b), 1 + b),
                random.uniform(max(0, 1 - c), 1 + c),
                random.uniform(max(0, 1 - s), 1 + s),
                random.uniform(-h, h),
            )

        def apply_one(img, params):
            br, co, sa, hu = params
            img = ImageEnhance.Brightness(img).enhance(br)
            img = ImageEnhance.Contrast(img).enhance(co)
            img = ImageEnhance.Color(img).enhance(sa)
            if abs(hu) > 1e-4:
                arr = np.array(img).astype(np.float32)
                hsv = _rgb_to_hsv(arr)
                hsv[..., 0] = (hsv[..., 0] + hu * 360) % 360
                img = Image.fromarray(np.clip(_hsv_to_rgb(hsv), 0, 255).astype(np.uint8))
            return img

        p_ref = draw_params()
        p_tgt = draw_params() if independent else p_ref
        return apply_one(ref, p_ref), apply_one(tgt, p_tgt), mask

    @staticmethod
    def _blur(op, ref, tgt, mask):
        lo, hi = op.get("radius_range", [0.0, 1.0])
        independent = bool(op.get("independent", True))
        def one(img):
            r = random.uniform(lo, hi)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        if independent:
            return one(ref), one(tgt), mask
        r = random.uniform(lo, hi)
        ref = ref.filter(ImageFilter.GaussianBlur(radius=r))
        tgt = tgt.filter(ImageFilter.GaussianBlur(radius=r))
        return ref, tgt, mask


# ─────────────────────────────────────────────────────────────────
# RGB ↔ HSV (vectorised)
# ─────────────────────────────────────────────────────────────────
def _rgb_to_hsv(rgb):
    rgb = rgb / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1); cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 1e-6
    # red is max
    mR = mask & (cmax == r)
    h[mR] = ((g[mR] - b[mR]) / delta[mR]) % 6
    mG = mask & (cmax == g)
    h[mG] = ((b[mG] - r[mG]) / delta[mG]) + 2
    mB = mask & (cmax == b)
    h[mB] = ((r[mB] - g[mB]) / delta[mB]) + 4
    h = (h * 60) % 360
    s = np.where(cmax > 0, delta / np.where(cmax > 0, cmax, 1), 0.0)
    v = cmax * 255.0
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v / 255.0 * s
    hp = (h / 60.0) % 6
    x = c * (1 - np.abs((hp % 2) - 1))
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    cond = [(hp < 1), (hp < 2) & (hp >= 1), (hp < 3) & (hp >= 2),
            (hp < 4) & (hp >= 3), (hp < 5) & (hp >= 4), (hp >= 5)]
    for i, m in enumerate(cond):
        if i == 0: r[m], g[m], b[m] = c[m], x[m], 0
        elif i == 1: r[m], g[m], b[m] = x[m], c[m], 0
        elif i == 2: r[m], g[m], b[m] = 0, c[m], x[m]
        elif i == 3: r[m], g[m], b[m] = 0, x[m], c[m]
        elif i == 4: r[m], g[m], b[m] = x[m], 0, c[m]
        else:        r[m], g[m], b[m] = c[m], 0, x[m]
    m_off = v / 255.0 - c
    return (np.stack([r + m_off, g + m_off, b + m_off], axis=-1)) * 255.0


# ─────────────────────────────────────────────────────────────────
# ValTransform (deterministic — reads normalization from config)
# ─────────────────────────────────────────────────────────────────
class ConfigurableValTransform:
    def __init__(self, config: Dict[str, Any]):
        self.img_size = int(config.get("test_img_size", config.get("img_size", 512)))
        norm = config.get("normalization") or {}
        self.mean = tuple(norm.get("mean", IMAGENET_MEAN))
        self.std = tuple(norm.get("std", IMAGENET_STD))

    def __call__(self, ref, tgt, mask):
        if mask.mode != "L":
            mask = mask.convert("L")
        if ref.size != (self.img_size, self.img_size):
            ref = ref.resize((self.img_size, self.img_size), Image.BILINEAR)
            tgt = tgt.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        ref_t = _normalize(_to_tensor(ref), self.mean, self.std)
        tgt_t = _normalize(_to_tensor(tgt), self.mean, self.std)
        mask_t = _mask_to_tensor(mask)
        return ref_t, tgt_t, mask_t
