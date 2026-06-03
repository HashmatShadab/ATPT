"""
visualize_transform_factors.py

Visualize image transforms using transform-specific, interpretable factor lists.

Compared with a single generic SEVERITY value, this script uses separate factor
lists for each transform. Each list is ordered from identity/no-change to
increasingly stronger transformation.

For transforms that can move in two directions, the script separates them into
clear one-direction variants:
    brightness_dark, brightness_bright
    contrast_low, contrast_high
    saturation_low, saturation_high
    sharpness_low, sharpness_high
    gamma_bright, gamma_dark
    hue_negative, hue_positive

Quick start:
    python visualize_transform_factors.py --image your_image.jpg

Run only selected transforms:
    python visualize_transform_factors.py --image your_image.jpg \
        --transforms rotation gaussian_blur jpeg

Output:
    transform_factor_figs/rotation.png
    transform_factor_figs/gaussian_blur.png
    transform_factor_figs/jpeg.png

Identity/no-change values:
    noise          : 0
    blur           : 0
    rotation       : 0 degrees
    translation    : 0 pixels
    brightness     : factor 1.0
    contrast       : factor 1.0
    saturation     : factor 1.0
    sharpness      : factor 1.0
    gamma          : gamma 1.0
    hue            : shift 0.0
    posterize      : drop 0 bits
    solarize       : strength 0.0
    downsample     : factor 1.0
    jpeg           : compression drop 0, quality 100
"""

# ============================ CONFIG (edit me) ============================
IMAGE_PATH  = "check.JPEG"       # input image path
IMG_SIZE    = 224                    # resize to this; set to None/0 to keep original
NUM_ANCHORS = 1                      # anchor samples per factor level
OUT_DIR     = "visualize_anchors"
SEED        = 0                      # int seed for reproducibility; -1 = random
DPI         = 130
# ===========================================================================

import argparse
import io
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


# Default transforms. Each one is ordered from identity to stronger distortion.
TRANSFORMS = [
    "gaussian_noise", "uniform_noise",
    "brightness_dark", "brightness_bright",
    "contrast_low", "contrast_high",
    "saturation_low", "saturation_high",
    "sharpness_low", "sharpness_high",
    "gamma_bright", "gamma_dark",
    "hue_negative", "hue_positive",
    "gaussian_blur", "rotation", "translation",
    "posterize", "solarize", "downsample", "jpeg",
]

# ---------------------------------------------------------------------------
# Transform-specific factor lists.
# Important: every list starts with the identity/no-change value.
# The order is by increasing transform strength, not always by numerical value.
# For example, brightness_dark goes 1.0 -> 0.3 because darkness increases as
# the brightness factor decreases.
# ---------------------------------------------------------------------------
FACTOR_LEVELS: Dict[str, List[float]] = {
    # Additive noise in normalized image scale [0, 1].
    # 0.00 = identity; larger = more noise.
    "gaussian_noise": [0.00, 0.01, 0.03, 0.05, 0.08, 0.12],
    "uniform_noise":  [0.00, 0.01, 0.03, 0.05, 0.08, 0.12],

    # Brightness direct factor. 1.0 = identity.
    # Split into two monotonic visual directions.
    "brightness_dark":   [1.00, 0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.10],
    "brightness_bright": [1.00,  1.25, 1.50, 1.75, 2.00, 2.50, 3.0, 3.50],

    # Contrast direct factor. 1.0 = identity.
    "contrast_low":  [1.00, 0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.10],
    "contrast_high": [1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.0, 3.50],

    # Saturation direct factor. 1.0 = identity; 0.0 = grayscale.
    "saturation_low":  [1.00, 0.90, 0.75, 0.50, 0.25, 0.00],
    "saturation_high": [1.00, 1.50, 2.00, 2.50, 3.00, 4.00],

    # Sharpness direct factor. 1.0 = identity; 0.0 is heavily softened/blurred.
    "sharpness_low":  [1.00, 0.75, 0.50, 0.25, 0.10, 0.00],
    "sharpness_high": [1.00, 1.25, 1.50, 2.00, 3.00, 4.00],

    # Gamma direct value. 1.0 = identity.
    # In torchvision/PIL convention: gamma < 1 brightens; gamma > 1 darkens.
    "gamma_bright": [1.00, 0.90, 0.75, 0.60, 0.45, 0.30, 0.20, 0.10],
    "gamma_dark":   [1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00],

    # Hue shift. 0.0 = identity. torchvision valid range is [-0.5, 0.5].
    "hue_negative": [0.00, -0.03, -0.06, -0.10, -0.20, -0.30, -0.40, -0.50],
    "hue_positive": [0.00,  0.03,  0.06,  0.10,  0.20,  0.30, 0.40, 0.50],

    # Spatial transforms.
    # Blur sigma: 0 = identity; larger = more blur.
    # Rotation angle: 0 = identity; larger = more clockwise rotation.
    # Translation: 0 = identity; larger = more right-shift in pixels.
    "gaussian_blur": [0.00, 0.25, 0.50, 1.00, 2.00, 4.00, 6.00],
    "rotation":      [0, 15, 30, 40, 60, 90, 120],
    "translation":   [0, 2, 4, 8, 16, 32, 48, 64],

    # Quantization / compression.
    # Posterize: dropped low-order bits; 0 = identity; larger = fewer color bits.
    # Solarize: strength; 0 = identity; larger = lower threshold, more inversion.
    # Downsample: factor; 1.0 = identity; larger = stronger downsample/upsample.
    # JPEG: compression drop from quality 100; 0 = identity, larger = lower quality.
    "posterize":  [0, 1, 2, 3, 4, 5, 6, 7],
    "solarize":   [0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 0.90],
    "downsample": [1.00, 1.25, 1.50, 2.00, 4.00, 8.00, 12.00],
    "jpeg":       [0, 10, 25, 40, 60, 80, 90],
}

FACTOR_DESCRIPTIONS: Dict[str, str] = {
    "gaussian_noise": "Gaussian noise std in normalized [0,1] pixels; 0 = identity",
    "uniform_noise":  "Uniform noise max magnitude in normalized [0,1] pixels; 0 = identity",
    "brightness_dark":   "Brightness factor from identity to darker; 1.0 = identity",
    "brightness_bright": "Brightness factor from identity to brighter; 1.0 = identity",
    "contrast_low":  "Contrast factor from identity to lower contrast; 1.0 = identity",
    "contrast_high": "Contrast factor from identity to higher contrast; 1.0 = identity",
    "saturation_low":  "Saturation factor from identity to grayscale; 1.0 = identity, 0 = grayscale",
    "saturation_high": "Saturation factor from identity to stronger color; 1.0 = identity",
    "sharpness_low":  "Sharpness factor from identity to softer/blurrier; 1.0 = identity",
    "sharpness_high": "Sharpness factor from identity to sharper; 1.0 = identity",
    "gamma_bright": "Gamma from identity to brighter; 1.0 = identity, lower gamma brightens",
    "gamma_dark":   "Gamma from identity to darker; 1.0 = identity, higher gamma darkens",
    "hue_negative": "Hue shift from identity in the negative direction; 0 = identity",
    "hue_positive": "Hue shift from identity in the positive direction; 0 = identity",
    "gaussian_blur": "Gaussian blur sigma; 0 = identity/no blur",
    "rotation":      "Clockwise rotation angle in degrees; 0 = identity",
    "translation":   "Horizontal right translation in pixels; 0 = identity",
    "posterize":     "Dropped low-order bits; 0 = identity, larger = stronger quantization",
    "solarize":      "Solarize strength; 0 = identity, larger = stronger pixel inversion",
    "downsample":    "Downsampling factor; 1.0 = identity, larger = stronger resolution loss",
    "jpeg":          "JPEG compression drop from quality 100; 0 = identity, larger = stronger compression",
}

# Map split variants to the underlying torchvision operation.
BASE_TRANSFORM: Dict[str, str] = {
    "brightness_dark": "brightness",
    "brightness_bright": "brightness",
    "contrast_low": "contrast",
    "contrast_high": "contrast",
    "saturation_low": "saturation",
    "saturation_high": "saturation",
    "sharpness_low": "sharpness",
    "sharpness_high": "sharpness",
    "gamma_bright": "gamma",
    "gamma_dark": "gamma",
    "hue_negative": "hue",
    "hue_positive": "hue",
}


def base_transform(transform_type: str) -> str:
    return BASE_TRANSFORM.get(transform_type, transform_type)


def is_identity(transform_type: str, value: float) -> bool:
    base = base_transform(transform_type)
    if base in {"brightness", "contrast", "saturation", "sharpness", "gamma", "downsample"}:
        return abs(float(value) - 1.0) < 1e-12
    return abs(float(value)) < 1e-12


def format_factor(transform_type: str, value: float) -> str:
    """Human-readable row label for each factor level."""
    base = base_transform(transform_type)

    if is_identity(transform_type, value):
        if base == "downsample":
            return "identity / factor 1×"
        if base in {"brightness", "contrast", "saturation", "sharpness"}:
            return "identity / factor 1"
        if base == "gamma":
            return "identity / gamma 1"
        if base == "posterize":
            return "identity / drop 0 bits"
        if base == "jpeg":
            return "identity / quality 100"
        return "identity / 0"

    if base == "rotation":
        return f"{value:g}° clockwise"
    if base == "translation":
        return f"{value:g}px right"
    if base == "gaussian_blur":
        return f"σ={value:g}"
    if base == "posterize":
        drop = int(round(value))
        bits = max(1, 8 - drop)
        return f"drop {drop} bits / keep {bits} bits"
    if base == "solarize":
        threshold = 1.0 - float(value)
        return f"strength {value:g} / thr {threshold:g}"
    if base == "jpeg":
        quality = max(1, 100 - int(round(value)))
        return f"drop {int(value)} / quality {quality}"
    if base == "downsample":
        return f"factor {value:g}×"
    if base == "hue":
        return f"hue {value:+g}"
    if base in {"brightness", "contrast", "saturation", "sharpness"}:
        return f"factor {value:g}"
    if base == "gamma":
        return f"gamma {value:g}"
    return f"value {value:g}"


# ---------------------------------------------------------------------------
# Transform implementation.
# `factor` is now directly interpretable for each transform.
# ---------------------------------------------------------------------------
def transform_anchors(images: torch.Tensor,
                      transform_type: str,
                      factor: float,
                      num_anchors: int) -> torch.Tensor:
    B, C, H, W = images.shape
    NA = num_anchors
    N = NA * B
    device = images.device
    v = float(factor)
    base = base_transform(transform_type)

    if base == "gaussian_noise":
        noise = torch.randn(NA, B, C, H, W, device=device) * v
        return (images.unsqueeze(0) + noise).view(N, C, H, W).clamp(0, 1)

    if base == "uniform_noise":
        noise = (2 * torch.rand(NA, B, C, H, W, device=device) - 1) * v
        return (images.unsqueeze(0) + noise).view(N, C, H, W).clamp(0, 1)

    expanded = images.unsqueeze(0).expand(NA, -1, -1, -1, -1).reshape(N, C, H, W)

    def _per_sample(fn, params):
        out = torch.empty_like(expanded)
        for i in range(N):
            out[i] = fn(expanded[i:i + 1], params[i])[0]
        return out

    if base == "brightness":
        return _per_sample(TF.adjust_brightness, [max(0.0, v)] * N).clamp(0, 1)

    if base == "contrast":
        return _per_sample(TF.adjust_contrast, [max(0.0, v)] * N).clamp(0, 1)

    if base == "saturation":
        return _per_sample(TF.adjust_saturation, [max(0.0, v)] * N).clamp(0, 1)

    if base == "hue":
        hue_shift = max(-0.5, min(0.5, v))
        return _per_sample(TF.adjust_hue, [hue_shift] * N).clamp(0, 1)

    if base == "sharpness":
        return _per_sample(TF.adjust_sharpness, [max(0.0, v)] * N).clamp(0, 1)

    if base == "gamma":
        return _per_sample(TF.adjust_gamma, [max(1e-3, v)] * N).clamp(0, 1)

    if base == "gaussian_blur":
        if v <= 0:
            return expanded.clone()
        s = v
        ks = int(2 * math.ceil(3 * s) + 1)
        ks += (ks % 2 == 0)
        ks = max(ks, 3)
        return TF.gaussian_blur(expanded, kernel_size=[ks, ks], sigma=[s, s]).clamp(0, 1)

    if base == "rotation":
        out = torch.empty_like(expanded)
        for i in range(N):
            out[i] = TF.rotate(
                expanded[i:i + 1],
                angle=v,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )[0]
        return out.clamp(0, 1)

    if base == "translation":
        tx = int(round(v))
        ty = 0
        out = torch.empty_like(expanded)
        for i in range(N):
            out[i] = TF.affine(
                expanded[i:i + 1],
                angle=0.0,
                translate=[tx, ty],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )[0]
        return out.clamp(0, 1)

    if base == "posterize":
        drop = max(0, int(round(v)))
        bits = max(1, 8 - drop)
        u8 = (expanded.clamp(0, 1) * 255.0).to(torch.uint8)
        out = torch.empty_like(u8)
        for i in range(N):
            out[i] = TF.posterize(u8[i:i + 1], bits=bits)[0]
        return out.float() / 255.0

    if base == "solarize":
        # strength=0 -> threshold=1.0 -> identity; strength increases -> threshold lowers.
        strength = max(0.0, min(1.0, v))
        threshold = int((1.0 - strength) * 255)
        u8 = (expanded.clamp(0, 1) * 255.0).to(torch.uint8)
        out = torch.empty_like(u8)
        for i in range(N):
            out[i] = TF.solarize(u8[i:i + 1], threshold=threshold)[0]
        return out.float() / 255.0

    if base == "downsample":
        f = max(1.0, v)
        if f == 1.0:
            return expanded.clone()
        out = torch.empty_like(expanded)
        nh, nw = max(1, int(H / f)), max(1, int(W / f))
        for i in range(N):
            d = F.interpolate(expanded[i:i + 1], size=(nh, nw), mode="bilinear", align_corners=False)
            out[i] = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)[0]
        return out.clamp(0, 1)

    if base == "jpeg":
        # v is compression drop. v=0 -> quality=100; larger v -> lower quality.
        quality = max(1, min(100, 100 - int(round(v))))
        out = []
        for i in range(N):
            u8 = (expanded[i].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            pil = Image.fromarray(u8)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            dec = Image.open(buf).convert("RGB")
            arr = torch.from_numpy(np.array(dec)).permute(2, 0, 1).float() / 255.0
            out.append(arr)
        return torch.stack(out, 0).to(device)

    raise ValueError(f"Unknown transform_type: {transform_type}. Choose from: {TRANSFORMS}")


# ---------------------------------------------------------------------------
# I/O + plotting
# ---------------------------------------------------------------------------
def load_image(path: str, size: Optional[int] = None) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    if size is not None:
        pil = pil.resize((size, size), Image.BICUBIC)
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def show_factor_ranges(transforms: List[str]) -> None:
    print("\nTransform-specific factor ranges, ordered from identity to stronger effect:\n")
    for t in transforms:
        values = FACTOR_LEVELS[t]
        labels = ", ".join(format_factor(t, v) for v in values)
        print(f"- {t}: {FACTOR_DESCRIPTIONS[t]}")
        print(f"  levels: {labels}\n")


def visualize_transform(image: torch.Tensor,
                        transform: str,
                        factors: List[float],
                        num_anchors: int,
                        out_path: Path,
                        seed: Optional[int] = None,
                        dpi: int = 130) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    nrows = len(factors)
    ncols = num_anchors + 1  # first column is original

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.55 * ncols, 2.25 * nrows),
        squeeze=False,
    )

    for row, factor in enumerate(factors):
        if seed is not None:
            torch.manual_seed(seed + row)
            np.random.seed(seed + row)

        anchors = transform_anchors(image, transform, factor, num_anchors)

        axes[row, 0].imshow(image[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[row, 0].set_title("original" if row == 0 else "", fontsize=9)
        axes[row, 0].set_ylabel(format_factor(transform, factor), fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col in range(num_anchors):
            ax = axes[row, col + 1]
            ax.imshow(anchors[col].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"anchor {col + 1}", fontsize=9)

    fig.suptitle(
        f"{transform}\n{FACTOR_DESCRIPTIONS[transform]}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def visualize_all(image_path: str,
                  transforms: List[str],
                  num_anchors: int,
                  size: Optional[int],
                  out_dir: str,
                  seed: Optional[int],
                  dpi: int) -> None:
    image = load_image(image_path, size)
    out_root = Path(out_dir)

    show_factor_ranges(transforms)
    for idx, transform in enumerate(transforms):
        transform_seed = None if seed is None else seed + 1000 * idx
        out_path = out_root / f"{transform}.png"
        visualize_transform(
            image=image,
            transform=transform,
            factors=FACTOR_LEVELS[transform],
            num_anchors=num_anchors,
            out_path=out_path,
            seed=transform_seed,
            dpi=dpi,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to input image.")
    parser.add_argument("--transforms", nargs="+", default=TRANSFORMS, choices=TRANSFORMS,
                        help="Transforms to visualize. Default: all transforms.")
    parser.add_argument("--num_anchors", type=int, default=NUM_ANCHORS,
                        help="Number of anchors per factor level.")
    parser.add_argument("--size", type=int, default=IMG_SIZE,
                        help="Resize image to size x size. Use 0 to keep original size.")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Directory to save figures.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed. Use -1 for fully random.")
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    size = None if args.size == 0 else args.size
    seed = None if args.seed == -1 else args.seed

    visualize_all(
        image_path=args.image,
        transforms=args.transforms,
        num_anchors=args.num_anchors,
        size=size,
        out_dir=args.out_dir,
        seed=seed,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
