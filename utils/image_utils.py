"""Image utility helpers for GRB_Technology."""

import numpy as np
from PIL import Image
import cv2


def pad_to_aspect(img: Image.Image, target_w: int, target_h: int,
                   fill=(255, 255, 255)) -> Image.Image:
    """Pad image to target resolution without cropping."""
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    nw, nh = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (target_w, target_h), fill)
    canvas.paste(resized, ((target_w - nw) // 2, (target_h - nh) // 2))
    return canvas


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using rembg. Falls back to original if unavailable."""
    try:
        from rembg import remove
        return remove(img)
    except ImportError:
        return img


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological open+close to clean up a binary mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def alpha_blend(base: np.ndarray, overlay: np.ndarray,
                 alpha: np.ndarray) -> np.ndarray:
    """Alpha-blend two [H,W,3] uint8 arrays using a [H,W] float32 mask."""
    a = alpha[:, :, np.newaxis]
    return (overlay * a + base * (1 - a)).clip(0, 255).astype(np.uint8)


def make_grid(images: list, cols: int = 3) -> Image.Image:
    """Arrange PIL images in a grid for debugging / visualisation."""
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h), (30, 30, 30))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img.resize((w, h)), (c * w, r * h))
    return grid
