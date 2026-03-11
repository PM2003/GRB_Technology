"""
Preprocessing pipeline for GRB_Technology virtual try-on.
Handles resizing, body parsing one-hot encoding, pose heatmaps,
and building the person agnostic representation for the ViTON model.
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

TARGET_H = 256
TARGET_W = 192

_normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def resize_to_target(img: Image.Image) -> Image.Image:
    """Resize PIL image to model input resolution (192 x 256)."""
    return img.resize((TARGET_W, TARGET_H), Image.LANCZOS)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Normalise a PIL image and return a [1, 3, H, W] tensor."""
    return _normalize(img).unsqueeze(0)


def parse_map_to_tensor(parse_array: np.ndarray,
                         num_classes: int = 20) -> torch.Tensor:
    """
    Convert an integer label map to a one-hot tensor.
    parse_array : [H, W]  integer class labels
    Returns     : [1, num_classes, H, W]
    """
    h, w = parse_array.shape
    one_hot = np.zeros((num_classes, h, w), dtype=np.float32)
    for cls in range(num_classes):
        one_hot[cls] = (parse_array == cls).astype(np.float32)
    return torch.from_numpy(one_hot).unsqueeze(0)


def keypoints_to_heatmap(keypoints: list,
                          h: int = TARGET_H,
                          w: int = TARGET_W,
                          sigma: float = 6.0,
                          num_kp: int = 18) -> torch.Tensor:
    """
    Convert a list of OpenPose keypoints to Gaussian heatmaps.
    keypoints : list of (x, y, confidence) tuples
    Returns   : [1, num_kp, H, W]
    """
    maps = np.zeros((num_kp, h, w), dtype=np.float32)
    xs = np.arange(w)
    ys = np.arange(h)
    gx, gy = np.meshgrid(xs, ys)
    for i, (x, y, conf) in enumerate(keypoints[:num_kp]):
        if conf > 0.1:
            maps[i] = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma ** 2))
    return torch.from_numpy(maps).unsqueeze(0)


def build_person_agnostic(person_t: torch.Tensor,
                           parse_t: torch.Tensor,
                           pose_t: torch.Tensor) -> torch.Tensor:
    """
    Build the 22-channel person agnostic representation.
    Removes the clothing region so the network must use the provided garment.

    Channels:
      0-2  : person image with clothing area zeroed out (3 ch)
      3    : binary body shape mask                     (1 ch)
      4-21 : pose Gaussian heatmaps                    (18 ch)

    Clothing classes in LIP parsing: upper-clothes=5, dress=6, coat=7
    """
    cloth_idx = [5, 6, 7]
    cloth_mask = parse_t[:, cloth_idx].sum(dim=1, keepdim=True).clamp(0, 1)
    masked_person = person_t * (1.0 - cloth_mask)
    body_shape = (1.0 - parse_t[:, 0:1])   # foreground = not background
    return torch.cat([masked_person, body_shape, pose_t], dim=1)
