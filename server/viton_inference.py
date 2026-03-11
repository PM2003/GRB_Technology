"""
ViTON end-to-end inference for GRB_Technology.
Runs the full pipeline: preprocessing -> GMM -> TOM -> result image.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from networks.gmm import GMM
from networks.tom import TOM
from server.preprocessing import (
    resize_to_target, image_to_tensor,
    parse_map_to_tensor, keypoints_to_heatmap, build_person_agnostic
)
from server.cloth_segmentation import get_cloth_tensor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GMM_PATH = os.environ.get('GMM_WEIGHTS', 'checkpoints/gmm_final.pth')
TOM_PATH = os.environ.get('TOM_WEIGHTS', 'checkpoints/tom_final.pth')

_gmm = None
_tom = None


def _load():
    global _gmm, _tom
    if _gmm is None:
        gmm = GMM()
        if os.path.exists(GMM_PATH):
            gmm.load_state_dict(torch.load(GMM_PATH, map_location='cpu'), strict=False)
            print(f'[ViTON] GMM loaded from {GMM_PATH}')
        _gmm = gmm.eval().to(DEVICE)
    if _tom is None:
        tom = TOM()
        if os.path.exists(TOM_PATH):
            tom.load_state_dict(torch.load(TOM_PATH, map_location='cpu'), strict=False)
            print(f'[ViTON] TOM loaded from {TOM_PATH}')
        _tom = tom.eval().to(DEVICE)
    return _gmm, _tom


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).cpu().clamp(-1, 1)
    arr = ((t + 1) / 2 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def run_tryon(person_img: Image.Image,
              cloth_img: Image.Image,
              parse_array: np.ndarray = None,
              keypoints: list = None) -> Image.Image:
    """
    Full virtual try-on inference.

    person_img  : PIL RGB — full-body photo
    cloth_img   : PIL RGB — flat-lay clothing item
    parse_array : [H, W] integer body part labels (optional)
    keypoints   : list of (x, y, conf) from OpenPose (optional)

    Returns: PIL RGB result image
    """
    gmm, tom = _load()

    person_img = resize_to_target(person_img)
    cloth_img  = resize_to_target(cloth_img)

    person_t     = image_to_tensor(person_img).to(DEVICE)
    cloth_t      = image_to_tensor(cloth_img).to(DEVICE)
    cloth_mask_t = get_cloth_tensor(cloth_img).to(DEVICE)

    if parse_array is None:
        parse_array = np.zeros((256, 192), dtype=np.int32)
    if keypoints is None:
        keypoints = [(0, 0, 0.0)] * 18

    parse_t = parse_map_to_tensor(parse_array).to(DEVICE)
    pose_t  = keypoints_to_heatmap(keypoints).to(DEVICE)

    agnostic_t = build_person_agnostic(person_t, parse_t, pose_t)

    with torch.no_grad():
        warped_cloth, grid = gmm(agnostic_t, cloth_t)
        warped_mask = F.grid_sample(cloth_mask_t, grid, mode='nearest',
                                     padding_mode='zeros', align_corners=True)
        result, _, _ = tom(agnostic_t, warped_cloth, warped_mask)

    return _tensor_to_pil(result)
