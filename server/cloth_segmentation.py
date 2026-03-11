"""
Cloth segmentation using U2Net for GRB_Technology.
Generates a binary foreground mask for the clothing item image.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from networks.u2net import U2NET

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_PATH = os.environ.get('U2NET_WEIGHTS', 'checkpoints/cloth_segm_u2net.pth')

_model = None

_transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def _get_model():
    global _model
    if _model is None:
        net = U2NET(in_ch=3, out_ch=1)
        if os.path.exists(WEIGHTS_PATH):
            state = torch.load(WEIGHTS_PATH, map_location='cpu')
            clean = {k.replace('module.', ''): v for k, v in state.items()}
            net.load_state_dict(clean, strict=False)
            print(f'[ClothSeg] Loaded: {WEIGHTS_PATH}')
        else:
            print(f'[ClothSeg] WARNING: weights not found at {WEIGHTS_PATH}')
        _model = net.eval().to(DEVICE)
    return _model


def segment_cloth(cloth_img: Image.Image, threshold: float = 0.5) -> np.ndarray:
    """
    Generate a binary mask for the clothing item.
    Returns numpy array [H, W] with values 0 or 255 (uint8).
    """
    model = _get_model()
    orig_w, orig_h = cloth_img.size
    inp = _transform(cloth_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(inp)[0].squeeze().cpu().numpy()
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    return (np.array(mask) >= int(threshold * 255)).astype(np.uint8) * 255


def get_cloth_tensor(cloth_img: Image.Image) -> torch.Tensor:
    """
    Return the cloth binary mask as a [1, 1, H, W] float tensor in [0, 1].
    """
    mask_np = segment_cloth(cloth_img)
    t = torch.from_numpy(mask_np / 255.0).float()
    return t.unsqueeze(0).unsqueeze(0)
