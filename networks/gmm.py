"""
Geometric Matching Module (GMM) for ViTON.
Warps the clothing item to match the body shape of the person using learned TPS parameters.
Written independently for GRB_Technology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Extracts deep appearance features from an input image."""
    def __init__(self, in_ch=3, base_ch=64, num_layers=5):
        super().__init__()
        layers, ch = [], in_ch
        for i in range(num_layers):
            out_ch = base_ch * (2 ** min(i, 3))
            layers += [
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class CorrelationLayer(nn.Module):
    """
    Computes cross-correlation between person and cloth feature maps.
    This guides the spatial transformation learning.
    """
    def forward(self, f_person, f_cloth):
        b, c, h, w = f_person.shape
        fp = f_person.view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]
        fc = f_cloth.view(b, c, -1)                     # [B, C, HW]
        corr = torch.bmm(fp, fc).view(b, h * w, h, w)  # [B, HW, H, W]
        return F.relu(corr)


class TPSGridGenerator(nn.Module):
    """
    Predicts TPS (Thin Plate Spline) offset parameters from correlation features
    and generates a dense sampling grid for warping the cloth image.
    """
    def __init__(self, feat_dim=512, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        n_ctrl = grid_size * grid_size
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_ctrl * 2)
        )
        # Initialise to identity (zero offsets)
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, feat, img_h=256, img_w=192):
        B = feat.size(0)
        offsets = self.regressor(feat).view(B, self.grid_size, self.grid_size, 2)

        # Build regular control-point grid in [-1, 1]
        gy = torch.linspace(-1, 1, self.grid_size, device=feat.device)
        gx = torch.linspace(-1, 1, self.grid_size, device=feat.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, gs, gs, 2]

        moved = (base_grid + 0.1 * offsets.clamp(-1, 1))  # [B, gs, gs, 2]

        # Upsample to full image resolution
        moved_t = moved.permute(0, 3, 1, 2)  # [B, 2, gs, gs]
        full_grid = F.interpolate(moved_t, size=(img_h, img_w),
                                   mode='bilinear', align_corners=True)
        return full_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]


class GMM(nn.Module):
    """
    Full Geometric Matching Module.
    Inputs:
      person_repr : [B, 22, H, W] — agnostic person representation
      cloth       : [B,  3, H, W] — flat-lay clothing image
    Outputs:
      warped_cloth : [B, 3, H, W]
      grid         : [B, H, W, 2]  (sampling grid)
    """
    def __init__(self, in_person=22, in_cloth=3, base_ch=64, grid_size=5):
        super().__init__()
        self.person_enc = FeatureExtractor(in_ch=in_person, base_ch=base_ch)
        self.cloth_enc  = FeatureExtractor(in_ch=in_cloth,  base_ch=base_ch)
        self.corr = CorrelationLayer()
        self.compress = nn.Sequential(
            nn.Conv2d(64 * 64, 512, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.tps = TPSGridGenerator(feat_dim=512, grid_size=grid_size)

    def forward(self, person_repr, cloth):
        fp = self.person_enc(person_repr)
        fc = self.cloth_enc(cloth)
        corr = self.corr(fp, fc)
        feat = self.compress(corr).view(person_repr.size(0), -1)
        grid = self.tps(feat, img_h=person_repr.size(2), img_w=person_repr.size(3))
        warped = F.grid_sample(cloth, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        return warped, grid
