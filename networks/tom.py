"""
Try-On Module (TOM) for ViTON.
Blends the warped clothing onto the person using a UNet generator
and a learned composition mask.
Written independently for GRB_Technology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with reflection padding and instance normalisation."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch)
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.Identity() if first else nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch + skip_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        return self.block(torch.cat([x, skip], dim=1))


class TOM(nn.Module):
    """
    Try-On Module: UNet generator that takes the person agnostic representation
    and warped cloth, and outputs a rendered cloth image + composition mask.

    Input channels (26):
      22 — person agnostic (masked person image + body shape + pose heatmaps)
       3 — warped cloth
       1 — warped cloth mask

    Output channels (4):
       3 — rendered cloth texture
       1 — composition mask (alpha)
    """
    def __init__(self, in_ch=26, base_ch=64, depth=4, num_res=6):
        super().__init__()

        # Encoder
        enc_chs = [base_ch * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        ch = in_ch
        for i, out_ch in enumerate(enc_chs):
            self.encoders.append(EncoderBlock(ch, out_ch, first=(i == 0)))
            ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(*[ResBlock(ch) for _ in range(num_res)])

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_chs[i]
            out_ch  = enc_chs[i - 1] if i > 0 else base_ch
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, 4, 7)
        )

    def forward(self, person_agnostic, warped_cloth, warped_mask):
        x = torch.cat([person_agnostic, warped_cloth, warped_mask], dim=1)

        # Encode
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        # Decode with skips
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        out = self.out_conv(x)
        p_rendered  = torch.tanh(out[:, :3])
        m_composite = torch.sigmoid(out[:, 3:4])

        # Composite: mask blends warped cloth with rendered texture
        result = m_composite * warped_cloth + (1.0 - m_composite) * p_rendered
        return result, p_rendered, m_composite
