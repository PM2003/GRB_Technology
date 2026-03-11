"""
U2Net architecture for cloth segmentation.
Paper: U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection
Written independently for GRB_Technology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU block with optional dilation."""
    def __init__(self, in_ch, out_ch, dilate=1):
        super().__init__()
        pad = dilate
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RSUBlock(nn.Module):
    """
    Residual U-block (RSU) — the core building block of U2Net.
    `height` controls how many encoder-decoder levels are nested inside.
    """
    def __init__(self, in_ch, mid_ch, out_ch, height=7):
        super().__init__()
        self.height = height
        self.in_conv = ConvBnRelu(in_ch, out_ch)

        self.enc = nn.ModuleList([ConvBnRelu(out_ch, mid_ch)])
        for _ in range(1, height - 1):
            self.enc.append(ConvBnRelu(mid_ch, mid_ch))

        self.bottleneck = ConvBnRelu(mid_ch, mid_ch, dilate=2)

        self.dec = nn.ModuleList()
        for i in range(height - 2):
            self.dec.append(ConvBnRelu(mid_ch * 2, mid_ch))
        self.dec.append(ConvBnRelu(mid_ch * 2, out_ch))

    def forward(self, x):
        residual = self.in_conv(x)

        enc_feats = [self.enc[0](residual)]
        for i in range(1, len(self.enc)):
            enc_feats.append(self.enc[i](F.max_pool2d(enc_feats[-1], 2, ceil_mode=True)))

        d = self.bottleneck(enc_feats[-1])

        for i in range(len(self.dec) - 1, -1, -1):
            d = F.interpolate(d, size=enc_feats[i].shape[2:], mode='bilinear', align_corners=False)
            d = self.dec[len(self.dec) - 1 - i](torch.cat([d, enc_feats[i]], dim=1))

        return d + residual


class U2NET(nn.Module):
    """
    Full U2Net for salient object / cloth segmentation.
    Returns multi-scale side outputs used for deep supervision during training.
    """
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        # Encoder
        self.stage1 = RSUBlock(in_ch, 32,  64,  height=7)
        self.stage2 = RSUBlock(64,    32,  128, height=6)
        self.stage3 = RSUBlock(128,   64,  256, height=5)
        self.stage4 = RSUBlock(256,   128, 512, height=4)
        self.stage5 = RSUBlock(512,   256, 512, height=4)
        self.stage6 = RSUBlock(512,   256, 512, height=4)
        self.pool   = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Decoder
        self.dec5 = RSUBlock(1024, 256, 512, height=4)
        self.dec4 = RSUBlock(1024, 128, 256, height=4)
        self.dec3 = RSUBlock(512,  64,  128, height=5)
        self.dec2 = RSUBlock(256,  32,  64,  height=6)
        self.dec1 = RSUBlock(128,  16,  64,  height=7)

        # Side output heads (one per scale)
        self.side1 = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.fuse  = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Encoder pass
        e1 = self.stage1(x)
        e2 = self.stage2(self.pool(e1))
        e3 = self.stage3(self.pool(e2))
        e4 = self.stage4(self.pool(e3))
        e5 = self.stage5(self.pool(e4))
        e6 = self.stage6(self.pool(e5))

        # Decoder pass with skip connections
        def upsample(src, ref):
            return F.interpolate(src, size=ref.shape[2:], mode='bilinear', align_corners=False)

        d5 = self.dec5(torch.cat([upsample(e6, e5), e5], dim=1))
        d4 = self.dec4(torch.cat([upsample(d5, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([upsample(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([upsample(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([upsample(d2, e1), e1], dim=1))

        # Side outputs upsampled to input resolution
        def side_out(feat, head):
            return F.interpolate(head(feat), size=(h, w), mode='bilinear', align_corners=False)

        s1 = side_out(d1, self.side1)
        s2 = side_out(d2, self.side2)
        s3 = side_out(d3, self.side3)
        s4 = side_out(d4, self.side4)
        s5 = side_out(d5, self.side5)
        s6 = side_out(e6, self.side6)

        fused = self.fuse(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))
        return (torch.sigmoid(fused), torch.sigmoid(s1), torch.sigmoid(s2),
                torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(s5), torch.sigmoid(s6))
