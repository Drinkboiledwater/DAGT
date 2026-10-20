# models.py
"""Model zoo for the derained-image quality assessment pipeline.

Sections
--------
1. Building blocks (UNetBlock, StemConv, ResidualBlock, AttnConv, FixedSobel)
2. UNetGenerator / PatchGANDiscriminator  (kept for pretraining compatibility)
3. RDE  : Rain Distortion Extractor      (encoder-decoder, 384-ch representation)
4. SDE  : Structural Distortion Extractor (Sobel + stems + res + attn, 512-ch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Building blocks
# =============================================================================
class UNetBlock(nn.Module):
    """Conv (down) / ConvTranspose (up) + optional BN + ReLU."""

    def __init__(self, in_c, out_c, down=True, use_bn=True):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
        else:
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class StemConv(nn.Module):
    """3x3 conv + BN + ReLU with configurable stride."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two 3x3 convs with residual connection."""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(x + out)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, c, reduction=16):
        super().__init__()
        hidden = max(c // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class AttnConv(nn.Module):
    """3x3 conv + BN + ReLU, modulated by channel attention, + residual."""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(True)
        self.attn = ChannelAttention(c)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return x + self.attn(out)


class FixedSobel(nn.Module):
    """Fixed 3x3 Sobel kernels: returns [gx, gy] edge responses (2-ch)."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]])
        kernel = torch.stack([kx, ky]).unsqueeze(1)  # (2, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        return F.conv2d(gray, self.kernel, padding=1)  # (B, 2, H, W)


# =============================================================================
# 2. Pretraining models (kept for backward compatibility)
# =============================================================================
class UNetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3):
        super().__init__()
        self.down1 = UNetBlock(in_c, 64, down=True, use_bn=False)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.down5 = UNetBlock(512, 512)

        self.up1 = UNetBlock(512, 512, down=False)
        self.up2 = UNetBlock(1024, 256, down=False)
        self.up3 = UNetBlock(512, 128, down=False)
        self.up4 = UNetBlock(256, 64, down=False)
        self.final = nn.ConvTranspose2d(128, out_c, 4, 2, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bott = self.down5(d4)

        u1 = self.up1(bott)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        out = self.final(torch.cat([u4, d1], dim=1))
        return torch.tanh(out), bott


class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# 3. RDE: Rain Distortion Extractor
# =============================================================================
class RDE(nn.Module):
    """Rain Distortion Extractor.

    Encoder: 4 downsampling stages (32, 64, 128, 256).
    Decoder: 3 upsampling stages (256, 128, 64) with U-Net skips,
             plus a 32-channel output stage.
    Output : 384-channel distortion representation
             (three decoder features -> 1x1 conv (128) -> concat,
             all aligned to the coarsest decoder resolution H/8).
    """

    def __init__(self, in_c=3):
        super().__init__()
        # ---- Encoder ----
        self.enc1 = UNetBlock(in_c, 32, down=True, use_bn=False)   # 32  @ H/2
        self.enc2 = UNetBlock(32, 64, down=True)                   # 64  @ H/4
        self.enc3 = UNetBlock(64, 128, down=True)                  # 128 @ H/8
        self.enc4 = UNetBlock(128, 256, down=True)                 # 256 @ H/16

        # ---- Decoder ----
        self.up1 = UNetBlock(256, 256, down=False)                 # 256 @ H/8
        self.up2 = UNetBlock(256 + 128, 128, down=False)           # 128 @ H/4
        self.up3 = UNetBlock(128 + 64, 64, down=False)             # 64  @ H/2
        self.out_stage = UNetBlock(64 + 32, 32, down=False)        # 32  @ H

        # ---- 1x1 projections to 128 channels ----
        self.proj1 = nn.Conv2d(256, 128, 1)
        self.proj2 = nn.Conv2d(128, 128, 1)
        self.proj3 = nn.Conv2d(64, 128, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                 # 32  @ H/2
        e2 = self.enc2(e1)                # 64  @ H/4
        e3 = self.enc3(e2)                # 128 @ H/8
        e4 = self.enc4(e3)                # 256 @ H/16

        # Decoder (skips)
        d1 = self.up1(e4)                 # 256 @ H/8
        d2 = self.up2(torch.cat([d1, e3], dim=1))  # 128 @ H/4
        d3 = self.up3(torch.cat([d2, e2], dim=1))  # 64  @ H/2
        _ = self.out_stage(torch.cat([d3, e1], dim=1))  # 32 @ H

        # Project & align to H/8
        p1 = self.proj1(d1)               # 128 @ H/8
        p2 = self.proj2(d2)               # 128 @ H/4
        p3 = self.proj3(d3)               # 128 @ H/2
        p2 = F.avg_pool2d(p2, 2)          # 128 @ H/8
        p3 = F.avg_pool2d(p3, 4)          # 128 @ H/8

        return torch.cat([p1, p2, p3], dim=1)   # 384 @ H/8


# =============================================================================
# 4. SDE: Structural Distortion Extractor
# =============================================================================
class SDE(nn.Module):
    """Structural Distortion Extractor.

    Pipeline:
        FixedSobel(3x3) -> [gx, gy]
        StemConv x3 (32/s1, 64/s2, 128/s2)
        ResidualBlock x3 (128)
        AttnConv x2 (conv + channel attention)
        Concat 4 intermediate representations (each 128 @ H/4) -> 512 ch.
    """

    def __init__(self):
        super().__init__()
        self.sobel = FixedSobel()

        self.stem1 = StemConv(2, 32, stride=1)    # 32  @ H
        self.stem2 = StemConv(32, 64, stride=2)   # 64  @ H/2
        self.stem3 = StemConv(64, 128, stride=2)  # 128 @ H/4

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)

        self.attn1 = AttnConv(128)
        self.attn2 = AttnConv(128)

    def forward(self, x):
        edge = self.sobel(x)      # 2   @ H
        s1 = self.stem1(edge)     # 32  @ H
        s2 = self.stem2(s1)       # 64  @ H/2
        s3 = self.stem3(s2)       # 128 @ H/4

        r1 = self.res1(s3)        # 128 @ H/4
        r2 = self.res2(r1)        # 128 @ H/4
        r3 = self.res3(r2)        # 128 @ H/4

        a1 = self.attn1(r3)       # 128 @ H/4
        a2 = self.attn2(a1)       # 128 @ H/4

        # Four intermediate representations (all 128 @ H/4)
        return torch.cat([r1, r2, r3, a2], dim=1)   # 512 @ H/4
