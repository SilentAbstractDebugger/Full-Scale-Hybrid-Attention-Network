"""
models/unet.py
--------------
Full U-Net  (Ronneberger et al., 2015)
Encoder -> bottleneck -> decoder with skip connections.
Returns raw logits (apply sigmoid externally).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU  x2."""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),  nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    """MaxPool2d + DoubleConv – one encoder step."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.mp_conv(x)


class Up(nn.Module):
    """Bilinear upsample + concat skip + DoubleConv – one decoder step."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:],
                          mode="bilinear", align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Full U-Net for binary segmentation.

    Parameters
    ----------
    in_channels  : 3  (RGB)
    out_channels : 1  (binary)
    base_filters : 64 (channel width at first encoder level)
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super().__init__()
        f = base_filters
        # Encoder
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,     f*2)
        self.down2 = Down(f*2,   f*4)
        self.down3 = Down(f*4,   f*8)
        self.down4 = Down(f*8,   f*16)   # bottleneck
        # Decoder
        self.up1   = Up(f*16+f*8,  f*8)
        self.up2   = Up(f*8 +f*4,  f*4)
        self.up3   = Up(f*4 +f*2,  f*2)
        self.up4   = Up(f*2 +f,    f)
        # Output head
        self.outc  = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.inc(x)           # (B, 64,  256,256)
        e2 = self.down1(e1)        # (B, 128, 128,128)
        e3 = self.down2(e2)        # (B, 256,  64, 64)
        e4 = self.down3(e3)        # (B, 512,  32, 32)
        b  = self.down4(e4)        # (B,1024,  16, 16)  bottleneck
        d4 = self.up1(b,  e4)      # (B, 512,  32, 32)
        d3 = self.up2(d4, e3)      # (B, 256,  64, 64)
        d2 = self.up3(d3, e2)      # (B, 128, 128,128)
        d1 = self.up4(d2, e1)      # (B,  64, 256,256)
        return self.outc(d1)       # (B,   1, 256,256)  raw logits
